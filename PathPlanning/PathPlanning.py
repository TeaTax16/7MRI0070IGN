# import required libraries to run the code
from importlib.metadata import entry_points
import logging
import os
from typing import Annotated, Optional
import vtk
import SimpleITK as sitk
import numpy as np
import slicer
from slicer.util import getNode
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (parameterNodeWrapper, WithinRange)
from slicer import vtkMRMLLabelMapVolumeNode, vtkMRMLMarkupsFiducialNode, vtkMRMLUnitNode

# boiler plate code for acknowledgements and description of the code
class PathPlanning(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("PathPlanning")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Examples")]
        self.parent.dependencies = []
        self.parent.contributors = ["Takrim Titas (King's College London)"]
        self.parent.helpText = _("""
This is the completed PathPlanning algorithm used as part of the assessment criteria for the Image Guided Navigation module
""")
        self.parent.acknowledgementText = _("""
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.  Rachel Sparks has modified this, 
for part of Image-guide Navigation for Robotics taught through King's College London.
""")

# assign data types to each variables in the GUI
@parameterNodeWrapper
class PathPlanningParameterNode:
    inputTargetVolume: vtkMRMLLabelMapVolumeNode # Target Points must be in this label map
    inputCriticalVolume: vtkMRMLLabelMapVolumeNode # Trajectory line must avoid this label map
    inputEntryFiducials: vtkMRMLMarkupsFiducialNode # Fiducials containing Entry Points
    inputTargetFiducials: vtkMRMLMarkupsFiducialNode # Fiducials containing Target points
    lengthThreshold: Annotated[float, WithinRange(0, 100)] = 0 # Maximum length threshold
    outputFiducials: vtkMRMLMarkupsFiducialNode # Filtered Target points that are in the target label map

# mostly boiler plate code to link the inputs from the GUI to the 3D Slicer
class PathPlanningWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    def __init__(self, parent=None) -> None:
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None

    def setup(self) -> None:
        ScriptedLoadableModuleWidget.setup(self)
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/PathPlanning.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)
        uiWidget.setMRMLScene(slicer.mrmlScene)
        self.logic = PathPlanningLogic()
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)
        self.ui.applyButton.clicked.connect(self.onApplyButton)
        self.initializeParameterNode()

    def cleanup(self) -> None:
        self.removeObservers()

    def enter(self) -> None:
        self.initializeParameterNode()

    def exit(self) -> None:
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)

    def onSceneStartClose(self, caller, event) -> None:
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        if self.parent.isEntered:
            self.initializeParameterNode()

    # ensure all input nodes are correctly assigned and ready for processing
    def initializeParameterNode(self) -> None:
        self.setParameterNode(self.logic.getParameterNode())
        if not self._parameterNode.inputTargetVolume:
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLLabelMapVolumeNode")
            if firstVolumeNode:
                self._parameterNode.inputTargetVolume = firstVolumeNode

        if not self._parameterNode.inputCriticalVolume:
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLLabelMapVolumeNode")
            if firstVolumeNode:
                self._parameterNode.inputCriticalVolume = firstVolumeNode

        if not self._parameterNode.inputTargetFiducials:
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLMarkupsFiducialNode")
            if firstVolumeNode:
                self._parameterNode.inputTargetFiducials = firstVolumeNode

        if not self._parameterNode.inputEntryFiducials:
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLMarkupsFiducialNode")
            if firstVolumeNode:
                self._parameterNode.inputEntryFiducials = firstVolumeNode

    def setParameterNode(self, inputParameterNode: Optional[PathPlanningParameterNode]) -> None:
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
            self._checkCanApply()

    def _checkCanApply(self, caller=None, event=None) -> None:
        if self._parameterNode and self._parameterNode.inputTargetVolume and self._parameterNode.inputCriticalVolume and self._parameterNode.inputEntryFiducials and self._parameterNode.inputTargetFiducials:
            self.ui.applyButton.toolTip = _("Compute output volume")
            self.ui.applyButton.enabled = True
        else:
            self.ui.applyButton.toolTip = _("Select all input nodes")
            self.ui.applyButton.enabled = False
    
    # initialise Path Planning Logic code and ensure all necessary data is available 
    def onApplyButton(self) -> None:
        self.logic = PathPlanningLogic()
        self.logic.SetEntryPoints(self.ui.inputEntryFiducialSelector.currentNode())
        self.logic.SetTargetPoints(self.ui.inputTargetFiducialSelector.currentNode())
        self.logic.SetOutputPoints(self.ui.outputFiducialSelector.currentNode())
        self.logic.SetInputTargetImage(self.ui.inputTargetVolumeSelector.currentNode())
        self.logic.SetInputCriticalVolume(self.ui.inputCriticalVolumeSelector.currentNode())
        self.logic.SetLengthThreshold(self.ui.lengthThreshold.value)
        # filter target points by checking if they are in the target structure
        complete = self.logic.run()

        # once target points are filtered, create line nodes and filter through those
        if complete:
            criticalVolume = self.ui.inputCriticalVolumeSelector.currentNode()
            PickPointsMatrix().GetLinesE_T(self.logic.myEntries, self.logic.myOutputs, "lineNodes", criticalVolume, self.logic.lengthThreshold)

        # once lines have been filtered, hide the unnecessary fiducials to clear the slicer interface
        if self.ui.inputTargetFiducialSelector.currentNode():
            self.ui.inputTargetFiducialSelector.currentNode().SetDisplayVisibility(False)
            self.ui.inputEntryFiducialSelector.currentNode().SetDisplayVisibility(False)
            self.ui.outputFiducialSelector.currentNode().SetDisplayVisibility(False)

        if not complete:
            print('I encountered an error')
            

class PathPlanningLogic(ScriptedLoadableModuleLogic):
    # import the data inputted by the user to be used for the logic
    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self)
    def getParameterNode(self):
        return PathPlanningParameterNode(super().getParameterNode())
    def SetEntryPoints(self, entryNode):
        self.myEntries = entryNode
    def SetTargetPoints(self, targetNode):
        self.myTargets = targetNode
    def SetInputTargetImage(self, imageNode):
        if (self.hasImageData(imageNode)):
            self.myTargetImage = imageNode
    def SetInputCriticalVolume(self, criticalVolumeNode):
        self.myCriticalVolume = criticalVolumeNode
    def SetOutputPoints(self, outputNode):
        self.myOutputs = outputNode
    def SetLengthThreshold(self, lengthThreshold):
        self.lengthThreshold = lengthThreshold
    def hasImageData(self, volumeNode):
        if not volumeNode:
            logging.debug('hasImageData failed: no volume node')
            return False
        if volumeNode.GetImageData() is None:
            logging.debug('hasImageData failed: no image data in volume node')
            return False
        return True
    
    # validate the data added to make ensure integrity and completeness before the processing begins
    def isValidInputOutputData(self, inputTargetVolumeNode, inputCriticalVolumeNode, inputTargetFiducialsNode, inputEntryFiducialsNode, outputFiducialsNode):
        if not inputTargetVolumeNode:
            logging.debug('isValidInputOutputData failed: no input target volume node defined')
            return False
        if not inputCriticalVolumeNode:
            logging.debug('isValidInputOutputData failed: no input critical volume node defined')
            return False
        if not inputTargetFiducialsNode:
            logging.debug('isValidInputOutputData failed: no input target fiducials node defined')
            return False
        if not inputEntryFiducialsNode:
            logging.debug('isValidInputOutputData failed: no input entry fiducials node defined')
            return False
        if not outputFiducialsNode:
            logging.debug('isValidInputOutputData failed: no output fiducials node defined')
            return False
        if inputTargetFiducialsNode.GetID() == outputFiducialsNode.GetID():
            logging.debug('isValidInputOutputData failed: input and output fiducial nodes are the same. Create a new output to avoid this error.')
            return False
        return True
    def run(self):
        if not self.isValidInputOutputData(self.myTargetImage, self.myCriticalVolume, self.myTargets, self.myEntries, self.myOutputs):
            slicer.util.errorDisplay('Not all inputs are set.')
            return False
        if not self.hasImageData(self.myTargetImage):
            raise ValueError("Input target volume is not appropriately defined.")
        pointPicker = PickPointsMatrix()
        pointPicker.run(self.myTargetImage,  self.myTargets, self.myOutputs)
        return True


class PickPointsMatrix(ScriptedLoadableModuleLogic):
    def __init__(self):
        self.trajectory = []
        
    # function to filter target points by checking which are in the target structure and which are not
    def run(self, inputVolume, inputFiducials, outputFiducials):
        outputFiducials.RemoveAllControlPoints()
        # create a transformation matrix to convert nodes from RAS to IJK
        mat = vtk.vtkMatrix4x4()
        inputVolume.GetRASToIJKMatrix(mat)
        transform = vtk.vtkTransform()
        transform.SetMatrix(mat)
        # loop through each target point and convert to IJK
        for x in range(inputFiducials.GetNumberOfControlPoints()):
            pos = [0, 0, 0]
            inputFiducials.GetNthControlPointPosition(x, pos)
            ind = transform.TransformPoint(pos)
            # check pixel value for each target point and add to new MarkupsFiducial node holding valid target points
            pixelValue = inputVolume.GetImageData().GetScalarComponentAsDouble(int(ind[0]), int(ind[1]), int(ind[2]), 0)
            if pixelValue == 1:
                outputFiducials.AddControlPoint(vtk.vtkVector3d(pos[0], pos[1], pos[2]))

    def GetLinesE_T(self, inputEntryFiducials, outputFiducials, groupName, CriticalVolume, lengthThreshold):
        # initialise required variables and components for the line filtering
        lineNode = slicer.mrmlScene.CreateNodeByClass("vtkMRMLMarkupsLineNode")
        mat = vtk.vtkMatrix4x4()
        CriticalVolume.GetRASToIJKMatrix(mat)
        imageData = CriticalVolume.GetImageData()
        maxMinDistance = -np.inf
        bestEntryPoint = None
        bestTargetPoint = None
        bestEntryIndex = -1
        bestTargetIndex = -1
        bestLength = -1
        bestMinDistance = -1
    
        # start timer to measure efficiency of code
        import time
        startTime = time.time()
        print("Time Started")

        # calculate and output total number of valid entry and target points for validation
        numEntryPoints = inputEntryFiducials.GetNumberOfControlPoints()
        numTargetPoints = outputFiducials.GetNumberOfControlPoints()
        print(f"Number of entry points: {numEntryPoints}, Number of target points: {numTargetPoints}")
        # use danielsson distance map filter to create the distance map for the critical volume
        distanceMap = self.computeDistanceMap(CriticalVolume)
        # begin loop for going through each entry and target point pair
        for entryIndex in range(numEntryPoints):
            entryPointRAS = [0, 0, 0]
            inputEntryFiducials.GetNthControlPointPosition(entryIndex, entryPointRAS)
            # for validation, output which entry point is being tested
            print("==============================")
            print(f"E{entryIndex+1}: {entryPointRAS}")
            print("")
            for targetIndex in range(numTargetPoints):
                targetPointRAS = [0, 0, 0]
                outputFiducials.GetNthControlPointPosition(targetIndex, targetPointRAS)
                # for validation, output which target point is being tested
                print(f"T{targetIndex+1}: {targetPointRAS}")

                # calculate euclidean distance between entry target pair and compare to threshold
                # if above threshold exit loop and test next entry and target point pair
                length = np.linalg.norm(np.array(targetPointRAS) - np.array(entryPointRAS))
                if length > lengthThreshold:
                    print(f"    FAIL: Length {length:.2f}")
                    continue
                print(f"    PASS: Length {length:.2f}")

                # if within threshold, check intersection with critical structure
                # if collision is detected, exit loop and test next entry and target point pair
                if self.checkIntersection(entryPointRAS, targetPointRAS, mat, imageData):
                    print("    FAIL: Intersection")
                    continue
                print("    PASS: No Intersection")
                
                # use the distance map to calculate euclidean distance between line and critical structure
                # return the minimum distance 
                minDistance = self.computeMinimumDistance(entryPointRAS, targetPointRAS, mat, distanceMap)
                print(f"    Distance to Critical Structure: {minDistance:.2f}")

                # if all tests are passed, update the details for the best line node
                if minDistance > maxMinDistance:
                    maxMinDistance = minDistance
                    bestEntryPoint = entryPointRAS
                    bestTargetPoint = targetPointRAS
                    bestEntryIndex = entryIndex
                    bestTargetIndex = targetIndex
                    bestLength = length
                    bestMinDistance = minDistance

        if bestEntryPoint and bestTargetPoint:
            # for the best line, extract the entry and target points into a line node for visualisation
            lineNode.RemoveAllControlPoints()
            lineNode.AddControlPoint(vtk.vtkVector3d(bestEntryPoint[0], bestEntryPoint[1], bestEntryPoint[2]))
            lineNode.AddControlPoint(vtk.vtkVector3d(bestTargetPoint[0], bestTargetPoint[1], bestTargetPoint[2]))
            lineNode.SetAttribute("LineGroup", groupName)
            slicer.mrmlScene.AddNode(lineNode)
            # and then create a MarkupsFiducial node with the same entry and target point which will be sent to ROS
            bestFiducialNode = slicer.mrmlScene.CreateNodeByClass("vtkMRMLMarkupsFiducialNode")
            bestFiducialNode.SetName("Trajectory")
            slicer.mrmlScene.AddNode(bestFiducialNode)
            bestFiducialNode.AddControlPoint(bestEntryPoint)
            bestFiducialNode.AddControlPoint(bestTargetPoint)
            # for validation, output the details for the best line in the console
            print("==============================")
            print(f"Best Line Details:")
            print(f"E {bestEntryIndex+1}, {bestEntryPoint}")
            print(f"T {bestTargetIndex+1}, {bestTargetPoint}")
            print(f"L: {bestLength:.2f}")
            print(f"D: {bestMinDistance:.2f}")
            print("")

        # end the timer and output to console
        stopTime = time.time()
        print(f"Processing completed in {stopTime-startTime:.2f} seconds")
        
    def checkIntersection(self, startPointRAS, endPointRAS, mat, imageData):
        '''
        function to check if a line between an entry point and target point intersects with critical structure.
        '''
        # create a transform matrix to convert points from RAS to IJK and apply to the points being tested
        startIJK = [0, 0, 0, 1]
        endIJK = [0, 0, 0, 1]
        mat.MultiplyPoint(startPointRAS + [1], startIJK)
        mat.MultiplyPoint(endPointRAS + [1], endIJK)
        # find the direction vector
        directionVector = np.array(endIJK[:3]) - np.array(startIJK[:3])
        # find the line length
        distance = np.linalg.norm(directionVector)
        # check the line is valid
        if distance == 0:
            return False
        
        directionVector /= distance
        # create sample points on the line to test voxel value
        for i in np.arange(0, distance, 0.1):
            samplePointIJK = np.array(startIJK[:3]) + directionVector * i
            voxelValue = imageData.GetScalarComponentAsDouble(int(samplePointIJK[0]), int(samplePointIJK[1]), int(samplePointIJK[2]), 0)
            # if voxel value is greater than 0, intersection is true
            if voxelValue != 0:
                # return true for the intersection flag
                return True
        return False

    def computeDistanceMap(self, volume):
        '''
        function to create a distance map for a define volume using Danielsson Distance Map Filter algorithm
        '''
        import sitkUtils
        # pull critical volume from slicer for sitk
        sitkImage = sitkUtils.PullVolumeFromSlicer(volume.GetID())
        # convert volume to an sitk Image with a supported pixel format
        sitkImage = sitk.Cast(sitkImage, sitk.sitkUInt8)  
        # create distance map using DanielssonDistanceMapImageFilter function
        distanceFilter = sitk.DanielssonDistanceMapImageFilter()
        distanceMap = distanceFilter.Execute(sitkImage)
        # convert to a NumPy Array where each voxel represents a distance from critical structure
        return sitk.GetArrayFromImage(distanceMap)

    def computeMinimumDistance(self, startPointRAS, endPointRAS, mat, distanceMap):
        '''
        function to calculate the minimum distance between a line and a critical distance using a premade distance map
        '''
        # create a transform matrix to convert points from RAS to IJK and apply to the points being tested
        startIJK = [0, 0, 0, 1]
        endIJK = [0, 0, 0, 1]
        mat.MultiplyPoint(startPointRAS + [1], startIJK)
        mat.MultiplyPoint(endPointRAS + [1], endIJK)
        startIJK = startIJK[:3]
        endIJK = endIJK[:3]
        # find the direction vector
        directionVector = np.array(endIJK) - np.array(startIJK)
        # find the line length
        totalDistance = np.linalg.norm(directionVector)
        # exclude invalid lines
        if totalDistance == 0:
            return np.inf
        directionVector /= totalDistance
        # initialise minimum distance variable
        minDistance = np.inf
        # sample the line at specific intervals
        for i in np.arange(0, totalDistance, 0.1):
            samplePointIJK = np.array(startIJK) + directionVector * i
            # exclude invalid indexes
            ijk = np.round(samplePointIJK).astype(int)
            if np.any(ijk < 0) or np.any(ijk >= distanceMap.shape):
                continue
            # check if distance of sample point from critical structure is the minimum for the line and update accordingly
            voxelDistance = distanceMap[ijk[2], ijk[1], ijk[0]]
            if voxelDistance < minDistance:
                minDistance = voxelDistance
        return minDistance

# class used to run test code on slicer 
class PathPlanningTest(ScriptedLoadableModuleTest):
    # import necessary library for testing
    import vtk
    # tests that should be run when test is used on slicer by the user
    def runTest(self):
        self.test_LengthThreshold()
        self.test_TargetFilter()
        self.test_CollisionDetection()
        self.test_DistanceMap()

    
    def setUp(self):
        '''function to clear everything on the Slicer Scene'''
        slicer.mrmlScene.Clear(0)
        print("Clearing Scene")

    def test_LengthThreshold(self):
        '''
        test to verify length threshold checking works
        '''
        self.setUp()  
        print("Length Threshold Test Started")
        # create nodes that will be used to test length thresholding
        testPoint_1 = slicer.vtkMRMLMarkupsFiducialNode()
        slicer.mrmlScene.AddNode(testPoint_1)
        testPoint_1.AddControlPoint(0, 0, 0)
        
        testPoint_2 = slicer.vtkMRMLMarkupsFiducialNode()
        slicer.mrmlScene.AddNode(testPoint_2)
        testPoint_2.AddControlPoint(5, 5, 5)
        
        testPoint_3 = slicer.vtkMRMLMarkupsFiducialNode()
        slicer.mrmlScene.AddNode(testPoint_3)
        testPoint_3.AddControlPoint(100, 100, 100)
        
        # calculate line length, one that will be below the threshold and one that will be above threshold
        length_1_2 = np.linalg.norm(np.array(testPoint_1.GetNthControlPointPosition(0)) - np.array(testPoint_2.GetNthControlPointPosition(0)))
        length_1_3 = np.linalg.norm(np.array(testPoint_1.GetNthControlPointPosition(0)) - np.array(testPoint_3.GetNthControlPointPosition(0)))
        lengthThreshold = 50
        print("Length Threshold: 50")
        
        if length_1_2 < lengthThreshold:
            print(f"Length from P1 to P2 is below the threshold: {length_1_2:.2f}")
        if length_1_3 > lengthThreshold:
            print(f"Length from P1 to P3 is above the threshold: {length_1_3:.2f}")
        
        print("Length Threshold Test Completed")
        print("")
        
    def test_TargetFilter(self):
        '''
        test to check target point filtering
        '''
        self.setUp()
        print("Target Filter Test Started")
        # import Target Volume from user direction -CHANGE THIS TO YOUR DIRECTORY
        targetLabelMap = slicer.util.loadVolume("/Users/takrim/Library/Mobile Documents/com~apple~CloudDocs/University/MSc Healthcare Technologies/Image Guided Navigation/Submissions/Image Data/BrainParcellation/r_hippo.nii.gz")
        # check if target label map is imported correctly
        if not targetLabelMap:
            print("Target Label Map not uploaded")
        else:
            print("Target Label Map imported successfully")
            # create points that are inside and outside the target structure
            insidePoint = slicer.vtkMRMLMarkupsFiducialNode()
            slicer.mrmlScene.AddNode(insidePoint)
            insidePoint.AddControlPoint(146, 82, 133)
            outsidePoint = slicer.vtkMRMLMarkupsFiducialNode()
            slicer.mrmlScene.AddNode(outsidePoint)
            outsidePoint.AddControlPoint(166,74,138)

            # create node that will hold points that are within the target volume
            returnedPoints = slicer.vtkMRMLMarkupsFiducialNode()
            slicer.mrmlScene.AddNode(returnedPoints)
            # run target point filtering
            PickPointsMatrix().run(targetLabelMap, insidePoint, returnedPoints)
            # positive test
            if returnedPoints.GetNumberOfControlPoints() == 1:
                print("Inside point detected correctly")
            returnedPoints.RemoveAllControlPoints()
            PickPointsMatrix().run(targetLabelMap, outsidePoint, returnedPoints)
            # negative test
            if returnedPoints.GetNumberOfControlPoints() == 0:
                print("Outside point removed correctly")
            print("Target Filter Test Completed")
            print("")
            
    def test_CollisionDetection(self):
        '''
        test to check intersection with critical volume node
        '''
        self.setUp()
        print("Collision Detection Test Started")
        # import Critical Volume from user direction -CHANGE THIS TO YOUR DIRECTORY
        criticalLabelMap = slicer.util.loadVolume("/Users/takrim/Library/Mobile Documents/com~apple~CloudDocs/University/MSc Healthcare Technologies/Image Guided Navigation/Submissions/Image Data/BrainParcellation/vessels.nii.gz")
        # check if critical label map is imported correctly
        if not criticalLabelMap:
            print("Critical Label Map not uploaded")
        else:
            print("Critical Label Map imported successfully")
            # create coordinate of lines that are and are not in the critical structure
            nonIntersectingLineStart = [210.625, 71.609, 123.702]
            nonIntersectingLineEnd = [116, 90, 33]
            intersectingLineStart = [207.931, 58.976, 136.701]
            intersectingLineEnd = [158, 90, 128]
            # transform matrix to convert from RAS to IJK coordinate freame
            mat = vtk.vtkMatrix4x4()
            criticalLabelMap.GetRASToIJKMatrix(mat)
            imageData = criticalLabelMap.GetImageData()
            
            # test for collision with non intersecting line
            collisionDetected = PickPointsMatrix().checkIntersection(nonIntersectingLineStart, nonIntersectingLineEnd, mat, imageData)
            if not collisionDetected:
                print("Non-intersecting line detected correctly (no collision)")
            else:
                print("Non-intersecting line detected incorrectly (collision detected)")
            # test for collision with intersecting line
            collisionDetected = PickPointsMatrix().checkIntersection(intersectingLineStart, intersectingLineEnd, mat, imageData)
            if collisionDetected:
                print("Intersecting line detected correctly (collision detected)")
            else:
                print("Intersecting line detected incorrectly (no collision)")
            print("Collision Detection Test Completed")
            print("")
        

    def test_DistanceMap(self):
        '''
        test to check if distance map works as expected
        '''
        self.setUp()
        print("Distance Map Test Started")
        # import Critical Volume from user direction -CHANGE THIS TO YOUR DIRECTORY
        criticalLabelMap = slicer.util.loadVolume("/Users/takrim/Library/Mobile Documents/com~apple~CloudDocs/University/MSc Healthcare Technologies/Image Guided Navigation/Submissions/Image Data/BrainParcellation/vessels.nii.gz")
        # check if critical label map is imported correctly
        if not criticalLabelMap:
            print("Critical Label Map not uploaded")
        else:
            print("Critical Label Map imported successfully")
            # create a distance map for the critical label map
            distanceMap = PickPointsMatrix().computeDistanceMap(criticalLabelMap)
            # initialise two points- one for known surface point and one for known outside of surface point
            surfacePoint = (127, 165, 188)
            outsidePoint = (127, 184, 188)
            # create transform matrix to convert coordinated from RAS to IJK and apply to points creates
            mat = self.vtk.vtkMatrix4x4()
            criticalLabelMap.GetRASToIJKMatrix(mat)
            surfaceIJK = list(mat.MultiplyPoint(surfacePoint + (1,))[:3])
            outsideIJK = list(mat.MultiplyPoint(outsidePoint + (1,))[:3])
            
            # Round and convert to integer as they are voxel
            surfaceIJK = [int(round(coord)) for coord in surfaceIJK]
            outsideIJK = [int(round(coord)) for coord in outsideIJK]
            # calculate the distance for surface point which should be 0 and point outside structure which is sqrt(5) in this case
            surfaceDistance = distanceMap[surfaceIJK[2], surfaceIJK[1], surfaceIJK[0]]
            outsideDistance = distanceMap[outsideIJK[2], outsideIJK[1], outsideIJK[0]]
            
            print(f"Surface Distance should be 0, got {surfaceDistance}")
            print(f"Outside Distance should be 12, got {outsideDistance}")

            print("Distance Map Test Completed")