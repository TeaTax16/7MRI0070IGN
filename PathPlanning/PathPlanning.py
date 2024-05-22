from importlib.metadata import entry_points
import logging
import os
from typing import Annotated, Optional
import vtk
import numpy as np
import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (parameterNodeWrapper, WithinRange)
from slicer import vtkMRMLLabelMapVolumeNode, vtkMRMLMarkupsFiducialNode, vtkMRMLUnitNode

class PathPlanning(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("PathPlanning")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Examples")]
        self.parent.dependencies = []
        self.parent.contributors = ["Takrim Titas (King's College London)"]
        self.parent.helpText = _("""
This is the start of the path planning script with some helpers already implemented
""")
        self.parent.acknowledgementText = _("""
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.  Rachel Sparks has modified this, 
for part of Image-guide Navigation for Robotics taught through King's College London.
""")

@parameterNodeWrapper
class PathPlanningParameterNode:

    inputTargetVolume: vtkMRMLLabelMapVolumeNode # Target Points must be in this label map
    inputCriticalVolume: vtkMRMLLabelMapVolumeNode # Trajectory line must avoid this label map
    inputEntryFiducials: vtkMRMLMarkupsFiducialNode # Fiducials containing Entry Points
    inputTargetFiducials: vtkMRMLMarkupsFiducialNode # Fiducials containing Target points
    lengthThreshold: Annotated[float, WithinRange(0, 500)] = 100 # Maximum length threshold
    outputFiducials: vtkMRMLMarkupsFiducialNode # Filtered Target points that are in the target label map


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
    
    def onApplyButton(self) -> None:
        self.logic = PathPlanningLogic()
        self.logic.SetEntryPoints(self.ui.inputEntryFiducialSelector.currentNode())
        self.logic.SetTargetPoints(self.ui.inputTargetFiducialSelector.currentNode())
        self.logic.SetOutputPoints(self.ui.outputFiducialSelector.currentNode())
        self.logic.SetInputTargetImage(self.ui.inputTargetVolumeSelector.currentNode())
        self.logic.SetInputCriticalVolume(self.ui.inputCriticalVolumeSelector.currentNode())
        self.logic.SetLengthThreshold(self.ui.lengthThreshold.value)
        complete = self.logic.run()

        if complete:
            criticalVolume = self.ui.inputCriticalVolumeSelector.currentNode()
            PickPointsMatrix().GetLinesE_T(self.logic.myEntries, self.logic.myOutputs, "lineNodes", criticalVolume, self.logic.lengthThreshold)

        if self.ui.inputTargetFiducialSelector.currentNode():
            self.ui.inputTargetFiducialSelector.currentNode().SetDisplayVisibility(False)

        if not complete:
            print('I encountered an error')
            

class PathPlanningLogic(ScriptedLoadableModuleLogic):
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
    
    def run(self, inputVolume, inputFiducials, outputFiducials):
        outputFiducials.RemoveAllControlPoints()
        mat = vtk.vtkMatrix4x4()
        inputVolume.GetRASToIJKMatrix(mat)
        transform = vtk.vtkTransform()
        transform.SetMatrix(mat)

        for x in range(inputFiducials.GetNumberOfControlPoints()):
            pos = [0, 0, 0]
            inputFiducials.GetNthControlPointPosition(x, pos)
            ind = transform.TransformPoint(pos)

            pixelValue = inputVolume.GetImageData().GetScalarComponentAsDouble(int(ind[0]), int(ind[1]), int(ind[2]), 0)
            if pixelValue == 1:
                outputFiducials.AddControlPoint(pos[0], pos[1], pos[2])

    def GetLinesE_T(self, inputEntryFiducials, outputFiducials, groupName, CriticalVolume, lengthThreshold):
        lineNode = slicer.mrmlScene.CreateNodeByClass("vtkMRMLMarkupsLineNode")
        mat = vtk.vtkMatrix4x4()
        CriticalVolume.GetRASToIJKMatrix(mat)
        imageData = CriticalVolume.GetImageData()
        
        bestLineNode = None
        maxDistanceToCriticalStructure = -np.inf
        
        import time
        startTime = time.time()
        print("Time Started")
        
        for entryIndex in range(inputEntryFiducials.GetNumberOfControlPoints()):
            entryPointRAS = [0, 0, 0]
            inputEntryFiducials.GetNthControlPointPosition(entryIndex, entryPointRAS)
            for targetIndex in range(outputFiducials.GetNumberOfControlPoints()):
                targetPointRAS = [0, 0, 0]
                outputFiducials.GetNthControlPointPosition(targetIndex, targetPointRAS)

                lineNode.RemoveAllControlPoints()
                lineNode.AddControlPoint(entryPointRAS)
                lineNode.AddControlPoint(targetPointRAS)
                lineNode.SetAttribute("LineGroup", groupName)
                slicer.mrmlScene.AddNode(lineNode)
                print("")
                
                # insertionAngle = self.calculateInsertionAngle(entryPointRAS, targetPointRAS)
                # if insertionAngle > 55:
                #     slicer.mrmlScene.RemoveNode(lineNode)
                #     print(f"FAIL: Angle {insertionAngle:.2f}")
                #     continue
                # print(f"PASS: Angle {insertionAngle:.2f}")
                
                if self.checkIntersection(entryPointRAS, targetPointRAS, mat, imageData):
                    slicer.mrmlScene.RemoveNode(lineNode)
                    print("FAIL: Intersection")
                    continue
                print("PASS: No Intersection")
                length = np.linalg.norm(np.array(targetPointRAS) - np.array(entryPointRAS))
                if length > lengthThreshold:
                    slicer.mrmlScene.RemoveNode(lineNode)
                    print(f"FAIL: Length {length:.2f}")
                    continue
                print(f"PASS: Length {length:.2f}")

                distanceToCriticalStructure = self.calculateDistanceToCriticalStructure(lineNode, mat, imageData)
                if distanceToCriticalStructure > maxDistanceToCriticalStructure:
                    print("PASS: Safest Line")
                    if bestLineNode:
                        slicer.mrmlScene.RemoveNode(bestLineNode)
                    bestLineNode = slicer.mrmlScene.CreateNodeByClass("vtkMRMLMarkupsLineNode")
                    bestLineNode.Copy(lineNode)
                    maxDistanceToCriticalStructure = distanceToCriticalStructure
                    bestEntryPoint = entryPointRAS
                    bestTargetPoint = targetPointRAS
                else:
                    print("FAIL: Not Safest Line")
                slicer.mrmlScene.RemoveNode(lineNode)
        if bestLineNode:
            bestLineNode.SetDisplayVisibility(1)
            slicer.mrmlScene.AddNode(bestLineNode)
            slicer.mrmlScene.Modified()
            print(f"Best Entry Point: {bestEntryPoint}")
            print(f"Best Target Point: {bestTargetPoint}")
            trajectoryNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "trajectory")
            trajectoryNode.AddControlPoint(bestEntryPoint[0], bestEntryPoint[1], bestEntryPoint[2])
            trajectoryNode.AddControlPoint(bestTargetPoint[0], bestTargetPoint[1], bestTargetPoint[2])
            

        stopTime = time.time()
        print(f"Processing completed in {stopTime-startTime:.2f} seconds")

    def checkIntersection(self, startPointRAS, endPointRAS, mat, imageData):
        startIJK = [0, 0, 0, 1]
        endIJK = [0, 0, 0, 1]
        mat.MultiplyPoint(startPointRAS + [1], startIJK)
        mat.MultiplyPoint(endPointRAS + [1], endIJK)
        directionVector = np.array(endIJK[:3]) - np.array(startIJK[:3])
        distance = np.linalg.norm(directionVector)
        if distance == 0:
            return False

        directionVector /= distance

        for i in np.arange(0, distance, 0.1):
            samplePointIJK = np.array(startIJK[:3]) + directionVector * i
            voxelValue = imageData.GetScalarComponentAsDouble(int(samplePointIJK[0]), int(samplePointIJK[1]), int(samplePointIJK[2]), 0)
            if voxelValue != 0:
                return True
            
        return False

    def calculateInsertionAngle(self, entryPointRAS, targetPointRAS):

        directionVector = np.array(targetPointRAS) - np.array(entryPointRAS)
        directionVector /= np.linalg.norm(directionVector)
        cortexNormal = np.array([0, 0, 1])
        angle = np.arccos(np.clip(np.dot(directionVector, cortexNormal), -1.0, 1.0))
        return np.degrees(angle)

    def calculateDistanceToCriticalStructure(self, lineNode, mat, imageData):

        startPointRAS = [0, 0, 0]
        endPointRAS = [0, 0, 0]
        lineNode.GetNthControlPointPosition(0, startPointRAS)
        lineNode.GetNthControlPointPosition(1, endPointRAS)
        startIJK = [0, 0, 0, 1]
        endIJK = [0, 0, 0, 1]
        mat.MultiplyPoint(startPointRAS + [1], startIJK)
        mat.MultiplyPoint(endPointRAS + [1], endIJK)
        directionVector = np.array(endIJK[:3]) - np.array(startIJK[:3])
        distance = np.linalg.norm(directionVector)
        if distance == 0:
            return 0

        directionVector /= distance
        minDistance = np.inf

        for i in np.arange(0, distance, 0.1):
            samplePointIJK = np.array(startIJK[:3]) + directionVector * i
            voxelValue = imageData.GetScalarComponentAsDouble(int(samplePointIJK[0]), int(samplePointIJK[1]), int(samplePointIJK[2]), 0)
            if voxelValue != 0:
                distanceToSurface = np.linalg.norm(samplePointIJK - np.array(startIJK[:3]))
                if distanceToSurface < minDistance:
                    minDistance = distanceToSurface
        return minDistance


class PathPlanningTest(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear(0)
        print("Clearing Scene")
        
    def runTest(self):
        self.setUp()
        self.test_PathPlanningTestOutsidePoint()


    def test_PathPlanningTestOutsidePoint(self):
        print("Starting the test")
        print("Starting test points outside mask.")
        #
        # get our mask image node
        mask = slicer.util.getNode('r_hippoTest')

        # I am going to hard code two points -- both of which I know are not in my mask
        outsidePoints = slicer.vtkMRMLMarkupsFiducialNode()
        outsidePoints.AddControlPoint(-1, -1, -1)  # this is outside of our image bounds
        cornerPoint = mask.GetImageData().GetOrigin()
        outsidePoints.AddControlPoint(cornerPoint[0], cornerPoint[1], cornerPoint[2])  # we know our corner pixel is no 1

        #run our class
        returnedPoints = slicer.vtkMRMLMarkupsFiducialNode()
        PickPointsMatrix().run(mask, outsidePoints, returnedPoints)

        # check if we have any returned fiducials -- this should be empty
        if (returnedPoints.GetNumberOfControlPoints() > 0):
            print('Test failed. There are ' + str(returnedPoints.GetNumberOfControlPoints()) + ' return points.')
            return

        print('Test passed! No points were returned.')
