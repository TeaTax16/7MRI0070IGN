# Image-Guided Navigation for Robotics

## Table of Contents
1. [Introduction](#introduction)
2. [Components](#components)
3. [Installation](#installation)
4. [Running the System](#running-the-system)
5. [End-to-End Pipeline](#end-to-end-pipeline)


## Introduction
This project integrates imaging software and robotics simulation to enhance medical procedures.
By combining 3D slicer for image processing and ROS for robotic simulation, the system aims to improve surgical precision, presonalised treatment and training for medical professionals.

## Components
### 3D Slicer
3D Slicer is an open-source software platform used for image analysis and scientific visualisation. It is used for path planning by creating optimal trajectories for medical procedures.

### ROS (Robot Operating System)
ROS is an open-source set of software libraries and tools used to build robot applications. IT is used to simulate robot movements based on the trajectories planne din 3D Slicer

### OpenIGTLink
OpenIGTLink is an open-source network communication interface for image-guided interventions. It facilitates seamless data transfer between 3D Slicer and ROS.

## Installation
### Requirements
- Ubuntu
- Python
- ROS noetic
- 3D Slicer

### Step-by-Step Installation
### 1. 3D Slicer
download and install 3D Slicer from the [official website](https://www.slicer.org/)

### 2. Install OpenIGTLLink
From the extension manager >> IGT >> install SlicerOpenIGTLink >> restar Slicer

### 3. Setup PathPlanning Extension
From Developer tools >> Extension Wizard >> Select Extension >> navigate through the directory to find PathPlanning download location and import to Slicer. 

## Running the System
### Configure 3D Slicer
- After adding the extension to Slicer, add in the required data stypes:
    - vtkMRMLLabelMapVolumeNode (Target Structure, Critical Structure)
    - vtkMRMLMarkupsFiducialNode (Entry Fiducials, Target Fiducials)
Create a new Point List for Output Fidduals
Run the PathPlanning exntesion

### Configure IGTLink
Go to Modules > IGT > OpenIGTLinkIF
Create a new connection and set type to server
Set the homename to localhost and the post to 18994

Source the ROS setup script
Launch the ROS-IGTL-Bridge
Echo the IGTL topic in a new terminal

### Configure ROS
Using the provided URDF file for the robot, use the MoveIt Setup Assistance to configure collision checking and kinematics
Launch the simulation in Rviz

# End-to-end pipeline
## Step 1: Path Planning in 3D Slicer
load the medical images in 3D Slicer
use the path planning extension to define entry and target points
run oath planning algorithm to generate the optimal trajectory
send the trajectory points to ROS using OpenIGTLink

## Step 2: Data Transfer to ROS
ensure both 3D Slicer and ROS are connected via OpenIGTLink
verify that the trajectory points are correctly receive in ROS by checking the echo'd topic

## Step 3: Robot Movement in ROS
use use the received trajecotry points to simulate robot movement
launch the robot in rviz and visualise the planned path
execute the path planning algorithm to move the robot from the start pose to the goal pose