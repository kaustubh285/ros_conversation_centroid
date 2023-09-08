# ROS Conversation Centroid Detection

The project implementation in Master branch covers the requirements for the dissertation project. Please check out the experimental branch for the latest updates and development goals.

## Table of Contents

- Introduction
- Installation
- Dependencies
- Usage
- Video Demo

## Introduction

This ROS package, ros_conversation_centroid, is designed to detect and track humans in a scene using computer vision techniques. It employs RGB-D data to identify humans and their interactions, and subsequently computes the 2D and 3D centroids of conversations.

The primary features include:

1. Human detection and tracking
2. Social interaction classification (e.g., N-shape, Vis-a-vis)
3. 2D and 3D centroid calculation for conversations

## Installation

To install this package, clone the repository into your ROS workspace and build the project:

```
cd ~/catkin_ws/src
git clone https://github.com/kaustubh285/ros_conversation_centroid/
cd ..
catkin_make
source devel/setup.bash
```

## Dependencies

- ROS (Tested on ROS Noetic)
- OpenCV (Tested on version 4.x)
- Python 3.x
- NumPy

## Usage

To run the code, simply execute the following command:

```
rosrun ros_conversation_centroid conversation_centroid_finder.py
```

## Video Demo

A video demonstration of this project is available on YouTube. You can watch it [here](https://youtu.be/qk1SVzceGWw).
