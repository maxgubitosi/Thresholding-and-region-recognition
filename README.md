# Thresholding and Region Recognition

This repository contains the solutions to Practical Assignment 0 for the course “Artificial Vision.” The assignment focuses on thresholding and region recognition algorithms applied to images.

## Table of Contents
- [Introduction](#Introduction)
- [Problem Description](#Problem-Description)
  - [1. Thresholding](#Thresholding)
  - [2. Region Recognition](#Region-Recognition)
- [Results](#Results)
  - [Color Candy Recognition Results](#Color-Candy-Recognition-Results)
  - [Red Blood Cell Detection Results](#Red-Blood-Cell-Detection-Results)
- [Files and Organization](#Files-and-Organization)
- [Acknowledgments](#Acknowledgements)

## Introduction

In this practical assignment, thresholding techniques and region recognition algorithms were implemented to process and analyze images of colored candies and red blood cells. The implementation leverages the OpenCV library (cv2) in Python to extract meaningful features and detect regions of interest in binary images.

## Problem Description

1. Thresholding

The goal of this task was to apply thresholding techniques to convert the input images into binary ones. Various methods were explored to optimize binary image creation, ensuring the best results for region detection.

2. Region Recognition

Once thresholded, connected component analysis and other region recognition algorithms were applied to identify and label individual regions of interest (e.g., colored candies or blood cells). This involved calculating bounding boxes around each detected region.

## Results

### Color Candy Recognition Results

Bounding boxes were successfully drawn around regions corresponding to individual candies in the image:
![Confites detectados](imgs/outputs/boxed_3.jpg)

### Red Blood Cell Detection Results

Red blood cells in the sample image were identified and highlighted:
![Células rojas detectadas](imgs/outputs/red_blood_cells_detected_1.jpg)

## Files and Organization

This repository contains the following files and folders:
	•	binarizacion_regiones.ipynb: Main Jupyter Notebook that includes problem statements, visualizations, and results.
	•	functions.py: Contains the custom functions developed for this assignment.
	•	utils.py: Base utility functions provided by the course for preprocessing.
	•	imgs: Folder containing the input images used for the assignment.

## Acknowledgments

This practical assignment was developed as part of the course “Artificial Vision,” showcasing fundamental techniques for image processing and region recognition.
