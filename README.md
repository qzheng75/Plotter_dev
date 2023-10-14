# HackGT Plot Visualize & Analysis Project

![HackGT Logo](hack_gt_logo.png)

Welcome to the repository for our HackGT Plot Analysis Project! This project is part of HackGT and is scheduled to start on October 13th.

## Project Overview

The main goal of this project is to develop a comprehensive tool for analyzing and visualizing various types of plots and data series. The project is divided into three main components:

1. **Plot Classification**
   - We have fine-tuned a classifier determining whether a user-provided image contains a valid plot. If the image is classified as a valid plot, we proceed to identify the type of plot it represents using another fine-tuned classifier.
   - We also provided twenty images of different types in `test_images` for you to experiment with the validity of both classification models.

2. **Data Series Analysis**
   - Leveraging OpenCV and related tools, our project analyzes data series within each type of plot. This includes extracting data points, identifying trends, and performing relevant data manipulations to gain insights from the plot.

3. **Web-Based Visualization**
   - To make our results accessible and user-friendly, we have built a website demonstrating our plot analysis's outcomes. Users can upload images, view classification results, and explore the analyzed data series through intuitive visualizations.

## Getting Started

To get started with our project, follow these steps:

1. Clone the repository to your local machine:
   ```shell
   git clone https://github.com/qzheng75/Visualize_Plots_dev.git

## Datasets involved in this project
   - Plots used in plot classification network training: https://www.kaggle.com/competitions/benetech-making-graphs-accessible
   - Datasets involved in training the `is-plot-or-not` network:
      - Plot dataset mentioned above. 
      - Icons-50: https://www.kaggle.com/datasets/danhendrycks/icons50
      - Handwritten math symbols: https://www.kaggle.com/datasets/sagyamthapa/handwritten-math-symbols
      - A subset (due to the limitation of computational resources) of the COCO 2017 dataset: https://www.kaggle.com/datasets/sabahesaraki/2017-2017
