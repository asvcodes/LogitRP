# Logit Rain Prediction

## Overview

This repository contains code and resources for building a rain prediction model using logistic regression with Scikit-learn in Python. The model predicts whether it will rain tomorrow based on today's weather data. It utilizes a real-world dataset from Kaggle, specifically the "Rain in Australia" dataset, which includes about 10 years of daily weather observations from various Australian weather stations.

## Problem Statement

The goal of this project is to create a fully-automated system that can predict whether it will rain at a given location tomorrow using today's weather data. This is a binary classification problem, where the classes are "Will Rain" and "Will Not Rain". The model is trained using historical weather data to make predictions for future days.

## Dataset

The dataset used for training and evaluation is the "Rain in Australia" dataset obtained from Kaggle. It contains various features such as temperature, humidity, wind speed, etc., along with the target variable indicating whether it rained the following day. The dataset is preprocessed and split into training, validation, and test sets for model training and evaluation.

## Approach

### 1. Preprocessing

Data preprocessing involves handling missing values, scaling numeric features, and encoding categorical variables. Missing values are filled or imputed, numeric features are scaled to a (0,1) range, and categorical columns are encoded as one-hot vectors.

### 2. Model Training

Logistic regression model is trained using Scikit-learn's `LogisticRegression` class. The model is trained on the training dataset and tuned using the validation set.

### 3. Model Evaluation

The trained model is evaluated using the validation set to assess its performance in predicting rain. Evaluation metrics such as accuracy, precision, recall, and F1-score are calculated to measure the model's effectiveness.

### 4. Prediction

Once the model is trained and evaluated, it can be used to make predictions on new data. Given today's weather data, the model predicts whether it will rain tomorrow at the specified location.

## Files

- `Rain Prediction.ipynb`: Jupyter notebook containing the code for data preprocessing, model training, evaluation, and prediction.
- `weatherdata.csv`: CSV file containing the Rain in Australia dataset.
- `README.md`: Documentation providing an overview of the project, problem statement, approach, and files in the repository.

## Dependencies

The following Python libraries are required to run the code:

- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## Usage

1. Clone the repository:

   ```
   git clone https://github.com/asvcodes/Logit-Rain-Prediction-Model.git
   ```

2. Navigate to the repository directory:

   ```
   cd Logit-Rain-Prediction-Model
   ```

3. Open and run the `Rain Prediction.ipynb` notebook using Jupyter or any compatible environment.

## Acknowledgments

- Kaggle for providing the "Rain in Australia" dataset.
- Scikit-learn developers for the logistic regression implementation.
- Contributors to open-source libraries used in the project.
  
---
