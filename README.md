# NBA team logo idenification

This project uses concepts such as KNN to create a machine learning pipeline for training
and using a KNN based model for recognizing NBA team logos.

## Overview

This project is divided into 3 main files

### main.py

Handles model creation and training

Use this file to:
* Load and prepare dataset
* Train StandardScaler, PCA, and KNN models
* Export these models to be loaded in for use at a future time

### preProFuncts.py

Provides help functions to be used when preparing dataset and
functions to be used during the training phase

Includes
* Preprocessing utilites
* Data manipulation tools

### runDetection.py

Contains a class for importing and running an already trained model 

Once a model has been trained, use for:
* Load the saved model
* Get prediction by providing a path to an image
* Get top X most likely classes

#### Usage

Main class functions and applications
```console
x = KnnNBALogoClassifier(knn_path, scaler_path, pca_path)
provide these paths if they are not in the predefined path

x.setK(x) -- select nearest neighbors value

x.predict(file_path, num) -> str -- use the KNN model to predict the most likely team with a string name output

x.predict_top_results(file_path) -> (str<class name>, float{0-1}<confidence score>) -- use the KNN 
model to get the top 'num' most likely options for what it predicts the input image to be with a confidence score for each
```
