import joblib
import numpy as np
import preProFuncts


#config info
KNN_path = "./modelInfo/knnModel.pkl"
standardScaler_path = "./modelInfo/scaler_labeled.pkl"
PCA_path = "./modelInfo/scaler_labeled.pkl"

#load models
standardScaler = joblib.load(standardScaler_path)
PCA = joblib.load(PCA_path)
knn = joblib.load(KNN_path)


#get image path from user
filePath = input("Image file path: ")

#scale image
sizeCorrection = preProFuncts.get_unique_filename(filePath)

preProFuncts.scaleImage2(filePath, sizeCorrection, (500, 500))