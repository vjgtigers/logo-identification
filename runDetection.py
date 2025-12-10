import joblib
import numpy as np
import preProFuncts


#config info
KNN_path = "./modelInfo/knnModel.pkl"
standardScaler_path = "./modelInfo/scaler_labeled.pkl"
PCA_path = "./modelInfo/pca_labeled.pkl"

#load models
standardScaler = joblib.load(standardScaler_path)
PCA = joblib.load(PCA_path)
knn = joblib.load(KNN_path)


#get image path from user
#filePath = input("Image file path: ")
filePath = "C:/Users/vjgti/Desktop/oment_pic5.png"


#scale image
sizeCorrection = preProFuncts.get_unique_filename(filePath)

preProFuncts.scaleImage2(filePath, sizeCorrection, (500, 500))
image1DArray = preProFuncts.imageToPx(sizeCorrection)
reshaped = image1DArray.reshape(1, -1)
image_scaled = standardScaler.transform(reshaped)
image_pca = PCA.transform(image_scaled)

pred = knn.predict(image_pca)
print(pred)


class KNNImageClassifier2:

    def __init__(self, knn_path, scaler_path, pca_path, image_size=(500, 500)):
        self.knn = joblib.load(knn_path)
        self.scaler = joblib.load(scaler_path)
        self.pca = joblib.load(pca_path)
        self.image_size = image_size
        
        
    def _preprocess(self, file_path):
        oneD = preProFuncts.returnScaledImageIn1DArray(file_path, self.image_size)
        temp = np.asarray(oneD).reshape(1, -1)
        
        return temp

    def predict(self, file_path):

        preprocessed = self._preprocess(file_path)
        scaled = self.scaler.transform(preprocessed)
        pcad = self.pca.transform(scaled)
        return self.knn.predict(pcad)[0]


knnPred = KNNImageClassifier2(KNN_path, standardScaler_path, PCA_path)

print(knnPred.predict(filePath))