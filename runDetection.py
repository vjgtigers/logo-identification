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

    # load in pre computed preprocessing steps and knn model
    def __init__(self, knn_path, scaler_path, pca_path, image_size=(500, 500)):
        self.knn = joblib.load(knn_path)
        self.scaler = joblib.load(scaler_path)
        self.pca = joblib.load(pca_path)
        self.image_size = image_size #_ not particuarly needed since 500x500 is what the model was trained on, so anything else wouldnt work

    #change k value from default after init
    def setK(self, k):
        self.knn.set_params(n_neighbors=k)

    #preprocessing steps before standardScaler and PCA
    def _preprocess(self, file_path):

        oneD = preProFuncts.returnScaledImageIn1DArray(file_path, self.image_size)
        temp = np.asarray(oneD).reshape(1, -1)

        return temp

    #preprocessing and prediction
    def predict(self, file_path):
        preprocessed = self._preprocess(file_path)
        scaled = self.scaler.transform(preprocessed)
        pcad = self.pca.transform(scaled)

        return self.knn.predict(pcad)[0]

    #predict the top X results and their "confidence score" which is really just (number of votes that class recived / total votes avalible)
    def predict_top_results(self, file_path, num=3):

        preprocessed = self._preprocess(file_path)
        scaled = self.scaler.transform(preprocessed)
        pcad = self.pca.transform(scaled)

        probs = self.knn.predict_proba(pcad)[0]
        maxProbIndicies = probs.argsort()[-num:][::-1] #takes the probabilities, get the indicies that would sort probs min-max order, then take this and get the max X values from the end, and reverse that list

        tempList = []
        for i in maxProbIndicies:
            tempList.append((str(knn.classes_[i]), float(probs[i])))

        return tempList


knnPred = KNNImageClassifier2(KNN_path, standardScaler_path, PCA_path)

knnPred.setK(10)
print(knnPred.predict(filePath))
print(knnPred.predict_top_results(filePath))