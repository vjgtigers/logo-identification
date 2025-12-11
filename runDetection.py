import joblib
import numpy as np
import preProFuncts


#testing filepath(s)
filePath = "C:/Users/vjgti/PycharmProjects/cs3200_termProject/.venv/images/NBA_teams/unlabeled_sizeprocessed/96.png"

class KnnNBALogoClassifier:

    # load in pre computed preprocessing steps and knn model
    def __init__(self,
                 knn_path = "./model_defaults/knn_NBA_default.pkl",
                 scaler_path = "./model_defaults/scaler_NBA_default.pkl",
                 pca_path = "./model_defaults/pca_NBA_default.pkl",
                 image_size=(500, 500)):

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

    #predict the top X results (may return less if less than X get 'votes') and their "confidence score"
    #which is really just (number of votes that class recived / total votes avalible)
    def predict_top_results(self, file_path, num=3):

        preprocessed = self._preprocess(file_path)
        scaled = self.scaler.transform(preprocessed)
        pcad = self.pca.transform(scaled)

        probs = self.knn.predict_proba(pcad)[0]
        maxProbIndicies = probs.argsort()[-num:][::-1] #takes the probabilities, get the indicies that would sort probs min-max order, then take this and get the max X values from the end, and reverse that list

        tempList = []
        for i in maxProbIndicies:
            if probs[i] != 0.0:
                tempList.append((str(self.knn.classes_[i]), float(probs[i])))

        return tempList




knnPred = KnnNBALogoClassifier()

knnPred.setK(2)
print(knnPred.predict(filePath))
print(knnPred.predict_top_results(filePath))