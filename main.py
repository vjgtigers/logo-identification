from datetime import datetime

def printCurrTimeAndMessage(message):
    currTime = datetime.now().strftime("%H:%M:%S")
    print(f"[{currTime}] {message}")
printCurrTimeAndMessage("Program Start")


import numpy as np
import preProFuncts
import joblib
import sklearn
import time
import matplotlib.pyplot as plt
from runDetection import KnnNBALogoClassifier

printCurrTimeAndMessage("Done loading imports")

#config
seed = 1
testTrainSplit = True
showPCAgraphs = False
saveScalerAndPCA = False #one of these should be true and one should be false depending on the step up
loadScalersandPCAS = True #one of these should be true and one should be false depending on the step up
neighbors = 2
runKNN = True
saveKNN = True

if loadScalersandPCAS == False:
    printCurrTimeAndMessage("Loading dataset")
    #load in image data set

    images = []

    images = preProFuncts.folderToPx("./images/NBA_teams/labeled_sizeprocessed")
    ###images_unlabeled = preProFuncts.folderToPx("./images/NBA_teams/unlabeled_sizeprocessed")
    print(f"Done loading in dataset(s) {len(images)}")###, {len(images_unlabeled)}")


    print(f"images loaded: {len(images)}")

    printCurrTimeAndMessage("Pre preprocessing")

    data = []
    target = []
    for i in images:
        print(i, len(i[0]))
        data.append(i[0])
        target.append(i[1])
    print(target)
    for i in range(len(target)):
        p = target[i].split("_")
        target[i] = p[0]

    print(target)
    print(set(len(x) for x in data))
    X = np.array(data)
    y = np.array(target)
    y_num = sklearn.preprocessing.LabelEncoder().fit_transform(y)

    ###data_unlabeled = []
    ###for i in images_unlabeled:
    ###    data_unlabeled.append(i[0])
    ###X_un = np.array(data_unlabeled)

    printCurrTimeAndMessage("pre preprocessing done")
    #TODO: normalize/standardize pixel colors -- this should happen before converting to a one D array

    #test/train split
        #only do after adding in more images than basic ones

    printCurrTimeAndMessage("Test/Train split")
    X_train, X_test, y_train, y_test = X, X, y, y
    ###X_un_train, X_un_test= X_un, X_un
    if testTrainSplit == True:
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=.15, random_state=seed)
        ###X_un_train, X_un_test = sklearn.model_selection.train_test_split(X_un, test_size=15, random_state=seed)

    #apply pca
        #reduce down to a useable amount for KNN and K-means
        #determine pca components value


    printCurrTimeAndMessage("Begin preprocessing")

    printCurrTimeAndMessage("scalar/pca creation")
    scalerVar = sklearn.preprocessing.StandardScaler()
    pcaVar = sklearn.decomposition.PCA(n_components=.90)

    ###scaler_un = sklearn.preprocessing.StandardScaler()
    ###pca_un = sklearn.decomposition.PCA(n_components=.90)

    printCurrTimeAndMessage("StandardScaler labeled")
    X_centered_train = scalerVar.fit_transform(X_train)
    X_centered_test = scalerVar.transform(X_test)

    ###X_centered_un_train = scaler_un.fit_transform(X_un_train)
    ###X_centered_un_test = scaler_un.transform(X_un_test)

    printCurrTimeAndMessage("Done StandardScaling")
    printCurrTimeAndMessage("PCA labeled data")

    X_pca_train = pcaVar.fit_transform(X_centered_train)
    X_pca_test = pcaVar.transform(X_centered_test)

    ###printCurrTimeAndMessage("PCA unlabeled data")

    ###X_pca_un_train = pca_un.fit_transform(X_centered_un_train)
    ###X_pca_un_test = pca_un.transform(X_centered_un_test)

    printCurrTimeAndMessage("Done applying PCA")


if loadScalersandPCAS == True:
    #load scalers
    printCurrTimeAndMessage("Loading Scalers")
    scalerVar = joblib.load("./modelInfo/scaler_labeled.pkl")
    ###scaler_un = joblib.load("./modelInfo/scaler_unlabeled.pkl")

    #load PCA models
    printCurrTimeAndMessage("Loading PCA's")
    pcaVar = joblib.load("./modelInfo/pca_labeled.pkl")
    ###pca_un = joblib.load("./modelInfo/pca_unlabeled.pkl")

    #Load pca data
    printCurrTimeAndMessage("Loading preprocessed labeled dataset")
    X_pca_train = np.load("./modelInfo/X_lab_train_pca.npy")
    X_pca_test = np.load("./modelInfo/X_lab_test_pca.npy")

    #load y train/test
    y_train = np.load("./modelInfo/y_train.npy")
    y_test = np.load("./modelInfo/y_test.npy")

    #load unlabeled data
    ###printCurrTimeAndMessage("Loading preprocessed unlabeled dataset")
    ###X_pca_un_train = np.load("./modelInfo/X_unlab_train_pca.npy")
    ###X_pca_un_test = np.load("./modelInfo/X_unlab_test_pca.npy")



X_pca = X_pca_train

if loadScalersandPCAS == False:
    print(f"Original shape of X: {X.shape}")
    print(f"Shape of X after dim. reduction with PCA: {X_pca.shape}")
    print(f"Number of PCs: {pcaVar.n_components_}")


if saveScalerAndPCA == True:
    print("""# ============================================================
# SAVE EVERYTHING
# ============================================================
""")
    printCurrTimeAndMessage("Saving models and datasets")

    #save models
    printCurrTimeAndMessage("Saving models")
    joblib.dump(scalerVar, "./modelInfo/scaler_labeled.pkl")
    ###joblib.dump(scaler_un, "./modelInfo/scaler_unlabeled.pkl")

    joblib.dump(pcaVar, "./modelInfo/pca_labeled.pkl")
    ###joblib.dump(pca_un, "./modelInfo/pca_unlabeled.pkl")


    #save modified data
    printCurrTimeAndMessage("Saving modified datasets")
    np.save("./modelInfo/X_lab_train_pca.npy", X_pca_train)
    np.save("./modelInfo/X_lab_test_pca.npy",  X_pca_test)
    np.save("./modelInfo/y_train.npy", y_train)
    np.save("./modelInfo/y_test.npy",  y_test)

    ###np.save("./modelInfo/X_unlab_train_pca.npy", X_pca_un_train)
    ###np.save("./modelInfo/X_unlab_test_pca.npy",  X_pca_un_test)


    printCurrTimeAndMessage("Done Saving")



#PCA graph display pulled from sample code from canvas
#------------------------------------------------------
if showPCAgraphs == True:
    # 4. Explained variance to find the optimal number of PCs
    explained_variance_ratio = pcaVar.explained_variance_ratio_ #strategy#1
    cumulative_variance = explained_variance_ratio.cumsum() #strategy#2
    plt.figure(figsize=(8,4))
    plt.plot(cumulative_variance, marker='o')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Explained Variance on Digits Dataset')
    plt.grid(True)
    plt.show()
    temp = sklearn.preprocessing.LabelEncoder().fit_transform(y_train)
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(X_pca[:,0], X_pca[:,1], c=temp, cmap='tab10', s=30)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Teams dataset projected onto first 2 PCs')
    plt.legend(*scatter.legend_elements(), title="LOGOS")
    plt.grid(True)
    plt.show()

#------------------------------------------------------



#run KNN

if runKNN == True:
    printCurrTimeAndMessage("Beginning KNN")
    knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=neighbors)
    knn.fit(X_pca_train, y_train)
    if saveKNN == True:
        joblib.dump(knn, "./modelInfo/knnModel.pkl")
        joblib.dump(pcaVar, "./modelInfo/pca_labeled.pkl")
        joblib.dump(scalerVar, "./modelInfo/scaler_labeled.pkl")



#accuracy testing
if runKNN == True:
    if testTrainSplit == False:
        print("TEST TRAIN NOT ENABLED")
    prediction_test = knn.predict(X_pca_test)
    prediction_train = knn.predict(X_pca_train)
    printCurrTimeAndMessage(f"KNN Accuracy (test)= {sklearn.metrics.accuracy_score(y_test, prediction_test)}")
    printCurrTimeAndMessage(f"KNN Accuracy (train)= {sklearn.metrics.accuracy_score(y_train, prediction_train)}")


knnTester = KnnNBALogoClassifier("./modelInfo/knnModel.pkl", "./modelInfo/scaler_labeled.pkl", "./modelInfo/pca_labeled.pkl")

test_data = []
train_data = []

for i in range(1, 11):
    print("-"*50)
    print(f"K Value = {i}")
    knnTester.setK(i)

    pred_test = knnTester._predictPreprocessedData(X_pca_test)
    pred_train = knnTester._predictPreprocessedData(X_pca_train)
    test_temp = sklearn.metrics.accuracy_score(y_test, pred_test)
    train_temp = sklearn.metrics.accuracy_score(y_train, pred_train)
    test_data.append(test_temp)
    train_data.append(train_temp)
    print(f"Test accuracy = {test_temp}")
    print(f"Train accuracy = {train_temp}")


test_data_max_indicies = np.array(test_data).argsort()[::-1]
train_data_max_indicies = np.array(train_data).argsort()[::-1]
print(f"max train indicies = {train_data_max_indicies}")
print(f"max test indicies = {test_data_max_indicies}")

for i in test_data_max_indicies:
    print(test_data[i], train_data[i])