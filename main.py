import numpy as np
import preProFuncts

import sklearn

import matplotlib.pyplot as plt

#select how many images from the image dataset i want
imagesToLoad = 0
    #to start just load in one basic image from each team
#randomly generate what images should be loaded

#load in image data set

images = []
if imagesToLoad == 0:
    images = preProFuncts.folderToPx("./images/NFL_teams/labeled_sizeprocessed")

#convert to a single one line vector - pillow
    #if doesnt work reduce pixel count to like a 32x32

print(f"images loaded: {len(images)}")

data = []
target = []
for i in images:
    data.append(i[0])
    target.append(i[1])
print(target)
X = np.array(data)
y = np.array(target)
y_num = sklearn.preprocessing.LabelEncoder().fit_transform(y)

scalerVar = sklearn.preprocessing.StandardScaler()
pcaVar = sklearn.decomposition.PCA(n_components=.95)

X_centered = scalerVar.fit_transform(X)

X_pca = pcaVar.fit_transform(X_centered)

print(f"Original shape of X: {X.shape}")
print(f"Shape of X after dim. reduction with PCA: {X_pca.shape}")
print(f"Number of PCs: {pcaVar.n_components_}")

#------------------------------------------------------

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

plt.figure(figsize=(8,6))
scatter = plt.scatter(X_pca[:,0], X_pca[:,1], c=y_num, cmap='tab10', s=30)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Teams dataset projected onto first 2 PCs')
plt.legend(*scatter.legend_elements(), title="LOGOS")
plt.grid(True)
plt.show()



#------------------------------------------------------




#test/train split
    #only do after adding in more images than basic ones
#normalize/standardize pixel colors

#apply pca
    #reduce down to a useable amount for KNN and K-means
    #determine pca components value

#run KNN
    #determine best K
#run K-means
    #amount of clusters is number of logos to identify

#accuracy testing (fine for KNN, alternative needed for K-means)

#use K-means to label data?

##-------
##When model fully built

#final accuracy testing with train/testing data (fine for KNN, alternative needed for K-means)

#confidence score

#better interface

##-------
##Things that dont belong in pipeline

#develop way to save all of the data after preprocessing if it takes a while to load to reduce loading times

#save model data after training so it doesnt need to be trained again if that model is needed again

#cluster id to label by majority vote?


##
##


#IMAGE CONVERSION TOOLS
##folderWebPtoPNG("C:\\Users\\vjgti\\Downloads\\NFL_teams", "C:\\Users\\vjgti\\PycharmProjects\\cs3200_termProject\\.venv\\images\\NFL_teams\\labeled")
##scaleFolder("C:\\Users\\vjgti\\PycharmProjects\\cs3200_termProject\\.venv\\images\\NFL_teams\\labeled", "C:\\Users\\vjgti\\PycharmProjects\\cs3200_termProject\\.venv\\images\\NFL_teams\\labeled_sizeprocessed", (500, 500))