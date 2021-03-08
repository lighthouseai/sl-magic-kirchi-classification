import cv2
import numpy as np 
import os
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=0)

# print(os.listdir("./data/kir"))


# img = cv2.imread("./data/kir/"+os.listdir("./data/kir")[10])
img = cv2.imread("./data/kir/"+"1599051453.3101153.jpg")
h,s,v = cv2.split(img)
# X = np.ravel(h)

# print(X.shape)
X = h
# X = h.reshape((1,h.shape[0],h.shape[1]))
X = h.flatten()

print(X.shape)

kmeans.fit(X)

print(kmeans.cluster_centers_.shape)

cv2.imshow("h",h)

cv2.imshow("image",img)

cv2.waitKey(0)