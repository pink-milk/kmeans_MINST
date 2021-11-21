import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

#------------------place file in dataframe, then preprocess----------------------
df = pd.read_csv(
    'new_test.txt', sep=",",header=None)

#remove cols that contain only 0 (784col->668col)
train_img=df.loc[:, (df != 0).any(axis=0)]
#print(train_img)

scaler = StandardScaler()

#we need to standardize the data
train_img=scaler.fit_transform(train_img)
#train_img = scaler.transform(train_img )

#get the PCA dim. reduction library and fit it onto standardize data
#we reduce 600 something cols into 4 cols
pca = PCA(.95)

#fit it onto our data, p is our new data set w 4 cols

#p=pca.fit_transform(train_img)
p=pca.fit_transform(train_img)

print(p)

kmeans = KMeans(n_clusters=10)
#k=kmeans.fit(p)
identified_clusters = kmeans.fit_predict(p)

print(identified_clusters)


a_file = open("output.txt", "w")

# for row in identified_clusters:
    # np.savetxt(a_file, row)

np.savetxt("output.txt", identified_clusters,fmt='%i')
# a_file.close()