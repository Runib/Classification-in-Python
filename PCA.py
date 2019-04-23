from sklearn.decomposition import PCA as sklearnPCA
import ReadAndClean as rd
import pandas as pd

def transformPCA(data):
    # extract attributes from dataset
    attributes = rd.extractAttributes(data)
    X_norm = attributes.loc[:, :]

    # transform data with PCA
    pca = sklearnPCA(n_components=2)
    dataTransformedPCA = pd.DataFrame(pca.fit_transform(X_norm))

    return dataTransformedPCA.join(how='left', other=data[['CLASS', 'NSP']])