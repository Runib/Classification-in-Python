from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import ReadAndClean as rd
import pandas as pd

def transformLDA(data, classColumn):
    # extract attributes from dataset
    attributes = rd.extractAttributes(data)
    X_norm = attributes.loc[:, :]

    # transform data with LDA
    lda = LDA(n_components=2)  # 2-dimensional LDA
    dataTransformedLDA = pd.DataFrame(lda.fit_transform(X_norm, data[classColumn]))

    return dataTransformedLDA.join(how='left', other=data[['CLASS', 'NSP']])