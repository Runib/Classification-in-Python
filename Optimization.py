from tpot import TPOTClassifier
import numpy as np

def svmOptimization(dataValTrain, dataValTest, nameOfClass):
    config_TPOT = {
        # Classifiers
        'sklearn.svm.SVC': {
            'C': np.logspace(-2, 5, 8),
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
            'degree ': range(1, 5),
            'gamma': np.logspace(-8, 3, 12)
        }
    }

    tpot_classifier = TPOTClassifier(generations=20, population_size=300, offspring_size=250,
                                     verbosity=2, early_stop=8, n_jobs=-1,
                                     config_dict=config_TPOT, cv=10, scoring='accuracy')
    tpot_classifier.fit(dataValTrain.values, dataValTest[nameOfClass].values)
  #  print(tpot_classifier.score(X_test, y_test))
    tpot_classifier.export('tpot_svm_pipeline.py')