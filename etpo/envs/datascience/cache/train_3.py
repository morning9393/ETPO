
import numpy as np

def build_model(x_train: np.ndarray, y_train: np.ndarray):
    """ Trained with x_train in shape (125, 4)."""
    from sklearn.svm import SVC
    model = SVC(gamma='auto')
    model.fit(x_train, y_train)
    return model
