
import numpy as np

def build_model(x_train: np.ndarray, y_train: np.ndarray):
    """ Trained with x_train in shape (125, 4)."""
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(C=0.001, solver='lbfgs', multi_class='multinomial', max_iter=1000)
    model.fit(x_train, y_train)
    return model
