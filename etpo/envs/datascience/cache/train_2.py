
import numpy as np

def build_model(x_train: np.ndarray, y_train: np.ndarray):
    """ Trained with x_train in shape (125, 4)."""
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    model = LogisticRegression()

    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.3, random_state=1)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")

    return model
