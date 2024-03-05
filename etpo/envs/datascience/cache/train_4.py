
import numpy as np

def build_model(x_train: np.ndarray, y_train: np.ndarray):
    """ Trained with x_train in shape (125, 4)."""
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import roc_auc_score

    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(x_train, y_train)

    # Predict
    y_pred = model.predict(x_test)

    # Evaluate
    print(
        f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')
    print(
        f'Classification Report
    return model
