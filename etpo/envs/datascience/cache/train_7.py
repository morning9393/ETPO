
import numpy as np

def build_model(x_train: np.ndarray, y_train: np.ndarray):
    """ Trained with x_train in shape (125, 4)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import plot_confusion_matrix

    # Split data into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.25, random_state=42)

    # Build model
    model = RandomForestClassifier(n_estimators=100)

    # Train model
    model.fit(x_train, y_train)

    # Make predictions
    y_pred = model.predict(x_test)

    # Evaluate predictions
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
    print(classification_report(y_test, y_pred))
    print
    return model
