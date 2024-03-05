
DS_CODE_PROMPT = '''
"""Classification with a {data_disc} Prediction Dataset"""
import numpy as np

def build_model(x_train: np.ndarray, y_train: np.ndarray):
    """ Import modules, return a scikit model trained with x_train in shape {feature_shape} and y_train."""
    <FILL_ME>
    return model
'''

DS_CODE_REFLECTION_PROMPT = '''
"""Classification with a {data_disc} Prediction Dataset"""
import numpy as np

""" An error was reported while executing this function: {error_message}. Please fix it."""
def build_model(x_train: np.ndarray, y_train: np.ndarray):
    """ Import modules, return a scikit model trained with x_train in shape {feature_shape} and y_train."""
    {pre_action}
    return model

def build_model(x_train: np.ndarray, y_train: np.ndarray):
    """ Fix the error in the previous function. """
    <FILL_ME>
    return model
'''

DS_CODE_FRAMEWORK = '''
import numpy as np

def build_model(x_train: np.ndarray, y_train: np.ndarray):
    """ Trained with x_train in shape {feature_shape}."""
    {action}
    return model
'''

LOG_CODE = '''
"""Classification with ROC AUC: {reward}, std: {std}, step: {step}."""
import numpy as np

def build_model(x_train: np.ndarray, y_train: np.ndarray):
    {action}
    return model


'''

CODE_EXAMPLE = """def train_model(x_train, y_train):
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier()
    model.fit(x_train, y_train)
    return model
"""

TASK_DESC = '''# This is a classification task. Include a train_model function only.

def train_model(x_train, y_train):
    # Import modules, build a scikit model, train and return it.
    <FILL_ME>
    return model
'''

TASK_DESC_CLASSIFICATION = "# This is a classification tasks of data science."

FEATURE_PROCESS_PROMPT = """
# Here is the feature processing function for classification task.
import numpy as np

def feature_process(features: np.ndarray):
    # Import necessary modules, preprocess features with shape {feature_shape}.
    <FILL_ME>
    return features
"""

TRAIN_MODEL_PROMPT = """
# Here is the model building function for classification task.
import numpy as np

def build_model(x_train: np.ndarray, y_train: np.ndarray):
    # Import modules, create a scikit model, train and return it.
    <FILL_ME>
    return model
"""

CODE_FRAMEWORK = """
# Here is a classification task with feature shape {feature_shape}.
import numpy as np

def feature_process(features: np.ndarray):
    {action_feature_process}
    return features

def build_model(x_train: np.ndarray, y_train: np.ndarray):
    {action_build_model}
    return model
"""
