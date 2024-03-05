import sys
# sys.path.append("../../../")
import numpy as np
import traceback
import sklearn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from etpo.envs.datascience.prompts.scikit_prompts import *
from etpo.envs.datascience.data_utils import *
from importlib import reload, import_module
import timeout_decorator
# import mappo.envs.datascience.cache.train as train

def load_dataset(dataset_name, split):
    if dataset_name == "kidney_stone":
        return load_kidney_stone(split=split)
    elif dataset_name == "pharyngitis":
        return load_pharyngitis(split=split)
    elif dataset_name == "health_insurance":
        return load_health_insurance(split=split)
    elif dataset_name == "spaceship_titanic":
        return load_spaceship_titanic(split=split)
    else:
        return load_caafe_dataset(dataset_name, split=split)

class ScikitEnv:

    def __init__(self, mode, rank, dataset_name, split=False):
        # super().__init__(**kwargs)
        self.split=split
        self.rank = rank
        
        if self.split:
            self.x_train, self.x_test, self.y_train, self.y_test, self.data_disc = load_dataset(dataset_name = dataset_name, split=self.split)
            self.feature_shape = self.x_train.shape
        else:
            self.x, self.y, self.data_disc = load_dataset(dataset_name = dataset_name, split=self.split)
            self.feature_shape = self.x.shape
        print("feature_shape: ", self.feature_shape)
        self.n_agents = 1
        self.max_step = 5

        self.code_file = f"../../etpo/envs/datascience/cache/{mode}_{rank}.py"
        # create the file if it doesn't exist
        with open(self.code_file, "w") as f:
            f.write("")
        self.cache_module = import_module(f"etpo.envs.datascience.cache.{mode}_{rank}")

        self.step_count = 0

    def reset(self):
        # n_features = self.dataset.data.shape[1]
        # n_classes = len(self.dataset.target_names)
        # task_desc = TASK_DESC
        # print("task_desc: ", task_desc)
        # obs = np.array([task_desc for _ in range(self.n_agents)], dtype=np.object_)
        obs1 = DS_CODE_PROMPT.format(data_disc=self.data_disc, feature_shape=self.feature_shape)
        obs = np.array([obs1], dtype=np.object_)
        self.step_count = 0

        return obs
    
    def step(self, action, full_action=False):
        self.step_count += 1
        if full_action:
            code = action
        else:
            action = action[0]
            code = DS_CODE_FRAMEWORK.format(feature_shape=self.feature_shape,
                                            action=action)
        # print("=========code=========\n", code)
        with open(self.code_file, "w") as f:
            f.write(code)
        status = "Code executes normally."
        std = 0.0
        dones = np.ones((self.n_agents), dtype=bool)
        obs = np.array([status for _ in range(self.n_agents)], dtype=np.object_)
        try:
            if "GridSearchCV(" in code:
                raise Exception("do not use GridSearchCV.")
            if self.split:
                score = self.run_ds_code()
            else:
                score, std = self.run_ds_code_cv5()
        except Exception as e:
            # traceback.print_exc()
            status = str(e)
            score = -1.0
            obs1 = DS_CODE_REFLECTION_PROMPT.format(data_disc=self.data_disc, error_message=status, 
                                                    feature_shape=self.feature_shape, pre_action=action)
            obs = np.array([obs1], dtype=np.object_)
            if self.step_count < self.max_step:
                dones = np.zeros((self.n_agents), dtype=bool)
        print("Thread: {}, Score: {}, Status: {}".format(self.rank, score, status))
        
        rewards = [[score] for _ in range(self.n_agents)]
        # dones = np.ones((self.n_agents), dtype=bool)
        # obs = np.array([status for _ in range(self.n_agents)], dtype=np.object_)
        infos = [{"status": status, "std": std} for _ in range(self.n_agents)]
        return obs, rewards, dones, infos
    
    @timeout_decorator.timeout(60)
    def run_ds_code(self):
        reload(self.cache_module)
        model = self.cache_module.build_model(self.x_train, self.y_train)
        y_pred = model.predict_proba(self.x_test)
        if np.shape(y_pred)[1] > 2:
            score = roc_auc_score(self.y_test, y_pred, multi_class='ovo')
        else:
            score = roc_auc_score(self.y_test, y_pred[:, 1])
        return score

    @timeout_decorator.timeout(300)
    def run_ds_code_cv5(self):
        reload(self.cache_module)
        scores = []
        for i in range(5):
            x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.5, random_state=i, shuffle=True)
            model = self.cache_module.build_model(x_train, y_train)
            y_pred = model.predict_proba(x_test)
            if np.shape(y_pred)[1] > 2:
                auc_score = roc_auc_score(y_test, y_pred, multi_class='ovo')
            else:
                auc_score = roc_auc_score(y_test, y_pred[:, 1])
            scores.append(auc_score)
        print("scores: ", scores)
        return np.mean(scores), np.std(scores)

    def render(self, **kwargs):
        # self.env.render(**kwargs)
        pass

    def close(self):
        pass

    def seed(self, seed):
        np.random.seed(seed)

    def get_env_info(self):

        env_info = {
                    "n_agents": self.n_agents
                    }
        return env_info


if __name__ == "__main__":
    print("Unit Test (ScikitEnv)")
    env = ScikitEnv(mode="train", rank=0)
    env.reset()

    action = "from sklearn.ensemble import RandomForestClassifier\n    model = RandomForestClassifier(n_estimators=100)\n    model.fit(x_train, y_train)"
    env.step([action])

