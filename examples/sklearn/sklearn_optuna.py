import os
from pprint import pprint

import numpy as np
import optuna
from sklearn.datasets import make_circles, make_classification, make_moons
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from animus import IExperiment


class Experiment(IExperiment):
    def on_tune_start(self):
        # setup data
        X, y = make_classification(
            n_features=2,
            n_redundant=0,
            n_informative=2,
            random_state=1,
            n_clusters_per_class=1,
        )
        rng = np.random.RandomState(2)
        X += 2 * rng.uniform(size=X.shape)
        linearly_separable = (X, y)
        self.datasets = {
            "moons": make_moons(noise=0.3, random_state=0),
            "circles": make_circles(noise=0.2, factor=0.5, random_state=1),
            "linear": linearly_separable,
        }

    def on_experiment_start(self, exp: "IExperiment"):
        super().on_experiment_start(exp)
        # setup model
        max_depth = self._trial.suggest_int("max_depth", 2, 32, log=True)
        n_estimators = self._trial.suggest_int("n_estimators", 2, 32, log=True)
        self.classifier = RandomForestClassifier(
            max_depth=max_depth, n_estimators=n_estimators, max_features=1
        )

    def run_dataset(self) -> None:
        X, y = self.dataset
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4, random_state=42
        )
        self.classifier.fit(X_train, y_train)
        score = self.classifier.score(X_test, y_test)
        self.dataset_metrics = {"score": score}

    def on_epoch_end(self, exp: "IExperiment") -> None:
        super().on_epoch_end(exp)
        pprint(self.epoch_metrics)
        self._score = np.mean([v["score"] for v in self.epoch_metrics.values()])

    def _objective(self, trial) -> float:
        self._trial = trial
        self.run()
        return self._score

    def tune(self, n_trials: int):
        self.on_tune_start()
        self.study = optuna.create_study(direction="maximize")
        self.study.optimize(self._objective, n_trials=n_trials, n_jobs=1)
        logfile = os.path.join(
            os.path.abspath(os.path.dirname(__file__)), "sklearn_optuna.csv"
        )
        df = self.study.trials_dataframe()
        df.to_csv(logfile, index=False)


if __name__ == "__main__":
    Experiment().tune(n_trials=100)
