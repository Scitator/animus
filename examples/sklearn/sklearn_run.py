from pprint import pprint

import numpy as np
from sklearn.datasets import make_circles, make_classification, make_moons
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from animus import IExperiment


# https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
class Experiment(IExperiment):
    def on_experiment_start(self, exp: "IExperiment"):
        super().on_experiment_start(exp)
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
        # setup model
        self.classifiers = {
            "Nearest Neighbors": KNeighborsClassifier(3),
            "Linear SVM": SVC(kernel="linear", C=0.025),
            "RBF SVM": SVC(gamma=2, C=1),
            "Gaussian Process": GaussianProcessClassifier(1.0 * RBF(1.0)),
            "Decision Tree": DecisionTreeClassifier(max_depth=5),
            "Random Forest": RandomForestClassifier(
                max_depth=5, n_estimators=10, max_features=1
            ),
            "Neural Net": MLPClassifier(alpha=1, max_iter=1000),
            "AdaBoost": AdaBoostClassifier(),
            "Naive Bayes": GaussianNB(),
            "QDA": QuadraticDiscriminantAnalysis(),
        }

    def run_dataset(self) -> None:
        X, y = self.dataset
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4, random_state=42
        )
        for name, clf in self.classifiers.items():
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
            self.dataset_metrics[name] = score

    def on_epoch_end(self, exp: "IExperiment") -> None:
        super().on_epoch_end(exp)
        pprint(self.epoch_metrics)


if __name__ == "__main__":
    Experiment().run()
