from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

DATASET_1_PATH = "/Users/amyliumacbook/Library/Application Support/JetBrains/PyCharm2024.3/scratches/Data/dataset_1.csv"
DATASET_2_PATH = "/Users/amyliumacbook/Library/Application Support/JetBrains/PyCharm2024.3/scratches/Data/dataset_2.csv"

PLOTS_ROOT = Path("plots")
PLOTS_ROOT.mkdir(exist_ok=True)

# -------------------------------------------------------------------
# 1. PREPROCESSOR CLASS
# -------------------------------------------------------------------

class Preprocessor:
    """
    Handles:
      - loading data from CSV
      - basic inspection / EDA-style info
      - splitting into train / test sets
      - constructing a preprocessing pipeline
        (imputation + standardisation)
    """

    def __init__(self, data_path: str, target_col: str):
        self.data_path = Path(data_path)
        self.target_col = target_col
        self.df: Optional[pd.DataFrame] = None

    # ---- data loading / info -------------------------------------------------

    def load_data(self) -> pd.DataFrame:   # Read the CSV file once and keep it in self.df for later use
        print(f"\n[Preprocessor] Loading data from {self.data_path} ...")
        self.df = pd.read_csv(self.data_path)
        print(f"[Preprocessor] Shape: {self.df.shape}")
        return self.df

    def show_basic_info(self) -> None:
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        print("\n[Preprocessor] Head:")
        print(self.df.head())
        print("\n[Preprocessor] Describe:")
        print(self.df.describe(include="all"))
        print("\n[Preprocessor] Missing values per column:")
        print(self.df.isna().sum())
        print("\n[Preprocessor] Target value counts:")
        print(self.df[self.target_col].value_counts())

    # ---- splitting -----------------------------------------------------------

    def get_features_and_target(self) -> Tuple[pd.DataFrame, pd.Series]:
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        # Separate the DataFrame into:
        #   X – all feature columns
        #   y – target label column
        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]
        return X, y

    def train_test_split(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        X, y = self.get_features_and_target()
        y_strat = y if stratify else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y_strat
        )
        print(
            f"\n[Preprocessor] Train shape: {X_train.shape}, "
            f"Test shape: {X_test.shape}"
        )
        return X_train, X_test, y_train, y_test

    # ---- preprocessing pipeline ---------------------------------------------

    def build_numeric_pipeline(
        self,
        selected_features: Optional[List[str]] = None,
    ) -> Tuple[ColumnTransformer, List[str]]:
        # Choose which numeric features to use and create a ColumnTransformer
        # that applies:
        #   1) SimpleImputer(median) to handle missing values
        #   2) StandardScaler() to normalise feature scales
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        if selected_features is None:
            feature_names = self.df.drop(columns=[self.target_col]).columns.tolist()
        else:
            feature_names = selected_features

        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, feature_names),
            ]
        )

        print(f"[Preprocessor] Using features: {feature_names}")
        return preprocessor, feature_names


# -------------------------------------------------------------------
# 2. CLASSIFIER (used as BinaryClassifier for dataset 1)
# -------------------------------------------------------------------

class Classifier:
    """
    Generic classifier wrapper which holds:
      - a scikit-learn model (e.g. LogisticRegression, RandomForest, KNN, SVM, DecisionTree)
      - the preprocessing pipeline
    It exposes .fit(), .predict() and .get_feature_importances().
    """

    def __init__(
        # Choose the underlying sklearn model based on a simple string name.
        # This makes it easy to loop over several models with one unified interface.
        self,
        model_name: str,
        preprocessor: ColumnTransformer,
        feature_names: List[str],
    ):
        self.model_name = model_name
        self.feature_names = feature_names

        # Map from name to sklearn model
        if model_name == "logistic":
            model = LogisticRegression(max_iter=1000)
        elif model_name == "random_forest":
            model = RandomForestClassifier(
                n_estimators=200, random_state=42
            )
        elif model_name == "knn_5":
            model = KNeighborsClassifier(n_neighbors=5)
        elif model_name == "svm":
            # SVM classifier, here using RBF kernel
            model = SVC(kernel="rbf", gamma="scale")
        elif model_name == "decision_tree":
            model = DecisionTreeClassifier(random_state=42)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        # Full pipeline: preprocessing + classifier
        self.pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", model),
            ]
        )

    # ---- basic API -----------------------------------------------------------

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "Classifier":
        print(f"\n[Classifier] Fitting model: {self.model_name}")
        self.pipeline.fit(X_train, y_train)
        return self

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict(X_test)

    # ---- feature importance --------------------------------------------------

    def get_feature_importances(self) -> Optional[Dict[str, float]]:
        """
        Extract a simple feature-importance measure when the underlying model
        supports it:
          - linear models: use absolute value of the learned coefficients
          - tree models  : use the model's built-in feature_importances_
        For models like KNN or kernel SVM, no feature importances are available.
        """
        clf = self.pipeline.named_steps["classifier"]

        # Linear-type models with coef_
        if hasattr(clf, "coef_"):
            coef = np.ravel(clf.coef_)
            importances = np.abs(coef)
            print("[Classifier] Using absolute coefficients as importance.")
        # Tree-based models
        elif hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_
            print("[Classifier] Using feature_importances_ from tree model.")
        else:
            print(
                "[Classifier] Model does not expose feature importances "
                f"({self.model_name})."
            )
            return None

        importance_dict = dict(zip(self.feature_names, importances))
        importance_dict = dict(
            sorted(importance_dict.items(), key=lambda kv: kv[1], reverse=True)
        )
        return importance_dict


# For Dataset 1 we can simply use Classifier as a BinaryClassifier:
BinaryClassifier = Classifier


# -------------------------------------------------------------------
# 3. EVALUATOR CLASS
# -------------------------------------------------------------------

class Evaluator:
    """
    Collects all evaluation and visualisation logic in one place:

      - metrics(): prints accuracy, precision, recall, F1 and a full
                   classification report.
      - plot_confusion_matrix(): saves confusion matrices as PNGs.
      - plot_feature_importances(): bar chart of feature importance.
      - plot_feature_subset_performance(): accuracy vs number of features
                                           (Dataset 1).
      - plot_learning_curve(): training / CV accuracy vs training set size
                               (Dataset 2).
      - plot_cv_accuracy_bar(): compares mean CV accuracy across models
                                (Dataset 2).
    """
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- metrics -------------------------------------------------------------

    def metrics(
        self, y_true: pd.Series, y_pred: np.ndarray, model_name: str
    ) -> Dict[str, float]:
        acc = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )

        print(f"\n[Evaluator] Metrics for {model_name}:")
        print(f"  Accuracy : {acc:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall   : {recall:.4f}")
        print(f"  F1-score : {f1:.4f}")
        print("\nClassification report:")
        print(classification_report(y_true, y_pred, zero_division=0))

        return {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    # ---- confusion matrix ----------------------------------------------------

    def plot_confusion_matrix(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        model_name: str,
        filename: str,
    ) -> None:
        labels = sorted(pd.unique(y_true))
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        fig, ax = plt.subplots(figsize=(5, 5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"Confusion matrix: {model_name}")
        plt.tight_layout()

        path = self.output_dir / filename
        plt.savefig(path)
        plt.close()
        print(f"[Evaluator] Saved confusion matrix to {path}")

    # ---- feature importance --------------------------------------------------

    def plot_feature_importances(
        self,
        feature_importances: Dict[str, float],
        title: str,
        filename: str,
        top_n: Optional[int] = None,
    ) -> None:
        if not feature_importances:
            print("[Evaluator] No feature importances to plot.")
            return

        items = list(feature_importances.items())
        if top_n is not None:
            items = items[:top_n]

        features, values = zip(*items)

        plt.figure(figsize=(8, 5))
        y_pos = np.arange(len(features))
        plt.barh(y_pos, values)
        plt.yticks(y_pos, features)
        plt.gca().invert_yaxis()
        plt.xlabel("Importance")
        plt.title(title)
        plt.tight_layout()

        path = self.output_dir / filename
        plt.savefig(path)
        plt.close()
        print(f"[Evaluator] Saved feature importance plot to {path}")

    # ---- feature subset performance (Dataset 1) ------------------------------

    def plot_feature_subset_performance(
        self,
        subset_sizes: List[int],
        accuracies: List[float],
        title: str,
        filename: str,
    ) -> None:
        plt.figure(figsize=(6, 4))
        plt.plot(subset_sizes, accuracies, marker="o")
        plt.xlabel("Number of features")
        plt.ylabel("Accuracy")
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()

        path = self.output_dir / filename
        plt.savefig(path)
        plt.close()
        print(f"[Evaluator] Saved feature-subset performance plot to {path}")

    # ---- learning curve (Dataset 2) -----------------------------------------

    def plot_learning_curve(
        self,
        train_sizes: np.ndarray,
        train_scores: np.ndarray,
        test_scores: np.ndarray,
        title: str,
        filename: str,
    ) -> None:
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        plt.figure(figsize=(7, 5))

        # Smaller markers and thinner lines so the dense curve is clearer
        plt.plot(
            train_sizes,
            train_mean,
            marker="o",
            linestyle="-",
            markersize=3,
            linewidth=1.0,
            label="Training score",
        )
        plt.plot(
            train_sizes,
            test_mean,
            marker="o",
            linestyle="-",
            markersize=3,
            linewidth=1.0,
            label="Cross-validation score",
        )

        plt.fill_between(
            train_sizes,
            train_mean - train_std,
            train_mean + train_std,
            alpha=0.15,
        )
        plt.fill_between(
            train_sizes,
            test_mean - test_std,
            test_mean + test_std,
            alpha=0.15,
        )

        plt.xlabel("Number of training samples")
        plt.ylabel("Accuracy")
        plt.title(title)
        plt.legend(loc="best")
        plt.grid(True)
        plt.tight_layout()

        path = self.output_dir / filename
        plt.savefig(path)
        plt.close()
        print(f"[Evaluator] Saved learning curve to {path}")

    # ---- CV accuracy bar chart (Dataset 2) ----------------------------------

    def plot_cv_accuracy_bar(
        # Bar chart for mean CV accuracy of each candidate model.
        # Y-axis is restricted to 0.90–1.01 so small differences
        # between 0.93 and 1.00 are clearly visible.
        self,
        model_names: List[str],
        cv_means: List[float],
        title: str,
        filename: str,
    ) -> None:
        """
        Bar chart of mean cross-validation accuracy for each model.
        Y-axis is zoomed so differences between ~0.93–1.00 are visible.
        """
        plt.figure(figsize=(7, 5))

        indices = np.arange(len(model_names))
        plt.bar(indices, cv_means)

        plt.xticks(indices, model_names, rotation=15)
        plt.ylabel("Mean CV accuracy")
        plt.title(title)

        # Zoom y-axis to highlight differences
        plt.ylim(0.90, 1.00)

        plt.grid(axis="y", linestyle="--", alpha=0.4)
        plt.tight_layout()

        path = self.output_dir / filename
        plt.savefig(path)
        plt.close()
        print(f"[Evaluator] Saved CV accuracy bar chart to {path}")


# -------------------------------------------------------------------
# 4. PIPELINES FOR DATASET 1 AND DATASET 2
# -------------------------------------------------------------------

PLOTS_ROOT = Path("plots")
PLOTS_ROOT.mkdir(exist_ok=True)


def run_dataset1_pipeline():
    """
    High-level pipeline for Dataset 1:

    1. Load and inspect the materials dataset.
    2. Build a numeric preprocessing pipeline.
    3. Train Logistic Regression and Random Forest models.
    4. Compare their test accuracy and keep the best model.
    5. For the best model:
         - plot feature importances
         - rerun Logistic Regression with progressively fewer top features
           and plot accuracy vs number of features.
    """
    print("\n" + "=" * 70)
    print("DATASET 1: Binary classification and feature selection")
    print("=" * 70)

    # --- step 1: preprocessing / splitting -----------------------------------
    pre = Preprocessor(DATASET_1_PATH, target_col="label")
    pre.load_data()
    pre.show_basic_info()
    preprocessor_all, feature_names = pre.build_numeric_pipeline()
    X_train, X_test, y_train, y_test = pre.train_test_split(test_size=0.2, stratify=True)

    evaluator = Evaluator(PLOTS_ROOT / "dataset1")

    # --- step 2: try two binary classifiers ----------------------------------
    models = {
        "LogisticRegression": BinaryClassifier(
            "logistic", preprocessor_all, feature_names
        ),
        "RandomForest": BinaryClassifier(
            "random_forest", preprocessor_all, feature_names
        ),
    }

    best_name = None
    best_acc = -np.inf
    best_model: Optional[BinaryClassifier] = None

    for name, clf in models.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        metrics = evaluator.metrics(y_test, y_pred, name)
        evaluator.plot_confusion_matrix(
            y_test, y_pred, model_name=name, filename=f"confusion_{name}.png"
        )

        if metrics["accuracy"] > best_acc:
            best_acc = metrics["accuracy"]
            best_name = name
            best_model = clf

    print(
        f"\n[Dataset 1] Best test accuracy: {best_acc:.4f} "
        f"with model {best_name}"
    )

    # --- step 3: feature importance and subsets ------------------------------

    if best_model is None:
        return

    importances = best_model.get_feature_importances()
    if importances is None:
        return

    evaluator.plot_feature_importances(
        importances,
        title=f"Feature importances ({best_name})",
        filename="feature_importances.png",
    )

    # ranked_features is ordered by importance (most important first)
    ranked_features = list(importances.keys())
    n_features = len(ranked_features)

    # Evaluate ALL subset sizes: n_features, n_features-1, ..., 1
    subset_sizes = list(range(n_features, 0, -1))
    subset_accuracies = []

    print("\n[Dataset 1] Feature subset experiments (all sizes):")
    for k in subset_sizes:
        top_feats = ranked_features[:k]
        print(f"  Using top {k} features: {top_feats}")

        preproc_k, feats_k = pre.build_numeric_pipeline(selected_features=top_feats)
        clf_k = BinaryClassifier("logistic", preproc_k, feats_k)
        clf_k.fit(X_train, y_train)
        y_pred_k = clf_k.predict(X_test)
        metrics_k = evaluator.metrics(y_test, y_pred_k, f"LogReg_top_{k}")
        subset_accuracies.append(metrics_k["accuracy"])

    evaluator.plot_feature_subset_performance(
        subset_sizes,
        subset_accuracies,
        title="Accuracy vs number of features (LogReg)",
        filename="feature_subset_accuracy.png",
    )


def run_dataset2_pipeline():
    """
    High-level pipeline for Dataset 2:

    1. Load and inspect the binary dataset (labels 0/1).
    2. Build a numeric preprocessing pipeline.
    3. Define five candidate models:
         - Logistic Regression
         - KNN (k = 5)
         - Random Forest
         - SVM with RBF kernel
         - Decision Tree
    4. For each model:
         - compute 5-fold CV accuracy on the training set
         - train on the full training set
         - evaluate on the held-out test set and save confusion matrix
         - track which model achieves the best test accuracy.
    5. Plot a bar chart comparing mean CV accuracy across all models.
    6. For the best model:
         - compute a learning curve using increasing training sizes
         - plot training vs cross-validation accuracy
         - estimate the minimum number of samples required to reach
           70% accuracy.
    """
    print("\n" + "=" * 70)
    print("DATASET 2: Model comparison and learning curve")
    print("=" * 70)

    pre = Preprocessor(DATASET_2_PATH, target_col="label")
    pre.load_data()
    pre.show_basic_info()
    preprocessor_all, feature_names = pre.build_numeric_pipeline()
    X_train, X_test, y_train, y_test = pre.train_test_split(test_size=0.25, stratify=True)

    evaluator = Evaluator(PLOTS_ROOT / "dataset2")

    # --- step 1: define candidate models -------------------------------------

    candidates = {
        "LogisticRegression": Classifier("logistic", preprocessor_all, feature_names),
        "KNN_5": Classifier("knn_5", preprocessor_all, feature_names),
        "RandomForest": Classifier("random_forest", preprocessor_all, feature_names),
        "SVM": Classifier("svm", preprocessor_all, feature_names),
        "DecisionTree": Classifier("decision_tree", preprocessor_all, feature_names),
    }

    best_name = None
    best_acc = -np.inf
    best_pipeline: Optional[Pipeline] = None

    # store mean CV accuracies for bar chart
    cv_means_dict: Dict[str, float] = {}

    # --- step 2: cross-validation + test evaluation --------------------------

    for name, clf in candidates.items():
        print(f"\n[Dataset 2] === {name} ===")
        # cross-validation on training set
        cv_scores = cross_val_score(
            clf.pipeline, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1
        )
        mean_cv = cv_scores.mean()
        std_cv = cv_scores.std()
        print(
            f"  CV accuracy: mean={mean_cv:.4f}, "
            f"std={std_cv:.4f}"
        )

        # save mean CV accuracy for plotting later
        cv_means_dict[name] = mean_cv

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        metrics = evaluator.metrics(y_test, y_pred, name)
        evaluator.plot_confusion_matrix(
            y_test, y_pred, model_name=name, filename=f"confusion_{name}.png"
        )

        if metrics["accuracy"] > best_acc:
            best_acc = metrics["accuracy"]
            best_name = name
            best_pipeline = clf.pipeline

    print(
        f"\n[Dataset 2] Best test accuracy: {best_acc:.4f} "
        f"with model {best_name}"
    )

    # --- step 2.5: bar chart of CV accuracies --------------------------------

    evaluator.plot_cv_accuracy_bar(
        model_names=list(cv_means_dict.keys()),
        cv_means=list(cv_means_dict.values()),
        title="Dataset 2: Cross-validation accuracy by model",
        filename="cv_accuracy_bar.png",
    )

    # --- step 3: learning curve for best model -------------------------------

    if best_pipeline is None:
        return

    X, y = pre.get_features_and_target()

    print("\n[Dataset 2] Computing learning curve for best model...")

    # learning_curve requires that the number of samples used at each step
    # is at least `cv` (here 5) and at most n_samples * (cv-1)/cv.
    n_samples = X.shape[0]  # 400 in your dataset
    cv_folds = 5
    min_train_size = cv_folds  # smallest possible size = 5
    max_train_size = int(n_samples * (cv_folds - 1) / cv_folds)  # 400 * 4/5 = 320

    # Absolute train sizes: 5, 6, 7, ..., 320
    train_sizes = np.arange(min_train_size, max_train_size + 1, 1)

    train_sizes, train_scores, test_scores = learning_curve(
        best_pipeline,
        X,
        y,
        cv=cv_folds,
        train_sizes=train_sizes,
        scoring="accuracy",
        n_jobs=-1,
    )

    evaluator.plot_learning_curve(
        train_sizes,
        train_scores,
        test_scores,
        title=f"Learning curve ({best_name})",
        filename="learning_curve.png",
    )

    test_mean = np.mean(test_scores, axis=1)
    threshold = 0.70
    min_samples = None

    for n, score in zip(train_sizes, test_mean):
        print(f"  Train size {n:4d}: CV accuracy = {score:.4f}")
        if score >= threshold and min_samples is None:
            min_samples = int(n)

    if min_samples is not None:
        print(
            f"\n[Dataset 2] Minimum samples to reach "
            f"{threshold * 100:.0f}% accuracy ≈ {min_samples}"
        )
    else:
        print(
            f"\n[Dataset 2] Accuracy did not reach {threshold * 100:.0f}% "
            f"in the explored training sizes."
        )


# -----------------------------------------------------------
# Main entry point
# -----------------------------------------------------------

if __name__ == "__main__":
    run_dataset1_pipeline()
    run_dataset2_pipeline()
