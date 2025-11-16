# Dataset 1 – Conductive vs Non-conductive Classification

## 1. Aim and data

The goal is to predict whether a material is **conductive** or **non-conductive** from ten measured properties (density, vacancy content, melting temperature, heat conductivity, band gap, crystallinity index, thermal expansion coefficient, Young’s modulus, hardness, lattice parameter). The dataset contains 5000 rows and 11 columns (10 features + `label`). The classes are imbalanced: 4506 non-conductive and 494 conductive samples.

## 2. Methods

A `Preprocessor` class loads `dataset_1.csv`, prints basic exploratory statistics, and reports missing values. The target column is `label`. The data are split into training and test sets using an 80/20 stratified split (4000 train, 1000 test) so that the class balance is preserved.

All input features are treated as numeric. A `ColumnTransformer` applies a pipeline of `SimpleImputer(strategy="median")` followed by `StandardScaler` to impute missing values and normalise the features.

A `BinaryClassifier` class wraps scikit-learn models inside a `Pipeline(preprocessor, classifier)`. Two classifiers were compared:

- **Logistic Regression** (linear model, `max_iter=1000`)
- **Random Forest** (`n_estimators=200`, `random_state=42`)

An `Evaluator` class computes accuracy, precision, recall, F1-score and the full classification report, and saves confusion matrices and feature-importance plots.

## 3. Results

On the held-out test set, both models achieved perfect performance:

- Logistic Regression: accuracy 1.00, precision 1.00, recall 1.00, F1-score 1.00
- Random Forest: accuracy 1.00, precision 1.00, recall 1.00, F1-score 1.00

The confusion matrices for both classifiers show **zero misclassifications**: all 99 conductive and 901 non-conductive samples in the test set are correctly classified.

Feature importance (from the best model) indicates that a small subset of features dominates the decision: band gap, lattice parameter, crystallinity index, Young’s modulus, thermal expansion coefficient and hardness have the highest weights. A bar chart visualises these importances.

To investigate redundancy, we ranked features by importance and retrained logistic regression using every subset size from 10 down to 1 feature. The “accuracy vs number of features” plot shows that test accuracy remains at 1.00 even when only the top few features are used; performance only degrades when very few features are retained.

## 4. Discussion and recommendation

The results suggest that Dataset 1 is **almost perfectly separable** using the provided features. Even a simple linear model (logistic regression) can classify the materials without error on the test set, and reducing the input to the most informative 4–6 features does not hurt performance. This is unlikely in noisy real-world measurements, so the dataset should be interpreted as an idealised scenario.

For a practical deployment based on this dataset, I would recommend:

- Using **logistic regression** as the main classifier, due to its simplicity and interpretability.
- Focusing on the most important features (especially band gap, lattice parameter and crystallinity index) to reduce measurement cost while keeping high predictive power.

Overall, the pipeline meets the brief by including preprocessing, model comparison, confusion matrices, feature-importance visualisation and a systematic feature-subset study.
