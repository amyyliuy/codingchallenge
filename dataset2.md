# Dataset 2 Report – Multi-class Classification and Data Requirements

## 1. Problem and Data

Dataset 2 consists of **400 samples** described by 8 numerical features (`feature_1`–`feature_8`) and a binary label (`0` or `1`). The classes are perfectly balanced (**200** samples per class). The aim is to compare several classifiers and estimate the **minimum number of training samples** required to reach at least **70% accuracy**.

## 2. Methods

A `Preprocessor` class loads `dataset_2.csv`, reports basic statistics and missing values, and performs a stratified train–test split (75% train / 25% test → 300 train, 100 test). Missing values in the features are imputed using `SimpleImputer(strategy="median")`, and all 8 features are standardised with `StandardScaler` inside a `ColumnTransformer`.

A generic `Classifier` class wraps different scikit-learn models inside a pipeline. We evaluated:

- Logistic Regression (`logistic`)
- Support Vector Classifier with RBF kernel (`svc_rbf`)
- k-Nearest Neighbours with k = 5 (`knn_5`)
- Random Forest (`random_forest`)

For each model we computed:

- **5-fold cross-validation accuracy** on the training set (`cross_val_score`)
- **Test-set accuracy, precision, recall, F1**, and confusion matrices.

For the best model we also computed a **learning curve** via `learning_curve`, using 5-fold CV and eight training sizes between 10% and 100% of the data.

## 3. Results

Training-set cross-validation (mean ± std):

- Logistic Regression: **0.97 ± 0.02**
- SVC (RBF): **0.99 ± 0.01**
- KNN (k=5): **0.93 ± 0.03**
- Random Forest: **1.00 ± 0.00**

Test-set accuracy:

- Logistic Regression: **0.99**
- SVC (RBF): **1.00**
- KNN (k=5): **0.97**
- Random Forest: **1.00**

Confusion matrices show that SVC and Random Forest classify all 100 test samples correctly, while Logistic Regression and KNN make only one or a few errors. Overall, all models perform very well; non-linear models have a small advantage.

The learning curve for the best model (SVC with RBF kernel) shows CV accuracy quickly rising above 95%. The first training size at which CV accuracy exceeds **70%** is **32 training samples**, and performance stabilises close to 99–100% for larger training sizes.

## 4. Discussion and Recommendation

All models can separate the two classes extremely well on this synthetic data. Given the almost perfect cross-validation and test performance, the limiting factor is not model capacity but the inherent separability of the dataset.

For deployment on data with similar characteristics, we recommend:

- Using a **non-linear model** such as **SVC (RBF kernel)** or **Random Forest**, which achieve perfect accuracy and robust cross-validation performance.
- Ensuring a training set of at least **32 labelled samples** to reliably exceed 70% accuracy, noting that more data (≈ 200–300 samples) pushes performance close to 100%.

In more realistic scenarios with noisier measurements, Logistic Regression or Random Forest would be attractive due to their simplicity and interpretability, but on this dataset all four algorithms already perform near optimally.
