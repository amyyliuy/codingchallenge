# Dataset 1 Report – Conductive vs Non-conductive Classification

## 1. Problem and Data

The goal is to predict whether a material is **conductive** or **non-conductive** from 10 measured properties (density, vacancy content, melting temperature, heat conductivity, band gap, crystallinity index, thermal expansion coefficient, Young’s modulus, hardness, lattice parameter). The dataset contains **5000** rows and **11** columns (10 features + `label`). The classes are imbalanced: **4506 non-conductive** and **494 conductive** samples.

## 2. Methods

### Preprocessing

A `Preprocessor` class handles:

- Loading the CSV with pandas.
- Basic inspection (head, summary statistics, missing values).
- Train–test splitting (`train_test_split`, test size 0.2, stratified by `label` → 4000 train, 1000 test).
- Building a scikit-learn `ColumnTransformer` that applies:
  - `SimpleImputer(strategy="median")` to handle missing values in each feature.
  - `StandardScaler` to normalise all numeric features.

### Models

A `BinaryClassifier` class wraps scikit-learn models inside a `Pipeline(preprocessor + classifier)`. For Dataset 1 we tested:

- **Logistic Regression** (`LogisticRegression(max_iter=1000)`)
- **Random Forest** (`RandomForestClassifier(n_estimators=200, random_state=42)`)

An `Evaluator` class computes accuracy, precision, recall and F1-score, prints the classification report and saves confusion-matrix and feature-importance plots.

## 3. Results

Both models achieve **perfect performance on the held-out test set**:

- Logistic Regression: accuracy **1.00**, precision **1.00**, recall **1.00**, F1 **1.00**
- Random Forest: accuracy **1.00**, precision **1.00**, recall **1.00**, F1 **1.00**

Confusion matrices show **zero misclassifications** for both conductive (99 test samples) and non-conductive (901 test samples).

Feature importance (from logistic-regression coefficients / random-forest importances) indicates that classification is dominated by a subset of features, in particular:

- `band_gap`
- `lattice_parameter`
- `crystallinity_index`
- `young_modulus`
- `thermal_expansion_coeff`
- `hardness`

Feature-subset experiments using logistic regression with only the top 10, 8, 6 and 4 features all still give **100% test accuracy**, and the plot of accuracy vs number of features is flat at 1.0.

## 4. Discussion and Recommendation

The pipeline suggests that Dataset 1 is **almost perfectly separable** in the space of the measured properties: even a simple linear model (logistic regression) is able to classify all test materials correctly. In practice, real experimental data would contain noise and imperfect labels, so we would not expect 100% accuracy; these results should be interpreted as an optimistic upper bound on achievable performance for this synthetic dataset.

Given that logistic regression matches or exceeds the performance of the more complex random forest while being simpler and more interpretable, we recommend:

- Using **logistic regression** as the production classifier.
- Measuring at least the **top 4–6 features** (`band_gap`, `lattice_parameter`, `crystallinity_index`, `young_modulus`, plus optionally `thermal_expansion_coeff` and `hardness`). This subset already achieves perfect performance on the test set and would reduce experimental measurement cost without sacrificing accuracy on this dataset.
