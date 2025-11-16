# Dataset 1 – Conductivity classification driven by band gap

## 1. Aim and data

The goal is to predict whether a material is **conductive** or **non-conductive** from ten measured properties: density, vacancy content, melting temperature, heat conductivity, band gap, crystallinity index, thermal expansion coefficient, Young’s modulus, hardness and lattice parameter. The dataset has 5000 rows and an imbalanced target (`label`: 4506 non-conductive, 494 conductive).

## 2. Methods

A `Preprocessor` class loads `dataset_1.csv`, prints summary statistics and missing values, and performs an 80/20 stratified train–test split (4000 train, 1000 test). All features are treated as numeric. A `ColumnTransformer` applies a pipeline of median imputation (`SimpleImputer`) and standardisation (`StandardScaler`) to all input features.

For classification we use a `BinaryClassifier` wrapper which builds a scikit-learn `Pipeline(preprocessor, classifier)`. Two models are compared:

- Logistic Regression (`max_iter=1000`)
- Random Forest (`n_estimators=200`, `random_state=42`)

An `Evaluator` class reports accuracy, precision, recall, F1-score, confusion matrices, and a **feature-importance bar plot**. For the best model we also perform a **feature subset study**, retraining logistic regression while reducing the number of features stepwise and plotting accuracy vs number of features (`feature_subset_accuracy.png`).

## 3. Results

On the held-out test set, both models achieve perfect metrics (accuracy/precision/recall/F1 = 1.00). The confusion matrices show zero misclassified samples.

The key result is the **feature-importance plot** for the best model. The importance of **`band_gap`** is dramatically larger than that of all other features; the remaining nine are essentially negligible by comparison. This means the classifier has effectively learned a **one-dimensional decision rule**: whether a material is conductive or not is determined almost entirely by its band-gap value.

Physically, this matches the standard picture from solid-state physics:  
- materials with a **small or zero band gap** allow electrons to move easily → **conductive**;  
- materials with a **large band gap** have no available states near the Fermi level → **non-conductive** (insulators).

The **feature_subset_accuracy** plot reinforces this. We ranked features by importance and retrained logistic regression using subsets from all 10 features down to only the single most important feature. Test accuracy stays at 1.00 for almost all subset sizes, and crucially, **using only `band_gap` already achieves perfect classification**. Adding the remaining features does not improve performance; it only adds redundancy.

## 4. Discussion and recommendation

Dataset 1 is therefore essentially a **threshold problem in band gap**: once the model knows the band-gap value, it can place a decision boundary at some critical band-gap range and correctly label almost every material as conductive or non-conductive. The other descriptors provide very little additional information about conductivity in this dataset.

For a practical model based on these data, I would recommend:

- Using a simple, interpretable classifier (logistic regression) with **`band_gap` as the primary input**.
- Optionally including a few secondary features for robustness, but recognising that they contribute almost nothing to predictive power compared to band gap.

This analysis shows that both the feature-importance bar chart and the feature-subset-accuracy curve tell a consistent story: **conductivity in Dataset 1 is almost completely determined by the material’s band gap.**
