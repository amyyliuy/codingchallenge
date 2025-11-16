Dataset 2 report – model comparison and learning curve

Dataset 2 consists of 400 samples with eight anonymous numerical features (feature_1 … feature_8) and a balanced binary label (200 zeros and 200 ones). As with dataset 1, missing feature values are imputed with the median and all variables are standardised inside a preprocessing pipeline. A stratified 75/25 split produces 300 training and 100 test instances.

Four classifiers were evaluated on Dataset 2: logistic regression, SVC with RBF kernel, k-nearest neighbours (k = 5), and random forest. Five-fold cross-validation on the training set gives:


KNN: mean accuracy ≈ 0.93 (lowest)


Logistic regression: mean accuracy ≈ 0.97


SVC_rbf: mean accuracy ≈ 0.987


Random forest: mean accuracy = 1.00


On the held-out test set, the same pattern appears. KNN reaches 0.97 accuracy, misclassifying 3 positive samples as class 0. Logistic regression achieves 0.99 accuracy with only one positive misclassified. Both SVC_rbf and random forest obtain 1.00 accuracy and perfectly clean confusion matrices, correctly labelling all 50 class-0 and 50 class-1 samples.

Because SVC_rbf combines excellent cross-validation accuracy with perfect test performance, it is chosen for the learning-curve analysis. The learning curve is computed by increasing the training size from 5 up to 320 samples and measuring training and cross-validation scores with five-fold CV at each step. With small training sets of 5 to 9 samples, the cross-validation accuracy fluctuates around 0.58–0.68, indicating high variance and unreliable generalisation when data are scarce. From around 10 samples onward the accuracy quickly rises above 0.8 and then into the 0.9+ range. The script reports that approximately 10 training samples are sufficient to exceed 70% accuracy, and as expected performance continues to improve as more data are added.

Beyond about 60–80 training samples, both the training and cross-validation curves stabilise around 0.98–0.99, with only a small gap between them. This suggests that the SVC model is powerful enough to fit the underlying structure but not severely overfitting: adding more data yields diminishing returns, but does not cause performance to collapse. A feature-importance plot derived from the random forest model (using its built-in feature_importances_) indicates that predictive power is shared across several features rather than being dominated by a single variable, consistent with the idea that the decision boundary in dataset 2 is genuinely multi-dimensional.

Overall, dataset 2 shows that several modern classifiers can achieve near-perfect performance on this problem, with SVC_rbf offering an excellent trade-off between accuracy and robustness. The learning curve confirms that a modest amount of data (tens of samples) is enough to train a highly accurate model, while additional data mainly refine the estimate and reduce variance.

<img width="385" height="384" alt="image" src="https://github.com/user-attachments/assets/f830f112-cfb0-41ea-a273-8fcafb0363c3" />
<img width="378" height="382" alt="image" src="https://github.com/user-attachments/assets/7b51c1c3-9eba-4d1c-af13-df4e5c8fd661" />
<img width="376" height="384" alt="image" src="https://github.com/user-attachments/assets/13859419-5ee4-49b8-ad07-2c23aaa57247" />
<img width="378" height="383" alt="image" src="https://github.com/user-attachments/assets/fa4d8e28-4328-42ae-a3a4-1301cbfd59a2" />
<img width="551" height="386" alt="image" src="https://github.com/user-attachments/assets/5b9147f5-d594-4b9b-8c7c-a5101523bb16" />
