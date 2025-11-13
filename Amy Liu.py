# Person C — Dataset 1: Feature importance & Recommendation

#Run L1-LogReg, RFECV, and permutation_importance on best_pipe.
#Produce ranking_df with columns: feature, method, rank/score; also an ensemble rank (average of normalized ranks).
#Implement accuracy_vs_k_curve using ensemble rank; evaluate k=1..10 via CV; plot.
#Write d1_feature_recommendation.txt:
#- Best CV accuracy
#- Smallest k within 1% of best
#- Final feature list in order, plus 2–3 sentence rationale.
#- Study: RFECV, permutation importance pitfalls (correlated features), CI vs overfitting in wrapper methods.

#import class
import pandas as pd
data1 = pd.read_csv('Dataset1.csv')
data2 = pd.read_csv('Dataset2.csv')
print(data1.head(), data1.info(), data1.describe())

#Preprocessor class
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class Preprocessor:
    def __init__(self, df, target_col):
        self.df = df
        self.target_col = target_col

    def clean(self):
        self.df = self.df.dropna()
        return self

    def split(self, test_size=0.2, random_state=42):
        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state)
        self.scaler = StandardScaler().fit(X_train)
        return (self.scaler.transform(X_train),
                self.scaler.transform(X_test),
                y_train, y_test,
                X.columns)

#classifier class
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

class Classifier:
    def __init__(self, model_name='logreg'):
        if model_name=='logreg':
            self.model = LogisticRegression(max_iter=200)
        elif model_name=='rf':
            self.model = RandomForestClassifier(n_estimators=200)
        elif model_name=='svm':
            self.model = SVC(kernel='rbf', probability=True)
        else:
            raise ValueError('Unknown model')

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def feature_importance(self, feature_names):
        if hasattr(self.model, 'coef_'):
            return dict(zip(feature_names, self.model.coef_[0]))
        elif hasattr(self.model, 'feature_importances_'):
            return dict(zip(feature_names, self.model.feature_importances_))

#Evaluator class
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np

class Evaluator:
    @staticmethod
    def evaluate(y_true, y_pred):
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted')
        }
        return metrics

    @staticmethod
    def plot_confusion(y_true, y_pred, title):
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(cm)
        disp.plot(cmap='Blues')
        plt.title(title)
        plt.show()

#Feature Importance (Dataset 1)
importances = clf.feature_importance(feature_names)
plt.bar(importances.keys(), importances.values())
plt.xticks(rotation=60)
plt.title("Feature importance (Dataset 1)")
plt.show()




