from multiprocessing.reduction import register

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, PrecisionRecallDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

data = pd.read_csv("task_data.csv")

for col in ["CTR - Cardiothoracic Ratio", "Inscribed circle radius", "Heart perimeter"]:
    data[col] = data[col].str.replace(pat=',',repl='.', regex=False).astype(float)

x = data[["Heart width", "Lung width", "CTR - Cardiothoracic Ratio",
          "xx", "yy", "xy", "normalized_diff", "Inscribed circle radius",
          "Polygon Area Ratio", "Heart perimeter", "Heart area ", "Lung area"]]

y = data["Cardiomegaly"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

pipe_knn = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("model", KNeighborsClassifier(
        n_neighbors=5, weights="uniform", metric="manhattan"
    ))
])

pipe_knn.fit(x_train, y_train)

cv_score = np.round(cross_val_score(pipe_knn, x_test, y_test), 2)

print("Scores of data cross-validation each fold")
list(map(print, cv_score))
print(f"\nCross validation mean score: {np.mean(cv_score):.3}")
print(f"Standard deviation of CV score: {np.std(cv_score):.3}")

'''
param_grid = {
    "model__n_neighbors": [3, 5, 7, 11, 15],
    "model__weights": ["uniform", "distance"],
    "model__metric": ["minkowski", "manhattan", "euclidean", "chebyshev"],
}

rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=100, random_state=None)

pipe_knn = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("model", KNeighborsClassifier())
])

grid_search =GridSearchCV(pipe_knn, param_grid=param_grid, cv=rskf, verbose=1, n_jobs=-1)
grid_search.fit(x_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best accuracy: {grid_search.best_score_}")
'''

y_prediction_knn = pipe_knn.predict(x_test)

acc_knn = accuracy_score(y_test, y_prediction_knn)
prec_knn = precision_score(y_test, y_prediction_knn)
rcl_knn = recall_score(y_test, y_prediction_knn)
f1c_knn = f1_score(y_test, y_prediction_knn)

fpr, tpr, thresholds = metrics.roc_curve(y_test, y_prediction_knn)
roc_auc = metrics.auc(fpr, tpr)
display1 = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, name='KNeighborsClassifier')

precision, recall, _ = metrics.precision_recall_curve(y_test, y_prediction_knn)
display2 = metrics.PrecisionRecallDisplay(precision=precision, recall=recall)
display1.plot()
display2.plot()

print(f"Accuracy of KNN classifier: {acc_knn:.3f}")
print(f"Precision of KNN classifier: {prec_knn:.3f}")
print(f"recall of KNN classifier: {rcl_knn:.3f}")
print(f"F1-score of KNN classifier: {f1c_knn:.3f}")

plt.show()