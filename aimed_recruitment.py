from multiprocessing.reduction import register

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline

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
        n_neighbors=3, weights="distance", metric="manhattan"
    ))
])

pipe_knn.fit(x_train, y_train)

cv_score = np.round(cross_val_score(pipe_knn, x_test, y_test), 2)

print("Scores of data cross-validation each fold")
list(map(print, cv_score))
print(f"\nCross validation mean score: {np.mean(cv_score):.3}")
print(f"Standard deviation of CV score: {np.std(cv_score):.3}")