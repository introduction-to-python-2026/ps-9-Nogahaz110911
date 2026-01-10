import pandas as pd

df = pd.read_csv("parkinsons.csv")
X = df[["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)"]]
y = df["status"]

print("X sample:")
print(X.head())

print("\nY sample:")
print(y.head())

from sklearn.preprocessing import MinMaxScaler
import pandas as pd

df = pd.read_csv("parkinsons.csv")

X = df[["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)"]]

scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(X)

X_scaled = pd.DataFrame(X_scaled, columns=["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)"])

print("After scaling:")
print(X_scaled.head())

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, 
    y, 
    test_size=0.2,
    random_state=42
)

print("Train shape:", X_train.shape)
print("Validation shape:", X_val.shape)

from sklearn.svm import SVC

model = SVC(kernel='rbf', C=1, gamma='scale')

print("Selected model:", model)

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

model = SVC(kernel='rbf', C=167, gamma=300)
model.fit(X_train, y_train)

y_pred = model.predict(X_val)

accuracy = accuracy_score(y_val, y_pred)
print("Validation Accuracy:", accuracy)

import joblib

joblib.dump(model, 'my_model.joblib')
print(X_scaled.head())
