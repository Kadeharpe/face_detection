import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#dataset
df = pd.read_csv("netflix_subscription_prediction_30_samples.csv")

train = df.iloc[:22]
test = df.iloc[22:]

#features
X_train = train.drop("Subscribed", axis=1)
y_train = train["Subscribed"]

X_test = test.drop("Subscribed", axis=1)
y_test = test["Subscribed"]

models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC(),
    "Random Forest": RandomForestClassifier()
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results[name] = acc

print("Accuracy Results:\n")
for model, acc in results.items():
    print(f"{model}: {acc:.2f}")