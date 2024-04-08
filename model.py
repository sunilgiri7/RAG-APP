from xgboost import XGBClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data['data'], data['target'], test_size=0.2, random_state=42)

xg = XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, objective="binary:logistic")
xg.fit(X_train, y_train)

y_pred = xg.predict(X_test)

print("Actual", X_test[:10])
print("Predicted", y_pred[:10])