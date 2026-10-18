import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

df = pd.read_csv("../data/Titanic-Dataset.csv")

X = df.drop(columns=['Survived', 'PassengerId', 'Name', 'Cabin', 'Ticket'])
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

age_median=X_train['Age'].median()
embarked_mode=X_train['Embarked'].mode()[0]

X_train['Age'] = X_train['Age'].fillna(age_median)
X_train['Embarked'] = X_train['Embarked'].fillna(embarked_mode)

X_test['Age'] = X_test['Age'].fillna(age_median)
X_test['Embarked'] = X_test['Embarked'].fillna(embarked_mode)

X_train = pd.get_dummies(
    X_train,
    columns=['Sex', 'Embarked'],
    dtype=int
)

X_test = pd.get_dummies(
    X_test,
    columns=['Sex', 'Embarked'],
    dtype=int
)

X_test = X_test.reindex(
    columns=X_train.columns,
    fill_value=0
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

pred = model.predict(X_test)

accuracy = accuracy_score(y_test, pred)
precision = precision_score(y_test, pred)
recall = recall_score(y_test, pred)
f1 = f1_score(y_test, pred)

print("Model Evaluation")
print("----------------")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")


cm = confusion_matrix(y_test, pred)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=['Did not survive', 'Survived']
)

fig, ax = plt.subplots(figsize=(7, 5))

display.plot(
    ax=ax,
    cmap='Blues',
    values_format='d',
    colorbar=False
)

ax.set_title('Titanic Survival Predictor - Confusion Matrix')
ax.set_xlabel('Predicted Outcome')
ax.set_ylabel('Actual Outcome')

plt.tight_layout()
plt.show()