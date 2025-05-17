# train_svm.py

import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)
from data_processing import load_data, preprocess_data, split_data

# === 1. Load and preprocess data ===
df = load_data("your_dataset.csv")  # Change to your dataset filename
X, y = preprocess_data(df, target_column="label")  # Change 'label' if needed
X_train, X_test, y_train, y_test = split_data(X, y)

# === 2. Train SVM model ===
model = SVC(probability=True)  # Needed for ROC curve
model.fit(X_train, y_train)

# === 3. Make predictions ===
y_pred = model.predict(X_test)
y_scores = model.predict_proba(X_test)[:, 1]  # for ROC AUC

# === 4. Evaluate ===
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("AUC Score: {:.3f}".format(roc_auc_score(y_test, y_scores)))

# === 5. Plot ROC Curve ===
fpr, tpr, _ = roc_curve(y_test, y_scores)
plt.plot(fpr, tpr, label="SVM (AUC = {:.3f})".format(roc_auc_score(y_test, y_scores)))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid()
plt.show()
