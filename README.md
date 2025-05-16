# The-credit-card-fraud-dection

💳 Credit Card Fraud Detection using SVM & Decision Trees
This project applies machine learning techniques to detect fraudulent credit card transactions using a real-world, highly imbalanced dataset. The models implemented include Support Vector Machine (SVM) and Decision Tree (DT) classifiers.

📂 Dataset
Source: Kaggle / IBM

Description: Contains transactions made by European cardholders in September 2013.

Rows: 284,807 transactions

Features: 30 features (V1-V28 from PCA, plus Time, Amount, Class)

Target Variable:

0 → Legitimate

1 → Fraudulent

🔍 Project Structure
bash
Copy
Edit
📁 credit-card-fraud-detection/
│
├── 📊 EDA.ipynb                # Exploratory Data Analysis & Visualizations
├── 📈 SVM_Model.ipynb          # Support Vector Machine implementation
├── 🌳 DecisionTree_Model.ipynb # Decision Tree implementation
├── data/
│   └── creditcard.csv          # Local dataset file
├── 📄 README.md                # Project description and guide
└── requirements.txt            # Python packages (optional)
📊 Exploratory Data Analysis (EDA)
Visualized class distribution using:

Pie chart

Bar chart

Boxplot (transaction amount by class)

Found that:

Only ~0.17% of transactions are fraudulent

Data is highly imbalanced, requiring special handling

🧠 Models Used
1. Support Vector Machine (SVM)
Scikit-learn's SVC

Used with balanced class weights

Tuned for kernel type, regularization, etc.

2. Decision Tree Classifier
Applied entropy and Gini-based splits

Visualized tree and evaluated with metrics

📈 Evaluation Metrics
Since accuracy is misleading for imbalanced datasets, we used:

Precision

Recall

F1-Score

Confusion Matrix

ROC AUC

🛠 How to Run
Clone the repo:

bash
Copy
Edit
git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Open notebooks in Jupyter or VS Code and run step-by-step:

EDA.ipynb

SVM_Model.ipynb

DecisionTree_Model.ipynb

💡 Future Improvements
Try Random Forest or XGBoost

Apply SMOTE or undersampling

Deploy as an API using Flask or FastAPI

📚 References
Scikit-learn Documentation

Imbalanced-learn

Kaggle Dataset

