# 🚢 Titanic Survival Predictor

The Titanic Survival Predictor is a machine learning project that predicts whether a passenger survived the Titanic disaster based on features such as passenger class, sex, age, number of siblings or spouses aboard, number of parents or children aboard, and fare.

The project was built to practice end-to-end machine learning workflows, including data preprocessing, missing-value handling, categorical encoding, model training, and evaluation using Python and Scikit-learn.

---

## ✨ Features

**Data Preprocessing**  
Prepares the dataset for machine learning by selecting relevant features and handling missing values.

**Missing Value Handling**  
Handles missing values in:
- Age using median imputation
- Embarked using mode imputation

**Categorical Encoding**  
Uses One-Hot Encoding to convert categorical features into numerical representations:
- Sex
- Embarked

**Machine Learning Model**  
Implements Logistic Regression to predict whether passengers survived the Titanic disaster.

**Model Evaluation**  
Evaluates classification performance using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

**Data Visualization**  
Uses Matplotlib to visualize the confusion matrix and examine the model's classification performance.

---

## 📊 Model Performance

The model was evaluated using a test set containing 20% of the dataset.

- **Accuracy:** ~80.45%

The model correctly classified approximately 80.45% of passengers in the test set. Precision, recall, F1-score, and the confusion matrix provide additional insight into its performance and classification errors.

---

## ⚙️ Technologies Used

**Python** – Core programming language  
**Pandas** – Data manipulation and preprocessing  
**Scikit-learn** – Data preprocessing, model training, and evaluation  
**Logistic Regression** – Classification machine learning model  
**Matplotlib** – Data visualization  

---

## 🚀 How to Run

1. Clone the repository:
```bash
git clone <repo-url>
cd Titanic-Survival-Predictor
```

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the model:
```bash
python main.py
```

---

## 📬 Contact

**LinkedIn**: https://www.linkedin.com/in/mariam-buzaladze-aa22b6297/

**Email**: mariambuzaladze10@gmail.com
