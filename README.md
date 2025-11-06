#  Heart Disease Prediction using Machine Learning

This project uses **machine learning** to predict the likelihood of heart disease based on patient health data.  
It was developed by **Sophia Pimpa Akuffo** as a practical implementation of her and Spencer's undergraduate project:  
> *“Predicting Myocardial Infarction Using Machine Learning Algorithms”*

---

##  Project Overview

Cardiovascular diseases (CVDs) are the world’s leading cause of death.  
This project explores how **machine learning** can assist medical practitioners in early detection of heart disease risk, enabling timely intervention.

---

##  Technologies Used

- **Python 3**
- **Jupyter Notebook**
- **Pandas**, **NumPy** — data manipulation  
- **Scikit-learn** — model building & evaluation  
- **Matplotlib** — visualizations
- **Joblib** — model saving/loading

  Heart-disease-prediction

└─ heart_disease_dataset.csv # Dataset used for model training
└─ test1.ipynb # Jupyter Notebook with step-by-step analysis
└─ heart_disease_model.joblib # Saved trained model
└── README.md # Project documentation

---

##  Project Structure


## Dataset Information

The dataset (`heart_disease_dataset.csv`) includes patient health metrics such as:
- Age  
- Sex  
- Chest pain type (`cp`)  
- Resting blood pressure (`trestbps`)  
- Cholesterol (`chol`)  
- Maximum heart rate achieved (`thalach`)  
- Exercise-induced angina (`exang`)  
- Fasting blood sugar (`fbs`)  
- BMI, Smoking, Diabetes indicators  
- Target: `heart_disease` (1 = presence, 0 = absence)

---

##  How to Run the Project


1️⃣ Clone the Repository
```bash
git clone https://github.com/Sophia-Akuffo/heart-disease-prediction.git
cd heart-disease-prediction

2️⃣ Install Required Packages

Make sure you have Python 3 installed. Then run:
pip install pandas numpy scikit-learn matplotlib joblib


3️⃣ Open the Notebook

Run Jupyter Notebook:
jupyter notebook
Then open test1.ipynb and run all the cells sequentially.

4️⃣  (Optional) Predict with the Saved Model
from joblib import load
import pandas as pd

model = load("heart_disease_model.joblib")

# Example patient data (replace with actual values)
new_patient = pd.DataFrame([{
    'age': 55,
    'sex': 1,
    'cp': 2,
    'trestbps': 130,
    'chol': 250,
    'fbs': 0,
    'restecg': 1,
    'thalach': 150,
    'exang': 0,
    'oldpeak': 1.2,
    'slope': 2,
    'ca': 0,
    'thal': 3,
    'smoking': 0,
    'diabetes': 0,
    'bmi': 27.5
}])

prediction = model.predict(new_patient)[0]
print("Predicted class (1 = heart disease, 0 = no disease):", prediction)

Model Summary

Model Used: Logistic Regression (baseline)

Alternative Model: Random Forest Classifier

Evaluation Metrics:

Accuracy ≈ 0.62

ROC AUC ≈ 0.64

Confusion matrix and ROC curve plotted for interpretation


Future Improvements

Implement model tuning with GridSearchCV

Deploy as a web app using Streamlit or Flask

Train with larger, region-specific datasets (e.g., Ghanaian health data)

Compare multiple models (Decision Tree, Random Forest, XGBoost)


Author

Sophia Pimpa Akuffo
🎓MSc. Data Science - University of Aberdeen
🎓 BSc. Information Technology — Ghana Technology University College
💻 Passionate about Machine Learning & AI 
