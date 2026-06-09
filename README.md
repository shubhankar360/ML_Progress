# Wine Quality Prediction 🍷

An end-to-end machine learning project that predicts the quality of red wine — tackled from two angles: as a regression problem (exact score) and as a binary classification problem (Good vs. Not Good).

---

## What This Project Does

Wine quality is typically assessed by expert tasters, but what if we could predict it from the wine's chemical composition alone? This project explores that question using real-world data on Portuguese "Vinho Verde" red wine.

Rather than treating it as a single problem, this project approaches wine quality prediction in two ways:

- **Regression** — Predict the exact quality score on a scale of 0–10
- **Classification** — Predict whether a wine is *Good* (quality ≥ 7) or *Not Good* (quality < 7)

Running both tasks on the same dataset made for a richer comparison and a more complete picture of what the data can tell us.

---

## Dataset

The data comes from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality) and contains **1,599 samples** of red wine, each described by 11 physicochemical properties.

| Feature | What It Represents |
|---|---|
| `fixed acidity` | Tartaric acid content — affects taste and stability |
| `volatile acidity` | Acetic acid — too much gives a vinegar taste |
| `citric acid` | Adds freshness and flavor |
| `residual sugar` | Sugar remaining after fermentation |
| `chlorides` | Salt content |
| `free sulfur dioxide` | Active SO₂ that prevents microbial growth |
| `total sulfur dioxide` | Total SO₂ content |
| `density` | Density of the wine |
| `pH` | Acidity/alkalinity level |
| `sulphates` | Wine additive used for preservation |
| `alcohol` | Alcohol content (%) |
| `quality` | **Sensory score (0–10) ← target variable** |

---

## Approach

The project follows a structured, end-to-end ML workflow:

1. **Data Exploration (EDA)** — Understand distributions, correlation patterns, and outliers
2. **Feature Engineering** — Create new features to capture chemical interactions
3. **Train-Test Split** — 80/20 split with stratification for classification
4. **Regression Modelling** — Train and compare 7 regression algorithms
5. **Classification Modelling** — Train and compare 6 classification algorithms
6. **Cross-Validation** — 5-fold CV for a more honest estimate of performance
7. **Hyperparameter Tuning** — GridSearchCV on the best model
8. **Feature Importance** — Understand what the model actually learned

---

## Feature Engineering

Five new features were created to help models capture chemical interactions that the raw features alone might miss:

| New Feature | Logic |
|---|---|
| `total_acidity` | Sum of fixed, volatile, and citric acid |
| `sugar_to_alcohol_ratio` | Balance between sweetness and strength |
| `sulphate_to_alcohol` | Preservation potential relative to alcohol |
| `pH_to_acidity` | Interaction between pH and total acidity |
| `free_to_total_so2_ratio` | Efficiency of sulfur dioxide as a preservative |

---

## Models Used

### Regression (predicting exact quality score)

| Model | Why It's Here |
|---|---|
| Linear Regression | Simple interpretable baseline |
| Ridge Regression | Handles multicollinearity with L2 regularization |
| Lasso Regression | Performs implicit feature selection via L1 |
| Decision Tree | Captures non-linear patterns |
| Random Forest | Robust ensemble; strong out-of-the-box |
| Gradient Boosting | Often top performer on tabular data |
| SVR | Effective in high-dimensional spaces |

**Evaluation metrics:** RMSE, MAE, R² Score

### Classification (Good vs. Not Good)

| Model | Why It's Here |
|---|---|
| Logistic Regression | Fast, interpretable baseline |
| K-Nearest Neighbors | Simple, non-parametric |
| Decision Tree | Easy to interpret, handles non-linearity |
| Random Forest | High accuracy, handles class imbalance well |
| Gradient Boosting | State-of-the-art for tabular classification |
| SVM | Strong margin-based classifier |

**Evaluation metrics:** Accuracy, Precision, Recall, F1-Score, ROC-AUC

> **Note on class imbalance:** Only ~13.6% of wines in the dataset are rated 'Good' (quality ≥ 7). Because of this, F1-Score and ROC-AUC are more meaningful metrics here than accuracy alone.

---

## Tech Stack

```
Python 3.x
├── NumPy
├── Pandas
├── Matplotlib
├── Seaborn
└── Scikit-learn
    ├── LinearRegression, Ridge, Lasso
    ├── DecisionTreeRegressor / Classifier
    ├── RandomForestRegressor / Classifier
    ├── GradientBoostingRegressor / Classifier
    ├── SVR / SVC
    ├── LogisticRegression
    ├── KNeighborsClassifier
    ├── StandardScaler
    ├── train_test_split, cross_val_score
    └── GridSearchCV
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/shubhankar360/wine-quality-prediction.git
cd wine-quality-prediction
```

### 2. Install dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib jupyter
```

### 3. Run the notebook

```bash
jupyter notebook Wine_Quality_Prediction_ML_Project.ipynb
```

The dataset is loaded directly from the UCI repository in the notebook, so no manual download is needed.

---

## Using the Saved Model

After running the notebook, the tuned Random Forest classifier and the scaler are saved to disk:

```python
import joblib

# Load model and scaler
model = joblib.load('wine_quality_classifier.pkl')
scaler = joblib.load('scaler.pkl')

# Predict on new data
prediction = model.predict(scaler.transform(new_data))
# Output: 1 = Good wine, 0 = Not Good
```

---

## Project Structure

```
📁 wine-quality-prediction/
├── Wine_Quality_Prediction_ML_Project.ipynb   # Main notebook
├── wine_quality_classifier.pkl                # Saved tuned classifier
├── scaler.pkl                                 # Saved StandardScaler
└── README.md
```

---

## Key Takeaways

- **Alcohol content** and **volatile acidity** are consistently the strongest predictors of wine quality — across both regression and classification tasks
- **Gradient Boosting** and **Random Forest** outperformed all linear and simpler models by a noticeable margin
- The class imbalance (~13.6% Good wines) makes this a harder classification problem than it looks — raw accuracy can be misleading here
- **Feature engineering helped** — the engineered chemical ratio features improved model performance compared to using raw features alone
- **5-fold cross-validation** confirmed that the results were stable and not just a product of a favorable train-test split

---

## What's Next

A few directions worth exploring from here:

- **Handle class imbalance more explicitly** — try SMOTE oversampling or adjusting class weights
- **Stack multiple models** — ensemble Gradient Boosting and Random Forest for better predictions
- **Try a neural network** — a simple MLP might squeeze out extra performance
- **Build a Streamlit app** — turn the saved model into an interactive web demo
- **Extend to white wine** — the UCI repository has a white wine dataset; comparing the two would be interesting

---

## Author

**Shubhankar**
- GitHub: [@shubhankar360](https://github.com/shubhankar360)

---

*Built as part of a hands-on machine learning portfolio. Feedback welcome.*
