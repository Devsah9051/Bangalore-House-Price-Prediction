# 🏠 Bangalore House Price Prediction using Machine Learning

## 📌 Project Overview
This project focuses on predicting house prices in Bangalore using Machine Learning techniques. The goal is to build a robust predictive model that estimates property prices based on important features such as location, total square feet area, number of bathrooms, and BHK (Bedroom-Hall-Kitchen) count.

The project includes:
- Data Cleaning & Preprocessing
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Outlier Detection & Removal
- Model Training & Evaluation
- Hyperparameter Tuning
- Model Serialization for Deployment

---

## 🎯 Problem Statement
Real estate prices depend on multiple factors and can be difficult to estimate manually. This project aims to solve this problem by building a machine learning model that predicts house prices accurately using historical housing data.

---

## 📂 Dataset Information
The dataset contains Bangalore housing data with the following features:

| Feature | Description |
|---------|-------------|
| area_type | Type of property area |
| availability | Availability status |
| location | Property location |
| size | Number of BHK/Bedrooms |
| society | Society name |
| total_sqft | Total area in square feet |
| bath | Number of bathrooms |
| balcony | Number of balconies |
| price | House price in Lakhs |

### Dataset Summary
- Total Records: **13,320**
- Features: **9**
- Final Cleaned Records: **9,898**

---

## 🛠 Technologies Used
- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Pickle

---

## 📊 Workflow

### 1. Data Preprocessing
Performed several preprocessing steps:
- Removed duplicate entries
- Handled missing values
- Dropped irrelevant columns:
  - `area_type`
  - `availability`
  - `society`

Missing value handling:
- Missing `location` filled using most frequent area
- Missing `size` filled with default `2 BHK`
- Missing `bath` filled using median

---

### 2. Feature Engineering
Created new meaningful features:

#### BHK Extraction
Extracted BHK value from `size` column.

Example:
```python
2 BHK → 2
4 Bedroom → 4
```

#### Total Square Feet Conversion
Some entries had ranges such as:
```python
1133 - 1384
```
Converted them into average values:
```python
(1133 + 1384)/2
```

#### Price Per Square Feet
Created new feature:
```python
price_per_sqft = price * 100000 / total_sqft
```

---

### 3. Dimensionality Reduction
Reduced high-cardinality locations.

- Original locations: **1294+**
- Grouped rare locations (<10 occurrences) into:
```python
Others
```

This improved model performance and reduced overfitting.

---

### 4. Outlier Removal
Removed unrealistic data points using domain knowledge.

#### Rule 1: Minimum Area per BHK
A property should have at least:
```python
300 sqft per BHK
```

Removed entries violating:
```python
total_sqft / bhk < 300
```

#### Rule 2: Price per Square Feet Outliers
For each location:
- Calculated mean
- Calculated standard deviation
- Removed values outside:
```python
(mean ± std)
```

#### Rule 3: BHK Price Anomalies
Removed cases where:
- 3 BHK price < average 2 BHK price in same location

---

## 🔍 Final Features Used for Training
After preprocessing:

| Feature |
|---------|
| location |
| total_sqft |
| bath |
| bhk |

Target Variable:
```python
price
```

---

## 🤖 Machine Learning Models Used
Three regression models were trained:

### 1. Linear Regression
Simple baseline model.

### 2. Lasso Regression
Useful for regularization and feature selection.

### 3. Ridge Regression
Handles multicollinearity using L2 regularization.

---

## ⚙ Pipeline Used
Built a preprocessing + training pipeline using:

- OneHotEncoder → for categorical feature (`location`)
- StandardScaler → for numerical scaling
- Regression Model

Pipeline ensures smooth training and deployment.

---

## 📈 Model Performance

| Model | R² Score |
|-------|----------|
| Linear Regression | 0.7701 |
| Lasso Regression | 0.6575 |
| Ridge Regression | 0.7747 |

---

## 🚀 Hyperparameter Tuning
Used **GridSearchCV (5-fold cross-validation)** to improve performance.

### Ridge Parameters
```python
alpha = [0.1, 1, 10, 100]
```

### Lasso Parameters
```python
alpha = [0.001, 0.01, 0.1, 1, 10]
```

---

## 🏆 Final Model Performance

| Model | R² Score |
|-------|----------|
| Ridge Regression (Tuned) | **0.8129** |
| Lasso Regression (Tuned) | 0.8128 |
| Ridge Regression | 0.7747 |
| Linear Regression | 0.7701 |
| Lasso Regression | 0.6575 |

### Best Model
✅ **Tuned Ridge Regression**

Final Accuracy:
```python
R² Score = 81.29%
```

---

## 💾 Model Saving
Saved the trained model using Pickle:

```python
pickle.dump(model, open('RidgeModel.pkl', 'wb'))
```

This allows easy deployment in web applications.

---

## 🧪 Sample Prediction
Input:
```python
Location: Gunjur
Total Sqft: 1140
Bath: 2
BHK: 2
```

Prediction:
```python
Predicted Price = ₹42.53 Lakhs
Actual Price = ₹49.11 Lakhs
```

The model prediction is close to actual price, showing good performance.

---

## 📌 Key Insights
- Location heavily influences house prices.
- Total square footage strongly correlates with price.
- Outlier removal significantly improves performance.
- Ridge Regression performed best for this dataset.

---

## 🔮 Future Improvements
Possible enhancements:
- Use advanced models like:
  - Random Forest
  - XGBoost
  - CatBoost
- Build a web application using:
  - :contentReference[oaicite:1]{index=1}
  - :contentReference[oaicite:2]{index=2}
- Deploy model on cloud
- Add interactive price visualization dashboard

---

## 👨‍💻 Author
**Dev Sah**  
Aspiring Data Scientist | Machine Learning Enthusiast | AI Learner

Passionate about solving real-world problems using Data Science, Machine Learning, and AI.
