<div align="center">

# 🏥 Medical Insurance Cost Prediction

**Predict yearly medical insurance charges from a customer's profile using Machine Learning.**

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Linear%20Regression-F7931E?logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)
![R2 Score](https://img.shields.io/badge/R%C2%B2%20Score-0.8068-success)

[🔗 **Live App**](PASTE_YOUR_STREAMLIT_LINK_HERE) &nbsp;|&nbsp; [📂 **Repository**](https://github.com/jawad2005-mj/medical-insurance-cost-prediction) &nbsp;|&nbsp; [📓 **Notebook**](notebooks/insurance_cost_prediction.ipynb)

<img src="images/app_interface.png" alt="Medical Insurance Cost Predictor app" width="620">

</div>

---

## Table of Contents
- [Overview](#overview)
- [Key Results](#key-results)
- [Dataset](#dataset)
- [Workflow](#workflow)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Model and Evaluation](#model-and-evaluation)
- [Sample Predictions](#sample-predictions)
- [Project Structure](#project-structure)
- [How to Run Locally](#how-to-run-locally)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Tech Stack](#tech-stack)
- [Author](#author)

---

## Overview

Insurance companies need to estimate the expected medical cost of a customer to set fair premiums. This project turns that business problem into a **regression problem**: given a customer's age, gender, BMI, number of children, smoking status and region, predict the insurance `charges`.

| | |
|---|---|
| **Problem type** | Regression |
| **Target variable** | `charges` |
| **Input features** | `age`, `sex`, `bmi`, `children`, `smoker`, `region` |
| **Algorithm** | Multiple Linear Regression |
| **Interface** | Streamlit web app |

## Key Results

| R² Score | MAE | RMSE | Test Samples |
|:---:|:---:|:---:|:---:|
| **0.8068** | **$4,182.35** | **$5,957.61** | **268** |

- The model explains about **81%** of the variation in insurance charges.
- On average, a prediction is about **$4,182** away from the real charge.
- **Smoking** is the strongest factor: smokers pay about **4x more** than non-smokers.

## Dataset

- **Name:** [Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance) (Kaggle, by Miri Choi)
- **Size:** 1,338 rows and 7 columns (1,337 after removing 1 duplicate row)
- **Missing values:** none

| Column | Description | Type |
|---|---|---|
| `age` | Age of the customer | Number |
| `sex` | female / male | Category |
| `bmi` | Body Mass Index | Number |
| `children` | Number of children covered | Number |
| `smoker` | yes / no | Category |
| `region` | northeast, northwest, southeast, southwest | Category |
| `charges` | Medical insurance cost (target) | Number |

## Workflow

1. **Understand the problem:** define the target and the features.
2. **Explore the dataset:** shape, data types and statistics.
3. **Clean and preprocess:**
   - Checked missing values (none found).
   - Removed 1 duplicate row.
   - Applied **label encoding** to the categorical columns:

     | Column | Encoding |
     |---|---|
     | `sex` | female = 0, male = 1 |
     | `smoker` | no = 0, yes = 1 |
     | `region` | northeast = 0, northwest = 1, southeast = 2, southwest = 3 |

4. **Exploratory Data Analysis:** charts and correlation analysis.
5. **Train the model:** 80% / 20% train-test split (`random_state=42`), 1,069 training rows and 268 test rows.
6. **Evaluate:** MAE, MSE, RMSE and R² score.
7. **Deploy:** Streamlit app for live predictions.

## Exploratory Data Analysis

**Key findings**
- Smokers pay about **$32,050** on average, while non-smokers pay about **$8,441**.
- `smoker` has the strongest correlation with charges (**0.79**).
- `age` (0.30) and `bmi` (0.20) also increase charges.
- `children` (0.07), `sex` (0.06) and `region` (-0.01) have very little effect.
- Charges are right-skewed: a few customers have very high bills.

<table>
  <tr>
    <td align="center"><b>Charges Distribution</b><br><img src="images/01_charges_distribution.png" width="400"></td>
    <td align="center"><b>Smoker vs Charges</b><br><img src="images/02_smoker_vs_charges.png" width="400"></td>
  </tr>
  <tr>
    <td align="center"><b>Age vs Charges</b><br><img src="images/03_age_vs_charges.png" width="400"></td>
    <td align="center"><b>BMI vs Charges</b><br><img src="images/04_bmi_vs_charges.png" width="400"></td>
  </tr>
</table>

## Model and Evaluation

**Model:** `sklearn.linear_model.LinearRegression`

**Learned coefficients**

| Feature | Coefficient |
|---|---:|
| `smoker` | 23,052.15 |
| `children` | 534.12 |
| `bmi` | 312.61 |
| `age` | 248.76 |
| `region` | -237.63 |
| `sex` | -99.70 |

**Evaluation on the test set**

| Metric | Value |
|---|---:|
| MAE | 4,182.35 |
| MSE | 35,493,102.61 |
| RMSE | 5,957.61 |
| R² Score | 0.8068 |

<table>
  <tr>
    <td align="center"><b>Correlation Heatmap</b><br><img src="images/05_correlation_heatmap.png" width="400"></td>
    <td align="center"><b>Actual vs Predicted</b><br><img src="images/06_actual_vs_predicted.png" width="400"></td>
  </tr>
</table>

## Sample Predictions

| Age | Gender | BMI | Children | Smoker | Region | Predicted Cost |
|:---:|:---:|:---:|:---:|:---:|:---:|---:|
| 30 | male | 27.0 | 1 | no | southeast | $4,814.85 |
| 45 | female | 32.0 | 2 | yes | northwest | $34,032.96 |

<!-- After deployment, add your prediction screenshot to images/ and uncomment:
<div align="center">
<img src="images/prediction_result.png" alt="Prediction result" width="620">
</div>
-->

## Project Structure

```
medical-insurance-cost-prediction/
├── app.py                # Streamlit prediction app
├── train_model.py        # Training and evaluation script
├── requirements.txt      # Python packages
├── README.md
├── .gitignore
├── data/
│   └── insurance.csv     # Dataset
├── model/
│   └── model.pkl         # Trained model
├── notebooks/
│   └── insurance_cost_prediction.ipynb   # Full workflow
└── images/               # App screenshot and graphs
```

## How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/jawad2005-mj/medical-insurance-cost-prediction.git
cd medical-insurance-cost-prediction

# 2. Install packages
pip install -r requirements.txt

# 3. (Optional) Train the model again
python train_model.py

# 4. Run the app
streamlit run app.py
```

## Limitations

- Linear Regression assumes a straight-line relationship, so it makes bigger errors for customers with very high charges.
- The dataset is small (1,337 rows) and comes from the USA, so results may not match insurance costs in other countries.
- Label encoding gives `region` an order (0 to 3) that does not really exist. One-hot encoding may work slightly better.
- The model does not use medical history, income or pre-existing conditions.
- Predictions are estimates for learning only. They are not real insurance quotes.

## Future Improvements

- [ ] Try Random Forest, Gradient Boosting or XGBoost
- [ ] Use one-hot encoding for `region`
- [ ] Apply a log transform on `charges`
- [ ] Add cross-validation and hyperparameter tuning

## Tech Stack

`Python` · `Pandas` · `NumPy` · `Scikit-learn` · `Matplotlib` · `Seaborn` · `Streamlit`

## Author

**Muhammad Jawad**
Machine Learning Developer

[![GitHub](https://img.shields.io/badge/GitHub-jawad2005--mj-181717?logo=github)](https://github.com/jawad2005-mj)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-muhammad--jawadofficial-0A66C2?logo=linkedin&logoColor=white)](https://linkedin.com/in/muhammad-jawadofficial)

---

<div align="center">
⭐ If you found this project useful, consider giving it a star!
</div>
