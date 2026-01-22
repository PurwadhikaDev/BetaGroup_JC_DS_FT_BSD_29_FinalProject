# E-Commerce Customer Churn Analysis and Prediction

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ecommerce-customer-churn-prediciton-finpro-beta.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.9-blue)
![Library](https://img.shields.io/badge/Library-Scikit_Learn-orange)
![Status](https://img.shields.io/badge/Status-Completed-green)

**Final Project - Data Science & Machine Learning Program**
**Institution:** Purwadhika Digital School
**Group Beta:**
- Muhamad Farhan Budiana
- Muhammad Giffari Putra Pradana
- Shadrina Putri Nabila

---

## Project Overview

Customer churn presents a significant challenge in the e-commerce sector, directly impacting revenue stability and increasing operational costs due to the high expense of acquiring new customers compared to retaining existing ones. This project aims to develop a robust machine learning model capable of predicting customer churn, thereby enabling proactive retention strategies.

The analysis follows a comprehensive end-to-end data science workflow, including data cleaning, rigorous preprocessing to prevent data leakage, handling class imbalance, and hyperparameter tuning to optimize model performance.

## Business Context and Objective

### Problem Statement
<img width="613" alt="Churn Context Illustration" src="https://github.com/user-attachments/assets/5d9daa6d-9dd1-49d0-90c3-25e8b87856e5">

The company is currently experiencing a customer churn rate of approximately **16.8%**. The lack of a predictive mechanism results in a reactive approach to customer retention, leading to inefficient marketing spend and lost revenue opportunities.

### Objectives
1.  **Predictive Modeling:** Develop a binary classification model to identify customers at high risk of churn.
2.  **Metric Optimization:** Prioritize **Recall** to minimize False Negatives (identifying as many potential churners as possible).
3.  **Business Insight:** Analyze feature importance to understand key drivers of customer attrition.

## Analytical Approach

To ensure model validity and performance, the following methodology was applied:

1.  **Data Quality & Leakage Prevention:**
    - Removed lagging indicators (e.g., `DaySinceLastOrder`) to prevent look-ahead bias and data leakage.
    - Standardized categorical values to reduce dimensionality and improve consistency.

2.  **Preprocessing Pipeline:**
    - **Data Splitting:** Applied stratified sampling to maintain class distribution in training and testing sets.
    - **Imputation:** Implemented median and mode imputation within a pipeline to prevent statistical leakage from the test set.
    - **Scaling & Encoding:** Utilized StandardScaler for distance-based algorithms and OneHotEncoder for categorical variables.

3.  **Handling Class Imbalance:**
    - Integrated SMOTE (Synthetic Minority Over-sampling Technique) within the cross-validation process to address the minority class (churners) without overfitting.

4.  **Model Selection:**
    - Benchmarked multiple algorithms: Logistic Regression, K-Nearest Neighbors (KNN), Random Forest, and XGBoost.
    - Selected the optimal model based on Recall performance on the validation set.

## Key Results

### Final Model Performance
The **K-Nearest Neighbors (KNN)** algorithm was selected as the final model after hyperparameter tuning.

-   **Hyperparameters:** `n_neighbors=11`, `weights='distance'`, `metric='euclidean'`
-   **Recall (Test Set):** ~91%
-   **Interpretation:** The model successfully identifies approximately **91%** of customers who are actually at risk of churning. This high recall ensures that retention efforts cover the vast majority of at-risk customers.

## Conclusion and Recommendation

### Technical Conclusion
Based on the evaluation results on the Test Set, the project concludes the following:
1.  **Best Model:** KNN was selected as the best-performing model due to its high sensitivity to minority class patterns.
2.  **Metric Performance:** The model achieved a Recall score of ~91% on the Churn class.
3.  **Risk Mitigation:** The False Negative rate is very low, ensuring that the business does not miss the opportunity to retain at-risk customers.

### Business Impact Analysis
Using a simulation with 10,000 customers and a 16.8% churn rate:
* **Reactive Approach (Without Model):** Estimated loss of ~$84,000 due to undetected churn.
* **Proactive Approach (With Model):** Total cost reduced to ~$22,840 (Retention Cost + Remaining Loss).
* **Potential Savings:** ~$61,160 (saving approximately 73% of churn-associated costs).

### Strategic Recommendations
1.  **Focus on the "Golden Period":** Churn risk is highest at low Tenure (new customers). Create a special Onboarding Program for the first 3 months.
2.  **Improve Complaint Mechanism:** Customers who file a Complaint have a high tendency to churn. Implement a priority handling system ("Service Recovery Paradox").
3.  **Optimize Mobile App:** Users who rarely open the app (low `HourSpendOnApp`) are at risk. Use personalized push notifications to re-engage inactive users.

## Model Deployment

The final model has been deployed as an interactive web application using Streamlit to facilitate real-time prediction for business stakeholders.

**Access the Application:**
[https://ecommerce-customer-churn-prediciton-finpro-beta.streamlit.app/](https://ecommerce-customer-churn-prediciton-finpro-beta.streamlit.app/)

Preview 
<img width="1910" height="898" alt="image" src="https://github.com/user-attachments/assets/3baa665d-5922-416e-ba67-81957488a5eb" />
<img width="1908" height="901" alt="image" src="https://github.com/user-attachments/assets/2b376f3e-f004-438b-a26c-d67c8852815c" />



**Application Features:**
* **Input Interface:** User-friendly sidebar for entering customer data.
* **Real-time Prediction:** Instant classification (Churn vs. Stay).
* **Probability Score:** Displays the model's confidence level for the prediction.

## Repository Structure

```text
├── data/                            # Dataset files
├── model/                           # Serialized model artifacts (.pkl)
├── notebooks/                       # Jupyter notebooks for analysis
├── app.py                           # Streamlit Application Source Code
├── README.md                        # Project Documentation
└── requirements.txt                 # Python dependencies

```

## Installation and Usage

To replicate the analysis or run the application locally, follow these steps:

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your-username/ecommerce-customer-churn-prediciton-finpro.git](https://github.com/your-username/ecommerce-customer-churn-prediciton-finpro.git)
    ```

2.  **Install Dependencies:**
    It is recommended to use a virtual environment. Install the required libraries using:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Analysis (Jupyter Notebook):**
    To view the detailed data exploration and model training process:
    ```bash
    jupyter notebook
    ```

4.  **Run the Web Application (Streamlit):**
    To launch the interactive prediction dashboard locally:
    ```bash
    streamlit run app.py
    ```

## Tools and Technologies

- **Programming Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-Learn (v1.5.2), Imbalanced-Learn, XGBoost, Matplotlib, Seaborn
- **Deployment:** Streamlit Cloud
- **Model Serialization:** Pickle
- **Editor:** Jupyter Notebook, Visual Studio Code
