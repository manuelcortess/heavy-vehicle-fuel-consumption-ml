# Heavy-Vehicle On-Road Fuel Consumption & Driving Pattern Analysis

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn%20%7C%20Ensembles-orange)
![Data Processing](https://img.shields.io/badge/Data-Pandas%20%7C%20NumPy-green)

A data science project evaluating predictive regression models and pattern classification architectures on heavy-vehicle telematics and spatial-temporal data.

📖 **[View the Academic Poster (PDF)](docs/Hands-on%20Machine%20Learning%20and%20Data%20Science.pdf)**

## 📌 Project Overview
Optimizing fleet logistics requires an accurate understanding of vehicular physics and operational efficiency. Inspired by sustainable urban logistics pilot frameworks—such as the *Îlot Voyageur* initiative aimed at reducing truck traffic and greenhouse gases—this project utilizes a dataset containing **46,476 distinct driving records** to solve two distinct machine learning tasks:
1. **Fuel Consumption Prediction (Regression):** Estimating continuous fuel consumption (liters) using spatial, dynamic, and payload variables.
2. **Driving Pattern Classification (Binary):** Characterizing operational driving modes based on real-time vehicular mechanics.

The tracking telemetry consists of 8 features: spatial data (`Latitude`, `Longitude`), vehicle dynamics (`Distance(m)`, `Speed(m/s)`, `Acceleration(m/s²)`), and vehicle load configurations ranging between 4,500 kg and 8,465 kg.

## 🚀 Key Engineering Challenges & Solutions
To ensure model robustness and statistical integrity across both the training and testing phases, several data anomalies and distribution characteristics were addressed:
* **Schema Alignment & Data Cleaning:** Detected a systemic engineering unit typo within the testing dataset schema (`Acceleration(m^2)` vs. `Acceleration(m/s²)`). Standardized features to establish strict data types and structural compatibility before ingestion.
* **Feature Engineering for Non-Linearities:** Addressed severe non-linear relationships between fuel consumption, payload mass, and acceleration rates by implementing **Polynomial Feature Transformations** and strategic log-scaling (`np.log1p`).
* **Imbalance Class Mitigation:** The driving pattern target variable exhibited significant class skew. The classification pipeline was modified to optimize for **Precision, Recall, Specificity, and F1-Scores** rather than standard accuracy, preventing the models from defaulting to a naive baseline classifier.

## 📊 Key Findings
The predictive pipelines were rigorously validated using cross-validation sweeps to establish generalizability:
* **Regularized Regression Performance:** Evaluated Ordinary Least Squares (OLS), Ridge Regression, and Lasso Regression. Utilizing a 5-Fold Cross-Validation hyperparameter grid search allowed for fine-tuning the regularizing parameter ($\alpha$), which minimized overfitting caused by the multicollinearity of spatial metrics.
* **Tree-Based Ensembles Outperform Distance Metrics:** For the classification task, Random Forest Classifiers and Decision Trees significantly outperformed K-Nearest Neighbors (KNN) in predicting high-frequency driving patterns. This demonstrates that hierarchical splitting handles the non-linear boundaries of mixed payload profiles better than localized distance calculations.

## 🛠️ Repository Structure
* `/data`: Contains the underlying `training_dataset.csv` and `testing_dataset.csv`.
* `/docs`: Contains the final `Hands-on Machine Learning and Data Science.pdf` presentation poster.
* `/notebooks`: Contains `topic_05.ipynb`, the complete, self-contained interactive engineering pipeline.

*Note: The Jupyter Notebook in the `/notebooks` directory is fully executed. GitHub natively renders `.ipynb` files, allowing you to view all code cells, data summaries, and comparative visualizations directly in your browser without running local kernels.*

## 💻 Tech Stack
* **Data Processing & Analytics:** `Pandas`, `NumPy`, `Tabulate`
* **Machine Learning & Evaluation:** `Scikit-Learn` (Linear Models, KNN, Tree Ensembles, Metrics)
* **Visualization:** `Matplotlib`, `Seaborn`