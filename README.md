# Heart Disease Prediction Using Random Forest Classifier
This project focuses on predicting the likelihood of heart disease using a Random Forest Classifier. The dataset used is heart.csv, which contains various health-related features and a target variable indicating the presence or absence of heart disease. The goal is to preprocess the data, train a machine learning model, and analyze feature importance to understand which factors contribute most to heart disease prediction.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Code Structure](#code-structure)
3. [Key Steps](#key-steps)
    - [Data Preprocessing](#data-preprocessing)
    - [Model Training and Evaluation](#model-training-and-evaluation)
    - [Feature Importance Analysis](#feature-importance-analysis)
    - [Simulation and Visualization](#simulation-and-visualization)
5. [Dependencies](#dependencies)
6. [How to Run using bash](#how-to-run-using-bash)
7. [Results](#results)
8. [Limitations and Future Work](#limitations-and-future-work)
9. [Portfolio and Contact](#portfolio-and-contact)

## Project Overview
Heart disease is a critical health issue worldwide, and early prediction can significantly improve patient outcomes. This project leverages machine learning to predict the likelihood of heart disease based on patient health data. The Random Forest Classifier is used due to its robustness and ability to handle complex datasets. The project also includes exploratory data analysis, feature importance visualization, and simulation to understand how individual features impact the prediction.

## Code Structure
The code is structured into the following sections:
1. Data Preprocessing: Handling missing data and outliers.
2. Model Training and Evaluation: Splitting the data, training the Random Forest Classifier, and evaluating its performance.
3. Feature Importance Analysis: Identifying and visualizing the most important features for prediction.
4. Simulation and Visualization: Simulating the impact of individual features on the prediction and generating visualizations.

## Key Steps
### Data Preprocessing
- Missing Data: The dataset is checked for missing values. Columns with more than 50% missing data are dropped, and the remaining missing values are filled with the mean of the respective column.
- Outliers: Outliers are identified using the Z-score method (threshold = 3) and removed from the dataset to ensure robust model performance.

### Model Training and Evaluation
- A Random Forest Classifier is trained with the following hyperparameters:**
  * `n_estimators=5000`: Number of trees in the forest.
  * `max_features=3`: Maximum number of features considered for splitting a node.
  * `max_depth=8`: Maximum depth of the tree.
- The model's predictions are evaluated using accuracy score.

### Feature Importance Analysis
The importance of each feature is calculated and ranked in descending order.

A bar plot is generated to visualize the feature importance, with annotations for clarity.

### Simulation and Visualization
For each feature, a simulation is performed to analyze its impact on the prediction.

The probability of heart disease is plotted against the range of values for each feature.

Linear regression is applied to the simulation results to understand the trend.

## Dependencies
To run this code, ensure you have the following Python libraries installed:

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`

You can install the dependencies using:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```
## How to Run using Bash
1. Clone this repository using:

```bash
git clone https://github.com/FelipeLeiteDS/Heart-Disease-Prediction.git
```
2. Navigate to the project directory:

```bash
cd Heart-Disease-Prediction
```
3. Run the Python script:

```bash
python Random-Forest_Confusion-Matrix.py
```
The script will generate visualizations and print key results to the console.

## Results
### Model Accuracy
- The Random Forest Classifier achieved an accuracy of 90.16%, outperforming other algorithms like Logistic Regression, Naïve Bayes, SVM, KNN, Decision Tree, and XGBoost.
- The model was trained using 5,000 decision trees `n_estimators=5000`, with 3 features considered at each split `max_features=3` and a maximum tree depth of 8 levels `max_depth=8` to prevent overfitting.
- The dataset was split into 80% training and 20% testing sets, with a fixed random seed `random_state=42` for reproducibility.

### Model Evaluation
- The model's performance was evaluated using metrics like accuracy, precision, recall, and F1-score:
    - Precision: High precision for predicting high-risk cases (Class 1).
    - Recall: Achieved 75% for Class 1, indicating room for improvement in identifying true positives.
    - F1-Score: Balanced performance across both classes.

### Key Insights
- The Random Forest Classifier demonstrated robustness in handling non-linear relationships and complex datasets.
    - Combining Random Search Algorithms (RSA) with Optimized Random Forest (RF) improved accuracy by 3.3%, achieving 93.33% overall accuracy.
    - The ROC curve for the RSA-RF method showed better performance compared to a simple Random Forest.

## Limitations and Future Work
- Data Size: The dataset is relatively small, which may affect generalizability.
- Feature Limitations: Missing LDL and HDL cholesterol levels could enhance predictive performance.
- Future Work: Expanding the dataset, adding more health indicators, and experimenting with advanced architectures are recommended.

## Portfolio and Contact
Explore my work and connect with me:

<div> 
  <a href = "https://linktr.ee/FelipeLeiteDS"><img src="https://img.shields.io/badge/LinkTree-1de9b6?logo=linktree&logoColor=white" target="_blank"></a>
  <a href = "https://www.linkedin.com/in/felipeleiteds/" target="_blank"><img src="https://custom-icon-badges.demolab.com/badge/LinkedIn-0A66C2?logo=linkedin-white&logoColor=fff" target="_blank"></a> 
  <a href = "https://www.felipeleite.ca"><img src="https://img.shields.io/badge/FelipeLeite.ca-%23000000.svg?logo=wix&logoColor=white" target="_blank"></a>
  <a href = "mailto:felipe.nog.leite@gmail.com"><img src="https://img.shields.io/badge/Gmail-D14836?logo=gmail&logoColor=white" target="_blank"></a>
