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
5. [Dependencies](#Dependencies)
6. [How to Run using bash](#how-to-run-using-bash)
7. [Results](#results)
8. [Contact](#contact)

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
git clone https://github.com/your-username/your-repo-name.git
```
2. Navigate to the project directory:

```bash
cd your-repo-name
```
3. Run the Python script:

```bash
python heart_disease_prediction.py
```
The script will generate visualizations and print key results to the console.

## Results
- Model Accuracy: The accuracy of the Random Forest Classifier is printed to the console.
- Feature Importance: A bar plot is generated to visualize the importance of each feature.
- Simulation Plots: For each feature, a plot is generated to show how changes in the feature value affect the predicted probability of heart disease.

## Contact
For questions or collaboration opportunities, feel free to reach out:
Name: Felipe Leite
Email: felipe.nog.leite@gmail.com
LinkedIn: [Felipe Leite](https://www.linkedin.com/in/felipeleiteds/)
Portfolio: [FelipeLeite.ca](https://www.felipeleite.ca/)
