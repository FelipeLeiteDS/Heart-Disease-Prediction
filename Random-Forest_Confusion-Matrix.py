# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 17:21:43 2023

@author: Felipe Leite
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

df= pd.read_csv("./heart.csv")

# # # # Data Preprocession

### ************** Missing Data
# Check for number of missing data 
missingData = df.isna()
percentage = missingData.mean(axis=0) * 100

# Handle missing data   
to_drop = percentage[percentage > 50]
df.drop(to_drop.index, axis=1, inplace=True)
df.fillna(df.mean(), inplace=True)
#df = pd.get_dummies(df, columns=["city"])

### ************** Outliers
# identifying outliers
threshold = 3
numeric_data = df.select_dtypes(include=['float64', 'int64'])
z_scores = np.abs((numeric_data - numeric_data.mean(axis=0)) / numeric_data.std(axis=0))
is_outlier = z_scores > threshold
outliers = df[is_outlier.any(axis=1)]
print("Outliers status:\n", is_outlier.sum(axis=0))
print("Outliers:\n", outliers)

# Handle outliers - Dropping outliers from the original dataset
df.drop(outliers.index, inplace=True)


# # # # **************  Random Forest Classifier  *************************************

### ************** Model Traininf and testing
y = df["output"]
x = df.drop("output", axis=1)

x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=42)
x_mean = x_train.mean().copy()

model_rf = RFC(n_estimators=5000,max_features=3,max_depth=8)
model_rf.fit(x_train,y_train)
y_pred = model_rf.predict(x_test)
m_pred = model_rf.predict(x_test).copy()
print("------------------------")    
print("pred_model: \n", m_pred)

model_rf.predict_proba(x_test)

def sigmoid(input, threshold):
    result = (input > threshold).astype(int)
    return result

y_pred = sigmoid(model_rf.predict(x_test),0.5)
accuracy_score(y_test, y_pred)

print("------------------------")    
print("y_pred: \n", y_pred)

# # # # **************  Simulation  *************************************
### ************** Feature Importance
importance = model_rf.feature_importances_
print("feature importance: \n", model_rf.feature_importances_)

# Sort features importance in descending order
indices = np.argsort(importance)[::-1]
s = importance.sum()
print("Sum")
# Print feature ranking
print("Feature ranking:")


for i, idx in enumerate(indices):
    print(f"{i + 1}. Feature {idx} ({x.columns[idx]}): {importance[idx]}")
# Plot feature importances
plt.figure()
plt.bar(range(x.shape[1]), importance[indices], align="center")
plt.xticks(range(x.shape[1]), np.array(x.columns)[indices], rotation=90)

# Add text labels
for i, v in enumerate(importance[indices]):
    plt.text(i, v - 0.01, str(round(v, 2)), ha="center", color='white')


plt.title("Features Importance")
plt.ylabel("Importance")
plt.xlabel("Features")
plt.tight_layout()
# Create a legend
plt.legend()
plt.show()

# Compute the mean values for each variable
mean_values = x_test.mean()
# Generate data and figures for each feature
for feature_name in x_test.columns:
    
    predictions = [[], []]
    column_data_types = x_test.dtypes
    
    if feature_name in x_test.columns:
        range_min = np.min(x_test[feature_name])
        range_max = np.max(x_test[feature_name])
        
        feature_range = np.arange(range_min, range_max + 1)
        input_values = pd.DataFrame(np.repeat(mean_values.values.reshape(1, -1), len(feature_range), axis=0), columns=mean_values.index)
        input_values[feature_name] = feature_range
        #y_actual = y_test[x_test[feature_name].iloc]
        prediction = model_rf.predict(input_values)
        prediction_prob = model_rf.predict_proba(input_values)[:, 0]  # Assuming positive class is at index 1
        #a = accuracy_score(y_test , prediction)
        #print(f"accuracy '{feature_name}': ", a)
        
        
        predictions[0] = feature_range
        predictions[1] = prediction_prob
        
        plt.plot(predictions[0], predictions[1])
        # Perform linear regression
        slope, intercept = np.polyfit(predictions[0], predictions[1], 1)
        regression_eq = f'y = {slope:.2f}x + {intercept:.2f}'
        plt.plot(predictions[0], slope * predictions[0] + intercept, color='green')
        plt.xlabel('Feature Random Samples')
        plt.ylabel('Predicted Heart Attack Probability')
        plt.title(f'Prediction based on the changes of: {feature_name}')
        plt.legend()
        plt.show()
        
    else:
        print(f"Column '{feature_name}' not found in x_train.")
        
    #input_values.index = x_train.index
    predictions[0] = input_values[feature_name].copy()
    
    # Check the values and data types in input_values[feature_name]
    print("input_values[feature_name] values:", input_values[feature_name].values)
    print("input_values[feature_name] data type:", input_values[feature_name].dtype)
     
    # Perform the prediction using model_rf
    prediction = model_rf.predict(input_values)
    prediction_prob = model_rf.predict_proba(input_values)
   
    # Get the classes in the classifier
    classes = model_rf.classes_
    # Check the index of the positive class (e.g., heart attack = 1)
    positive_class_index = np.where(classes == 1)[0][0]
    # Get the probabilities for the positive class
    positive_class_probabilities = prediction_prob[:, positive_class_index]
    prediction_prob = model_rf.predict_proba(input_values)[:,positive_class_index]
    
    print("prediction:", prediction)
    print("prediction_prob:", prediction_prob)
