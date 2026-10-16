import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# reading data from the csv file
DATA_PATH = Path(__file__).resolve().parent / "data" / "heart.csv"
df = pd.read_csv(DATA_PATH)
pd.set_option('display.max_columns', None)
print(df.head())
print(df.info())

print(df.isnull().sum())
print(df.describe())

# counting values for each category and printing it
print("LEGEND: \n0 indicates the absence of heart diseases\n1 indicates the presence of heart diseases")
print("COUNTING...:")
print(df.target.value_counts())

countDisease = len(df[df.target == 1])
countNoDisease = len(df[df.target == 0])
print("People with heart diseases: ", countDisease)
print("People without heart diseases: ", countNoDisease)
print("Percentage of people with heart Disease: {:.2f}%".format((countDisease / (len(df.target)) * 100)))
print("Percentage of people without heart Disease: {:.2f}%".format((countNoDisease / (len(df.target)) * 100)))

# count plot of people with or without heart diseases
sns.countplot(x='target', data=df)
plt.xlabel("Target (0 = no heart disease, 1 = heart disease)")
plt.show()

countFemale = len(df[df.sex == 0])
countMale = len(df[df.sex == 1])
print("Number of females: ", countFemale)
print("Number of males:", countMale)
print("Percentage of Females: {:.2f}%".format((countFemale / (len(df.sex)) * 100)))
print("Percentage of Males: {:.2f}%".format((countMale / (len(df.sex)) * 100)))

#data visualization
correlation = (df.corr())
plt.figure(figsize=(24,12))
sns.heatmap(correlation, annot=True)
plt.show()



# Splitting data and features
X = df.drop(columns='target')
Y = df['target']

print(X)
print(Y)
# Splitting the Data into Training data & Test Data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, stratify=Y, random_state=1)

print(X.shape, X_train.shape, X_test.shape)

model = LogisticRegression(max_iter=1000)

# training the LogisticRegression model with Training data
model.fit(X_train, Y_train)

# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy on Training data : ', training_data_accuracy)

# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy on Test data : ', test_data_accuracy)

input_data = (62, 1, 3, 140, 238, 1, 0, 150, 0, 2.6, 0, 0, 2)

# wrap the single sample in a DataFrame with the same feature names used for training
input_data_reshaped = pd.DataFrame([input_data], columns=X.columns)

prediction = model.predict(input_data_reshaped)
print(prediction)

if prediction[0] == 0:
    print('The Person does not have a Heart Disease')
else:
    print('The Person has Heart Disease')
