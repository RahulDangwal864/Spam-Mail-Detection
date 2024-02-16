# Importing dependencies
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Data collection & pre-processing
# Loading the data from csv file to a pandas dataframe
raw_mail_data = pd.read_csv(r'C:\Users\rahul\Desktop\vscode\SpamMailDetection\mail_data.csv')
print(raw_mail_data)

# Handling missing values
mail_data = raw_mail_data.fillna('')

# Printing the first 13 rows of the data frame
mail_data.head()

# Checking the number of rows and columns in the dataframe
mail_data.shape

# Label Encoding
# Label spam mail as 0; ham mail as 1;
mail_data['Category'] = mail_data['Category'].replace({'spam': 0, 'ham': 1})

# Separating the data as texts and label
X = mail_data['Message']
Y = mail_data['Category']

print(X)
print(Y)

# Splitting the data into training data & test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)
print(X.shape)
print(X_train.shape)
print(X_test.head())

# Feature Extraction
# Transform the text data to feature vectors using TF-IDF
tfidf_vectorizer = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_features = tfidf_vectorizer.fit_transform(X_train)
X_test_features = tfidf_vectorizer.transform(X_test)

# Convert Y_train and Y_test values to integers
Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

# Training a Logistic Regression model
model_lr = LogisticRegression()
model_lr.fit(X_train_features, Y_train)

# Evaluating the Logistic Regression model
y_pred_train_lr = model_lr.predict(X_train_features)
accuracy_train_lr = accuracy_score(Y_train, y_pred_train_lr)

y_pred_test_lr = model_lr.predict(X_test_features)
accuracy_test_lr = accuracy_score(Y_test, y_pred_test_lr)

print('\nLogistic Regression Model:')
print('Accuracy on training data:', accuracy_train_lr)
print('Accuracy on test data:', accuracy_test_lr)

# Training a Random Forest classifier
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train_features, Y_train)

# Evaluating the Random Forest model
y_pred_train_rf = model_rf.predict(X_train_features)
accuracy_train_rf = accuracy_score(Y_train, y_pred_train_rf)

y_pred_test_rf = model_rf.predict(X_test_features)
accuracy_test_rf = accuracy_score(Y_test, y_pred_test_rf)

print('\nRandom Forest Model:')
print('Accuracy on training data:', accuracy_train_rf)
print('Accuracy on test data:', accuracy_test_rf)

# Classification report on test data for Random Forest model
print('\nClassification Report for Random Forest Model on test data:')
print('\n-------------------------------------------------------------------\n')
print(classification_report(Y_test, y_pred_test_rf))
print('\n-------------------------------------------------------------------\n')

# Building a Predictive System with Logistic Regression
input_mail_lr = ["This is the 2nd time we have tried 2 contact u..."]
input_data_features_lr = tfidf_vectorizer.transform(input_mail_lr)
prediction_lr = model_lr.predict(input_data_features_lr)

if prediction_lr[0] == 1:
    print('\nLogistic Regression Model Prediction: Ham mail')
else:
    print('\nLogistic Regression Model Prediction: Spam mail')

# Building a Predictive System with Random Forest
input_mail_rf = ["This is the 2nd time we have tried 2 contact u..."]
input_data_features_rf = tfidf_vectorizer.transform(input_mail_rf)
prediction_rf = model_rf.predict(input_data_features_rf)

if prediction_rf[0] == 1:
    print('Random Forest Model Prediction: Ham mail')
else:
    print('Random Forest Model Prediction: Spam mail')
