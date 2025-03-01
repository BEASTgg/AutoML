import streamlit as st
import pandas as pd
import os
import numpy as np
import pickle

from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet, BayesianRidge, SGDClassifier, PassiveAggressiveClassifier, Perceptron
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier, ExtraTreesRegressor, ExtraTreesClassifier, AdaBoostClassifier, BaggingClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVR, SVC, NuSVC, LinearSVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, ComplementNB
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, classification_report

import pycaret.classification as classification
import pycaret.regression as regression


with st.sidebar:
    st.image('icon.png')
    st.title('Automated ML')
    choice = st.radio("Navigation", ['Upload', 'Data Analysis', 'Regression Modelling', 'Classification Modelling', 'Regressor Testing', 'Classifier Testing'])
    st.info('An AI-driven AutoML web app that automates the process of building, training, and deploying machine learning models, making data science accessible to everyone.')

if os.path.exists('original_data.csv'):
    df = pd.read_csv('original_data.csv', index_col=None)

if choice == 'Upload':
    file = st.file_uploader('Upload your Data here')
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv('original_data.csv', index=None)
        st.dataframe(df)


def handle_missing_values(df):
    df = df.fillna(df.median(numeric_only=True))
    df = df.fillna(df.mode().iloc[0])
    return df


if choice == 'Data Analysis':
    st.title('Exploratiory Data Analysis')
    report = ProfileReport(df)
    st_profile_report(report)


if choice == 'Regression Modelling':
    st.title('Manual Regression Model Training')
    target = st.selectbox('Select Target Parameter', df.columns)
    
    if st.button('Train Model'):
        df = handle_missing_values(df)
        X = df.drop(columns=[target])
        y = df[target]

        regression.setup(df, target=target)
        
        X = pd.get_dummies(X, drop_first=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(),
            'Lasso Regression': Lasso(),
            'Elastic Net': ElasticNet(),
            'Bayesian Ridge': BayesianRidge(),
            'Random Forest Regressor': RandomForestRegressor(),
            'Gradient Boosting Regressor': GradientBoostingRegressor(),
            'Extra Trees Regressor': ExtraTreesRegressor(),
            'SVR': SVR(),
            'KNN Regressor': KNeighborsRegressor()
        }
        
        best_model = None
        best_mae = float('inf')
        results = {}
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            mae = mean_absolute_error(y_test, predictions)
            results[name] = mae
            
            if mae < best_mae:
                best_mae = mae
                best_model = model
        
        st.info('Model Performance')
        st.write(results)
        st.success(f'Best Model: {best_model}')

        regression.save_model(best_model, 'best_regressor')
        
        with open('best_regressor.pkl', 'wb') as f:
            pickle.dump(best_model, f)
        with open('best_regressor.pkl', 'rb') as f:
            st.download_button('Download Model', f, file_name='best_regressor.pkl')


if choice == 'Classification Modelling':
    st.title('Manual Classification Model Training')
    target = st.selectbox('Select Target Parameter', df.columns)
    
    if st.button('Train Model'):
        df = handle_missing_values(df)
        X = df.drop(columns=[target])
        y = df[target]

        classification.setup(df, target=target)
        
        if y.dtype == 'O':
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        X = pd.get_dummies(X, drop_first=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        models = {
            'Logistic Regression': LogisticRegression(),
            'Ridge Classifier': RidgeClassifier(),
            'KNN Classifier': KNeighborsClassifier(),
            'Decision Tree Classifier': DecisionTreeClassifier(),
            'Random Forest Classifier': RandomForestClassifier(),
            'Gradient Boosting Classifier': GradientBoostingClassifier(),
            'Extra Trees Classifier': ExtraTreesClassifier(),
            'SVC': SVC(),
            'Naïve Bayes': GaussianNB(),
            'Multinomial Naïve Bayes': MultinomialNB(),
            'Bernoulli Naïve Bayes': BernoulliNB(),
            'Complement Naïve Bayes': ComplementNB(),
            'AdaBoost Classifier': AdaBoostClassifier(),
            'Bagging Classifier': BaggingClassifier(),
            'HistGradientBoosting Classifier': HistGradientBoostingClassifier(),
            'SGD Classifier': SGDClassifier(),
            'Passive Aggressive Classifier': PassiveAggressiveClassifier(),
            'Perceptron': Perceptron()
        }
        
        results = {}
        best_model = None
        best_acc = 0
        
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            predictions = model.predict(X_test_scaled)
            acc = accuracy_score(y_test, predictions)
            results[name] = acc
            
            if acc > best_acc:
                best_acc = acc
                best_model = model
        
        st.info('Model Performance')
        st.write(results)
        st.success(f'Best Model: {best_model}')

        classification.save_model(best_model, 'best_classifier')
        
        with open('best_classifier.pkl', 'wb') as f:
            pickle.dump(best_model, f)
        with open('best_classifier.pkl', 'rb') as f:
            st.download_button('Download Model', f, file_name='best_classifier.pkl')

if choice == 'Regressor Testing':
    st.title('Model Testing')
    st.info('Upload your test dataset')
    test = st.file_uploader('Upload Test csv file')
    if test is not None:  
        df2 = pd.read_csv(test)
        st.info('This is your test dataset')
        st.dataframe(df2)
        
        pipeline = regression.load_model('trained_regressor')
        st.info('This is the dataframe with the prediction labels')
        st.dataframe(regression.predict_model(pipeline, df2))
    else:
        st.warning('Please upload a test dataset before proceeding.') 

if choice == 'Classifier Testing':
    st.title('Model Testing')
    st.info('Upload your test dataset')
    test = st.file_uploader('Upload Test csv file')
    if test is not None:  
        df2 = pd.read_csv(test)
        st.info('This is your test dataset')
        st.dataframe(df2)
        
        pipeline = classification.load_model('trained_classifier')
        st.info('This is the dataframe with the prediction labels')
        st.dataframe(classification.predict_model(pipeline, df2))
    else:
        st.warning('Please upload a test dataset before proceeding.') 