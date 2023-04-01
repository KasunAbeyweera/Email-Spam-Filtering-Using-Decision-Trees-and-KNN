# Email-Spam-Filtering-Using-Decision-Trees-and-KNN
This repository contains code for Email Spam Filtering using two machine learning algorithms, Decision Trees and K-Nearest Neighbors (KNN). The code includes data preprocessing, model creation, hyperparameter tuning, and model evaluation. The models are trained on a dataset of emails labeled as spam or non-spam.
Spam Detection using Machine Learning

In this project, we have developed a machine learning model to classify emails as spam or not spam. The dataset used for this task is the SpamBase dataset, which contains features extracted from 4,601 emails, with a total of 57 features, and a binary target variable indicating whether an email is spam or not.
Requirements

    Python 3.6+
    Pandas
    Numpy
    Matplotlib
    Seaborn
    Scikit-learn
    Imblearn

Steps

    Importing Required Libraries
    Loading Dataset
    Data Exploration And Preprocessing
        Checking Missing Values
        Check the class distribution
        Distribution of each feature
        Checking correlation between the features
        Visualize the distribution of each feature
        Finding Outliers
        Removing Duplicate Records
        Data Oversampling
    Feature Scaling
    Dimensionality Reduction
    Model Development
    Model Evaluation

Usage

To run this project, clone the repository and open the Jupyter notebook spam_detection.ipynb. The notebook contains all the code for the project, and you can run each cell sequentially to reproduce the results.

Results

The best performing model was a KNN without PCA and the Qt, with an accuracy of 95% and F1-score of 0.94. 
Conclusion

In this project, we have developed a machine learning model to classify emails as spam or not spam. We explored the dataset, preprocessed the data, applied feature scaling and dimensionality reduction, and oversampled the minority class. We trained and evaluated various models and identified a KNN as the best performing model. The model achieved an accuracy of 95% and F1-score of 0.94.
