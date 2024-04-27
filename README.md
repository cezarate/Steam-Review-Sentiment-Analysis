
# Steam Review Sentiment Analysis

Hello! This is a simple sentiment analysis project that uses steam reviews as a dataset. This project uses TF-IDF to transform the reviews for training the models. The machine learning models used are the following:

    - Logistic Regression
    - Naive Bayes
    - Support Vector Machines[^1]
    - Random Forest[^1]

[^1]: Takes too long to train. 
## Run Locally

Clone the project

```bash
  git clone https://link-to-project
```

Go to the project directory

```bash
  cd my-project
```

Install dependencies

```bash
  poetry init
```

Start the server

```bash
  python -m analytics
```

## Features

A simple menu in the cli is provided to easily navigate the program. This project is capable of:

    - Retrieving steam reviews by game_code.
    - Transformation of reviews to TF-IDF, including exporting and importing of the vectorizer.
    - Training models that find and use the best parameters using sklearn's GridSearchCV. Includes exporting and importing of the trained models.
    - Displaying the accuracy of the model with different model parameters using matplotlib.
    - Testing models with user-typed reviews.

Retraining models will be implemented next.
## Environment Variables

To run this project, you will need to add the following environment variables to your .env file

    LOG_DIRECTORY="analytics/logs"
    ML_MODEL_DIRECTORY="analytics/exported_models"
    TFIDF_VECTORIZER_FILENAME
    LOGISTIC_REGRESSION_FILENAME
    NAIVE_BAYES_FILENAME
    RANDOM_FOREST_FILENAME
    SVM_FILENAME
    N_GRAMS


## 🚀 About Me
I'm a software developer.

