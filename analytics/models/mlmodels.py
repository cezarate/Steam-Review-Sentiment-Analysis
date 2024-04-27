import pandas as pd
from sklearn.model_selection import GridSearchCV
from analytics.settings import settings
from analytics.utils import display_runtime
import joblib
import os


class MLModel:
    

    def __init__(self):
        self.model_type = None
        self.params = {}
        self.model_filename = ""
        self.gs_model = None

    @display_runtime
    def train(self, train_features, train_labels, test_features, test_labels):
        
        print("Training model...")

        self.gs_model = GridSearchCV(estimator=self.model_type, param_grid=self.params, cv=10)
        self.gs_model.fit(train_features, train_labels)
        
        print(self.gs_model.score(test_features, test_labels))
        print(self.gs_model.best_params_)

        self.export_model()

        df = pd.DataFrame(self.gs_model.cv_results_)
        df.to_csv("results.csv")


    def visualize(self):
        print(f"Please implement this method in {self.__class__}")


    def export_model(self):
        print("Exporting model...")
        joblib.dump(self.gs_model, os.path.join(settings.ml_model_directory, self.model_filename))


    def import_model(self):
        print("Importing model...")
        self.gs_model = joblib.load(os.path.join(settings.ml_model_directory, self.model_filename))


    def evaluate(self, review):
        prediction = self.gs_model.predict(review)
        if prediction[0] == 1:
            print("The model detects this as a positive review!")
        else:
            print("The model detects this as a negative review")
        return prediction
