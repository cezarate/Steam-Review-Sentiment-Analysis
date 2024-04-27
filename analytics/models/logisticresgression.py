import math
import matplotlib.pyplot as plt
from analytics.models.mlmodels import MLModel
from sklearn.linear_model import LogisticRegression as LR
from analytics.settings import settings
import pandas as pd

class LogisticRegression(MLModel):

    def __init__(self):
        self.model_type = LR()
        self.params = {
            "penalty": ["l1", "l2", "elasticnet", None]
        }
        self.model_filename = settings.logistic_regression_filename

    def visualize(self):
        data = pd.read_csv("results.csv")

        penalties = [str(eval(row["params"].replace("'", "\"")).get("penalty")) for index, row in data.iterrows()]
        scores = [row["mean_test_score"] if not math.isnan(row["mean_test_score"]) else 0 for index, row in data.iterrows()]
        
        fig, ax = plt.subplots()

        bar_colors = ['tab:red', 'tab:blue', 'tab:red', 'tab:orange']

        ax.bar(penalties, scores, label=penalties, color=bar_colors)

        ax.set_ylabel('Accuracy')
        ax.set_title('Logistic Regression')
        ax.legend(title='Penalties')
        plt.ylim((0, 1))

        plt.show()