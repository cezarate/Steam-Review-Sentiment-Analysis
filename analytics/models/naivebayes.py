import numpy as np
import json
import matplotlib.pyplot as plt
from analytics.models.mlmodels import MLModel
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
from analytics.settings import settings

class NaiveBayes(MLModel):

    def __init__(self):
        self.model_type = MultinomialNB()
        self.params = {
            "alpha": np.logspace(-1, 1, 10)
        }
        self.model_filename = settings.naive_bayes_filename
        self.gs_model = None

    def visualize(self):
        data = pd.read_csv("results.csv")

        alphas = [json.loads(row["params"].replace("'", "\""))["alpha"] for index, row in data.iterrows()]
        scores = [row["mean_test_score"] for index, row in data.iterrows()]
        
        # plt.xticks(alphas)
        plt.ylabel('Score')
        plt.xlabel('Alpha Value')
        plt.grid(True)
        plt.ylim((0, 1))
        plt.xticks(alphas)
        plt.plot(alphas, scores, linestyle='--', marker='o', color='b', label='line with marker')
        plt.legend()
        plt.show()      

