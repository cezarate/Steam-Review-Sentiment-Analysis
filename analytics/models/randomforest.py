from analytics.models.mlmodels import MLModel
from sklearn.ensemble import RandomForestClassifier
from analytics.settings import settings

class RandomForest(MLModel):

    def __init__(self):
        self.model_type = RandomForestClassifier()
        self.params = {
            "criterion": ["gini"],
            "max_features": ["sqrt"]
        }
        self.model_filename = settings.random_forest_filename
        self.gs_model = None

