from sklearn.svm import SVC
from analytics.models.mlmodels import MLModel
from analytics.settings import settings

class SVM(MLModel):

    def __init__(self):
        self.model_type = SVC()
        self.params = {
            "kernel": ["linear"],
            "gamma": ["scale"],
        }
        self.model_filename = settings.svm_filename
        self.gs_model = None

