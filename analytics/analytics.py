from analytics.puller import pull_reviews
from analytics.utils import cleanup
from analytics.feature_extractor import get_bag_of_ngrams, import_vectorizer, has_export
from analytics.models.naivebayes import NaiveBayes
from analytics.models.mlmodels import MLModel
from analytics.models.logisticresgression import LogisticRegression
from analytics.settings import settings

# from analytics.models.svm import SVM
# from analytics.models.randomforest import RandomForest

def start():
    selection = 0

    while selection not in (1, 2, 3):
        print("Hello! This is a simple sentiment analysis project based on Steam reviews")
        print("1. Train a model")
        print("2. Test a model")
        print("3. Exit")
        selection = int(input())
        print(selection)
        match selection:
            case 1:
                dataset_selection()
            case 2:
                model = model_selection()

                if model:
                    evaluate(model)
                
            case 3:
                break
            case _:
                print("Invalid input")
                

def evaluate(model):

    try:
        vectorizer = import_vectorizer()
        model.import_model()
        review = input("Enter a review to test sentiment analysis:\n")
        analysis = model.evaluate(vectorizer.transform([review]))
        print(analysis)
    except FileNotFoundError as e:
        print("pkl files do not yet exist. Please train the model first.")


def dataset_selection():
    selection = 0
    while selection not in (1, 2, 3):
        print("Configure dataset")
        print("1. Load Dataset")
        print("2. New Dataset")
        print("3. Back")
        selection = int(input())
        match selection:
            case 1:
                cleanup(settings.ml_model_directory)
                break
            case 2:
                cleanup(settings.ml_model_directory)
                cleanup(settings.log_directory)
                data_retrieval()
                break
            case 3:
                return
            case _:
                print("Invalid input")
    train_tfidfed, train_labels, test_tfidfed, test_labels, _ = get_bag_of_ngrams(n=settings.n_grams)
    model = model_selection()

    if model:
        model.train(train_tfidfed, train_labels, test_tfidfed, test_labels)
        model.visualize()

    return
                
def data_retrieval():
    game_codes = []
    game_code = None
    while game_code != 0:
        game_code = int(input("Enter steam game code: (Enter 0 to exit): "))
        if game_code == 0:
            break
        elif isinstance(game_code, int):
            game_codes.append(game_code)
        else:
            print("Invalid input")
    if game_codes != []:
        for game_code in game_codes:
            pull_reviews(game_code)

# 865610
# 1716740

def model_selection() -> MLModel:
    selection = 0
    model = None
    while selection not in (1, 2, 3):
        print("Select a model")
        print("1. Naive Bayes")
        print("2. Logistic Regression")
        print("3. Back")
        selection = int(input())
        match selection:
            case 1:
                model = NaiveBayes()
                break
            case 2:
                model = LogisticRegression()
                break
            case 3:
                return None
            case _:
                print("Invalid input")
    return model
