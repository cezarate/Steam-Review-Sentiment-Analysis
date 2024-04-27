import os
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from analytics.settings import settings
import joblib
from analytics.utils import display_runtime

delchars = ''.join(c for c in map(chr, range(256)) if not c.isalnum() and c != ' ')


def strip_content(text):
    return text.translate(str.maketrans('','',delchars))


def extract_features(log_file):

    with open(f"analytics\\log\\{log_file}", "r", encoding="utf-8") as f:
        review_details = json.loads(f.read()).get("reviews")
        reviews = [strip_content(review_detail["review"]) for review_detail in review_details]
        votes = [1 if review_detail["voted_up"] else 0 for review_detail in review_details]

        return reviews, votes
    

@display_runtime
def get_bag_of_ngrams(n):

    print("Extracting features...")

    log_files = [file for file in os.listdir("analytics\\log\\") if "summary" not in file]
    all_votes = []
    all_reviews = []

    for log_file in log_files:
        reviews, votes = extract_features(log_file)
        all_reviews += reviews
        all_votes += votes

    train_features, test_features, train_labels, test_labels = train_test_split(all_reviews, all_votes, test_size=0.2)

    vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, n), stop_words=None)
    train_tfidfed = vectorizer.fit_transform(train_features)
    test_tfidfed = vectorizer.transform(test_features)

    export_vectorizer(vectorizer)

    return train_tfidfed, train_labels, test_tfidfed, test_labels, vectorizer


def export_vectorizer(vectorizer):
    joblib.dump(vectorizer, os.path.join(settings.ml_model_directory, settings.tfidf_vectorizer_filename))


def import_vectorizer():
    return joblib.load(os.path.join(settings.ml_model_directory, settings.tfidf_vectorizer_filename))


def has_export():
    return os.path.exists(os.path.join(settings.ml_model_directory, settings.tfidf_vectorizer_filename))
