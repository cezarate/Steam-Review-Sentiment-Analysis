from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    log_directory: str
    ml_model_directory: str
    tfidf_vectorizer_filename: str
    logistic_regression_filename: str
    naive_bayes_filename: str
    random_forest_filename: str
    svm_filename : str
    n_grams: int

    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

settings = Settings()