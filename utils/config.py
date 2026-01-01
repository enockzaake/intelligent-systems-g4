from pathlib import Path


class Config:
    DATA_PATH = Path("data/cleaned_delivery_data.csv")
    RAW_DATA_PATH = Path("data/raw.xlsx")
    
    OUTPUT_DIR = Path("outputs")
    MODELS_DIR = OUTPUT_DIR / "models"
    PREPROCESSOR_DIR = OUTPUT_DIR / "preprocessor"
    RESULTS_DIR = OUTPUT_DIR / "results"
    
    RANDOM_SEED = 42
    TEST_SIZE = 0.2
    
    LOGISTIC_REGRESSION_CONFIG = {
        "max_iter": 1000,
        "random_state": RANDOM_SEED
    }
    
    RANDOM_FOREST_CLASSIFIER_CONFIG = {
        "n_estimators": 100,
        "max_depth": 15,
        "random_state": RANDOM_SEED,
        "n_jobs": -1
    }
    
    RANDOM_FOREST_REGRESSOR_CONFIG = {
        "n_estimators": 100,
        "max_depth": 15,
        "random_state": RANDOM_SEED,
        "n_jobs": -1
    }
    
    LSTM_CLASSIFIER_CONFIG = {
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.2,
        "learning_rate": 0.001,
        "sequence_length": 10,
        "epochs": 50,
        "batch_size": 64
    }
    
    LSTM_REGRESSOR_CONFIG = {
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.2,
        "learning_rate": 0.001,
        "sequence_length": 10,
        "epochs": 50,
        "batch_size": 64
    }
    
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    
    @classmethod
    def create_directories(cls):
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        cls.PREPROCESSOR_DIR.mkdir(parents=True, exist_ok=True)
        cls.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_model_path(cls, model_name):
        return cls.MODELS_DIR / f"{model_name}"
    
    @classmethod
    def get_preprocessor_path(cls, filename):
        return cls.PREPROCESSOR_DIR / filename
    
    @classmethod
    def get_results_path(cls, filename):
        return cls.RESULTS_DIR / filename

