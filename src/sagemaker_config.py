import os
from dotenv import load_dotenv

load_dotenv()

class SageMakerConfig:
    """Configuration for SageMaker training jobs
    """

    # Get sensitive info from environment variables
    ROLE_ARN = os.environ.get("SAGEMAKER_ROLE_ARN")
    S3_BUCKET = os.environ.get("S3_BUCKET")

    # Training parameters
    INSTANCE_TYPE = "ml.m5.xlarge"
    INSTANCE_COUNT = 10
    FRAMEWORK_VERSION = "2.13"
    PYTHON_VERSION = "py310"

    # Hyperparameters
    HYPERPARAMETERS = {
        "epochs": 1,
        "batch-size": 32,
        "learning-rate": 0.001,
    }

    # Local source dir
    SOURCE_DIR = "src"
    ENTRY_POINT = "train_model.py"

SETTINGS = SageMakerConfig()