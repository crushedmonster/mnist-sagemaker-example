import sagemaker
from sagemaker.tensorflow import TensorFlow

from sagemaker_config import SETTINGS

def run_training():
    
    session = sagemaker.Session()
    role = SETTINGS.ROLE_ARN

    estimator = TensorFlow(
        entry_point=SETTINGS.ENTRY_POINT,
        source_dir=SETTINGS.SOURCE_DIR,
        role=role,
        instance_type=SETTINGS.INSTANCE_TYPE,
        instance_count=SETTINGS.INSTANCE_COUNT,
        framework_version=SETTINGS.FRAMEWORK_VERSION,
        py_version=SETTINGS.PYTHON_VERSION,
        hyperparameters=SETTINGS.HYPERPARAMETERS,
        output_path=f"{SETTINGS.S3_BUCKET}",
        base_job_name="mnist-training-job"
    )

    # Launch training
    estimator.fit()

if __name__ == "__main__":
    run_training()
