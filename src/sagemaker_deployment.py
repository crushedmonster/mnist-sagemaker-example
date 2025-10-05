from sagemaker.tensorflow import TensorFlowModel 

from sagemaker_config import SETTINGS

def deploy():

    role = SETTINGS.ROLE_ARN

    model = TensorFlowModel( 
        model_data=f"{SETTINGS.S3_BUCKET}/mnist-example/mobilenetv2.tar.gz", 
        role=role, 
        framework_version=SETTINGS.FRAMEWORK_VERSION,
    ) 
        
    predictor = model.deploy( 
        initial_instance_count=1, 
        instance_type=SETTINGS.DEPLOYMENT_INSTANCE_TYPE,
        endpoint_name="mnist-mobilenetv2-inference"
    )

if __name__ == "__main__": 
    deploy()