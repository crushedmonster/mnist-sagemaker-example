## Pre-requisite
1. Install [UV](https://docs.astral.sh/uv/getting-started/installation/)
2. Ensure you have access to SageMaker Studio

## Quick Start
### Setup
1. Clone this repository and navigate to the project directory.
2. Copy the example environment file and configure it:
```bash
cp .example.env .env
```
3. Populate .env with your specific values (eg. SageMaker role ARN, S3 bucket)

### Model training
#### Run in SageMaker Studio Code Editor
Sync environment with UV:
```bash
uv sync
```

Run training locally (within the Studio editor environment):
```bash
# Test on studio code editor locally
uv run src/train_model.py --local
```

Submit training to SageMaker:
```bash
uv run src/sagemaker_training.py
```

### Model Deployment
#### Important: SavedModel Format
SageMaker TensorFlow Serving requires models in the **Tensorflow SavedModel** format and organised into **numeric versioned directories**.

The directory structure must look like this:
```
model/
    1/
        saved_model.pb
        variables/
```

- If you train in SageMaker, the training job automatically saves the model to /opt/ml/model in the correct structure, and SageMaker uploads it to S3.
- If you train locally, you must package the model before uploading.

#### Packaging a Locally Trained Model

Create a .tar.gz archive of the model:
```sh
tar -czf model.tar.gz -C model_checkpoint .
```
#### Uploading to S3
Upload the tarball to your chosen S3 bucket. Replace the placeholders with your values:
```sh
aws s3 cp model.tar.gz s3://<your-sagemaker-bucket>/models/<model-name>/model.tar.gz
```
#### Deploying in SageMaker
Run the deployment script:
```bash
uv run src/sagemaker_deployment.py
```
- This will create a SageMaker endpoint for inference.
