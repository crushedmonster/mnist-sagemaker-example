## Pre-requisite
1. Install [UV](https://docs.astral.sh/uv/getting-started/installation/)
2. Ensure you have access to SageMaker Studio

## Quick Start
1. Clone this repository and navigate to the project directory.
2. Copy the example environment file and configure it:
```bash
cp example.env .env
```
3. Populate .env with your specific values (eg. SageMaker role ARN, S3 bucket)

### Model training
#### Sagemaker Studio Code Editor
Sync environment with UV:
```bash
uv sync
```

Run on the terminal of Sagemaker Studio Code editor:
```bash
# Test on studio code editor locally
uv run src/train_model.py --local
```

Log training to Sagemaker:
```bash
uv run src/sagemaker_training.py
```