import os
import argparse
import logging

from mnist import model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(batch_size, epochs, learning_rate, is_local, model_checkpoint):

    clf = model.MNISTModel(
        batch_size=batch_size,
        epochs=epochs,
        base_learning_rate=learning_rate,
    )

    # Load dataset
    logger.info("Loading dataset...")
    train_ds, val_ds, test_ds = clf.load_data()

    # Train
    logger.info("Starting training...")
    clf.train(train_ds, val_ds)
    logger.info("Training completed.")

    # # Save model to SageMaker's model dir (uploaded to S3 automatically)
    # # model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    # save_path = os.path.join(model_dir, "saved_model")
    # clf.save_model(save_path)
    # logger.info(f"Model saved at {save_path}")

    # Save model (different paths for local vs SageMaker) 
    if is_local: 
        os.makedirs(model_checkpoint, exist_ok=True) 
        save_path = os.path.join(model_checkpoint, "saved_model") 
        logger.info(f"Saving model locally to: {save_path}") 
    else: 
        sm_model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model") 
        save_path = os.path.join(sm_model_dir, "saved_model")
        logger.info(f"Saving model to SageMaker model dir: {save_path}") 
    
    clf.save_model(save_path)

    # Evaluate
    loss, acc = clf.evaluate(test_ds)
    logger.info(f"Test Accuracy: {acc:.4f}, Loss: {loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Hyperparameters
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.001)

    # SageMaker automatically passes this
    parser.add_argument("--model_dir", type=str, default="/opt/ml/model")

    # Mode selection: default False means SageMaker
    parser.add_argument(
        "--local",
        action="store_true",
        default=False,
        help="Run locally instead of SageMaker. Default: False (runs on SageMaker)"
    )

    # Local-only checkpoint directory
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        default="./model_checkpoint",
        help="Directory to save model when running locally"
    )
    
    args = parser.parse_args()

    main( 
        batch_size=args.batch_size, 
        epochs=args.epochs, 
        learning_rate=args.learning_rate,
        # model_dir=args.model_dir
        is_local=args.local,
        model_checkpoint=args.model_checkpoint
    )