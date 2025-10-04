import os
import argparse
import logging

from mnist import model 


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(batch_size, epochs, learning_rate, model_dir):

    clf = model.MNISTModel(
        batch_size=batch_size,
        epochs=epochs,
        base_learning_rate=learning_rate,
    )

    # Load dataset
    logger.info("Loading data...")
    train_ds, val_ds, test_ds = clf.load_data()

    # Train
    logger.info("Starting training...")
    clf.train(train_ds, val_ds)
    logger.info("Training finished!")

    # Save model
    clf.save(model_dir)
    logger.info(f"Saved model to {model_dir}")

    # Evaluate
    loss, acc = clf.evaluate(test_ds)
    logger.info(f"Test Accuracy: {acc:.4f}, Loss: {loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Hyperparameters from SageMaker
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.environ.get("SM_MODEL_DIR", "model_checkpoint")  # Local fallback
    )

    args = parser.parse_args()

    main(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        model_dir=args.model_dir
    )