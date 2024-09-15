import argparse
import os
import joblib
import logging


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log the environment variables
logger.info(f"SM_MODEL_DIR: {os.environ.get('SM_MODEL_DIR')}")


def model_fn(model_dir):
    """Load the model from the model_dir during inference."""
    model_path = os.path.join(model_dir, "model.joblib")
    logger.info(f"Loading model from {model_path}")
    clf = joblib.load(model_path)
    return clf


if __name__ == '__main__':
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser()

    # Add argument for model directory (default from environment variable)
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    args, _ = parser.parse_known_args()

    # Assuming you already have a trained model object `model`
    # Replace this with your actual trained model object.
    # model = joblib.load("path/to/your/trained/model.joblib")  # Load your existing trained model

    # Save the model to the specified model directory
    model_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")
    print(f"Model saved at: {model_path}")
