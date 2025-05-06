import pickle
import os
import logging

# Function to save the trained model
def save_model(model, model_filename):
    with open(model_filename, 'wb') as file:
        pickle.dump(model, file)
    logging.info(f"Model saved as {model_filename}")

# Function to load a saved model
def load_model(model_filename):
    if os.path.exists(model_filename):
        with open(model_filename, 'rb') as file:
            model = pickle.load(file)
        logging.info(f"Model loaded from {model_filename}")
        return model
    else:
        logging.error(f"Model file {model_filename} not found!")
        return None

# Function to log the evaluation metrics
def log_metrics(accuracy, classification_report, confusion_matrix):
    logging.info(f"Accuracy: {accuracy}")
    logging.info(f"Classification Report: {classification_report}")
    logging.info(f"Confusion Matrix: {confusion_matrix}")
