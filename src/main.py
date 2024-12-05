from pathlib import Path
from config import TrainingConfig
from pipeline import CIFAR10Pipeline


def main():
    # Load or create configuration
    config_path = Path("config.yaml")
    if config_path.exists():
        config = TrainingConfig.load_config(config_path)
    else:
        config = TrainingConfig()
        config.save_config(config_path)

    # Initialize pipeline
    pipeline = CIFAR10Pipeline(config)

    # Step 1: Preprocess and augment data
    # Comment out if preprocessing is already done
    # pipeline.preprocess_data()


    # Step 2: Train model
    pipeline.train_model(hypertune=True)

    # Step 3: Make predictions
    pipeline.load_trained_model("models/best_model.pt")
    # metrics_report, confusion_mat = pipeline.trainer.compute_validation_metrics()
    metrics = pipeline.trainer.compute_validation_metrics()
    print("Validation metrics: ", metrics, "\n")
    # predictions_df = pipeline.predict_test_directory("data/predictions.csv")


if __name__ == "__main__":
    main()
