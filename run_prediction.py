import sys
import os
from aes_ml_pipeline.predict_next_scan import predict_next_scan
import matplotlib.pyplot as plt


def main():

    if len(sys.argv) < 2:
        print("Usage: python run_prediction.py <EXPERIMENT_NAME>")
        sys.exit(1)

    experiment_name = sys.argv[1]

    data_root = "/Users/chloeisabella/Desktop/Files for Yash"
    experiment_folder = os.path.join(data_root, experiment_name)

    if not os.path.exists(experiment_folder):
        print(f"Experiment folder not found: {experiment_folder}")
        sys.exit(1)

    print(f"\nRunning prediction for experiment: {experiment_name}")

    energy, predicted = predict_next_scan(experiment_folder)

    print("Prediction complete.")
    print(f"Spectrum length: {len(predicted)}")

    plt.plot(energy, predicted, label="Predicted Spectrum")
    plt.xlabel("Energy (eV)")
    plt.ylabel("Normalized Intensity")
    plt.title(f"Predicted Ti MVV Spectrum ({experiment_name})")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()