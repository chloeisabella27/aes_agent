import os
import subprocess
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI()

# Path to dataset
DATA_ROOT = "/Users/chloeisabella/Desktop/Files for Yash"

SYSTEM_PROMPT = """
You are an AI assistant for Auger Electron Spectroscopy (AES) analysis.

You are connected to a machine learning pipeline that predicts the next
Ti MVV Auger spectrum for experiments.

You are only allowed to perform these tasks:

1. Predict the next Ti MVV spectrum for an experiment
2. List available experiments
3. Explain prediction results

If the user asks for something outside these capabilities,
respond exactly with:

"I'm sorry, I can't perform that task with the current AES analysis tools."

Never invent experimental data or results.
Only use the available prediction tools.
"""


# --------------------------------------------------
# Utility: list experiments from dataset
# --------------------------------------------------

def list_experiments():

    if not os.path.exists(DATA_ROOT):
        return []

    experiments = [
        d for d in os.listdir(DATA_ROOT)
        if os.path.isdir(os.path.join(DATA_ROOT, d))
    ]

    experiments.sort()
    return experiments


# --------------------------------------------------
# Guardrail: validate experiment name
# --------------------------------------------------

def validate_experiment(exp):

    experiments = list_experiments()

    if exp not in experiments:
        return False

    return True


# --------------------------------------------------
# Tool: run ML prediction
# --------------------------------------------------

def run_prediction(experiment):

    if not validate_experiment(experiment):
        print(f"\nAgent: Experiment '{experiment}' not found.\n")
        return

    print(f"\n[TOOL CALL] run_prediction.py {experiment}\n")

    subprocess.run(
        ["python", "run_prediction.py", experiment],
        check=False
    )


# --------------------------------------------------
# Guardrail: detect allowed intent
# --------------------------------------------------

def detect_intent(user_input):

    text = user_input.lower()

    if "predict" in text:
        return "predict"

    if "list" in text or "experiments" in text:
        return "list"

    return "unsupported"


# --------------------------------------------------
# Extract experiment name (TF268 style)
# --------------------------------------------------

def extract_experiment(user_input):

    words = user_input.split()

    for w in words:
        if w.upper().startswith("TF") or w.upper().startswith("HF"):
            return w.strip()

    return None


# --------------------------------------------------
# Main agent loop
# --------------------------------------------------

def run_agent():

    print("\nAES Analysis Agent Ready")
    print("Type 'quit' to exit.\n")

    while True:

        user_input = input("User: ")

        if user_input.lower() == "quit":
            break

        intent = detect_intent(user_input)

        # ---------------------------------
        # LIST EXPERIMENTS
        # ---------------------------------

        if intent == "list":

            experiments = list_experiments()

            if not experiments:
                print("\nAgent: No experiments found.\n")
                continue

            print("\nAvailable experiments:\n")

            for e in experiments:
                print(e)

            print()
            continue

        # ---------------------------------
        # PREDICT NEXT SPECTRUM
        # ---------------------------------

        if intent == "predict":

            experiment = extract_experiment(user_input)

            if experiment is None:
                print("\nAgent: Please specify an experiment name (e.g. TF268).\n")
                continue

            if not validate_experiment(experiment):
                print(f"\nAgent: Experiment '{experiment}' not found.\n")
                continue

            run_prediction(experiment)

            print("\nAgent: Prediction completed.\n")

            continue

        # ---------------------------------
        # UNSUPPORTED REQUEST
        # ---------------------------------

        print("\nAgent: I'm sorry, I can't perform that task with the current AES analysis tools.\n")


# --------------------------------------------------
# Run agent
# --------------------------------------------------

if __name__ == "__main__":
    run_agent()