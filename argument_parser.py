import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_id", default=0, type=int)
    parser.add_argument("--perform_expert_hook", action="store_true")
    parser.add_argument("--perform_safety_classifications", action="store_true")

    arguments = parser.parse_args()

    print(f"Configuration:\n'{vars(arguments)}'\n")

    return arguments