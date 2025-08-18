import numpy as np
from scipy.stats import entropy
import json

OLMO_PREDICTIONS_PATH = "../olmo_predictions/1000-3000__was.were.is.are.will__suzeva_olmo2-1b-4xH100-2ndtry-step-10000__In_[year]_there/olmo_predictions.json"
OLMO_TRAINING_DATA_FILE = "../olmo_training_data/1000-3000__was.were.is.are.will__allenai_OLMo-2-0425-1B/aggregated/steps0-10000/analytics/1000-3000__was.were.is.are.will__aggregated_results_steps0-10000.json"

def get_relative_probabilities(olmo_predictions, olmo_training_data):
    # turn olmo_predictions into nice format
    year_to_relative_predictions = {}
    for year in olmo_predictions.keys():
        if year == "metadata":
            continue
        
        relative_predictions = olmo_predictions[year]["relative"]
        past = relative_predictions[" was"] + relative_predictions[" were"]
        present = relative_predictions[" is"] + relative_predictions[" are"]
        future = relative_predictions[" will"]

        year_to_relative_predictions[year] = {
            "past": past,
            "present": present,
            "future": future
        }

    # turn olmo_training_data into nice format
    year_to_relative_counts = {}
    relative_counts = olmo_training_data["in_year_there_word_counts"]
    for year in relative_counts.keys():
        counts_per_year = relative_counts[year]
        total_be_tense_counts = counts_per_year.get("was", 0) + counts_per_year.get("were", 0) + counts_per_year.get("is", 0) + counts_per_year.get("are", 0) + counts_per_year.get("will", 0)
        if total_be_tense_counts == 0:
            year_to_relative_counts[year] = {
                "past": 0,
                "present": 0,
                "future": 0
            }
            continue

        past = (counts_per_year.get("was", 0) + counts_per_year.get("were", 0)) / total_be_tense_counts
        present = (counts_per_year.get("is", 0) + counts_per_year.get("are", 0)) / total_be_tense_counts
        future = counts_per_year.get("will", 0) / total_be_tense_counts

        year_to_relative_counts[year] = {
            "past": past,
            "present": present,
            "future": future
        }

    # print summary as sanity check
    print("\nModel predictions:")
    print("1001:", json.dumps(year_to_relative_predictions["1001"], indent=4))
    print("1980:", json.dumps(year_to_relative_predictions["1980"], indent=4))
    print("2030:", json.dumps(year_to_relative_predictions["2030"], indent=4))

    print("\nTraining data distributions:")
    print("1001:", json.dumps(year_to_relative_counts["1001"], indent=4))
    print("1980:", json.dumps(year_to_relative_counts["1980"], indent=4))
    print("2030:", json.dumps(year_to_relative_counts["2030"], indent=4))

    return year_to_relative_predictions, year_to_relative_counts

if __name__ == "__main__":
    # python kl_divergence.py
    # load data
    with open(OLMO_PREDICTIONS_PATH, "r") as f:
        olmo_predictions = json.load(f)

    with open(OLMO_TRAINING_DATA_FILE, "r") as f:
        olmo_training_data = json.load(f)

    year_to_relative_predictions, year_to_relative_counts = get_relative_probabilities(olmo_predictions, olmo_training_data)
    """
    Output of this will look something like this:

    Model predictions:
    1001: {
        "past": 0.8894760612136261,
        "present": 0.06551307810828974,
        "future": 0.0450108606780842
    }
    1980: {
        "past": 0.9861653467751059,
        "present": 0.012863822370620615,
        "future": 0.0009708308542735161
    }
    2030: {
        "past": 0.033143502475075054,
        "present": 0.20145175507142032,
        "future": 0.7654047424535046
    }

    Training data distributions:
    1001: {
        "past": 0,
        "present": 0,
        "future": 0
    }
    1980: {
        "past": 0.9629629629629629,
        "present": 0.02962962962962963,
        "future": 0.007407407407407408
    }
    2030: {
        "past": 0.0,
        "present": 0.09090909090909091,
        "future": 0.9090909090909091
    }
    
    Note: this will add 0 0 0 for years that don't exist in the training data we consider
    """

 


    
    
    
        

