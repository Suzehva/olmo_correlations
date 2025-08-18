import numpy as np
from scipy.stats import entropy
import json
import matplotlib.pyplot as plt

OLMO_PREDICTIONS_PATH = "../olmo_predictions/1000-3000__was.were.is.are.will__suzeva_olmo2-1b-4xH100-2ndtry-step-10000__In_[year]_there/olmo_predictions.json"
OLMO_TRAINING_DATA_FILE = "../olmo_training_data/1000-3000__was.were.is.are.will__allenai_OLMo-2-0425-1B/aggregated/steps0-10000/analytics/1000-3000__was.were.is.are.will__aggregated_results_steps0-10000.json"

def kl_divergence_from_dicts(dist_p, dist_q, epsilon=1e-12):
    """Calculate KL divergence KL(P || Q) between two distributions from dictionaries."""
    all_categories = set(dist_p.keys()) | set(dist_q.keys())
    categories = sorted(all_categories)
    
    p_values = [dist_p.get(cat, 0) for cat in categories]
    q_values = [dist_q.get(cat, 0) for cat in categories]
    
    p = np.array(p_values)
    q = np.array(q_values)
    
    # Apply epsilon smoothing and normalize
    p_smooth = np.maximum(p, epsilon)
    q_smooth = np.maximum(q, epsilon)
    p_smooth = p_smooth / np.sum(p_smooth)
    q_smooth = q_smooth / np.sum(q_smooth)
    
    return entropy(p_smooth, q_smooth)

def plot_kl_divergence(years, kl_values, olmo_training_data, data_source="in_year_there_word_counts", 
                      count_bins=None):
    """Plot KL divergence by year, colored by total word count in training data.
    
    Args:
        years: List of years
        kl_values: List of KL divergence values
        olmo_training_data: Training data dictionary
        data_source: Either "in_year_there_word_counts" or "co_occurrence_frequency_word_boundaries"
        count_bins: List of bin edges (e.g., [0, 1, 4, 11])
    """
    
    # Set default values based on data source
    if data_source == "co_occurrence_frequency_word_boundaries":
        title_suffix = "co-occurrence frequency"
        default_count_bins = [0, 1, 5000, 15000, 25000, 35000, 45000, 55000]
    else:  # "in_year_there_word_counts"
        title_suffix = "In [year] there"
        default_count_bins = [0, 1, 4, 11]
    
    # Use provided count_bins or defaults
    if count_bins is None:
        count_bins = default_count_bins
    
    # Generate bin labels automatically based on count_bins
    bin_labels = []
    for i in range(len(count_bins) - 1):
        if i == len(count_bins) - 2:  # Last bin
            bin_labels.append(f"{count_bins[i]}+")
        else:
            if count_bins[i+1] - count_bins[i] == 1:
                bin_labels.append(str(count_bins[i]))
            else:
                bin_labels.append(f"{count_bins[i]}-{count_bins[i+1]-1}")
    
    # Set standard colors
    colors = ['lightgray', 'orange', 'blue', 'darkblue', 'darkred'][:len(bin_labels)]
    
    # Get total counts for each year
    relative_counts = olmo_training_data[data_source]
    total_counts = []
    
    for year in years:
        counts_per_year = relative_counts[year]
        total_count = (counts_per_year.get("was", 0) + 
                        counts_per_year.get("were", 0) + 
                        counts_per_year.get("is", 0) + 
                        counts_per_year.get("are", 0) + 
                        counts_per_year.get("will", 0))

        total_counts.append(total_count)
    
    # Create plot
    plt.figure(figsize=(20, 6))
    
    
    # Convert years to integers for proper plotting
    years_int = [int(year) for year in years]
    
    # Assign colors based on count bins
    for i, (bin_start, bin_end, label, color) in enumerate(zip(count_bins[:-1], count_bins[1:], bin_labels[:-1], colors[:-1])):
        mask = [(bin_start <= count < bin_end) for count in total_counts]
        x_bin = [year for year, m in zip(years_int, mask) if m]
        y_bin = [kl for kl, m in zip(kl_values, mask) if m]
        if x_bin:  # Only plot if there are points
            plt.scatter(x_bin, y_bin, s=10, c=color, label=label, alpha=0.7)
    
    # Handle the last bin (highest values)
    mask = [count >= count_bins[-1] for count in total_counts]
    x_bin = [year for year, m in zip(years_int, mask) if m]
    y_bin = [kl for kl, m in zip(kl_values, mask) if m]
    if x_bin:
        plt.scatter(x_bin, y_bin, s=10, c=colors[-1], label=bin_labels[-1], alpha=0.7)
    
    # Set nice x-axis formatting
    min_year = min(years_int)
    max_year = max(years_int)
    
    # Set x-axis ticks at regular intervals (every 200 years for range 1000-3000)
    tick_interval = 200
    x_ticks = range(min_year - (min_year % tick_interval), max_year + tick_interval, tick_interval)
    plt.xticks(x_ticks, rotation=45)
    
    # Set x-axis limits with some padding
    plt.xlim(min_year - 50, max_year + 50)
    
    plt.xlabel('Year')
    plt.ylabel('KL Divergence (training || model)')
    plt.title(f'Per-Year KL Divergence\nYears 1000-3000 | {title_suffix}')
    plt.grid(True, alpha=0.3)
    plt.legend(title='Total word count in training data', loc='upper right')
    plt.tight_layout()
    
    # Create filename based on data source
    filename = f"kl_divergence_{data_source}.png"
    plt.savefig(filename)
    print(f"Plot saved as {filename}")

def get_relative_probabilities(olmo_predictions, olmo_training_data, data_source="in_year_there_word_counts"):
    """Get relative probabilities from predictions and training data.
    
    Args:
        olmo_predictions: Model predictions dictionary
        olmo_training_data: Training data dictionary  
        data_source: Either "in_year_there_word_counts" or "co_occurrence_frequency_word_boundaries"
    """
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
    relative_counts = olmo_training_data[data_source]
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
    print(f"\nUsing data source: {data_source}")
    print("\nModel predictions:")
    print("1001:", json.dumps(year_to_relative_predictions["1001"], indent=4))
    print("1980:", json.dumps(year_to_relative_predictions["1980"], indent=4))
    print("2030:", json.dumps(year_to_relative_predictions["2030"], indent=4))

    print("\nTraining data distributions:")
    print("1001:", json.dumps(year_to_relative_counts["1001"], indent=4))
    print("1980:", json.dumps(year_to_relative_counts["1980"], indent=4))
    print("2030:", json.dumps(year_to_relative_counts["2030"], indent=4))

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

    return year_to_relative_predictions, year_to_relative_counts

if __name__ == "__main__":
    # Configuration - modify these as needed
    # DATA_SOURCE = "in_year_there_word_counts"  
    DATA_SOURCE = "co_occurrence_frequency_word_boundaries"
    
    # Custom bins for different data sources (optional - will use defaults if None)
    CUSTOM_COUNT_BINS = None  # e.g., [0, 10, 50, 100, 500] for co-occurrence
    
    # python kl_divergence.py
    # load data
    with open(OLMO_PREDICTIONS_PATH, "r") as f:
        olmo_predictions = json.load(f)

    with open(OLMO_TRAINING_DATA_FILE, "r") as f:
        olmo_training_data = json.load(f)

    year_to_relative_predictions, year_to_relative_counts = get_relative_probabilities(
        olmo_predictions, olmo_training_data, data_source=DATA_SOURCE
    )
    
    # Calculate KL divergences
    kl_values = []
    for year in sorted(year_to_relative_predictions.keys()):
        assert year in year_to_relative_counts, f"Year {year} not found in training data"
        # TODO only include years that have any counts in training data

        training_dist = year_to_relative_counts[year]
        model_dist = year_to_relative_predictions[year]
        kl = kl_divergence_from_dicts(training_dist, model_dist)
        kl_values.append(kl)
    
    print(f"\nAverage KL divergence (training || model): {np.mean(kl_values):.4f}")
    print(f"Number of years: {len(kl_values)}")

    plot_kl_divergence(
        sorted(year_to_relative_predictions.keys()), 
        kl_values, 
        olmo_training_data,
        data_source=DATA_SOURCE,
        count_bins=CUSTOM_COUNT_BINS
    )