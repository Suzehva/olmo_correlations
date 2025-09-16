import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

PYTHIA_PREDICTIONS_FILE = "../olmo_predictions/model_predictions__In__year__there/EleutherAI_pythia-1.4b-deduped_rev_step{cp}/pythia-1.4b-deduped_rev_step{cp}_predictions.json"
OLMO_PREDICTIONS_FILE = "../olmo_predictions/output_checkpoints_olmo/checkpoint_{cp}.json" # this is the 2nd try model

OLMO_TRAINING_DATA_STRING_MATCHING = "../olmo_training_data/1000-3000__was.were.is.are.will__allenai_OLMo-2-0425-1B/aggregated/steps0-{cp}/analytics/1000-3000__was.were.is.are.will__aggregated_results_steps0-{cp}.json"
OLMO_TRAINING_DATA_CO_OCCURRENCE = "../olmo_training_data/1000-3000__was.were.is.are.will__allenai_OLMo-2-0425-1B/aggregated/steps0-{cp}/extra_strict_analytics/1000-3000__was.were.is.are.will__extra_strict_aggregated_results_steps0-{cp}.json"
PYTHIA_TRAINING_DATA = "../olmo_training_data/1000-3000__was.were.is.are.will__EleutherAI_pythia-1.4b-deduped/aggregated/steps0-{cp}/analytics/1000-3000__was.were.is.are.will__aggregated_results_steps0-{cp}.json"

TOTAL_YEARS = (1000, 3000) # goes until 2999
TENSE_MAPPING = {
    "was": "past",
    "were": "past",
    "is": "presfut",
    "are": "presfut",
    "will": "presfut",
}
TENSES = list(set(TENSE_MAPPING.values()))
TENSE_ORDER = ["past", "presfut"]
TENSE_COLORS = {"past": "orange", "future": "green", "present": "#4b0082", "presfut": "#4b0082"}

# Display names for data types used in plotting
DISPLAY_NAMES = {
    "in_year_tense_sentence_counts": "Co-occurrence model",
    "in_year_there_word_counts": "Exact string match model",
    "Laplace-smoothed n-gram": "N-gram model",
    "Next-token predictions": "Next-token predictions",
}

# Model display names for plot titles
MODEL_DISPLAY_NAMES = {
    "pythia": "Pythia-1.4B-deduped",
    "olmo": "OLMo2-1B",
}

# Training-derived data types that should show corpus names instead of model names
TRAINING_DERIVED_DISPLAY_TYPES = {"Co-occurrence model", "Exact string match model", "N-gram model"}
MODEL_CORPUS_OVERRIDES = {
    "pythia": "The Pile",
    "olmo": "OLMo-mix-1124",
}

def get_model_display_name(model_key, data_type):
    """Resolve the display name for a model, with overrides for training-derived plots."""
    display_data_type = DISPLAY_NAMES.get(data_type, data_type)
    if display_data_type in TRAINING_DERIVED_DISPLAY_TYPES:
        return MODEL_CORPUS_OVERRIDES.get(model_key, model_key)
    return MODEL_DISPLAY_NAMES.get(model_key, model_key)

OLMO_CUTOFF = 2024
OLMO_CHECKPOINTS = list(range(250, 10001, 250))
PYTHIA_CUTOFF = 2020
PYTHIA_CHECKPOINTS = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 15000, 20000, 40000, 60000, 80000, 90000, 100000, 110000, 120000, 130000, 143000]


class AnalyzerClass:
    def __init__(self):
        # these all stored per tense (past/present/future), not per verb
        self.olmo_training_data = {}
        self.olmo_predictions = {} 

        self.pythia_training_data = {}
        self.pythia_predictions = {} 

        # Laplace-smoothed n-gram distributions derived from in_year_there_word_counts
        self.olmo_relative_ngram = {}
        self.pythia_relative_ngram = {}
        self.laplace_alpha = 1.0

        # this is per future/present/past 
        self.olmo_relative_gold_distribution = {} # has entries 0 or 1 for past and future
        self.pythia_relative_gold_distribution = {} # see olmo


        print("loading olmo in_year_there_word_counts ") # always has past or presfut entries
        self.olmo_training_data["in_year_there_word_counts"] = self.load_training_data(OLMO_TRAINING_DATA_STRING_MATCHING, "in_year_there_word_counts", 250)
        print("loading olmo in_year_tense_sentence_counts") # always has past or presfut entries
        self.olmo_training_data["in_year_tense_sentence_counts"] = self.load_training_data(OLMO_TRAINING_DATA_CO_OCCURRENCE, "in_year_tense_sentence_counts", 250)
        print("loading pythia in_year_there_word_counts") # always has past or presfut entries
        self.pythia_training_data["in_year_there_word_counts"] = self.load_training_data(PYTHIA_TRAINING_DATA, "in_year_there_word_counts", 1000)
        print("loading pythia in_year_tense_sentence_counts") # always has past or presfut entries
        self.pythia_training_data["in_year_tense_sentence_counts"] = self.load_training_data(PYTHIA_TRAINING_DATA, "in_year_tense_sentence_counts", 1000)
        print("loading olmo predictions") 
        self.olmo_predictions = self.load_model_predictions(OLMO_PREDICTIONS_FILE, OLMO_CHECKPOINTS)
        print("loading pythia predictions")
        self.pythia_predictions = self.load_model_predictions(PYTHIA_PREDICTIONS_FILE, PYTHIA_CHECKPOINTS)
        print("populating gold data")
        self.populate_gold_data()

        print("loading olmo relative ngram") # always has past or presfut entries, add to 1.0; missing cutoff year
        self.olmo_relative_ngram = self._compute_laplace_smoothed_ngram(self.olmo_training_data["in_year_there_word_counts"])
        print("loading pythia relative ngram") # always has past or presfut entries, add to 1.0; missing cutoff year
        self.pythia_relative_ngram = self._compute_laplace_smoothed_ngram(self.pythia_training_data["in_year_there_word_counts"])
        print("finished loading all data")




    def load_training_data(self, file_template, data_source, step_size):
            results_absolute = {}

            for cp_base in range(step_size, 10001, step_size):
                cp_num = cp_base
                if cp_num % 100 == 50: # to deal with training data inconsistency for olmo
                    cp_num += 10
                filepath = file_template.format(cp=cp_num)
                if not os.path.exists(filepath):
                    raise FileNotFoundError(f"could not find {filepath}")
                with open(filepath, "r") as f:
                    data = json.load(f)

                year_to_counts_absolute = {}
                
                
                # Then process the actual data
                for year in range(TOTAL_YEARS[0], TOTAL_YEARS[1]):
                    year_to_counts_absolute[str(year)] = {
                        "past": 0,
                        "presfut": 0,
                    }
                for year, counts_per_verb in data[data_source].items():
                    # Group by tense categories (raw absolute counts)
                    tense_counts_absolute = year_to_counts_absolute[year]
                    for verb, tense in TENSE_MAPPING.items():
                        if verb in counts_per_verb: # in_year_there_word_counts only has entries if there are occurrences so we need this statement
                            year_to_counts_absolute[year][tense] += counts_per_verb[verb]
                    year_to_counts_absolute[year] = tense_counts_absolute

                results_absolute[cp_num] = year_to_counts_absolute
            return results_absolute

    def load_model_predictions(self, file_template, checkpoints):
        results_absolute = {}
        for cp in checkpoints:
            filepath = file_template.format(cp=cp)
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"could not find {filepath}")
            with open(filepath, "r") as f:
                data = json.load(f)
            print(f"Loaded {filepath}")
            
            year_to_counts_absolute = {}
            
            
            # Then process the actual data
            if file_template == OLMO_PREDICTIONS_FILE:
                for year, verb_count_dict in data["data"].items():
                    year_to_counts_absolute[year] = {
                        "past": 0,
                        "presfut": 0,
                    }
                    for verb, tense in TENSE_MAPPING.items():
                        year_to_counts_absolute[year][tense] += verb_count_dict[verb]

            elif file_template == PYTHIA_PREDICTIONS_FILE:
                for year in data.keys():
                    if year == "metadata":
                        continue
                    verb_count_dict = data[year]["absolute"]
                    # Remove leading spaces from verb keys for Pythia predictions
                    verb_count_dict_cleaned = {k.strip(): v for k, v in verb_count_dict.items()}
                    year_to_counts_absolute[year] = {
                        "past": 0,
                        "presfut": 0,
                    }
                    for verb, tense in TENSE_MAPPING.items():
                        year_to_counts_absolute[year][tense] += verb_count_dict_cleaned[verb]

            results_absolute[cp] = year_to_counts_absolute
        return results_absolute

    def populate_gold_data(self):
        def populate_gold_distribution(cutoff):
            gold_per_year = {}
            for year in range(TOTAL_YEARS[0], TOTAL_YEARS[1]):
                year_dist = {}
                if year < cutoff:
                    year_dist = {"past": 1.0, "future": 0.0}
                if year > cutoff:
                    year_dist = {"past": 0.0, "future": 1.0}
                if year == cutoff:
                    continue
                gold_per_year[str(year)] = year_dist
            return gold_per_year


        for cp in OLMO_CHECKPOINTS:
            self.olmo_relative_gold_distribution[cp] = populate_gold_distribution(OLMO_CUTOFF)
        for cp in PYTHIA_CHECKPOINTS:
            self.pythia_relative_gold_distribution[cp] = populate_gold_distribution(PYTHIA_CUTOFF)


    def _compute_laplace_smoothed_ngram(self, absolute_counts_per_cp):
        smoothed_per_cp = {}
        for cp, year_map in absolute_counts_per_cp.items():
            year_to_probs = {}
            for year_str, counts_per_tense in year_map.items():
                year_total = sum(counts_per_tense.values())
                denom = year_total + self.laplace_alpha * len(TENSES)
                new_counts_per_tense = {}
                for tense, counts in counts_per_tense.items():
                    new_counts_per_tense[tense] = (counts + self.laplace_alpha) / denom
                year_to_probs[year_str] = new_counts_per_tense
            smoothed_per_cp[cp] = year_to_probs

        return smoothed_per_cp


    def save_all_data_to_file(self, filepath="analyzer_data.json"):
        """
        Save all analyzer data to a single JSON file.
        
        Args:
            filepath: Path to save the data file
        """
        all_data = {
            'metadata': {
                'tense_mapping': TENSE_MAPPING,
                'tense_order': TENSE_ORDER,
                'tense_colors': TENSE_COLORS,
                'total_years': TOTAL_YEARS,
                'laplace_alpha': self.laplace_alpha,
                'export_timestamp': datetime.now().isoformat(),
            },
            'olmo_training_data': self.olmo_training_data,
            'olmo_predictions': self.olmo_predictions,
            'olmo_relative_ngram': self.olmo_relative_ngram,
            'olmo_relative_gold_distribution': self.olmo_relative_gold_distribution,
            'pythia_training_data': self.pythia_training_data,
            'pythia_predictions': self.pythia_predictions,
            'pythia_relative_ngram': self.pythia_relative_ngram,
            'pythia_relative_gold_distribution': self.pythia_relative_gold_distribution,
        }
        
        with open(filepath, 'w') as f:
            json.dump(all_data, f, indent=2)
        
        print(f"All data saved to: {filepath}")
        return filepath

    # ----------------------------- PLOTTING -----------------------------
    
    def _get_folder_name(self, model, checkpoint, data_type):
        """Generate folder name for saving plots."""
        return f"{model}_checkpoint{checkpoint}"
    
    def _make_relative_distributions(self, dist_dict):
        """Convert absolute counts to relative distributions (probabilities that sum to 1)."""
        relative_dict = {}
        for cp, year_data in dist_dict.items():
            relative_dict[cp] = {}
            for year, tense_counts in year_data.items():
                total = sum(tense_counts.values())
                if total > 0:
                    relative_dict[cp][year] = {tense: count/total for tense, count in tense_counts.items()}
                else:
                    # If no data, keep as zeros (don't artificially split)
                    relative_dict[cp][year] = {tense: 0.0 for tense in TENSE_ORDER}
        return relative_dict

    def bar_plot(self, dist_dict, model, data_type, checkpoint, year_start, year_end, make_relative=True):
        """Plot stacked bars for a distribution at a specific checkpoint.
        Adapted for the new tense structure using 'past' and 'presfut'.
        
        Args:
            dist_dict: Dictionary containing distribution data
            model: Model name 
            data_type: Type of data being plotted
            checkpoint: Checkpoint number
            year_start: Start year (inclusive)
            year_end: End year (inclusive)
            make_relative: If True, convert counts to probabilities that sum to 1
        """
        if checkpoint not in dist_dict:
            raise ValueError(f"Checkpoint {checkpoint} not found in {dist_dict}")
        
        # Convert to relative distributions if requested
        if make_relative:
            dist_dict = self._make_relative_distributions(dist_dict)
        
        output_dir = self._get_folder_name(model, checkpoint, data_type)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        cp_data = dist_dict[checkpoint]
        all_years = sorted(cp_data.keys())
        
        # Filter years based on year_start and year_end
        filtered_years = []
        for year_str in all_years:
            year = int(year_str)
            if year < year_start or year > year_end:
                continue
            filtered_years.append(year_str)
        years = filtered_years
        
        # Plot
        fig, ax = plt.subplots(figsize=(5.2, 3))
        bottom = np.zeros(len(years))
        
        for tense in TENSE_ORDER:
            vals = np.array([cp_data[y].get(tense, 0) for y in years])
            ax.bar(range(len(years)), vals, bottom=bottom, 
                  label=tense.title(), color=TENSE_COLORS[tense], width=1.0)
            bottom += vals
        
        # Format - using same logic as kl_divergence_checkpoints.py but simplified
        display_data_type = DISPLAY_NAMES.get(data_type, data_type)
        model_display = get_model_display_name(model, data_type)
        
        title = f"{model_display} — {display_data_type}"
        safe_data_type = data_type.replace(' ', '_')
        filename = f"{model}_checkpoint{checkpoint}_{safe_data_type}_{year_start}-{year_end}"
        
        ax.set_title(title)
        # Use 10-year intervals for x-axis ticks; fallback if none present
        tick_indices = [i for i, y in enumerate(years) if int(y) % 10 == 0]
        if tick_indices:
            ax.set_xticks(tick_indices)
            ax.set_xticklabels([years[i] for i in tick_indices], rotation=45)
        else:
            step = max(1, len(years)//20)
            ax.set_xticks(range(0, len(years), step))
            ax.set_xticklabels(years[::step], rotation=45)
        ax.set_ylabel("Probability")
        ax.legend(loc='lower left')
        
        # Remove whitespace around plot
        ax.set_xlim(-0.5, len(years) - 0.5)  # Remove left/right whitespace
        ax.set_ylim(0, 1)  # Remove top whitespace, set to max probability
        ax.margins(0)  # Remove all margins
        
        # Save
        save_path = Path(output_dir) / f"{filename}.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=600)
        plt.close()
        print(f"Saved: {save_path}")
        


if __name__ == "__main__":
    # python get_paper_results.py
    cp = 10000
    start_year = 1950
    years_end = 2050

    analyzer = AnalyzerClass()
    filepath = analyzer.save_all_data_to_file()
    print(f"Data export completed. File saved: {filepath}")

    # plot_training_data
    analyzer.bar_plot(analyzer.olmo_training_data["in_year_tense_sentence_counts"], "olmo", "in_year_tense_sentence_counts", cp, start_year, years_end)
    analyzer.bar_plot(analyzer.olmo_training_data["in_year_there_word_counts"], "olmo", "in_year_there_word_counts", cp, start_year, years_end)
    analyzer.bar_plot(analyzer.pythia_training_data["in_year_tense_sentence_counts"], "pythia", "in_year_tense_sentence_counts", cp, start_year, years_end)
    analyzer.bar_plot(analyzer.pythia_training_data["in_year_there_word_counts"], "pythia", "in_year_there_word_counts", cp, start_year, years_end)
    analyzer.bar_plot(analyzer.olmo_relative_ngram, "olmo", "Laplace-smoothed n-gram", cp, start_year, years_end)
    analyzer.bar_plot(analyzer.pythia_relative_ngram, "pythia", "Laplace-smoothed n-gram", cp, start_year, years_end)

    # plot_model_predictions
    analyzer.bar_plot(analyzer.olmo_predictions, "olmo", "Next-token predictions", cp, start_year, years_end)
    analyzer.bar_plot(analyzer.pythia_predictions, "pythia", "Next-token predictions", cp, start_year, years_end)


    

