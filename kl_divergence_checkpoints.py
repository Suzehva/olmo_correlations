
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

TENSE_MAPPING = {
    "was": "past",
    "were": "past",
    "is": "present",
    "are": "present",
    "will": "future",
}
TENSE_ORDER = ["past", "present", "future"]  # strict plotting order
TENSE_COLORS = {"past": "orange", "future": "green", "present": "#4b0082"}
CROSS_ENTROPY_COLORS = ["grey", "blue", "red"]
TOTAL_YEARS = (1000, 3000) # goes until 2999

# Standardized display names for data types used throughout plotting
DISPLAY_NAMES = {
    "in_year_tense_sentence_counts": "Co-occurrence model",
    "in_year_tense_counts": "Co-occurrence model",
    "in_year_there_word_counts": "Exact string match model",
    "Laplace-smoothed n-gram": "N-gram model",
    "Next-token predictions": "Next-token predictions",
    "Next-token Predictions": "Next-token predictions",
}

# Nice names for model identifiers used in plot titles
MODEL_DISPLAY_NAMES = {
    "pythia": "Pythia-1.4b-deduped",
    "olmo": "OLMo2-1B",
}

OLMO_CUTOFF = 2024
OLMO_CHECKPOINTS = list(range(250, 10001, 250))
PYTHIA_CUTOFF = 2020
PYTHIA_CHECKPOINTS = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 15000, 20000, 40000, 60000, 80000, 90000, 100000, 110000, 120000, 130000, 143000]

# Single-experiment constants
EXPERIMENT_KEY = 'experiment1_past_vs_present_future'
EXPERIMENT_TITLE = 'Binary Classification: Past vs (Present + Future)'
EXPERIMENT_SHORT_NAME = EXPERIMENT_KEY.replace('experiment1_', '')

class AnalyzerClass:
    def __init__(self, ngram_alpha=1.0):
        # these all stored per tense (past/present/future), not per verb
        self.olmo_training_data = {}
        # in_year_there_word_counts: saves an entry like past/present/future, only entries if there are occurrences
        # in_year_tense_sentence_counts: always has all entries, for past/present/future. If no occurences it's set to 0

        self.olmo_relative_training_data = {}
        # in_year_there_word_counts: has dictionaries for apst/present/future even if no occurrences (then set to 0)
        # in_year_tense_sentence_counts: ^

        self.olmo_predictions = {} # always has some probability for past, present, future
        self.olmo_relative_predictions = {} # same but adds to one

        self.pythia_training_data = {} # see olmo
        self.pythia_relative_training_data = {} # see olmo
        self.pythia_predictions = {} # see olmo
        self.pythia_relative_predictions = {} # see olmo

        # Laplace-smoothed n-gram distributions derived from in_year_there_word_counts
        self.olmo_relative_ngram = {}
        self.pythia_relative_ngram = {}

        # Smoothing parameter (add-alpha)
        self.ngram_alpha = ngram_alpha

        # this is per future/present/past 
        self.olmo_gold_distribution = {} # has entries 0 or 1 for past and future
        self.pythia_gold_distribution = {} # see olmo

        print("loading all data")
        self.load_training_data()
        self.load_model_predictions()
        self.populate_gold_data()
        print("finished loading all data")

    def load_other_prompts(self, prompt_names):
        prompt_to_model_to_predictions = {}
        for prompt_name in prompt_names:
            prompt_to_model_to_predictions[prompt_name] = {}
            for model_name, just_model_name in [("allenai_OLMo-2-0425-1B", "OLMo-2-0425-1B"), ("EleutherAI_pythia-1.4b-deduped", "pythia-1.4b-deduped")]:
                model_prediction_file = f"../olmo_predictions/model_predictions__{prompt_name}/{model_name}/{just_model_name}_all_verbs_predictions.json"
                with open(model_prediction_file, "r") as f:
                    data = json.load(f)
                year_to_counts_absolute = {}
                for year in data.keys():
                    if year == "metadata":
                        continue
                    verb_count_dict = data[year]["absolute"]
                    # Remove leading spaces from verb keys for predictions
                    verb_count_dict_cleaned = {k.strip(): v for k, v in verb_count_dict.items()}
                    
                    tense_counts_absolute = {}
                    for verb, tense in TENSE_MAPPING.items():
                        if verb in verb_count_dict_cleaned:
                            tense_counts_absolute.setdefault(tense, 0)
                            tense_counts_absolute[tense] += verb_count_dict_cleaned[verb]

                    year_to_counts_absolute[year]= tense_counts_absolute
                prompt_to_model_to_predictions[prompt_name][model_name] = {"final": year_to_counts_absolute}
        return prompt_to_model_to_predictions
            
        
    
    def load_other_model_predictions(self, model_names):
        model_name_to_predictions = {}
        for model_name in model_names:
            just_model_name = model_name.split("_")[-1]
            model_prediction_file = f"../olmo_predictions/model_predictions__In__year__there/{model_name}/{just_model_name}_predictions.json"
            with open(model_prediction_file, "r") as f:
                data = json.load(f)

            year_to_counts_absolute = {}
            for year in data.keys():
                if year == "metadata":
                    continue
                verb_count_dict = data[year]["absolute"]
                # Remove leading spaces from verb keys for Pythia predictions
                verb_count_dict_cleaned = {k.strip(): v for k, v in verb_count_dict.items()}
                
                tense_counts_absolute = {}
                for verb, tense in TENSE_MAPPING.items():
                    if verb in verb_count_dict_cleaned:
                        tense_counts_absolute.setdefault(tense, 0)
                        tense_counts_absolute[tense] += verb_count_dict_cleaned[verb]

                year_to_counts_absolute[year]= tense_counts_absolute
            model_name_to_predictions[model_name] = {"final": year_to_counts_absolute}
        return model_name_to_predictions
            



    def _normalize_tense_distribution(self, tense_counts):
        all_tenses = list(set(TENSE_MAPPING.values()))
        complete_counts = {tense: tense_counts.get(tense, 0.0) for tense in all_tenses}
        
        total = sum(complete_counts.values())
        if total == 0:
            return {tense: 0.0 for tense in all_tenses}
        
        normalized = {tense: count / total for tense, count in complete_counts.items()}
        
        return normalized



    def _compute_laplace_smoothed_ngram(self, absolute_counts_per_cp):
        """
        Build Laplace-smoothed P(tense | year) per checkpoint from absolute counts.

        Args:
            absolute_counts_per_cp: dict[checkpoint -> dict[year(str) -> dict[tense -> count]]]
            alpha: smoothing parameter (defaults to self.ngram_alpha)

        Returns:
            dict[checkpoint -> dict[year(str) -> dict[tense -> probability]]]
        """

        tenses = list(set(TENSE_MAPPING.values()))
        vocab_size = len(tenses)

        smoothed_per_cp = {}
        for cp, year_map in absolute_counts_per_cp.items():
            year_to_probs = {}
            for year_str, counts in year_map.items():
                # counts may be empty dict; treat missing as zero
                year_total = sum(counts.values()) if counts else 0
                denom = year_total + self.ngram_alpha * vocab_size
                probs = {}
                for tense in tenses:
                    tense_count = counts.get(tense, 0)
                    probs[tense] = (tense_count + self.ngram_alpha) / denom
                year_to_probs[year_str] = probs
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
                'ngram_alpha': self.ngram_alpha,
                'export_timestamp': datetime.now().isoformat(),
            },
            'olmo_training_data': self.olmo_training_data,
            'olmo_relative_training_data': self.olmo_relative_training_data,
            'olmo_predictions': self.olmo_predictions,
            'olmo_relative_predictions': self.olmo_relative_predictions,
            'olmo_relative_ngram': self.olmo_relative_ngram,
            'olmo_gold_distribution': self.olmo_gold_distribution,
            'pythia_training_data': self.pythia_training_data,
            'pythia_relative_training_data': self.pythia_relative_training_data,
            'pythia_predictions': self.pythia_predictions,
            'pythia_relative_predictions': self.pythia_relative_predictions,
            'pythia_relative_ngram': self.pythia_relative_ngram,
            'pythia_gold_distribution': self.pythia_gold_distribution,
        }
        
        with open(filepath, 'w') as f:
            json.dump(all_data, f, indent=2)
        
        print(f"All data saved to: {filepath}")
        return filepath

    def compute_cross_entropy_over_range(self, dist_dict, which_model, checkpoint, year_start, year_end, min_samples=0):
        def compute_losses(pred_dist, gold_dist): 
            assert abs(sum(pred_dist) - 1) < 1e-6, f"pred_dist should sum to approximately 1, got {sum(pred_dist)} for model {which_model} at checkpoint {checkpoint} and year {year_str}"
            assert abs(sum(gold_dist) - 1) < 1e-6
            assert len(pred_dist) == 2
            
            # Compute cross-entropy: -sum(target * log(pred))
            # Add small epsilon to avoid log(0)
            epsilon = 1e-12
            ce_loss = - gold_dist[0] * np.log(pred_dist[0] + epsilon) - gold_dist[1] * np.log(pred_dist[1] + epsilon)

            # Fix floating-point precision errors: cross-entropy should never be negative
            if ce_loss < 0:
                if abs(ce_loss) < 1e-10:  # If it's just floating-point error
                    ce_loss = 0.0  # Perfect predictions get exactly 0 cross-entropy
                else:
                    raise ValueError(f"Unexpected negative cross-entropy {ce_loss} for model {which_model} at checkpoint {checkpoint} and year {year_str}")
            
            # Check for invalid values
            if not np.isfinite(ce_loss):
                raise ValueError(f"ce_loss is not finite for model {which_model} at checkpoint {checkpoint} and year {year_str}")
                
            return ce_loss

        # Get data for the specified checkpoint
        dist_cp = dist_dict[checkpoint]
        if which_model == "olmo":
            gold_cp = self.olmo_gold_distribution[checkpoint]
            # Get raw counts to check minimum sample requirement (only for training data)
            if dist_dict == self.olmo_relative_training_data["in_year_there_word_counts"]:
                raw_counts_cp = self.olmo_training_data["in_year_there_word_counts"][checkpoint]
            elif dist_dict == self.olmo_relative_training_data["in_year_tense_sentence_counts"]:
                raw_counts_cp = self.olmo_training_data["in_year_tense_sentence_counts"][checkpoint]
            elif dist_dict == self.olmo_relative_predictions:
                raw_counts_cp = None  # Predictions don't have raw sample counts
            elif dist_dict == self.olmo_relative_ngram:
                raw_counts_cp = self.olmo_training_data["in_year_there_word_counts"][checkpoint]
            else:
                raise ValueError(f"Unknown OLMO distribution: {dist_dict}")
        elif which_model == "pythia":
            gold_cp = self.pythia_gold_distribution[checkpoint]
            # Get raw counts to check minimum sample requirement (only for training data)
            if dist_dict == self.pythia_relative_training_data["in_year_there_word_counts"]:
                raw_counts_cp = self.pythia_training_data["in_year_there_word_counts"][checkpoint]
            elif dist_dict == self.pythia_relative_training_data["in_year_tense_sentence_counts"]:
                raw_counts_cp = self.pythia_training_data["in_year_tense_sentence_counts"][checkpoint]
            elif dist_dict == self.pythia_relative_predictions:
                raw_counts_cp = None  # Predictions don't have raw sample counts
            elif dist_dict == self.pythia_relative_ngram:
                raw_counts_cp = self.pythia_training_data["in_year_there_word_counts"][checkpoint]
            else:
                raise ValueError(f"Unknown Pythia distribution: {dist_dict}")
        else:
            raise ValueError(f"Invalid model: {which_model}")
        
        # Separate storage for each experiment
        experiment1_losses = {}  # present + future combined
        years_used = []
        
        for year in range(year_start, year_end + 1):
            year_str = str(year)
            # Exclude cutoff year from CE computations and plotting
            if (which_model == "olmo" and year == OLMO_CUTOFF) or (which_model == "pythia" and year == PYTHIA_CUTOFF):
                continue
            dist_data = dist_cp[year_str]
            
            # Skip years with no data (distributions that sum to zero)
            if sum(dist_data.values()) == 0:
                continue
                
            # Check minimum sample requirement if raw counts are available
            if raw_counts_cp is not None:
                raw_counts = raw_counts_cp[year_str]
                total_samples = sum(raw_counts.values())
                if total_samples < min_samples:
                    print(f"Skipping year {year_str} for {which_model}: only {total_samples} samples (min required: {min_samples})")
                    continue
            assert abs(sum(dist_data.values()) - 1) < 1e-6, f"dist_data should sum to 1, got {sum(dist_data.values())}"
            gold_data = gold_cp[year_str]
            assert abs(sum(gold_data.values()) - 1) < 1e-6, f"gold_data should sum to 1, got {sum(gold_data.values())}"
            gold_dist = [gold_data["past"], gold_data["future"]]
            
            # experiment 1: combine present and future tense
            pred_dist = [dist_data["past"], dist_data["present"] + dist_data["future"]]
            
            # Skip if prediction distribution sums to zero
            if sum(pred_dist) == 0:
                continue
                
            ce_loss = compute_losses(pred_dist, gold_dist)
            if ce_loss is not None:
                experiment1_losses[year_str] = ce_loss
                years_used.append(year_str)


        # Compute average losses for each experiment
        avg_loss_exp1 = sum(experiment1_losses.values()) / len(experiment1_losses)
  
        return {
            'experiment1_past_vs_present_future': {
                'per_year_losses': experiment1_losses,
                'average_loss': avg_loss_exp1,
                'years_used': years_used,
                'years_skipped': set(range(year_start, year_end+1)) - set(years_used),
                'description': 'Binary classification: past vs (present + future)'
            },
            
        }

    def _get_folder_name(self, model, checkpoint, data_type):
        """
        Generate standardized folder name for outputs.
        
        Args:
            model: "olmo" or "pythia"
            checkpoint: int, checkpoint number
            data_type: str, type of data being plotted
            
        Returns:
            str: standardized folder name
        """
        return f"{model}_checkpoint{checkpoint}"

    def load_training_data(self):
        mappings = [
            (OLMO_TRAINING_DATA_STRING_MATCHING, "in_year_there_word_counts", 250),
            (OLMO_TRAINING_DATA_CO_OCCURRENCE, "in_year_tense_sentence_counts", 250),
            (PYTHIA_TRAINING_DATA, "in_year_there_word_counts", 1000),
            (PYTHIA_TRAINING_DATA, "in_year_tense_sentence_counts", 1000),
        ]
        for file_template, data_source, step_size in mappings:
            results_absolute = {}
            results_relative = {}

            for cp_base in range(step_size, 10001, step_size):
                cp_num = cp_base
                if cp_num % 100 == 50: # to deal with training data inconsistency
                    cp_num += 10
                filepath = file_template.format(cp=cp_num)
                if not os.path.exists(filepath):
                    raise FileNotFoundError(f"could not find {filepath}")
                with open(filepath, "r") as f:
                    data = json.load(f)

                year_to_counts_absolute = {}
                year_to_counts_relative = {}
                
                # First, ensure all years in TOTAL_YEARS range are present with empty dicts
                for year in range(TOTAL_YEARS[0], TOTAL_YEARS[1]):
                    year_str = str(year)
                    year_to_counts_absolute[year_str] = {}
                    year_to_counts_relative[year_str] = {tense: 0.0 for tense in set(TENSE_MAPPING.values())}
                
                # Then process the actual data
                for year, counts in data[data_source].items():
                    # Group by tense categories
                    tense_counts_absolute = {}
                    for verb, tense in TENSE_MAPPING.items():
                        if verb in counts:
                            tense_counts_absolute.setdefault(tense, 0)
                            tense_counts_absolute[tense] += counts[verb]
                    
                    # Use the normalization helper for accurate normalization
                    tense_counts_relative = self._normalize_tense_distribution(tense_counts_absolute)
                    
                    year_to_counts_absolute[year] = tense_counts_absolute
                    year_to_counts_relative[year] = tense_counts_relative

                results_relative[cp_num] = year_to_counts_relative
                results_absolute[cp_num] = year_to_counts_absolute

            # Build Laplace-smoothed n-gram distribution from in_year_there_word_counts
            if data_source == "in_year_there_word_counts":
                smoothed = self._compute_laplace_smoothed_ngram(results_absolute)
                if file_template == OLMO_TRAINING_DATA_STRING_MATCHING:
                    self.olmo_relative_ngram = smoothed
                else:
                    # PYTHIA_TRAINING_DATA
                    self.pythia_relative_ngram = smoothed

            if file_template == OLMO_TRAINING_DATA_STRING_MATCHING or file_template == OLMO_TRAINING_DATA_CO_OCCURRENCE:
                self.olmo_training_data[data_source] = results_absolute
                self.olmo_relative_training_data[data_source] = results_relative
            else:
                self.pythia_training_data[data_source] = results_absolute
                self.pythia_relative_training_data[data_source] = results_relative


    def load_model_predictions(self):
        def divide_into_tenses(year, verb_count_dict):
            tense_counts_absolute = {}
            for verb, tense in TENSE_MAPPING.items():
                if verb in verb_count_dict:
                    tense_counts_absolute.setdefault(tense, 0)
                    tense_counts_absolute[tense] += verb_count_dict[verb]
            
            # Use the normalization helper for accurate normalization
            tense_counts_relative = self._normalize_tense_distribution(tense_counts_absolute)
        
            return tense_counts_absolute, tense_counts_relative
            

        def load_model_predictions(file_template, checkpoints):
            results_absolute = {}
            results_relative = {}
            for cp in checkpoints:
                filepath = file_template.format(cp=cp)
                if not os.path.exists(filepath):
                    raise FileNotFoundError(f"could not find {filepath}")
                with open(filepath, "r") as f:
                    data = json.load(f)
                print(f"Loaded {filepath}")
                
                year_to_counts_absolute = {}
                year_to_counts_relative = {}
                
                # First, ensure all years in TOTAL_YEARS range are present with empty dicts
                for year in range(TOTAL_YEARS[0], TOTAL_YEARS[1]):
                    year_str = str(year)
                    year_to_counts_absolute[year_str] = {}
                    year_to_counts_relative[year_str] = {tense: 0.0 for tense in set(TENSE_MAPPING.values())}
                
                # Then process the actual data
                if file_template == OLMO_PREDICTIONS_FILE:
                    for year, verb_count_dict in data["data"].items():
                        year_to_counts_absolute[year], year_to_counts_relative[year] = divide_into_tenses(year, verb_count_dict)

                elif file_template == PYTHIA_PREDICTIONS_FILE:
                    for year in data.keys():
                        if year == "metadata":
                            continue
                        verb_count_dict = data[year]["absolute"]
                        # Remove leading spaces from verb keys for Pythia predictions
                        verb_count_dict_cleaned = {k.strip(): v for k, v in verb_count_dict.items()}
                        year_to_counts_absolute[year], year_to_counts_relative[year] = divide_into_tenses(year, verb_count_dict_cleaned)

                results_absolute[cp] = year_to_counts_absolute
                results_relative[cp] = year_to_counts_relative
            return results_absolute, results_relative

        self.olmo_predictions, self.olmo_relative_predictions = load_model_predictions(OLMO_PREDICTIONS_FILE, OLMO_CHECKPOINTS)
        self.pythia_predictions, self.pythia_relative_predictions = load_model_predictions(PYTHIA_PREDICTIONS_FILE, PYTHIA_CHECKPOINTS)


    def populate_gold_data(self):
        def populate_gold_distribution(cutoff):
            gold_per_year = {}
            for year in range(TOTAL_YEARS[0], TOTAL_YEARS[1]):
                year_dist = {}
                if year <= cutoff:
                    year_dist = {"past": 1.0, "future": 0.0}
                else:
                    year_dist = {"past": 0.0, "future": 1.0}
                gold_per_year[str(year)] = year_dist
            return gold_per_year


        for cp in OLMO_CHECKPOINTS:
            self.olmo_gold_distribution[cp] = populate_gold_distribution(OLMO_CUTOFF)
        for cp in PYTHIA_CHECKPOINTS:
            self.pythia_gold_distribution[cp] = populate_gold_distribution(PYTHIA_CUTOFF)



    # --- PLOTTING FUNCTIONS ---

    def bar_plot(self, dist_dict, model, data_type, checkpoint, year_start, year_end):
        """Plot stacked bars for a distribution at a specific checkpoint.
        
        Args:
            dist_dict: Dictionary containing distribution data
            model: Model name 
            data_type: Type of data being plotted
            checkpoint: Checkpoint number
            year_start: Start year (inclusive)
            year_end: End year (inclusive)
        """
        if checkpoint not in dist_dict:
            raise ValueError(f"Checkpoint {checkpoint} not found in {dist_dict}")
        
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
            
        tenses = [t for t in TENSE_ORDER if t in set(TENSE_MAPPING.values())]
        
        # Plot
        fig, ax = plt.subplots(figsize=(6, 3))
        bottom = np.zeros(len(years))
        
        for tense in tenses:
            vals = np.array([cp_data[y].get(tense, 0) for y in years])
            ax.bar(range(len(years)), vals, bottom=bottom, 
                  label=tense.title(), color=TENSE_COLORS[tense], width=1.0)
            bottom += vals
        
        # Format
        display_data_type = DISPLAY_NAMES.get(data_type, data_type)
        model_display = MODEL_DISPLAY_NAMES.get(model, str(model))
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

        
    def bar_plots_for_checkpoints(self, dist_dict, model, data_type, checkpoints, rows, cols, year_start, year_end):
        """
        Create a grid of stacked bar plots, one subplot per checkpoint.

        Args:
            dist_dict: Dictionary containing distribution data (keyed by checkpoint -> year -> tense -> prob)
            model: Model name (str)
            data_type: Type of data being plotted (str)
            checkpoints: List of checkpoint integers to plot
            rows: Number of rows in the subplot grid
            cols: Number of columns in the subplot grid
            year_start: Start year (inclusive)
            year_end: End year (inclusive)
            output_filename: Optional filename to save the combined figure
        """
        if len(checkpoints) > rows * cols:
            raise ValueError(f"Too many checkpoints ({len(checkpoints)}) for grid size {rows}×{cols}")

        fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 2.5 * rows))
        if rows == 1 and cols == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        tenses = [t for t in TENSE_ORDER if t in set(TENSE_MAPPING.values())]

        for idx, cp in enumerate(checkpoints):
            ax = axes[idx]

            if cp not in dist_dict:
                ax.text(0.5, 0.5, f"Checkpoint {cp} not found", ha='center', va='center', transform=ax.transAxes, fontsize=10)
                ax.set_title(f"Checkpoint {cp}")
                ax.grid(True, alpha=0.3)
                continue

            cp_data = dist_dict[cp]
            all_years = sorted(cp_data.keys())

            # Filter years based on year_start and year_end
            filtered_years = []
            for year_str in all_years:
                year = int(year_str)
                if year < year_start or year > year_end:
                    continue
                filtered_years.append(year_str)
            years = filtered_years

            bottom = np.zeros(len(years))
            for tense in tenses:
                vals = np.array([cp_data[y].get(tense, 0) for y in years])
                ax.bar(range(len(years)), vals, bottom=bottom,
                       label=tense.title() if idx == 0 else "",
                       color=TENSE_COLORS[tense], width=1.0)
                bottom += vals

            # Format subplot
            display_data_type = DISPLAY_NAMES.get(data_type, data_type)
            model_display = MODEL_DISPLAY_NAMES.get(model, str(model))
            ax.set_title(f"{model_display} — {display_data_type} | Checkpoint {cp}", fontsize=10)
            tick_step = max(1, len(years) // 20)
            ax.set_xticks(range(0, len(years), tick_step))
            ax.set_xticklabels(years[::tick_step], rotation=45, fontsize=9)
            if idx % cols == 0:
                ax.set_ylabel("Probability")
            ax.set_ylim(0, 1)
            ax.set_xlim(-0.5, len(years) - 0.5)
            ax.margins(0)
            ax.grid(True, alpha=0.3)

        # Hide any unused subplots
        for idx in range(len(checkpoints), len(axes)):
            axes[idx].set_visible(False)

        # Add a single legend for the whole figure
        fig.legend([t.title() for t in tenses], loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=len(tenses), fontsize=12)

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.12)

        # Save combined figure
        year_range = f"{year_start}-{year_end}"
        cps_str = f"{len(checkpoints)}cps"
        safe_data_type = DISPLAY_NAMES.get(data_type, data_type).replace(' ', '_')
        output_filename = f"{model}_{safe_data_type}_{cps_str}_{rows}x{cols}_{year_range}.png"

        output_dir = "checkpoint_bar_plots"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        save_path = Path(output_dir) / output_filename
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")


def plot_cross_entropies(ce_results_list, labels_list, model_name, year_start=1950, year_end=2050, output_dir="cross_entropy_plots"):
    """Plot cross-entropy losses as lines for multiple distributions.
    
    Args:
        ce_results_list: List of cross-entropy result dictionaries from compute_cross_entropy_over_range
        labels_list: List of labels for each distribution
        model_name: Model name for the plot title and filename
        year_start: Start year (inclusive)
        year_end: End year (inclusive)
        output_dir: Directory to save the plots
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Single experiment only
    exp_key = EXPERIMENT_KEY
    exp_title = EXPERIMENT_TITLE
    
    fig, ax = plt.subplots(figsize=(9, 5))
    
    all_years_plotted = []
    all_losses_plotted = []
    
    for i, (ce_result, label) in enumerate(zip(ce_results_list, labels_list)):
        exp_data = ce_result[exp_key]
        per_year_losses = exp_data['per_year_losses']
        
        # Filter years and prepare data for plotting
        years = []
        losses = []
        
        for year in exp_data['years_used']:
            year_int = int(year)
            # Filter by year_start and year_end
            if year_start <= year_int <= year_end:
                years.append(year_int)
                loss_value = per_year_losses[year]
                losses.append(loss_value)
    
        if years and losses:
            color = CROSS_ENTROPY_COLORS[i % len(CROSS_ENTROPY_COLORS)]
            
            avg_loss = exp_data['average_loss']
            avg_display = f"{avg_loss:.4f}"
            
            ax.scatter(years, losses, 
                      label=f"{label} (avg: {avg_display}, num_years: {len(years)})", 
                      color=color, s=20, alpha=0.7)  # Increased alpha for visibility
            
            # Collect all plotted data for axis limits
            all_years_plotted.extend(years)
            all_losses_plotted.extend(losses)
        else:
            print(f"  WARNING: No data to plot for {label}")
    
    # Format the plot
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Cross-Entropy Loss', fontsize=12)
    ax.set_title(f"{model_name.upper()} - {exp_title}\nYears {year_start}-{year_end}", fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Use linear scale for y-axis to show zero values
    # ax.set_yscale('log')  # Commented out - using linear scale
    
    # Set axis limits based on actual plotted data
    if all_years_plotted and all_losses_plotted:
        # X-axis: set to requested range with some padding
        ax.set_xlim(year_start - 5, year_end + 5)
        
        # Y-axis: set based on actual data with some padding
        y_min, y_max = min(all_losses_plotted), max(all_losses_plotted)
        # For linear scale, use additive padding
        y_range = y_max - y_min
        y_padding = max(0.05 * y_range, 0.1)  # At least 5% padding or 0.1 units
        ax.set_ylim(max(0, y_min - y_padding), y_max + y_padding)
    
    # --- Per-figure summary (printed) ---
    # Compute figure-specific averages (restricted to the plotted year range)
    # and also averages restricted to the string-match years within the range
    # Identify string-match index from labels
    string_match_idx = None
    for idx, lbl in enumerate(labels_list):
        if "string match" in lbl.lower():
            string_match_idx = idx
            break
    
    print(f"\nSummary: {model_name.upper()} - {exp_title} | Years {year_start}-{year_end}")
    
    # Collect string-match years used (restricted to range), if available
    string_match_years_in_range = []
    if string_match_idx is not None:
        sm_exp_data = ce_results_list[string_match_idx][exp_key]
        for y in sm_exp_data['years_used']:
            yi = int(y)
            if year_start <= yi <= year_end:
                string_match_years_in_range.append(yi)
    
    for i, (ce_result, label) in enumerate(zip(ce_results_list, labels_list)):
        exp_data = ce_result[exp_key]
        # Years used for this dataset within range
        years_in_range = []
        for y in exp_data['years_used']:
            yi = int(y)
            if year_start <= yi <= year_end:
                years_in_range.append(yi)
        
        # Average over its own years within range
        own_losses = [exp_data['per_year_losses'][str(y)] for y in years_in_range if str(y) in exp_data['per_year_losses']]
        if own_losses:
            own_avg = sum(own_losses) / len(own_losses)
            own_avg_str = f"{own_avg:.4f}"
        else:
            own_avg_str = "N/A"
        
        # Average over string-match years within range
        if string_match_years_in_range:
            sm_losses = [exp_data['per_year_losses'][str(y)] for y in string_match_years_in_range if str(y) in exp_data['per_year_losses']]
            if sm_losses:
                sm_avg = sum(sm_losses) / len(sm_losses)
                sm_avg_str = f"{sm_avg:.4f}"
                sm_count = len(sm_losses)
            else:
                sm_avg_str = "N/A"
                sm_count = 0
        else:
            sm_avg_str = "N/A"
            sm_count = 0
        
        # Average over years NOT in string-match within range
        non_sm_years_in_range = [y for y in years_in_range if y not in set(string_match_years_in_range)]
        non_sm_losses = [exp_data['per_year_losses'][str(y)] for y in non_sm_years_in_range if str(y) in exp_data['per_year_losses']]
        if non_sm_losses:
            non_sm_avg = sum(non_sm_losses) / len(non_sm_losses)
            non_sm_avg_str = f"{non_sm_avg:.4f}"
            non_sm_count = len(non_sm_losses)
        else:
            non_sm_avg_str = "N/A"
            non_sm_count = 0
        
        print(f"- {label}: avg={own_avg_str} over {len(own_losses)} yrs; avg@string-match={sm_avg_str} over {sm_count} yrs; avg@non-string-match={non_sm_avg_str} over {non_sm_count} yrs")
    
    # Save the plot
    exp_name = EXPERIMENT_SHORT_NAME
    num_dists = len(labels_list)
    filename = f"{model_name}_cross_entropy_{exp_name}_{num_dists}dists_{year_start}_{year_end}.png"
    save_path = Path(output_dir) / filename
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_cross_entropy_averages_over_checkpoints(analyzer, model_name, year_start=1950, year_end=2050, output_dir="cross_entropy_across_checkpoints"):
    """Plot average cross-entropy vs checkpoints.

    For each checkpoint, compute the average CE over (a) all available years
    within [year_start, year_end] for each distribution and (b) the same CE
    but restricted to the years used by the string-match distribution.

    Produces separate figures for the two experiments and reuses the same
    colors per distribution with different linestyles for the two averaging modes.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)


    if model_name == "olmo":
        dist_pred = analyzer.olmo_relative_predictions
        dist_ngram = analyzer.olmo_relative_ngram
        dist_co = analyzer.olmo_relative_training_data["in_year_tense_sentence_counts"]
    elif model_name == "pythia":
        dist_pred = analyzer.pythia_relative_predictions
        dist_ngram = analyzer.pythia_relative_ngram
        dist_co = analyzer.pythia_relative_training_data["in_year_tense_sentence_counts"]
    else:
        raise ValueError("model_name must be 'olmo' or 'pythia'")

    # Use only checkpoints present in all three distributions
    checkpoints = sorted(set(dist_pred.keys()) & set(dist_ngram.keys()) & set(dist_co.keys()))

    # Single experiment only
    exp_key = EXPERIMENT_KEY
    exp_title = EXPERIMENT_TITLE

    labels = ["Next-token predictions", "N-gram model", "Co-occurrence model"]
    color_map = {
        "Next-token predictions": CROSS_ENTROPY_COLORS[0],
        "N-gram model": CROSS_ENTROPY_COLORS[1],
        "Co-occurrence model": CROSS_ENTROPY_COLORS[2],
    }

    def _avg_specific_years(exp_data, years_list):
        assert len(years_list) > 0, "Expected non-empty year list for average"
        losses = [exp_data['per_year_losses'][str(y)] for y in years_list if str(y) in exp_data['per_year_losses']]
        assert len(losses) > 0, "No losses found for provided years"
        return float(sum(losses) / len(losses))

    x_cps = []

    # Each dict maps label -> list of averages over checkpoints
    full_avgs = {lbl: [] for lbl in labels}
    ngram_avgs = {lbl: [] for lbl in labels}

    for cp in checkpoints:
        ce_pred = analyzer.compute_cross_entropy_over_range(dist_pred, model_name, cp, year_start, year_end)
        ce_ngram = analyzer.compute_cross_entropy_over_range(dist_ngram, model_name, cp, year_start, year_end)
        ce_co = analyzer.compute_cross_entropy_over_range(dist_co, model_name, cp, year_start, year_end)

        # Find common years across all three distributions for this checkpoint
        pred_years = set(int(y) for y in ce_pred[exp_key]['years_used'] if year_start <= int(y) <= year_end)
        ngram_years = set(int(y) for y in ce_ngram[exp_key]['years_used'] if year_start <= int(y) <= year_end)
        co_years = set(int(y) for y in ce_co[exp_key]['years_used'] if year_start <= int(y) <= year_end)
        common_years = sorted(pred_years & ngram_years & co_years)

        # Compute averages over common years only
        full_avgs["Next-token predictions"].append(_avg_specific_years(ce_pred[exp_key], common_years))
        full_avgs["N-gram model"].append(_avg_specific_years(ce_ngram[exp_key], common_years))
        full_avgs["Co-occurrence model"].append(_avg_specific_years(ce_co[exp_key], common_years))

        # Ngram-restricted averages (use ngram years as reference since it has most complete data)
        ngram_years_list = [int(y) for y in ce_ngram[exp_key]['years_used'] if year_start <= int(y) <= year_end]
        ngram_avgs["Next-token predictions"].append(_avg_specific_years(ce_pred[exp_key], ngram_years_list))
        ngram_avgs["N-gram model"].append(_avg_specific_years(ce_ngram[exp_key], ngram_years_list))
        ngram_avgs["Co-occurrence model"].append(_avg_specific_years(ce_co[exp_key], ngram_years_list))

        x_cps.append(cp)

    # Plot
    fig, ax = plt.subplots(figsize=(9, 5))

    for lbl in labels:
        color = color_map[lbl]
        ax.plot(x_cps, full_avgs[lbl], color=color, linestyle='-', marker='.', markersize=8, label=f"{lbl} (full)")
        ax.plot(x_cps, ngram_avgs[lbl], color=color, linestyle='--', marker='.', markersize=8, label=f"{lbl} (ngram-years)")

    ax.set_xlabel("Checkpoint", fontsize=12)
    ax.set_ylabel("Average Cross-Entropy", fontsize=12)
    ax.set_title(f"{model_name.upper()} - {exp_title}\nYears {year_start}-{year_end}", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # y-limits based on data
    all_vals = []
    for d in [full_avgs, ngram_avgs]:
        for v in d.values():
            all_vals.extend(v)
    y_min, y_max = min(all_vals), max(all_vals)
    y_range = y_max - y_min
    y_pad = max(0.05 * y_range, 0.05)
    ax.set_ylim(max(0, y_min - y_pad), y_max + y_pad)

    plt.tight_layout()
    exp_name = EXPERIMENT_SHORT_NAME
    fname = f"{model_name}_avg_ce_vs_checkpoint_{exp_name}_{year_start}_{year_end}.png"
    save_path = Path(output_dir) / fname
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ------------------------------------------------------------------------------------------------



# def plot_prediction_ce_prepost_over_checkpoints(model_name, year_start=None, year_end=None, output_dir="cross_entropy_predictions_prepost"):
#     """Plot predictions-only average CE vs checkpoints, split pre/post cutoff.

#     Uses all available years by default (TOTAL_YEARS). Creates separate figures for
#     experiment1 (past vs present+future).
#     """
#     if year_start is None:
#         year_start = TOTAL_YEARS[0]
#     if year_end is None:
#         year_end = TOTAL_YEARS[1] - 1

#     Path(output_dir).mkdir(parents=True, exist_ok=True)

#     analyzer = AnalyzerClass()

#     if model_name == "olmo":
#         dist_pred = analyzer.olmo_relative_predictions
#         cutoff = OLMO_CUTOFF
#     elif model_name == "pythia":
#         dist_pred = analyzer.pythia_relative_predictions
#         cutoff = PYTHIA_CUTOFF
#     else:
#         raise ValueError("model_name must be 'olmo' or 'pythia'")

#     checkpoints = sorted(dist_pred.keys())

#     # Single experiment only
#     exp_key = EXPERIMENT_KEY
#     exp_title = EXPERIMENT_TITLE

#     x_cps = []
#     pre_avgs = []  # years < cutoff
#     post_avgs = [] # years > cutoff

#     for cp in checkpoints:
#         ce_pred = analyzer.compute_cross_entropy_over_range(dist_pred, model_name, cp, year_start, year_end)
#         exp_data = ce_pred[exp_key]

#         years_int = [int(y) for y in exp_data['years_used'] if year_start <= int(y) <= year_end]
#         pre_years = [y for y in years_int if y < cutoff]
#         post_years = [y for y in years_int if y > cutoff]

#         if not pre_years or not post_years:
#             continue

#         pre_losses = [exp_data['per_year_losses'][str(y)] for y in pre_years if str(y) in exp_data['per_year_losses']]
#         post_losses = [exp_data['per_year_losses'][str(y)] for y in post_years if str(y) in exp_data['per_year_losses']]



#         pre_avgs.append(sum(pre_losses) / len(pre_losses))
#         post_avgs.append(sum(post_losses) / len(post_losses))
#         x_cps.append(cp)

#     if not x_cps:
#         print(f"No checkpoints with both pre- and post-cutoff years for {model_name} {exp_key}")
#         return

#     fig, ax = plt.subplots(figsize=(12, 7))
#     color = CROSS_ENTROPY_COLORS[0]

#     ax.plot(x_cps, pre_avgs, color=color, linestyle='-', marker='.', markersize=8, label=f"predictions (< {cutoff})")
#     ax.plot(x_cps, post_avgs, color=color, linestyle='--', marker='.', markersize=8, label=f"predictions (> {cutoff})")

#     ax.set_xlabel("Checkpoint", fontsize=12)
#     ax.set_ylabel("Average Cross-Entropy", fontsize=12)
#     ax.set_title(f"{model_name.upper()} - {exp_title}\nAll years ({year_start}-{year_end}), split by cutoff {cutoff}", fontsize=14)
#     ax.grid(True, alpha=0.3)
#     ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

#     all_vals = pre_avgs + post_avgs
#     y_min, y_max = min(all_vals), max(all_vals)
#     y_range = y_max - y_min
#     y_pad = max(0.05 * y_range, 0.05)
#     ax.set_ylim(max(0, y_min - y_pad), y_max + y_pad)

#     plt.tight_layout()
#     exp_name = EXPERIMENT_SHORT_NAME
#     fname = f"{model_name}_predictions_avg_ce_prepost_{exp_name}_{year_start}_{year_end}.png"
#     save_path = Path(output_dir) / fname
#     plt.savefig(save_path, dpi=600, bbox_inches='tight')
#     plt.close()
#     print(f"Saved: {save_path}")

def plot_prediction_ce_all_years_over_checkpoints(analyzer, model_name, year_start=None, year_end=None, output_dir="cross_entropy_predictions_all_years"):
    """Plot predictions-only CE per year across checkpoints with a gradient colorbar.

    - One line per year (within [year_start, year_end]), colored by year using a continuous colormap.
    
    - Separate figures for experiment1.
    - No per-line legend; use a colorbar labeled 'Year'.
    """
    if year_start is None:
        year_start = TOTAL_YEARS[0]
    if year_end is None:
        year_end = TOTAL_YEARS[1] - 1

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if model_name == "olmo":
        dist_pred = analyzer.olmo_relative_predictions
    elif model_name == "pythia":
        dist_pred = analyzer.pythia_relative_predictions
    else:
        raise ValueError("model_name must be 'olmo' or 'pythia'")

    checkpoints = sorted(dist_pred.keys())

    # Single experiment only
    exp_key = EXPERIMENT_KEY
    exp_title = EXPERIMENT_TITLE

    # Color mapping from year to color via a colormap
    cmap = plt.get_cmap('viridis')
    norm = plt.Normalize(vmin=year_start, vmax=year_end)

    # Build series per year
    year_to_series = {y: {'x': [], 'y': []} for y in range(year_start, year_end + 1)}

    for cp in checkpoints:
        ce_pred = analyzer.compute_cross_entropy_over_range(dist_pred, model_name, cp, year_start, year_end)
        exp_data = ce_pred[exp_key]
        for y_str, loss in exp_data['per_year_losses'].items():
            yi = int(y_str)
            if year_start <= yi <= year_end:
                year_to_series[yi]['x'].append(cp)
                year_to_series[yi]['y'].append(loss)

    # Plot
    fig, ax = plt.subplots(figsize=(9, 6))
    all_vals = []

    for yi in range(year_start, year_end + 1):
        xs = year_to_series[yi]['x']
        ys = year_to_series[yi]['y']
        if not xs:
            continue
        color = cmap(norm(yi))
        ax.plot(xs, ys, color=color, linewidth=1.2)
        all_vals.extend(ys)

    ax.set_xlabel("Checkpoint", fontsize=12)
    ax.set_ylabel("Cross-Entropy", fontsize=12)
    ax.set_title(f"{model_name.upper()} - {exp_title}\nPer-year CE across checkpoints ({year_start}-{year_end})", fontsize=14)
    ax.grid(True, alpha=0.3)

    # Add colorbar indicating year
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Year')

    # y-limits based on data
    if all_vals:
        y_min, y_max = min(all_vals), max(all_vals)
        y_range = y_max - y_min
        y_pad = max(0.05 * y_range, 0.05)
        ax.set_ylim(max(0, y_min - y_pad), y_max + y_pad)

    plt.tight_layout()
    exp_name = EXPERIMENT_SHORT_NAME
    fname = f"{model_name}_predictions_ce_by_year_{exp_name}_{year_start}_{year_end}.png"
    save_path = Path(output_dir) / fname
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_prediction_ce_averages_over_checkpoints(analyzer, model_name, checkpoints, year_start=1950, year_end=2050, output_dir="cross_entropy_prediction_avgs"):
    """Plot predictions-only average CE vs specified checkpoints.

    - Uses only model predictions (no training data) over [year_start, year_end].
    - Creates separate figures for experiment1.
    - Plots a single line with dot markers for the provided checkpoints.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)


    if model_name == "olmo":
        dist_pred = analyzer.olmo_relative_predictions
    elif model_name == "pythia":
        dist_pred = analyzer.pythia_relative_predictions
    else:
        raise ValueError("model_name must be 'olmo' or 'pythia'")

    # Keep only checkpoints present in predictions
    available_cps = set(dist_pred.keys())
    cps = [cp for cp in checkpoints if cp in available_cps]
    for cp in checkpoints:
        if cp not in available_cps:
            print(f"Skipping checkpoint {cp}: not found in predictions for {model_name}")

    if not cps:
        print("No valid checkpoints to plot.")
        return

    # Single experiment only
    exp_key = EXPERIMENT_KEY
    exp_title = EXPERIMENT_TITLE

    x_cps = []
    avgs = []

    for cp in cps:
        ce_pred = analyzer.compute_cross_entropy_over_range(dist_pred, model_name, cp, year_start, year_end)
        avg_loss = float(ce_pred[exp_key]['average_loss'])
        x_cps.append(cp)
        avgs.append(avg_loss)

    if not x_cps:
        print(f"No data to plot for {model_name} {exp_key}")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    color = CROSS_ENTROPY_COLORS[0]
    ax.plot(x_cps, avgs, color=color, linestyle='-', marker='.', markersize=8, label="Next-token predictions")

    ax.set_xlabel("Checkpoint", fontsize=12)
    ax.set_ylabel("Average Cross-Entropy", fontsize=12)
    ax.set_title(f"{model_name.upper()} - {exp_title}\nYears {year_start}-{year_end}", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

    y_min, y_max = min(avgs), max(avgs)
    y_range = y_max - y_min
    y_pad = max(0.05 * y_range, 0.05)
    ax.set_ylim(max(0, y_min - y_pad), y_max + y_pad)

    plt.tight_layout()
    exp_name = EXPERIMENT_SHORT_NAME
    cps_str = f"{len(x_cps)}cps"
    fname = f"{model_name}_predictions_avg_ce_vs_checkpoint_{exp_name}_{year_start}_{year_end}_{cps_str}.png"
    save_path = Path(output_dir) / fname
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

# ------------------------------------------------------------

if __name__ == "__main__":
    # python kl_divergence_checkpoints.py
    cp = 10000
    start_year = 1950
    years_end = 2050

    
    # saev_all_analyzer_data
    analyzer = AnalyzerClass()
    filepath = analyzer.save_all_data_to_file()
    print(f"Data export completed. File saved: {filepath}")

    if True:
        # plot_training_data
        analyzer.bar_plot(analyzer.olmo_relative_training_data["in_year_tense_sentence_counts"], "olmo", "in_year_tense_sentence_counts", cp, start_year, years_end)
        analyzer.bar_plot(analyzer.olmo_relative_training_data["in_year_there_word_counts"], "olmo", "in_year_there_word_counts", cp, start_year, years_end)
        analyzer.bar_plot(analyzer.pythia_relative_training_data["in_year_tense_sentence_counts"], "pythia", "in_year_tense_sentence_counts", cp, start_year, years_end)
        analyzer.bar_plot(analyzer.pythia_relative_training_data["in_year_there_word_counts"], "pythia", "in_year_there_word_counts", cp, start_year, years_end)
        analyzer.bar_plot(analyzer.olmo_relative_ngram, "olmo", "Laplace-smoothed n-gram", cp, start_year, years_end)
        analyzer.bar_plot(analyzer.pythia_relative_ngram, "pythia", "Laplace-smoothed n-gram", cp, start_year, years_end)

    
        # plot_model_predictions
        analyzer.bar_plot(analyzer.olmo_relative_predictions, "olmo", "Next-token predictions", cp, start_year, years_end)
        analyzer.bar_plot(analyzer.pythia_relative_predictions, "pythia", "Next-token predictions", cp, start_year, years_end)


        # plot_model_predictions absolute with other models
        model_names = ["allenai_OLMo-2-0425-1B", "EleutherAI_pythia-1b-deduped", "EleutherAI_pythia-1.4b-deduped","EleutherAI_pythia-6.9b-deduped", "allenai_OLMo-2-1124-7B", "meta-llama_Llama-3.1-8B"]
        model_name_to_predictions = analyzer.load_other_model_predictions(model_names)
        for model_name in model_names:
            analyzer.bar_plot(model_name_to_predictions[model_name], model_name, "Next-token predictions", "final", start_year, years_end)

        # plot other prompts for pythia and olmo
        prompt_names = [
            "During__year__there", "In__year__the_choir", "In__year__there", 
            "In__year__they", "In__year_,_at_the_dinner_table,_the_family", 
            "In__year_,_there", "In__year_,_with_a_knife,_he", "In__year_,_with_a_pen_to_paper,_she", 
            "In__year_,_with_his_credit_card,_he", "In_the_magic_show_in__year_,_there_magically",
        ]
        prompt_to_model_to_predictions = analyzer.load_other_prompts(prompt_names)
        for prompt, model_to_pred in prompt_to_model_to_predictions.items():
            for model_name, pred in model_to_pred.items():
                analyzer.bar_plot(prompt_to_model_to_predictions[prompt][model_name], model_name, f"Next-token predictions __{prompt}", "final", start_year, years_end)

    else:
        # compute_cross_entropies
        olmo_predictions_ce = analyzer.compute_cross_entropy_over_range(analyzer.olmo_relative_predictions, "olmo", cp, start_year, years_end)
        olmo_string_match_ce = analyzer.compute_cross_entropy_over_range(analyzer.olmo_relative_training_data["in_year_there_word_counts"], "olmo", cp, start_year, years_end)
        olmo_co_occurrence_ce = analyzer.compute_cross_entropy_over_range(analyzer.olmo_relative_training_data["in_year_tense_sentence_counts"], "olmo", cp, start_year, years_end)
        olmo_ngram_ce = analyzer.compute_cross_entropy_over_range(analyzer.olmo_relative_ngram, "olmo", cp, start_year, years_end)
        plot_cross_entropies([olmo_predictions_ce, olmo_ngram_ce, olmo_co_occurrence_ce], ["Next-token predictions", "N-gram model", "Co-occurrence model"], "olmo")
        plot_cross_entropies([olmo_string_match_ce], ["Exact string match model"], "olmo") # for appendix

        pythia_predictions_ce = analyzer.compute_cross_entropy_over_range(analyzer.pythia_relative_predictions, "pythia", cp, start_year, years_end)
        pythia_string_match_ce = analyzer.compute_cross_entropy_over_range(analyzer.pythia_relative_training_data["in_year_there_word_counts"], "pythia", cp, start_year, years_end)
        pythia_co_occurrence_ce = analyzer.compute_cross_entropy_over_range(analyzer.pythia_relative_training_data["in_year_tense_sentence_counts"], "pythia", cp, start_year, years_end)
        pythia_ngram_ce = analyzer.compute_cross_entropy_over_range(analyzer.pythia_relative_ngram, "pythia", cp, start_year, years_end)
        plot_cross_entropies([pythia_predictions_ce, pythia_ngram_ce, pythia_co_occurrence_ce], ["Next-token predictions", "N-gram model", "Co-occurrence model"], "pythia")
        plot_cross_entropies([pythia_string_match_ce], ["Exact string match model"], "pythia")

        # plot_training_dynamics
        analyzer.bar_plots_for_checkpoints(analyzer.olmo_relative_predictions, "olmo", "Next-token Predictions", [250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 3250, 3500, 3750, 4000, 4250, 4500, 4750, 5000, 5250, 5500, 5750, 6000, 6250, 6500, 6750, 7000, 7250, 7500, 7750, 8000, 8250, 8500, 8750, 9000, 9250, 9500, 9750, 10000], 8, 5, start_year, years_end)
        analyzer.bar_plots_for_checkpoints(analyzer.olmo_relative_training_data["in_year_tense_sentence_counts"], "olmo", "in_year_tense_sentence_counts", [260, 500, 760, 1000, 1260, 1500, 1760, 2000, 2260, 2500, 2760, 3000, 3260, 3500, 3760, 4000, 4260, 4500, 4760, 5000, 5260, 5500, 5760, 6000, 6260, 6500, 6760, 7000, 7260, 7500, 7760, 8000, 8260, 8500, 8760, 9000, 9260, 9500, 9760, 10000], 8, 5, start_year, years_end)
        analyzer.bar_plots_for_checkpoints(analyzer.olmo_relative_ngram, "olmo", "Laplace-smoothed n-gram", [260, 500, 760, 1000, 1260, 1500, 1760, 2000, 2260, 2500, 2760, 3000, 3260, 3500, 3760, 4000, 4260, 4500, 4760, 5000, 5260, 5500, 5760, 6000, 6260, 6500, 6760, 7000, 7260, 7500, 7760, 8000, 8260, 8500, 8760, 9000, 9260, 9500, 9760, 10000], 8, 5, start_year, years_end)
        
        analyzer.bar_plots_for_checkpoints(analyzer.pythia_relative_predictions, "pythia", "Next-token Predictions", [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000], 2, 5, start_year, years_end)
        analyzer.bar_plots_for_checkpoints(analyzer.pythia_relative_training_data["in_year_tense_sentence_counts"], "pythia", "in_year_tense_sentence_counts", [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000], 2, 5, start_year, years_end)
        analyzer.bar_plots_for_checkpoints(analyzer.pythia_relative_ngram, "pythia", "Laplace-smoothed n-gram", [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000], 2, 5, start_year, years_end)
        

        # ce_over_training()
        plot_cross_entropy_averages_over_checkpoints(analyzer, "olmo", start_year, years_end)
        plot_cross_entropy_averages_over_checkpoints(analyzer, "pythia", start_year, years_end)

        # ce_over_training_split()
        plot_prediction_ce_all_years_over_checkpoints(analyzer, "olmo", start_year, years_end)
        plot_prediction_ce_all_years_over_checkpoints(analyzer, "pythia", start_year, years_end)

        # ce_over_more_checkpoints_pythia()
        plot_prediction_ce_averages_over_checkpoints(analyzer, "pythia", PYTHIA_CHECKPOINTS, start_year, years_end)




        

        
