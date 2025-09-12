import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from scipy.stats import spearmanr
import torch.nn.functional as F
from matplotlib.lines import Line2D
from datetime import datetime
from fractions import Fraction

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

OLMO_CUTOFF = 2024
PYTHIA_CUTOFF = 2020

class AnalyzerClass:
    def __init__(self):
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

        # this is per future/present/past 
        self.olmo_gold_distribution = {} # has entries 0 or 1 for past and future
        self.pythia_gold_distribution = {} # see olmo

        print("loading all data")
        self.load_training_data()
        self.load_model_predictions()
        self.populate_gold_data()
        print("finished loading all data")
    


    def _normalize_tense_distribution(self, tense_counts):
        all_tenses = list(set(TENSE_MAPPING.values()))
        complete_counts = {tense: tense_counts.get(tense, 0.0) for tense in all_tenses}
        
        total = sum(complete_counts.values())
        if total == 0:
            return {tense: 0.0 for tense in all_tenses}
        
        normalized = {tense: count / total for tense, count in complete_counts.items()}
        
        return normalized



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
                'export_timestamp': datetime.now().isoformat(),
            },
            'olmo_training_data': self.olmo_training_data,
            'olmo_relative_training_data': self.olmo_relative_training_data,
            'olmo_predictions': self.olmo_predictions,
            'olmo_relative_predictions': self.olmo_relative_predictions,
            'olmo_gold_distribution': self.olmo_gold_distribution,
            'pythia_training_data': self.pythia_training_data,
            'pythia_relative_training_data': self.pythia_relative_training_data,
            'pythia_predictions': self.pythia_predictions,
            'pythia_relative_predictions': self.pythia_relative_predictions,
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

            if (gold_dist[0] == 1 and pred_dist[0] == 0) or (gold_dist[1] == 1 and pred_dist[1] == 0):
                # gold distribution and this distribution have zero overlap; cap ce_loss at 3
                print(f"Gold distribution and this distribution have zero overlap for model {which_model} at checkpoint {checkpoint} and year {year_str}")
                ce_loss = 2.5
            
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
            else:
                raise ValueError(f"Unknown Pythia distribution: {dist_dict}")
        else:
            raise ValueError(f"Invalid model: {which_model}")
        
        # Separate storage for each experiment
        experiment1_losses = {}  # present + future combined
        experiment2_losses = {}  # only past vs future
        years_used_exp1 = []
        years_used_exp2 = []
        
        for year in range(year_start, year_end + 1):
            year_str = str(year)
            # Exclude cutoff year from CE computations and plotting
            if (which_model == "olmo" and year == OLMO_CUTOFF) or (which_model == "pythia" and year == PYTHIA_CUTOFF):
                continue
            dist_data = dist_cp[year_str]
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
            if sum(pred_dist) == 0:
                continue    
            ce_loss = compute_losses(pred_dist, gold_dist)
            if ce_loss is not None:
                experiment1_losses[year_str] = ce_loss
                years_used_exp1.append(year_str)
            
            # experiment 2: disregard present tense
            pred_dist_sum = dist_data["past"] + dist_data["future"]
            if pred_dist_sum == 0:
                continue
            pred_dist = [dist_data["past"] / pred_dist_sum, dist_data["future"] / pred_dist_sum]
            ce_loss = compute_losses(pred_dist, gold_dist)
            if ce_loss is not None:
                experiment2_losses[year_str] = ce_loss
                years_used_exp2.append(year_str)

        # Compute average losses for each experiment
        avg_loss_exp1 = sum(experiment1_losses.values()) / len(experiment1_losses)
        avg_loss_exp2 = sum(experiment2_losses.values()) / len(experiment2_losses)
        
        return {
            'experiment1_past_vs_present_future': {
                'per_year_losses': experiment1_losses,
                'average_loss': avg_loss_exp1,
                'years_used': years_used_exp1,
                'years_skipped': set(range(year_start, year_end+1)) - set(years_used_exp1),
                'description': 'Binary classification: past vs (present + future)'
            },
            'experiment2_past_vs_future': {
                'per_year_losses': experiment2_losses,
                'average_loss': avg_loss_exp2,
                'years_used': years_used_exp2,
                'years_skipped': set(range(year_start, year_end+1)) - set(years_used_exp2),
                'description': 'Binary classification: past vs future (ignoring present)'
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
            

        def load_model_predictions(file_template, step_size):
            results_absolute = {}
            results_relative = {}
            for cp in range(step_size, 10001, step_size):
                filepath = file_template.format(cp=cp)
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

        self.olmo_predictions, self.olmo_relative_predictions = load_model_predictions(OLMO_PREDICTIONS_FILE, 250)
        self.pythia_predictions, self.pythia_relative_predictions = load_model_predictions(PYTHIA_PREDICTIONS_FILE, 1000)


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

        # Default checkpoint list
        olmo_cps = range(250, 10001, 250)
        pythia_cps = range(1000, 10001, 1000)

        for cp in olmo_cps:
            self.olmo_gold_distribution[cp] = populate_gold_distribution(OLMO_CUTOFF)
        for cp in pythia_cps:
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
        fig, ax = plt.subplots(figsize=(12, 6))
        bottom = np.zeros(len(years))
        
        for tense in tenses:
            vals = np.array([cp_data[y].get(tense, 0) for y in years])
            ax.bar(range(len(years)), vals, bottom=bottom, 
                  label=tense, color=TENSE_COLORS[tense], width=1.0)
            bottom += vals
        
        # Format
        title = f"{data_type} | Checkpoint {checkpoint} | Years {year_start}-{year_end}"
        safe_data_type = data_type.replace(' ', '_')
        filename = f"{model}_checkpoint{checkpoint}_{safe_data_type}_{year_start}-{year_end}"
        
        ax.set_title(title)
        ax.set_xticks(range(0, len(years), max(1, len(years)//20)))
        ax.set_xticklabels(years[::max(1, len(years)//20)], rotation=45)
        ax.set_ylabel("Probability")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
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

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
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
            ax.set_title(f"{data_type} | Checkpoint {cp}", fontsize=10)
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
        safe_data_type = data_type.replace(' ', '_')
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
    
    # Create separate plots for each experiment
    experiments = ['experiment1_past_vs_present_future', 'experiment2_past_vs_future']
    experiment_titles = [
        'Binary Classification: Past vs (Present + Future)',
        'Binary Classification: Past vs Future (ignoring Present)'
    ]
    
    for exp_key, exp_title in zip(experiments, experiment_titles):
        fig, ax = plt.subplots(figsize=(14, 8))
        
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
        ax.set_title(f'{model_name.upper()} - {exp_title}\nYears {year_start}-{year_end}', fontsize=14)
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
        exp_name = exp_key.replace('experiment1_', '').replace('experiment2_', '')
        filename = f"{model_name}_cross_entropy_{exp_name}_{year_start}_{year_end}.png"
        save_path = Path(output_dir) / filename
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")
        
        # --- Additional figure: plot only years where ALL distributions have data (fair comparison) ---
        # Compute intersection of years across all distributions for this experiment within the range
        common_years = None
        for ce_result in ce_results_list:
            exp_data_tmp = ce_result[exp_key]
            years_in_range_tmp = set()
            for y in exp_data_tmp['years_used']:
                yi = int(y)
                if year_start <= yi <= year_end:
                    years_in_range_tmp.add(yi)
            if common_years is None:
                common_years = years_in_range_tmp
            else:
                common_years &= years_in_range_tmp
        common_years = sorted(common_years) if common_years else []

        if common_years:
            fig2, ax2 = plt.subplots(figsize=(14, 8))
            all_years_common = []
            all_losses_common = []

            for i, (ce_result, label) in enumerate(zip(ce_results_list, labels_list)):
                exp_data = ce_result[exp_key]
                losses_common = [exp_data['per_year_losses'][str(y)] for y in common_years if str(y) in exp_data['per_year_losses']]
                if losses_common:
                    color = CROSS_ENTROPY_COLORS[i % len(CROSS_ENTROPY_COLORS)]
                    avg_common = sum(losses_common) / len(losses_common)
                    ax2.scatter(common_years, losses_common,
                                label=f"{label} (avg common: {avg_common:.4f}, n={len(losses_common)})",
                                color=color, s=20, alpha=0.7)
                    all_years_common.extend(common_years)
                    all_losses_common.extend(losses_common)

            # Format
            ax2.set_xlabel('Year', fontsize=12)
            ax2.set_ylabel('Cross-Entropy Loss', fontsize=12)
            ax2.set_title(f"{model_name.upper()} - {exp_title} (Common Years Only)\nYears {year_start}-{year_end}", fontsize=14)
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax2.grid(True, alpha=0.3)

            # Axis limits based on common data
            if all_years_common and all_losses_common:
                ax2.set_xlim(year_start - 5, year_end + 5)
                y_min_c, y_max_c = min(all_losses_common), max(all_losses_common)
                y_range_c = y_max_c - y_min_c
                y_pad_c = max(0.05 * y_range_c, 0.1)
                ax2.set_ylim(max(0, y_min_c - y_pad_c), y_max_c + y_pad_c)

            # Save common-years-only plot
            filename_common = f"{model_name}_cross_entropy_{exp_name}_common_{year_start}_{year_end}.png"
            save_path_common = Path(output_dir) / filename_common
            plt.tight_layout()
            plt.savefig(save_path_common, dpi=600, bbox_inches='tight')
            plt.close()
            print(f"Saved: {save_path_common}")
        else:
            print("No common years across distributions within the selected range; skipping common-years plot.")


def plot_cross_entropy_averages_over_checkpoints(model_name, year_start=1950, year_end=2050, output_dir="cross_entropy_across_checkpoints"):
    """Plot average cross-entropy vs checkpoints.

    For each checkpoint, compute the average CE over (a) all available years
    within [year_start, year_end] for each distribution and (b) the same CE
    but restricted to the years used by the string-match distribution.

    Produces separate figures for the two experiments and reuses the same
    colors per distribution with different linestyles for the two averaging modes.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    analyzer = AnalyzerClass()

    if model_name == "olmo":
        dist_pred = analyzer.olmo_relative_predictions
        dist_sm = analyzer.olmo_relative_training_data["in_year_there_word_counts"]
        dist_co = analyzer.olmo_relative_training_data["in_year_tense_sentence_counts"]
    elif model_name == "pythia":
        dist_pred = analyzer.pythia_relative_predictions
        dist_sm = analyzer.pythia_relative_training_data["in_year_there_word_counts"]
        dist_co = analyzer.pythia_relative_training_data["in_year_tense_sentence_counts"]
    else:
        raise ValueError("model_name must be 'olmo' or 'pythia'")

    # Use only checkpoints present in all three distributions
    checkpoints = sorted(set(dist_pred.keys()) & set(dist_sm.keys()) & set(dist_co.keys()))

    experiments = ['experiment1_past_vs_present_future', 'experiment2_past_vs_future']
    experiment_titles = [
        'Binary Classification: Past vs (Present + Future)',
        'Binary Classification: Past vs Future (ignoring Present)'
    ]

    labels = ["predictions", "string match", "co occurrence"]
    color_map = {
        "predictions": CROSS_ENTROPY_COLORS[0],
        "string match": CROSS_ENTROPY_COLORS[1],
        "co occurrence": CROSS_ENTROPY_COLORS[2],
    }

    def _avg_all_years(exp_data):
        # compute_cross_entropy_over_range already averaged over the requested range
        return float(exp_data['average_loss'])

    def _avg_specific_years(exp_data, years_list):
        assert len(years_list) > 0, "Expected non-empty year list for string-match average"
        losses = [exp_data['per_year_losses'][str(y)] for y in years_list]
        assert len(losses) > 0, "No losses found for provided years"
        return float(sum(losses) / len(losses))

    for exp_key, exp_title in zip(experiments, experiment_titles):
        x_cps = []

        # Each dict maps label -> list of averages over checkpoints
        full_avgs = {lbl: [] for lbl in labels}
        sm_avgs = {lbl: [] for lbl in labels}

        for cp in checkpoints:
            ce_pred = analyzer.compute_cross_entropy_over_range(dist_pred, model_name, cp, year_start, year_end)
            ce_sm   = analyzer.compute_cross_entropy_over_range(dist_sm,   model_name, cp, year_start, year_end)
            ce_co   = analyzer.compute_cross_entropy_over_range(dist_co,   model_name, cp, year_start, year_end)

            # Find common years across all three distributions for this checkpoint
            pred_years = set(int(y) for y in ce_pred[exp_key]['years_used'] if year_start <= int(y) <= year_end)
            sm_years = set(int(y) for y in ce_sm[exp_key]['years_used'] if year_start <= int(y) <= year_end)
            co_years = set(int(y) for y in ce_co[exp_key]['years_used'] if year_start <= int(y) <= year_end)
            common_years = sorted(pred_years & sm_years & co_years)

            # Compute averages over common years only
            full_avgs["predictions"].append(_avg_specific_years(ce_pred[exp_key], common_years))
            full_avgs["string match"].append(_avg_specific_years(ce_sm[exp_key], common_years))
            full_avgs["co occurrence"].append(_avg_specific_years(ce_co[exp_key], common_years))

            # String-match restricted averages
            sm_years_list = [int(y) for y in ce_sm[exp_key]['years_used'] if year_start <= int(y) <= year_end]
            sm_avgs["predictions"].append(_avg_specific_years(ce_pred[exp_key], sm_years_list))
            sm_avgs["string match"].append(_avg_specific_years(ce_sm[exp_key], sm_years_list))
            sm_avgs["co occurrence"].append(_avg_specific_years(ce_co[exp_key], sm_years_list))

            x_cps.append(cp)

        # Plot
        fig, ax = plt.subplots(figsize=(12, 7))

        for lbl in labels:
            color = color_map[lbl]
            ax.plot(x_cps, full_avgs[lbl], color=color, linestyle='-', marker='.', markersize=8, label=f"{lbl} (full)")
            ax.plot(x_cps, sm_avgs[lbl],   color=color, linestyle='--', marker='.', markersize=8, label=f"{lbl} (string-match)")

        ax.set_xlabel("Checkpoint", fontsize=12)
        ax.set_ylabel("Average Cross-Entropy", fontsize=12)
        ax.set_title(f"{model_name.upper()} - {exp_title}\nYears {year_start}-{year_end}", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # y-limits based on data
        all_vals = []
        for d in [full_avgs, sm_avgs]:
            for v in d.values():
                all_vals.extend(v)
        y_min, y_max = min(all_vals), max(all_vals)
        y_range = y_max - y_min
        y_pad = max(0.05 * y_range, 0.05)
        ax.set_ylim(max(0, y_min - y_pad), y_max + y_pad)

        plt.tight_layout()
        exp_name = exp_key.replace('experiment1_', '').replace('experiment2_', '')
        fname = f"{model_name}_avg_ce_vs_checkpoint_{exp_name}_{year_start}_{year_end}.png"
        save_path = Path(output_dir) / fname
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")



def plot_model_predictions():
    analyzer = AnalyzerClass()
    # currently only running for 10k steps of training
    analyzer.bar_plot(analyzer.olmo_relative_predictions, "olmo", "Next-token predictions", 10000, 1950, 2050)
    analyzer.bar_plot(analyzer.pythia_relative_predictions, "pythia", "Next-token predictions", 10000, 1950, 2050)

def plot_training_data():
    # currently only running for 10k steps of training
    analyzer = AnalyzerClass()
    analyzer.bar_plot(analyzer.olmo_relative_training_data["in_year_tense_sentence_counts"], "olmo", "in_year_tense_sentence_counts", 10000, 1950, 2050)
    analyzer.bar_plot(analyzer.olmo_relative_training_data["in_year_there_word_counts"], "olmo", "in_year_there_word_counts", 10000, 1950, 2050)
    analyzer.bar_plot(analyzer.pythia_relative_training_data["in_year_tense_sentence_counts"], "pythia", "in_year_tense_sentence_counts", 10000, 1950, 2050)
    analyzer.bar_plot(analyzer.pythia_relative_training_data["in_year_there_word_counts"], "pythia", "in_year_there_word_counts", 10000, 1950, 2050)
    
def compute_cross_entropies():
    analyzer = AnalyzerClass()
    cp = 10000
    olmo_predictions_ce = analyzer.compute_cross_entropy_over_range(analyzer.olmo_relative_predictions, "olmo", cp, 1950, 2050)
    olmo_string_match_ce = analyzer.compute_cross_entropy_over_range(analyzer.olmo_relative_training_data["in_year_there_word_counts"], "olmo", cp, 1950, 2050)
    olmo_co_occurrence_ce = analyzer.compute_cross_entropy_over_range(analyzer.olmo_relative_training_data["in_year_tense_sentence_counts"], "olmo", cp, 1950, 2050)
    plot_cross_entropies([olmo_predictions_ce, olmo_string_match_ce, olmo_co_occurrence_ce], ["olmo predictions", "olmo string match", "olmo co occurrence"], "olmo")

    pythia_predictions_ce = analyzer.compute_cross_entropy_over_range(analyzer.pythia_relative_predictions, "pythia", cp, 1950, 2050)
    pythia_string_match_ce = analyzer.compute_cross_entropy_over_range(analyzer.pythia_relative_training_data["in_year_there_word_counts"], "pythia", cp, 1950, 2050)
    pythia_co_occurrence_ce = analyzer.compute_cross_entropy_over_range(analyzer.pythia_relative_training_data["in_year_tense_sentence_counts"], "pythia", cp, 1950, 2050)
    plot_cross_entropies([pythia_predictions_ce, pythia_string_match_ce, pythia_co_occurrence_ce], ["pythia predictions", "pythia string match", "pythia co occurrence"], "pythia")

# ------------------------------------------------------------------------------------------------

def save_all_analyzer_data():
    """
    Example function to load and save all analyzer data.
    """
    analyzer = AnalyzerClass()
    filepath = analyzer.save_all_data_to_file()
    print(f"Data export completed. File saved: {filepath}")
    return filepath

def plot_training_dynamics():
    analyzer = AnalyzerClass()
    # for appendix
    analyzer.bar_plots_for_checkpoints(analyzer.olmo_relative_predictions, "olmo", "Next-token Predictions", [250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 3250, 3500, 3750, 4000, 4250, 4500, 4750, 5000, 5250, 5500, 5750, 6000, 6250, 6500, 6750, 7000, 7250, 7500, 7750, 8000, 8250, 8500, 8750, 9000, 9250, 9500, 9750, 10000], 8, 5, 1950, 2050)
    analyzer.bar_plots_for_checkpoints(analyzer.olmo_relative_training_data["in_year_tense_sentence_counts"], "olmo", "in_year_tense_sentence_counts", [260, 500, 760, 1000, 1260, 1500, 1760, 2000, 2260, 2500, 2760, 3000, 3260, 3500, 3760, 4000, 4260, 4500, 4760, 5000, 5260, 5500, 5760, 6000, 6260, 6500, 6760, 7000, 7260, 7500, 7760, 8000, 8260, 8500, 8760, 9000, 9260, 9500, 9760, 10000], 8, 5, 1950, 2050)
    analyzer.bar_plots_for_checkpoints(analyzer.olmo_relative_training_data["in_year_there_word_counts"], "olmo", "in_year_there_word_counts", [260, 500, 760, 1000, 1260, 1500, 1760, 2000, 2260, 2500, 2760, 3000, 3260, 3500, 3760, 4000, 4260, 4500, 4760, 5000, 5260, 5500, 5760, 6000, 6260, 6500, 6760, 7000, 7260, 7500, 7760, 8000, 8260, 8500, 8760, 9000, 9260, 9500, 9760, 10000], 8, 5, 1950, 2050)
    
    analyzer.bar_plots_for_checkpoints(analyzer.pythia_relative_predictions, "pythia", "Next-token Predictions", [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000], 2, 5, 1950, 2050)
    analyzer.bar_plots_for_checkpoints(analyzer.pythia_relative_training_data["in_year_tense_sentence_counts"], "pythia", "in_year_tense_sentence_counts", [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000], 2, 5, 1950, 2050)
    analyzer.bar_plots_for_checkpoints(analyzer.pythia_relative_training_data["in_year_there_word_counts"], "pythia", "in_year_there_word_counts", [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000], 2, 5, 1950, 2050)
    

    # for main section
    analyzer.bar_plots_for_checkpoints(analyzer.olmo_relative_predictions, "olmo", "Next-token Predictions", [1000, 3000, 7000], 1, 3, 1950, 2050)
    analyzer.bar_plots_for_checkpoints(analyzer.pythia_relative_predictions, "pythia", "Next-token Predictions", [1000, 3000, 7000], 1, 3, 1950, 2050)
    
def ce_over_training():
    plot_cross_entropy_averages_over_checkpoints("olmo", 1950, 2050)
    plot_cross_entropy_averages_over_checkpoints("pythia", 1950, 2050)

def ce_over_training_split():
    # plot_prediction_ce_prepost_over_checkpoints("olmo", 1950, 2050)
    # plot_prediction_ce_prepost_over_checkpoints("pythia", 1950, 2050)

    plot_prediction_ce_all_years_over_checkpoints("olmo", 1950, 2050)
    plot_prediction_ce_all_years_over_checkpoints("pythia", 1950, 2050)

def plot_prediction_ce_prepost_over_checkpoints(model_name, year_start=None, year_end=None, output_dir="cross_entropy_predictions_prepost"):
    """Plot predictions-only average CE vs checkpoints, split pre/post cutoff.

    Uses all available years by default (TOTAL_YEARS). Creates separate figures for
    experiment1 (past vs present+future) and experiment2 (past vs future).
    """
    if year_start is None:
        year_start = TOTAL_YEARS[0]
    if year_end is None:
        year_end = TOTAL_YEARS[1] - 1

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    analyzer = AnalyzerClass()

    if model_name == "olmo":
        dist_pred = analyzer.olmo_relative_predictions
        cutoff = OLMO_CUTOFF
    elif model_name == "pythia":
        dist_pred = analyzer.pythia_relative_predictions
        cutoff = PYTHIA_CUTOFF
    else:
        raise ValueError("model_name must be 'olmo' or 'pythia'")

    checkpoints = sorted(dist_pred.keys())

    experiments = ['experiment1_past_vs_present_future', 'experiment2_past_vs_future']
    experiment_titles = [
        'Binary Classification: Past vs (Present + Future)',
        'Binary Classification: Past vs Future (ignoring Present)'
    ]

    for exp_key, exp_title in zip(experiments, experiment_titles):
        x_cps = []
        pre_avgs = []  # years < cutoff
        post_avgs = [] # years > cutoff

        for cp in checkpoints:
            ce_pred = analyzer.compute_cross_entropy_over_range(dist_pred, model_name, cp, year_start, year_end)
            exp_data = ce_pred[exp_key]

            years_int = [int(y) for y in exp_data['years_used'] if year_start <= int(y) <= year_end]
            pre_years = [y for y in years_int if y < cutoff]
            post_years = [y for y in years_int if y > cutoff]

            if not pre_years or not post_years:
                continue

            pre_losses = [exp_data['per_year_losses'][str(y)] for y in pre_years]
            post_losses = [exp_data['per_year_losses'][str(y)] for y in post_years]

            pre_avgs.append(sum(pre_losses) / len(pre_losses))
            post_avgs.append(sum(post_losses) / len(post_losses))
            x_cps.append(cp)

        if not x_cps:
            print(f"No checkpoints with both pre- and post-cutoff years for {model_name} {exp_key}")
            continue

        fig, ax = plt.subplots(figsize=(12, 7))
        color = CROSS_ENTROPY_COLORS[0]

        ax.plot(x_cps, pre_avgs, color=color, linestyle='-', marker='.', markersize=8, label=f"predictions (< {cutoff})")
        ax.plot(x_cps, post_avgs, color=color, linestyle='--', marker='.', markersize=8, label=f"predictions (> {cutoff})")

        ax.set_xlabel("Checkpoint", fontsize=12)
        ax.set_ylabel("Average Cross-Entropy", fontsize=12)
        ax.set_title(f"{model_name.upper()} - {exp_title}\nAll years ({year_start}-{year_end}), split by cutoff {cutoff}", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        all_vals = pre_avgs + post_avgs
        y_min, y_max = min(all_vals), max(all_vals)
        y_range = y_max - y_min
        y_pad = max(0.05 * y_range, 0.05)
        ax.set_ylim(max(0, y_min - y_pad), y_max + y_pad)

        plt.tight_layout()
        exp_name = exp_key.replace('experiment1_', '').replace('experiment2_', '')
        fname = f"{model_name}_predictions_avg_ce_prepost_{exp_name}_{year_start}_{year_end}.png"
        save_path = Path(output_dir) / fname
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")

def plot_prediction_ce_all_years_over_checkpoints(model_name, year_start=None, year_end=None, output_dir="cross_entropy_predictions_all_years"):
    """Plot predictions-only CE per year across checkpoints with a gradient colorbar.

    - One line per year (within [year_start, year_end]), colored by year using a continuous colormap.
    - Separate figures for experiment1 and experiment2.
    - No per-line legend; use a colorbar labeled 'Year'.
    """
    if year_start is None:
        year_start = TOTAL_YEARS[0]
    if year_end is None:
        year_end = TOTAL_YEARS[1] - 1

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    analyzer = AnalyzerClass()

    if model_name == "olmo":
        dist_pred = analyzer.olmo_relative_predictions
    elif model_name == "pythia":
        dist_pred = analyzer.pythia_relative_predictions
    else:
        raise ValueError("model_name must be 'olmo' or 'pythia'")

    checkpoints = sorted(dist_pred.keys())

    experiments = ['experiment1_past_vs_present_future', 'experiment2_past_vs_future']
    experiment_titles = [
        'Binary Classification: Past vs (Present + Future)',
        'Binary Classification: Past vs Future (ignoring Present)'
    ]

    # Color mapping from year to color via a colormap
    cmap = plt.get_cmap('viridis')
    norm = plt.Normalize(vmin=year_start, vmax=year_end)

    for exp_key, exp_title in zip(experiments, experiment_titles):
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
        fig, ax = plt.subplots(figsize=(12, 8))
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
        exp_name = exp_key.replace('experiment1_', '').replace('experiment2_', '')
        fname = f"{model_name}_predictions_ce_by_year_{exp_name}_{year_start}_{year_end}.png"
        save_path = Path(output_dir) / fname
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")

def plot_prediction_ce_averages_over_checkpoints(model_name, checkpoints, year_start=1950, year_end=2050, output_dir="cross_entropy_prediction_avgs"):
    """Plot predictions-only average CE vs specified checkpoints.

    - Uses only model predictions (no training data) over [year_start, year_end].
    - Creates separate figures for experiment1 and experiment2.
    - Plots a single line with dot markers for the provided checkpoints.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    analyzer = AnalyzerClass()

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

    experiments = ['experiment1_past_vs_present_future', 'experiment2_past_vs_future']
    experiment_titles = [
        'Binary Classification: Past vs (Present + Future)',
        'Binary Classification: Past vs Future (ignoring Present)'
    ]

    for exp_key, exp_title in zip(experiments, experiment_titles):
        x_cps = []
        avgs = []

        for cp in cps:
            ce_pred = analyzer.compute_cross_entropy_over_range(dist_pred, model_name, cp, year_start, year_end)
            avg_loss = float(ce_pred[exp_key]['average_loss'])
            x_cps.append(cp)
            avgs.append(avg_loss)

        if not x_cps:
            print(f"No data to plot for {model_name} {exp_key}")
            continue

        fig, ax = plt.subplots(figsize=(12, 7))
        color = CROSS_ENTROPY_COLORS[0]
        ax.plot(x_cps, avgs, color=color, linestyle='-', marker='.', markersize=8, label="predictions")

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
        exp_name = exp_key.replace('experiment1_', '').replace('experiment2_', '')
        cps_str = f"{len(x_cps)}cps"
        fname = f"{model_name}_predictions_avg_ce_vs_checkpoint_{exp_name}_{year_start}_{year_end}_{cps_str}.png"
        save_path = Path(output_dir) / fname
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")

if __name__ == "__main__":
    # python kl_divergence_checkpoints.py
    
    
    # plot_training_data()
    # plot_model_predictions()
    # save_all_analyzer_data()
    # compute_cross_entropies()
    # plot_training_dynamics()
    # ce_over_training()
    ce_over_training_split()

    

    
