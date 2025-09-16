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



MODEL_DISPLAY_NAMES = {
    "pythia": "Pythia-1.4B-deduped",
    "EleutherAI_pythia-1b-deduped": "Pythia-1B-deduped",
    "EleutherAI_pythia-1.4b-deduped": "Pythia-1.4B-deduped",
    "olmo": "OLMo2-1B",
    "allenai_OLMo-2-0425-1B": "OLMo2-1B",
    "EleutherAI_pythia-6.9b-deduped": "Pythia-6.9B-deduped",
    "allenai_OLMo-2-1124-7B": "OLMo2-7B",
    "meta-llama_Llama-3.1-8B": "LLama3.1-8B",
}

# Training-derived data types that should show corpus names instead of model names
CO_OCCURR_NAME = "Co-occurrence model"
EXACT_STRING_MATCH_NAME = "Exact string match model"
NGRAM_NAME = "N-gram model"
NEXT_TOKEN_NAME = "Next-token predictions"
TRAINING_DERIVED_DISPLAY_TYPES = {CO_OCCURR_NAME, EXACT_STRING_MATCH_NAME, NGRAM_NAME}

# Consistent color mapping for cross-entropy plots
CROSS_ENTROPY_COLOR_MAPPING = {
    NEXT_TOKEN_NAME: "grey",
    NGRAM_NAME: "blue", 
    CO_OCCURR_NAME: "red",
}

MODEL_CORPUS_OVERRIDES = {
    "pythia": "The Pile",
    "olmo": "OLMo-mix-1124",
}

def get_model_display_name(model_key, data_type):
    """Resolve the display name for a model, with overrides for training-derived plots."""
    display_data_type = data_type
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

        # N-gram model distributions derived from in_year_there_word_counts
        self.olmo_relative_ngram = {}
        self.pythia_relative_ngram = {}
        self.laplace_alpha = 1.0

        # this is per future/present/past 
        self.olmo_relative_gold_distribution = {} # has entries 0 or 1 for past and future
        self.pythia_relative_gold_distribution = {} # see olmo


        print("loading olmo in_year_there_word_counts ") # always has past or presfut entries
        self.olmo_exact_string_match = self.load_training_data(OLMO_TRAINING_DATA_STRING_MATCHING, "in_year_there_word_counts", 250)
        print("loading olmo in_year_tense_sentence_counts") # always has past or presfut entries
        self.olmo_co_occurrence = self.load_training_data(OLMO_TRAINING_DATA_CO_OCCURRENCE, "in_year_tense_sentence_counts", 250)
        print("loading pythia in_year_there_word_counts") # always has past or presfut entries
        self.pythia_exact_string_match = self.load_training_data(PYTHIA_TRAINING_DATA, "in_year_there_word_counts", 1000)
        print("loading pythia in_year_tense_sentence_counts") # always has past or presfut entries
        self.pythia_co_occurrence = self.load_training_data(PYTHIA_TRAINING_DATA, "in_year_tense_sentence_counts", 1000)
        print("loading olmo predictions") 
        self.olmo_predictions = self.load_model_predictions(OLMO_PREDICTIONS_FILE, OLMO_CHECKPOINTS) # UP TO CHECKPOINT 10000; OUR OWN MODEL
        print("loading pythia predictions")
        self.pythia_predictions = self.load_model_predictions(PYTHIA_PREDICTIONS_FILE, PYTHIA_CHECKPOINTS) # UP TO CHECKPOINT 10000; NOT OUR MODEL
        print("populating gold data")
        self.populate_gold_data()

        print("loading olmo relative ngram") # always has past or presfut entries, add to 1.0; missing cutoff year
        self.olmo_relative_ngram = self._compute_laplace_smoothed_ngram(self.olmo_exact_string_match)
        print("loading pythia relative ngram") # always has past or presfut entries, add to 1.0; missing cutoff year
        self.pythia_relative_ngram = self._compute_laplace_smoothed_ngram(self.pythia_exact_string_match)
        print("finished loading all data")

        model_names = ["allenai_OLMo-2-0425-1B", "EleutherAI_pythia-1b-deduped", "EleutherAI_pythia-1.4b-deduped","EleutherAI_pythia-6.9b-deduped", "allenai_OLMo-2-1124-7B", "meta-llama_Llama-3.1-8B"]
        self.other_model_predictions = self.load_other_model_predictions(model_names)
        print("--------------------------------")
        print("finished loading all data")
        print("--------------------------------")




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
                        if verb in counts_per_verb: # exact string match only has entries if there are occurrences so we need this statement
                            year_to_counts_absolute[year][tense] += counts_per_verb[verb]
                    year_to_counts_absolute[year] = tense_counts_absolute

                results_absolute[cp_num] = year_to_counts_absolute
            return results_absolute

    def load_other_model_predictions(self, model_names):
        """Load predictions from other models with separate present and future tenses (not combined as presfut)."""
        model_name_to_predictions = {}
        # Hardcoded tense mapping for other models - keep present and future separate
        OTHER_MODEL_TENSE_MAPPING = {
            "was": "past",
            "were": "past", 
            "is": "present",
            "are": "present",
            "will": "future",
        }
        
        for model_name in model_names:
            print(f"Loading predictions for {model_name}")
            just_model_name = model_name.split("_")[-1]
            model_prediction_file = f"../olmo_predictions/model_predictions__In__year__there/{model_name}/{just_model_name}_predictions.json"
            
            if not os.path.exists(model_prediction_file):
                raise FileNotFoundError(f"could not find {model_prediction_file}")
                
            with open(model_prediction_file, "r") as f:
                data = json.load(f)

            year_to_counts_absolute = {}
            
            # Initialize all years with zero counts for past, present, future
            for year in range(TOTAL_YEARS[0], TOTAL_YEARS[1]):
                year_to_counts_absolute[str(year)] = {
                    "past": 0,
                    "present": 0,
                    "future": 0,
                }
            
            # Process the actual data
            for year in data.keys():
                if year == "metadata":
                    continue
                verb_count_dict = data[year]["absolute"]
                # Remove leading spaces from verb keys
                verb_count_dict_cleaned = {k.strip(): v for k, v in verb_count_dict.items()}
                
                for verb, tense in OTHER_MODEL_TENSE_MAPPING.items():
                    if verb in verb_count_dict_cleaned:
                        year_to_counts_absolute[year][tense] += verb_count_dict_cleaned[verb]

            # Store with "final" checkpoint key to match the expected format
            model_name_to_predictions[model_name] = {"final": year_to_counts_absolute}
        
        return model_name_to_predictions

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
                    year_dist = {"past": 1.0, "presfut": 0.0}
                if year > cutoff:
                    year_dist = {"past": 0.0, "presfut": 1.0}
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
    
    def _make_relative_distributions(self, dist_dict, separate_present_future=False):
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
                    if separate_present_future:
                        relative_dict[cp][year] = {"past": 0.0, "present": 0.0, "future": 0.0}
                    else:
                        relative_dict[cp][year] = {tense: 0.0 for tense in TENSE_ORDER}
        return relative_dict

    def bar_plot(self, dist_dict, model, data_type, checkpoint, year_start, year_end, make_relative=True, separate_present_future=False):
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
            separate_present_future: If True, plot past/present/future separately instead of past/presfut
        """
        if checkpoint not in dist_dict:
            raise ValueError(f"Checkpoint {checkpoint} not found in {dist_dict}")
        
        # Convert to relative distributions if requested
        if make_relative:
            dist_dict = self._make_relative_distributions(dist_dict, separate_present_future)
        
        output_dir = self._get_folder_name(model, checkpoint, data_type)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        cp_data = dist_dict[checkpoint]
        
        # Assert that distributions sum to 1 (with small tolerance for floating point errors)
        for year_str, year_data in cp_data.items():
            total = sum(year_data.values())
            if make_relative:
                assert abs(total - 1.0) < 1e-6 or total == 0.0, f"Distribution for year {year_str} sums to {total}, not 1.0: {year_data}"
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
        
        # Determine tense order and colors based on separate_present_future flag
        if separate_present_future:
            tense_order = ["past", "present", "future"]
            tense_colors = {"past": "orange", "present": "#4b0082", "future": "green"}
        else:
            tense_order = TENSE_ORDER
            tense_colors = TENSE_COLORS
        
        for tense in tense_order:
            vals = np.array([cp_data[y].get(tense, 0) for y in years])
            # Use "Present+Future" label for presfut, otherwise use title case
            label = "Present+Future" if tense == "presfut" else tense.title()
            ax.bar(range(len(years)), vals, bottom=bottom, 
                  label=label, color=tense_colors[tense], width=1.0)
            bottom += vals
        
        # Format - using same logic as kl_divergence_checkpoints.py but simplified
        display_data_type = data_type
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

    def bar_plots_for_checkpoints(self, dist_dict, model, data_type, checkpoints, rows, cols, year_start, year_end, subplot_width=3.5, subplot_height=2.2):
        """
        Create a grid of stacked bar plots, one subplot per checkpoint.
        Adapted for the new tense structure using 'past' and 'presfut'.

        Args:
            dist_dict: Dictionary containing distribution data (keyed by checkpoint -> year -> tense -> prob)
            model: Model name (str)
            data_type: Type of data being plotted (str)
            checkpoints: List of checkpoint integers to plot
            rows: Number of rows in the subplot grid
            cols: Number of columns in the subplot grid
            year_start: Start year (inclusive)
            year_end: End year (inclusive)
            subplot_width: Width multiplier for each subplot (default: 3.5)
            subplot_height: Height multiplier for each subplot (default: 2.2)
        """
        if len(checkpoints) > rows * cols:
            raise ValueError(f"Too many checkpoints ({len(checkpoints)}) for grid size {rows}×{cols}")

        # Convert to relative distributions
        dist_dict = self._make_relative_distributions(dist_dict)

        fig, axes = plt.subplots(rows, cols, figsize=(subplot_width * cols, subplot_height * rows))
        if rows == 1 and cols == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for idx, cp in enumerate(checkpoints):
            ax = axes[idx]

            cp_data = dist_dict[cp]
            
            # Assert that distributions sum to 1 (with small tolerance for floating point errors)
            for year_str, year_data in cp_data.items():
                total = sum(year_data.values())
                assert abs(total - 1.0) < 1e-6 or total == 0.0, f"Distribution for year {year_str} sums to {total}, not 1.0: {year_data}"
            
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
            for tense in TENSE_ORDER:
                vals = np.array([cp_data[y].get(tense, 0) for y in years])
                # Use "Present+Future" label for presfut, otherwise use title case
                label = "Present+Future" if tense == "presfut" else tense.title()
                ax.bar(range(len(years)), vals, bottom=bottom,
                       label=label if idx == 0 else "",
                       color=TENSE_COLORS[tense], width=1.0)
                bottom += vals

            # Format subplot
            ax.set_title(f"Checkpoint {cp}", fontsize=10)
            # Show years every 10 years; only label on bottom row to reduce clutter
            tick_indices = [i for i, y in enumerate(years) if int(y) % 10 == 0]
            if tick_indices:
                ax.set_xticks(tick_indices)
                if (idx // cols) == (rows - 1):
                    ax.set_xticklabels([years[i] for i in tick_indices], rotation=45, fontsize=9)
                else:
                    ax.set_xticklabels([])
            else:
                # Fallback if no years divisible by 10
                tick_step = max(1, len(years) // 10)
                ax.set_xticks(range(0, len(years), tick_step))
                if (idx // cols) == (rows - 1):
                    ax.set_xticklabels(years[::tick_step], rotation=45, fontsize=9)
                else:
                    ax.set_xticklabels([])
            if idx % cols == 0:
                ax.set_ylabel("Probability")
            ax.set_ylim(0, 1)
            ax.set_xlim(-0.5, len(years) - 0.5)
            ax.margins(0)
            ax.grid(True, alpha=0.3)

        # Hide any unused subplots
        for idx in range(len(checkpoints), len(axes)):
            axes[idx].set_visible(False)

        # Add a single legend for the whole figure with proper labels
        legend_labels = []
        for tense in TENSE_ORDER:
            if tense == "presfut":
                legend_labels.append("Present+Future")
            else:
                legend_labels.append(tense.title())
        
        # Add main title at the top
        model_display = get_model_display_name(model, data_type)
        
        # Dynamic layout tuning based on grid shape
        if rows == 1:
            # 1xN: give extra top margin so title never overlaps plots
            title_y = 0.99
            top_margin = 0.80
            pad = 2.0
        elif rows == 2:
            # 2xN: similar to 1xN but slightly more compact
            title_y = 0.985
            top_margin = 0.85
            pad = 2.0
        elif rows >= 6:
            # Tall grids (e.g., 8x5): keep title closer, give plots more space
            title_y = 0.98
            top_margin = 0.95
            pad = 2.2
        else:
            # Mid-size grids (3-5 rows)
            title_y = 0.975
            top_margin = 0.92
            pad = 2.0
        bottom_margin = 0.15
        hspace = 0.5 if rows > 2 else 0.4
        wspace = 0.3
        
        fig.suptitle(f"{model_display} — {data_type}", fontsize=16, y=title_y)
        
        # Apply layout with dynamic margins
        plt.tight_layout(pad=pad)
        plt.subplots_adjust(bottom=bottom_margin, top=top_margin, hspace=hspace, wspace=wspace)
        
        # Add legend at the bottom (dynamic vertical position)
        legend_y = 0.015 if rows == 1 else (0.12 if rows >= 6 else 0.0)
        fig.legend(legend_labels, loc='upper center', bbox_to_anchor=(0.5, legend_y), ncol=len(TENSE_ORDER), fontsize=12)

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

    def collect_years_with_no_data(self, dist_dict, checkpoint, year_start, year_end, model_name=""):
        """Print which years have no data in the given distribution dictionary.
        
        Args:
            dist_dict: Dictionary containing distribution data
            checkpoint: Checkpoint number to examine
            year_start: Start year (inclusive)
            year_end: End year (inclusive)
            model_name: Optional name for clearer output
        """
        if checkpoint not in dist_dict:
            print(f"Checkpoint {checkpoint} not found in distribution for {model_name}")
            return
        
        cp_data = dist_dict[checkpoint]
        years_with_no_data = []
        
        for year in range(year_start, year_end + 1):
            year_str = str(year)
            
            year_data = cp_data[year_str]
            total = sum(year_data.values())
            if total == 0:
                years_with_no_data.append(year)
        
        return years_with_no_data

    def compute_cross_entropy_over_range(self, dist_dict, model, checkpoint, year_start, year_end, allow_missing_data=False, specific_years=None):
        """Compute cross-entropy loss between model predictions and gold labels.
        
        Args:
            dist_dict: Dictionary containing distribution data (will be normalized to probabilities that sum to 1)
            model: Model name ("olmo" or "pythia")
            checkpoint: Checkpoint number
            year_start: Start year (inclusive)
            year_end: End year (inclusive)
            allow_missing_data: If True, skip years with no data instead of raising errors
            specific_years: Optional list of specific years to compute CE for (overrides year_start/year_end range)
            
        Returns:
            Dictionary with cross-entropy results including years_used
        """
        if checkpoint not in dist_dict:
            raise ValueError(f"Checkpoint {checkpoint} not found in {dist_dict}")
        
        # Normalize the input distribution to ensure probabilities sum to 1
        normalized_dist_dict = self._make_relative_distributions(dist_dict)
        
        # Get gold distribution for this model
        if model == "olmo":
            gold_cp = self.olmo_relative_gold_distribution[checkpoint]
        elif model == "pythia":
            gold_cp = self.pythia_relative_gold_distribution[checkpoint]
        else:
            raise ValueError(f"Invalid model: {model}")
        
        dist_cp = normalized_dist_dict[checkpoint]
        
        losses = {}
        years_used = []
        
        # Use specific years if provided, otherwise use the range
        if specific_years is not None:
            years_to_process = specific_years
        else:
            years_to_process = range(year_start, year_end + 1)
        
        for year in years_to_process:
            year_str = str(year)
            
            # Skip cutoff year
            if ((model == "olmo" and year == OLMO_CUTOFF) or 
                (model == "pythia" and year == PYTHIA_CUTOFF)):
                continue
                
            if year_str not in gold_cp:
                continue
                
            if year_str not in dist_cp:
                continue
            pred_data = dist_cp[year_str]
            gold_data = gold_cp[year_str]
            
            # Skip years with no prediction data
            if sum(pred_data.values()) == 0:
                if allow_missing_data:
                    continue
                raise ValueError(f"No prediction data for year {year_str}")
            
            # Convert to past vs future binary classification (new structure only)
            pred_past = pred_data["past"]
            pred_future = pred_data["presfut"]
            
            gold_past = gold_data["past"]
            gold_future = gold_data["presfut"]
            
            # Skip cases where predictions and gold are completely unrelated
            # (pred=1 for one class, gold=1 for the other class)
            if allow_missing_data:
                if ((pred_past == 1.0 and gold_past == 0.0) or 
                    (pred_future == 1.0 and gold_future == 0.0)):
                    continue
            
            # Compute cross-entropy: -sum(target * log(pred))
            epsilon = 1e-12  # Avoid log(0)
            ce_loss = -(gold_past * np.log(pred_past + epsilon) + 
                       gold_future * np.log(pred_future + epsilon))
            
            # Handle floating-point precision errors
            if ce_loss < 0 and abs(ce_loss) < 1e-10:
                ce_loss = 0.0
            
            if not np.isfinite(ce_loss):
                if allow_missing_data:
                    continue
                raise ValueError(f"Invalid cross-entropy loss for year {year_str}: {ce_loss}")
            
            losses[year_str] = ce_loss
            years_used.append(year_str)
        
        # Compute average loss
        avg_loss = sum(losses.values()) / len(losses)
        
        # Sanity check: print sample data for year 1950 if it exists
        # if "1950" in losses and "1950" in dist_cp and "1950" in gold_cp:
        #     print(f"Sanity check for {model} checkpoint {checkpoint}, year 1950:")
        #     print(f"  Normalized predictions: {dist_cp['1950']}")
        #     print(f"  Gold distribution: {gold_cp['1950']}")
        #     print(f"  Cross-entropy loss: {losses['1950']}")
        
        return {
            'per_year_losses': losses,
            'average_loss': avg_loss,
            'years_used': years_used,
        }


def plot_average_cross_entropies_over_checkpoints(analyzer, distributions_dict, model_name, checkpoints, year_start=1950, year_end=2050, output_dir="cross_entropy_over_checkpoints"):
    """Plot average cross-entropy losses over checkpoints for multiple distributions.
    
    Args:
        analyzer: AnalyzerClass instance
        distributions_dict: Dictionary mapping distribution names to distribution data dictionaries
        model_name: Model name ("olmo" or "pythia")
        checkpoints: List of checkpoint numbers to plot
        year_start: Start year (inclusive)
        year_end: End year (inclusive)
        output_dir: Directory to save the plot
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(9, 6))
    
    for dist_name, dist_dict in distributions_dict.items():
        # Only use checkpoints that exist in this distribution
        
        avg_losses = []
        used_checkpoints = []
        
        for cp in checkpoints:
            ce_result = analyzer.compute_cross_entropy_over_range(
                dist_dict, model_name, cp, year_start, year_end, allow_missing_data=True
            )
            avg_losses.append(ce_result['average_loss'])
            used_checkpoints.append(cp)
        
        # Use consistent colors from the cross-entropy color mapping
        color = CROSS_ENTROPY_COLOR_MAPPING.get(dist_name, "black")
        ax.plot(used_checkpoints, avg_losses, 
                label=dist_name, color=color, marker='o', linewidth=2, markersize=6)
                
        print(f"{dist_name}: {len(used_checkpoints)} checkpoints plotted")
    
    # Format the plot
    ax.set_xlabel('Checkpoint', fontsize=12)
    ax.set_ylabel('Average Cross-Entropy Loss', fontsize=12)
    model_display = MODEL_DISPLAY_NAMES.get(model_name, str(model_name))
    ax.set_title(f"{model_display} — Cross-Entropy vs Training Progress", fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Save the plot
    num_dists = len(distributions_dict)
    filename = f"{model_name}_avg_cross_entropy_over_checkpoints_{num_dists}dists_{year_start}_{year_end}.png"
    save_path = Path(output_dir) / filename
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Saved cross-entropy over checkpoints plot: {save_path}")


def plot_cross_entropies(ce_results_list, labels_list, model_name, year_start=1950, year_end=2050, output_dir="cross_entropy_plots"):
        """Plot cross-entropy losses as scatter plots for multiple distributions.
        
        Args:
            ce_results_list: List of cross-entropy result dictionaries from compute_cross_entropy_over_range
            labels_list: List of labels for each distribution
            model_name: Model name for the plot title and filename
            year_start: Start year (inclusive)
            year_end: End year (inclusive)
            output_dir: Directory to save the plots
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(4.4, 3))
        
        all_years_plotted = []
        all_losses_plotted = []
        
        for ce_result, label in zip(ce_results_list, labels_list):
            per_year_losses = ce_result['per_year_losses']
            
            # Filter years and prepare data for plotting
            years = []
            losses = []
            
            for year_str in ce_result['years_used']:
                year_int = int(year_str)
                # Filter by year_start and year_end
                if year_start <= year_int <= year_end:
                    years.append(year_int)
                    losses.append(per_year_losses[year_str])
            
            ax.scatter(years, losses, label=label, color=CROSS_ENTROPY_COLOR_MAPPING[label], s=2, alpha=1.0)
            
            # Collect all plotted data for axis limits
            all_years_plotted.extend(years)
            all_losses_plotted.extend(losses)
        
        # Format the plot
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Cross-Entropy Loss', fontsize=12)
        model_display = MODEL_DISPLAY_NAMES.get(model_name, str(model_name))
        ax.set_title(f"{model_display}", fontsize=14)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Set axis limits based on actual plotted data
        if all_years_plotted and all_losses_plotted:
            # X-axis: set to requested range with some padding
            ax.set_xlim(year_start - 5, year_end + 5)
            
            # Y-axis: set based on actual data with some padding
            y_min, y_max = min(all_losses_plotted), max(all_losses_plotted)
            y_range = y_max - y_min
            y_padding = max(0.05 * y_range, 0.01)  # At least 5% padding or 0.01 units
            ax.set_ylim(max(0, y_min - y_padding), y_max + y_padding)
        
        # Save the plot
        num_dists = len(labels_list)
        filename = f"{model_name}_cross_entropy_{num_dists}dists_{year_start}_{year_end}.png"
        save_path = Path(output_dir) / filename
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"Saved cross-entropy plot: {save_path}")


if __name__ == "__main__":
    # python get_paper_results.py
    cp = 10000
    start_year = 1950
    years_end = 2050

    analyzer = AnalyzerClass()

    
    # filepath = analyzer.save_all_data_to_file()
    # print(f"Data export completed. File saved: {filepath}")

    # # plot_training_data
    # analyzer.bar_plot(analyzer.olmo_co_occurrence, "olmo", CO_OCCURR_NAME, cp, start_year, years_end)
    # analyzer.bar_plot(analyzer.olmo_exact_string_match, "olmo", EXACT_STRING_MATCH_NAME, cp, start_year, years_end)
    # analyzer.bar_plot(analyzer.pythia_co_occurrence, "pythia", CO_OCCURR_NAME, cp, start_year, years_end)
    # analyzer.bar_plot(analyzer.pythia_exact_string_match, "pythia", EXACT_STRING_MATCH_NAME, cp, start_year, years_end)
    # analyzer.bar_plot(analyzer.olmo_relative_ngram, "olmo", NGRAM_NAME, cp, start_year, years_end)
    # analyzer.bar_plot(analyzer.pythia_relative_ngram, "pythia", NGRAM_NAME, cp, start_year, years_end)

    # # plot_model_predictions
    # analyzer.bar_plot(analyzer.olmo_predictions, "olmo", NEXT_TOKEN_NAME, cp, start_year, years_end)
    # analyzer.bar_plot(analyzer.pythia_predictions, "pythia", NEXT_TOKEN_NAME, cp, start_year, years_end)

    # # plot_model_predictions with other models
    # for model_name in analyzer.other_model_predictions.keys():
    #     analyzer.bar_plot(analyzer.other_model_predictions[model_name], model_name, NEXT_TOKEN_NAME, "final", start_year, years_end, make_relative=False, separate_present_future=True)


    # # compute losses for 10k checkpoint
    # olmo_pred_loss = analyzer.compute_cross_entropy_over_range(analyzer.olmo_predictions, "olmo", cp, start_year, years_end)
    # olmo_co_occurrence_loss = analyzer.compute_cross_entropy_over_range(analyzer.olmo_co_occurrence, "olmo", cp, start_year, years_end)
    # olmo_ngram_loss = analyzer.compute_cross_entropy_over_range(analyzer.olmo_relative_ngram, "olmo", cp, start_year, years_end)
    # olmo_exact_string_match_loss = analyzer.compute_cross_entropy_over_range(analyzer.olmo_exact_string_match, "olmo", cp, start_year, years_end, allow_missing_data=True)
    # print(f"Olmo predictions average loss: {olmo_pred_loss['average_loss']}")
    # print(f"Olmo co-occurrence average loss: {olmo_co_occurrence_loss['average_loss']}")
    # print(f"Olmo n-gram average loss: {olmo_ngram_loss['average_loss']}")
    # print(f"Olmo exact string match average loss (years used: {len(olmo_exact_string_match_loss['years_used'])}): {olmo_exact_string_match_loss['average_loss']}")
    # print("--------------------------------")
    
    # pythia_pred_loss = analyzer.compute_cross_entropy_over_range(analyzer.pythia_predictions, "pythia", cp, start_year, years_end)
    # pythia_co_occurrence_loss = analyzer.compute_cross_entropy_over_range(analyzer.pythia_co_occurrence, "pythia", cp, start_year, years_end)
    # pythia_ngram_loss = analyzer.compute_cross_entropy_over_range(analyzer.pythia_relative_ngram, "pythia", cp, start_year, years_end)
    # pythia_exact_string_match_loss = analyzer.compute_cross_entropy_over_range(analyzer.pythia_exact_string_match, "pythia", cp, start_year, years_end, allow_missing_data=True)
    # print(f"Pythia predictions average loss: {pythia_pred_loss['average_loss']}")
    # print(f"Pythia co-occurrence average loss: {pythia_co_occurrence_loss['average_loss']}")
    # print(f"Pythia n-gram average loss: {pythia_ngram_loss['average_loss']}")
    # print(f"Pythia exact string match average loss (years used: {len(pythia_exact_string_match_loss['years_used'])}): {pythia_exact_string_match_loss['average_loss']}")
    # print("--------------------------------")
    # # print summary of average losses for 10k checkpoint
    # olmo_exact_str_years = analyzer.collect_years_with_no_data(analyzer.olmo_exact_string_match, cp, start_year, years_end, "OLMo exact string match")
    # olmo_pred_loss_less_data = analyzer.compute_cross_entropy_over_range(analyzer.olmo_predictions, "olmo", cp, start_year, years_end, specific_years=olmo_exact_str_years)
    # print(f"Olmo predictions average loss (years used: {len(olmo_exact_str_years)}): {olmo_pred_loss_less_data['average_loss']}")
    # print("--------------------------------")
    
    # # print summary of loss if only using ones frome xact string match for each metods
    # pythia_exact_str_years = analyzer.collect_years_with_no_data(analyzer.pythia_exact_string_match, cp, start_year, years_end, "Pythia exact string match")
    # pythia_pred_loss_less_data = analyzer.compute_cross_entropy_over_range(analyzer.pythia_predictions, "pythia", cp, start_year, years_end, specific_years=pythia_exact_str_years)
    # print(f"Pythia predictions average loss (years used: {len(pythia_exact_str_years)}): {pythia_pred_loss_less_data['average_loss']}")
    # print("--------------------------------")

    # # plot cross-entropy losses
    # plot_cross_entropies([olmo_pred_loss, olmo_co_occurrence_loss, olmo_ngram_loss], [NEXT_TOKEN_NAME, CO_OCCURR_NAME, NGRAM_NAME], "olmo", start_year, years_end)
    # plot_cross_entropies([pythia_pred_loss, pythia_co_occurrence_loss, pythia_ngram_loss], [NEXT_TOKEN_NAME, CO_OCCURR_NAME, NGRAM_NAME], "pythia", start_year, years_end)
    

    analyzer.bar_plots_for_checkpoints(analyzer.olmo_predictions, "olmo", NEXT_TOKEN_NAME, [1000, 3000, 7000], 1, 3, start_year, years_end, subplot_width=4.5, subplot_height=2.6)
    analyzer.bar_plots_for_checkpoints(analyzer.pythia_predictions, "pythia", NEXT_TOKEN_NAME, [1000, 3000, 7000], 1, 3, start_year, years_end, subplot_width=4.5, subplot_height=2.6)
    
    # # Generate checkpoint grid plots (APPENDIX)
    analyzer.bar_plots_for_checkpoints(analyzer.olmo_predictions, "olmo", NEXT_TOKEN_NAME, [250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 3250, 3500, 3750, 4000, 4250, 4500, 4750, 5000, 5250, 5500, 5750, 6000, 6250, 6500, 6750, 7000, 7250, 7500, 7750, 8000, 8250, 8500, 8750, 9000, 9250, 9500, 9750, 10000], 8, 5, start_year, years_end)
    analyzer.bar_plots_for_checkpoints(analyzer.olmo_co_occurrence, "olmo", CO_OCCURR_NAME, [260, 500, 760, 1000, 1260, 1500, 1760, 2000, 2260, 2500, 2760, 3000, 3260, 3500, 3760, 4000, 4260, 4500, 4760, 5000, 5260, 5500, 5760, 6000, 6260, 6500, 6760, 7000, 7260, 7500, 7760, 8000, 8260, 8500, 8760, 9000, 9260, 9500, 9760, 10000], 8, 5, start_year, years_end)
    analyzer.bar_plots_for_checkpoints(analyzer.olmo_relative_ngram, "olmo", NGRAM_NAME, [260, 500, 760, 1000, 1260, 1500, 1760, 2000, 2260, 2500, 2760, 3000, 3260, 3500, 3760, 4000, 4260, 4500, 4760, 5000, 5260, 5500, 5760, 6000, 6260, 6500, 6760, 7000, 7260, 7500, 7760, 8000, 8260, 8500, 8760, 9000, 9260, 9500, 9760, 10000], 8, 5, start_year, years_end)
    analyzer.bar_plots_for_checkpoints(analyzer.pythia_predictions, "pythia", NEXT_TOKEN_NAME, [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000], 2, 5, start_year, years_end)
    analyzer.bar_plots_for_checkpoints(analyzer.pythia_co_occurrence, "pythia", CO_OCCURR_NAME, [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000], 2, 5, start_year, years_end)
    analyzer.bar_plots_for_checkpoints(analyzer.pythia_relative_ngram, "pythia", NGRAM_NAME, [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000], 2, 5, start_year, years_end)
    
    
    # # Plot average cross-entropies over checkpoints
    # olmo_distributions = {
    #     NEXT_TOKEN_NAME: analyzer.olmo_predictions,
    #     CO_OCCURR_NAME: analyzer.olmo_co_occurrence,
    #     NGRAM_NAME: analyzer.olmo_relative_ngram,
    # }
    # plot_average_cross_entropies_over_checkpoints(analyzer, olmo_distributions, "olmo", OLMO_CHECKPOINTS, start_year, years_end)
    
    # pythia_distributions = {
    #     NEXT_TOKEN_NAME: analyzer.pythia_predictions,
    #     CO_OCCURR_NAME: analyzer.pythia_co_occurrence,
    #     NGRAM_NAME: analyzer.pythia_relative_ngram,
    # }
    # plot_average_cross_entropies_over_checkpoints(analyzer, pythia_distributions, "pythia", PYTHIA_CHECKPOINTS, start_year, years_end)
     
         
