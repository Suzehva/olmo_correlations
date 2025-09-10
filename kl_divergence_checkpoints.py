import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from scipy.stats import spearmanr
import torch.nn.functional as F
from matplotlib.lines import Line2D


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
TOTAL_YEARS = (1000, 3000) # goes until 2999

class AnalyzerClass:
    def __init__(self):
        # these all stored per tense (past/present/future), not per verb
        self.olmo_training_data = {}
        self.olmo_relative_training_data = {}
        self.olmo_predictions = {}
        self.olmo_relative_predictions = {}

        self.pythia_training_data = {}
        self.pythia_relative_training_data = {}
        self.pythia_predictions = {}
        self.pythia_relative_predictions = {}

        # this is per future/present/past 
        self.olmo_gold_distribution = {}
        self.pythia_gold_distribution = {}

        print("loading all data")
        self.load_training_data()
        self.load_model_predictions()
        self.populate_gold_data(olmo_cutoff=2024, pythia_cutoff=2020)
        print("finished loading all data")
    
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
                for year, counts in data[data_source].items():
                    # Group by tense categories
                    tense_counts_absolute = {}
                    for verb, tense in TENSE_MAPPING.items():
                        if verb in counts:
                            tense_counts_absolute.setdefault(tense, 0)
                            tense_counts_absolute[tense] += counts[verb]
                    
                    total_absolute_counts = sum(tense_counts_absolute.values())
                    
                    if total_absolute_counts > 0:
                        tense_counts_relative = {tense: count/total_absolute_counts for tense, count in tense_counts_absolute.items()}
                    else:
                        tense_counts_relative = {tense: 0.0 for tense in set(TENSE_MAPPING.values())}
                    
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
            
            total_absolute_counts = sum(tense_counts_absolute.values())
            if total_absolute_counts > 0:
                tense_counts_relative = {tense: count / total_absolute_counts for tense, count in tense_counts_absolute.items()}
            else:
                tense_counts_relative = {tense: 0.0 for tense in set(TENSE_MAPPING.values())}
        
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


    def populate_gold_data(self, olmo_cutoff, pythia_cutoff):
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
            self.olmo_gold_distribution[cp] = populate_gold_distribution(olmo_cutoff)
        for cp in pythia_cps:
            self.pythia_gold_distribution[cp] = populate_gold_distribution(pythia_cutoff)



    # # --- PLOTTING FUNCTIONS ---

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
        filename = f"{model}_checkpoint{checkpoint}_{data_type}"
        
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


    # def plot_stacked_grid_over_checkpoints(
    #     self, dist_dict, checkpoints, output_dir="single_dist_grid",
    #     label="distribution", n_cols=5, n_rows=8
    # ):
    #     """
    #     Plot stacked bars for a single distribution over multiple checkpoints
    #     in a grid (n_cols x n_rows). Only show overall y-axis label and title.
    #     """
    #     print(f"Plotting stacked distribution grid: {label}")

    #     Path(output_dir).mkdir(parents=True, exist_ok=True)

    #     tenses_to_plot = [t for t in self.TENSE_ORDER if t in set(self.tense_mapping.values())]

    #     fig, axes = plt.subplots(
    #         n_rows, n_cols, figsize=(n_cols*3, n_rows*2.5), sharey=True
    #     )
    #     axes = axes.flatten()

    #     for i, ckpt in enumerate(checkpoints):
    #         ax = axes[i]
    #         if ckpt not in dist_dict:
    #             print(f"No data for checkpoint {ckpt}")
    #             ax.axis('off')
    #             continue

    #         cp_data = dist_dict[ckpt]
    #         years = sorted(cp_data.keys())
    #         ind = np.arange(len(years))
    #         bottom = np.zeros(len(years))

    #         for tense in tenses_to_plot:
    #             vals = np.array([cp_data[y].get(tense, 0) for y in years])
    #             ax.bar(ind, vals, bottom=bottom, label=tense, color=self.TENSE_COLORS[tense])
    #             bottom += vals

    #         ax.set_title(f"Checkpoint {ckpt}", fontsize=8)
    #         ax.axis('off')  # hide x/y ticks

    #     # Hide any extra subplots
    #     for j in range(i+1, n_cols*n_rows):
    #         axes[j].axis('off')

    #     # Overall labels and title
    #     fig.suptitle(
    #         f"{label} | Checkpoints {checkpoints[0]}-{checkpoints[-1]} | Years {self.year_range[0]}-{self.year_range[1]}",
    #         fontsize=14
    #     )
    #     fig.text(0.03, 0.5, "Probability", va='center', rotation='vertical', fontsize=12)

    #     # Adjust spacing to reduce outer margins but keep subplots separated
    #     plt.subplots_adjust(
    #         left=0.06, right=0.99, top=0.93, bottom=0.05,
    #         hspace=0.3, wspace=0.25
    #     )

    #     save_path = f"{output_dir}/{label}_stacked_grid_{checkpoints[0]}_{checkpoints[-1]}.png"
    #     plt.savefig(save_path, dpi=300)
    #     plt.close()



    # def plot_spearman_sliding_window(self, training_dict, tense, checkpoint, window=20, output_dir="spearman", label="exact_str_matching"):
    #     """
    #     Compute Spearman rank correlation between a training dict and model predictions
    #     for a specific tense over a sliding window of years.

    #     training_dict: dict like self.relative_training_data[...] or self.relative_human_gold
    #     tense: "past", "present", or "future"
    #     checkpoint: int checkpoint to use from model predictions
    #     window: size of sliding window in years
    #     """
    #     if tense not in self.TENSE_ORDER:
    #         raise ValueError(f"Tense '{tense}' not in defined tenses: {self.TENSE_ORDER}")
        
    #     if checkpoint not in self.relative_model_data:
    #         raise ValueError(f"Checkpoint {checkpoint} not available in model data")

    #     model_cp_data = self.relative_model_data[checkpoint]
    #     train_cp_data = training_dict[checkpoint]

    #     years = sorted(train_cp_data.keys())
    #     spearman_vals = []
    #     window_starts = []


    #     # inside your loop over windows
    #     for i in range(len(years) - window + 1):
    #         win_years = years[i:i+window]

    #         train_vals = [train_cp_data[y].get(tense, 0) for y in win_years]
    #         model_vals = [model_cp_data[y].get(tense, 0) for y in win_years]

    #         rho, _ = spearmanr(train_vals, model_vals)
    #         spearman_vals.append(rho)

    #         # Use center year instead of starting year
    #         center_year = int((int(win_years[0]) + int(win_years[-1])) / 2)
    #         window_starts.append(center_year)

    #     # Then plot as before
    #     plt.figure(figsize=(12, 5))
    #     plt.plot(window_starts, spearman_vals, marker='o', color=self.TENSE_COLORS[tense])
    #     plt.xlabel(f"Center Year of {window}-Year Window")
    #     plt.ylabel("Spearman Rank Correlation")
    #     plt.title(f"Spearman Rank Correlation ({tense}) | Checkpoint {checkpoint} | Counting method {label}")
    #     plt.grid(True)

    #     Path(f"{output_dir}").mkdir(parents=True, exist_ok=True)
    #     plt.tight_layout()
    #     filename = f"{output_dir}/{self.model_name}_spearman_{tense}_{label}_ckpt{checkpoint}_window{window}_years{self.year_range[0]}-{self.year_range[1]}.png"
    #     plt.savefig(filename, dpi=300)
    #     plt.close()


    # def compute_ce_loss_single(self, input_dict, checkpoint, label="training", compute_binary_losses=False):
    #     """
    #     Compute cross-entropy loss between predicted distribution (input_dict) 
    #     and human gold distribution (self.relative_human_gold).
    #     input_dict: dict like self.relative_training_data[...] or self.relative_model_data
    #     checkpoint: int
    #     compute_binary_losses: if True, also compute binary losses for each tense vs non-tense
    #     """
    #     gold_cp_data = self.relative_human_gold[checkpoint]
    #     cp_data = input_dict[checkpoint]
        
    #     # Get active tenses from tense_mapping
    #     active_tenses = sorted(set(self.tense_mapping.values()))
        
    #     probs = []
    #     gold = []
    #     years_used = 0
    #     years_skipped = 0

    #     for year in range(self.year_range[0], self.year_range[1] + 1):
    #         year_str = str(year)

    #         # Predicted distribution (normalize if necessary)
    #         d = cp_data.get(year_str, {tense: 0 for tense in active_tenses})
    #         s = sum(d.get(tense, 0) for tense in active_tenses)
    #         if s == 0:
    #             years_skipped += 1
    #             continue  # Skip this year if no data
    #         else:
    #             pred_dist = [d.get(tense, 0)/s for tense in active_tenses]
            
    #         # Gold distribution (already normalized)
    #         g = gold_cp_data.get(year_str, {tense: 0 for tense in active_tenses})
    #         gold_dist = [g.get(tense, 0) for tense in active_tenses]
            
    #         probs.append(pred_dist)
    #         gold.append(gold_dist)
    #         years_used += 1
        
    #     # Convert to tensors and compute loss
    #     probs_tensor = torch.tensor(probs, dtype=torch.float32)
    #     gold_tensor = torch.tensor(gold, dtype=torch.float32)
        
    #     log_probs = torch.log(probs_tensor + 1e-12)
    #     loss = -(gold_tensor * log_probs).sum(dim=1).mean()

    #     print(f"{label} | Checkpoint {checkpoint} | CE Loss: {loss:.4f} | Years used: {years_used} | Years skipped: {years_skipped}")
        
    #     # Compute binary losses if requested
    #     if compute_binary_losses:
    #         for i, target_tense in enumerate(active_tenses):
    #             # Create binary distributions: target tense vs all others
    #             binary_probs = []
    #             binary_gold = []
                
    #             for pred_row, gold_row in zip(probs, gold):
    #                 # Binary predicted: [target_tense_prob, sum_of_other_tenses_prob]
    #                 target_prob = pred_row[i]
    #                 other_prob = sum(pred_row[j] for j in range(len(pred_row)) if j != i)
    #                 binary_pred = [target_prob, other_prob]
                    
    #                 # Binary gold: [target_tense_gold, sum_of_other_tenses_gold]
    #                 target_gold = gold_row[i]
    #                 other_gold = sum(gold_row[j] for j in range(len(gold_row)) if j != i)
    #                 binary_gold_dist = [target_gold, other_gold]
                    
    #                 binary_probs.append(binary_pred)
    #                 binary_gold.append(binary_gold_dist)
                
    #             # Compute binary CE loss
    #             binary_probs_tensor = torch.tensor(binary_probs, dtype=torch.float32)
    #             binary_gold_tensor = torch.tensor(binary_gold, dtype=torch.float32)
                
    #             binary_log_probs = torch.log(binary_probs_tensor + 1e-12)
    #             binary_loss = -(binary_gold_tensor * binary_log_probs).sum(dim=1).mean()
                
    #             print(f"{label} | Checkpoint {checkpoint} | Binary CE Loss ({target_tense} vs non-{target_tense}): {binary_loss:.4f}")
        
    #     return loss.item()


    # def compute_spearman_window(self, dict1, dict2, checkpoint, start_year, window_size):
    #     """
    #     Compute Spearman correlation between two distributions over a sliding window.
    #     """
    #     years = [str(y) for y in range(start_year, start_year + window_size)]
    #     d1_vals, d2_vals = [], []

    #     for y in years:
    #         d1 = dict1[checkpoint].get(y, {"past": 0, "future": 0})
    #         d2 = dict2[checkpoint].get(y, {"past": 0, "future": 0})
    #         d1_vals.append([d1["past"], d1["future"]])
    #         d2_vals.append([d2["past"], d2["future"]])

    #     # flatten: (past + future) across window
    #     d1_flat = [v for pair in d1_vals for v in pair]
    #     d2_flat = [v for pair in d2_vals for v in pair]

    #     rho, _ = spearmanr(d1_flat, d2_flat)
    #     return rho if rho == rho else None  # return None if nan


    # def plot_spearman_over_checkpoints(
    #     self, dict1, dict2, window_size, start_years,
    #     label1="data 1", label2="data 2", output_dir="spearman_cp"
    # ):
    #     """
    #     Plot Spearman rank correlation over checkpoints for different start years (sliding windows).
    #     Skip points where rho is None.
    #     Use different marker shapes for start years before/after 2022.
    #     """
    #     checkpoints = sorted(dict1.keys())
    #     colors = plt.cm.tab10(np.linspace(0, 1, len(start_years)))

    #     plt.figure(figsize=(12, 6))

    #     for color, start_year in zip(colors, start_years):
    #         spearmans = []
    #         valid_cps = []
    #         for cp in checkpoints:
    #             rho = self.compute_spearman_window(dict1, dict2, cp, start_year, window_size)
    #             if rho is not None:
    #                 spearmans.append(rho)
    #                 valid_cps.append(cp)

    #         if spearmans:
    #             marker_shape = "s" if start_year >= 2022 else "o"  # square for future, circle for past
    #             plt.plot(valid_cps, spearmans, marker=marker_shape, color=color,
    #                     label=f"{start_year} - {start_year+window_size}", linestyle='-')

    #     plt.xlabel("Checkpoint")
    #     plt.ylabel("Spearman correlation")
    #     plt.title(f"Spearman correlation over checkpoints ({label1} vs {label2})")
    #     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    #     plt.ylim(-1, 1)

    #     os.makedirs(f"{output_dir}", exist_ok=True)
    #     save_path = f"{output_dir}/{self.model_name}_spearman_checkpoints_comparing_{label1}_{label2}_window{window_size}_years{self.year_range[0]}-{self.year_range[1]}.png"
    #     plt.savefig(save_path, dpi=300)
    #     plt.close()


    # def plot_avg_ce_over_checkpoints(self, dict1, dict2, window_size, start_years, 
    #                                 label1="data 1", label2="data 2", output_dir="ce_cp"):
    #     """
    #     Plot average cross-entropy over checkpoints for different start years (sliding windows).
    #     Only show first/last datalines in legend + marker shape meaning.
    #     """
    #     checkpoints = sorted(dict1.keys())

    #     cmap = plt.get_cmap("rainbow")
    #     colors = cmap(np.linspace(0, 1, len(start_years)))[::-1]

    #     plt.figure(figsize=(14, 6))

    #     handles, labels = [], []  # collect for custom legend

    #     for idx, (color, start_year) in enumerate(zip(colors, start_years)):
    #         avg_ces, valid_cps = [], []

    #         for cp in checkpoints:
    #             ce_values = []
    #             for year in range(start_year, start_year + window_size):
    #                 year_str = str(year)

    #                 # Dist 1
    #                 d1 = dict1[cp].get(year_str, {"past":0,"present":0,"future":0})
    #                 s1 = sum(d1.values())
    #                 dist1_tensor = torch.tensor(
    #                     [d1.get("past",0)/s1 if s1 else 1.0,
    #                     d1.get("present",0)/s1 if s1 else 0.0,
    #                     d1.get("future",0)/s1 if s1 else 0.0], dtype=torch.float32)

    #                 # Dist 2
    #                 d2 = dict2[cp].get(year_str, {"past":0,"present":0,"future":0})
    #                 s2 = sum(d2.values())
    #                 dist2_tensor = torch.tensor(
    #                     [d2.get("past",0)/s2 if s2 else 1.0,
    #                     d2.get("present",0)/s2 if s2 else 0.0,
    #                     d2.get("future",0)/s2 if s2 else 0.0], dtype=torch.float32)

    #                 ce = -(dist2_tensor * torch.log(dist1_tensor + 1e-12)).sum().item()
    #                 ce_values.append(ce)

    #             if ce_values:
    #                 avg_ces.append(np.mean(ce_values))
    #                 valid_cps.append(cp)

    #         if avg_ces:
    #             marker_shape = "s" if start_year >= 2022 else "o"
    #             line, = plt.plot(valid_cps, avg_ces, marker=marker_shape, 
    #                             color=color, linestyle='-')

    #             # Only add legend entry for first and last dataline
    #             if idx == 0 or idx == len(start_years)-1:
    #                 if window_size == 1:
    #                     legend_label = f"{start_year}"
    #                 else:
    #                     legend_label = f"{start_year} - {start_year + window_size}"
    #                 labels.append(legend_label)
    #                 handles.append(line)

    #     # Add marker-shape legend entries
    #     custom_handles = [
    #         Line2D([0], [0], color="k", marker="o", linestyle="", label="Past"),
    #         Line2D([0], [0], color="k", marker="s", linestyle="", label="Future"),
    #     ]
    #     handles.extend(custom_handles)
    #     labels.extend([h.get_label() for h in custom_handles])

    #     plt.legend(handles, labels,
    #             bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)

    #     plt.xlabel("Checkpoint")
    #     plt.ylabel("Average Cross-Entropy")
    #     plt.title(f"Average CE over checkpoints ({label1} vs {label2})")
    #     plt.ylim(0, 30)
    #     plt.grid(True)

    #     os.makedirs(f"{output_dir}", exist_ok=True)
    #     save_path = f"{output_dir}/avg_ce_{label1}_{label2}_window_{window_size}_years_{self.year_range[0]}_{self.year_range[1]}.png"
    #     plt.savefig(save_path, dpi=300, bbox_inches="tight")
    #     plt.close()

    # def plot_avg_ce_past_future_checkpoints(self, dict1, dict2, window_size, start_years,
    #                                         label1="data 1", label2="data 2", output_dir="ce_cp"):
    #     """
    #     Plot average cross-entropy over checkpoints for all start years collapsed into two lines:
    #     one for 'past' (<2022) and one for 'future' (>=2022).
    #     Legend includes year range and count.
    #     """
    #     checkpoints = sorted(dict1.keys())
    #     groups = {"past": [], "future": []}

    #     # separate years into past/future groups
    #     for start_year in start_years:
    #         if start_year >= 2022:
    #             groups["future"].append(start_year)
    #         else:
    #             groups["past"].append(start_year)

    #     plt.figure(figsize=(14, 6))

    #     for group_name, years in groups.items():
    #         if not years:
    #             continue

    #         years = sorted(years)
    #         group_ces = {cp: [] for cp in checkpoints}

    #         for start_year in years:
    #             for cp in checkpoints:
    #                 ce_values = []
    #                 for year in range(start_year, start_year + window_size):
    #                     year_str = str(year)

    #                     # Dist 1
    #                     d1 = dict1[cp].get(year_str, {"past":0,"present":0,"future":0})
    #                     s1 = sum(d1.values())
    #                     dist1_tensor = torch.tensor(
    #                         [d1.get("past",0)/s1 if s1 else 1.0,
    #                         d1.get("present",0)/s1 if s1 else 0.0,
    #                         d1.get("future",0)/s1 if s1 else 0.0], dtype=torch.float32)

    #                     # Dist 2
    #                     d2 = dict2[cp].get(year_str, {"past":0,"present":0,"future":0})
    #                     s2 = sum(d2.values())
    #                     dist2_tensor = torch.tensor(
    #                         [d2.get("past",0)/s2 if s2 else 1.0,
    #                         d2.get("present",0)/s2 if s2 else 0.0,
    #                         d2.get("future",0)/s2 if s2 else 0.0], dtype=torch.float32)

    #                     ce = -(dist2_tensor * torch.log(dist1_tensor + 1e-12)).sum().item()
    #                     ce_values.append(ce)

    #                 if ce_values:
    #                     group_ces[cp].append(np.mean(ce_values))

    #         # Average across all start years in this group
    #         avg_ces = []
    #         valid_cps = []
    #         for cp, vals in group_ces.items():
    #             if vals:
    #                 avg_ces.append(np.mean(vals))
    #                 valid_cps.append(cp)

    #         if avg_ces:
    #             marker_shape = "s" if group_name == "future" else "o"
    #             color = "red" if group_name == "future" else "blue"

    #             # legend label with range + count
    #             label = f"{group_name.capitalize()} ({years[0]}–{years[-1]}, avg of {len(years)} years)"

    #             plt.plot(valid_cps, avg_ces, marker=marker_shape, color=color, label=label)

    #     plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)
    #     plt.xlabel("Checkpoint")
    #     plt.ylabel("Average Cross-Entropy")
    #     plt.title(f"Average CE over checkpoints ({label1} vs {label2}) — Past vs Future")
    #     plt.ylim(0, 30)
    #     plt.grid(True)

    #     os.makedirs(f"{output_dir}", exist_ok=True)
    #     save_path = f"{output_dir}/avg_ce_{label1}_{label2}_window_{window_size}_past_future.png"
    #     plt.savefig(save_path, dpi=300, bbox_inches="tight")
    #     plt.close()


    # def plot_ce_vs_gold(self, dist_a, dist_b, gold_dist, checkpoint, output_dir="ce_plots", labels=("A vs Gold", "B vs Gold")):
    #     """
    #     Plot CE loss per year for dist_a and dist_b against a gold distribution on the same plot.

    #     dist_a, dist_b, gold_dist: dict[checkpoint][year] -> {"past":..., "present":..., "future":...}
    #     checkpoint: int
    #     labels: tuple of labels for the legend
    #     """
    #     ce_a = []
    #     ce_b = []
    #     valid_years = []

    #     for year in range(self.year_range[0], self.year_range[1] + 1):
    #         year_str = str(year)

    #         # Gold distribution
    #         d_gold = gold_dist[checkpoint].get(year_str, {"past":0, "present":0, "future":0})
    #         s_gold = sum(d_gold.values())
    #         if s_gold == 0:
    #             continue  # skip year entirely
    #         gold_tensor = torch.tensor([
    #             d_gold.get("past", 0)/s_gold,
    #             d_gold.get("present", 0)/s_gold,
    #             d_gold.get("future", 0)/s_gold
    #         ], dtype=torch.float32)

    #         # Distribution A
    #         d_a = dist_a[checkpoint].get(year_str, {"past":0, "present":0, "future":0})
    #         s_a = sum(d_a.values())
    #         if s_a == 0:
    #             continue
    #         a_tensor = torch.tensor([
    #             d_a.get("past", 0)/s_a,
    #             d_a.get("present", 0)/s_a,
    #             d_a.get("future", 0)/s_a
    #         ], dtype=torch.float32)

    #         # Distribution B
    #         d_b = dist_b[checkpoint].get(year_str, {"past":0, "present":0, "future":0})
    #         s_b = sum(d_b.values())
    #         if s_b == 0:
    #             continue
    #         b_tensor = torch.tensor([
    #             d_b.get("past", 0)/s_b,
    #             d_b.get("present", 0)/s_b,
    #             d_b.get("future", 0)/s_b
    #         ], dtype=torch.float32)

    #         # CE (gold as target)
    #         ce_a.append(-(gold_tensor * torch.log(a_tensor + 1e-12)).sum().item())
    #         ce_b.append(-(gold_tensor * torch.log(b_tensor + 1e-12)).sum().item())
    #         valid_years.append(year)

    #     # Plot only valid years
    #     plt.figure(figsize=(12,5))
    #     plt.plot(valid_years, ce_a, marker='o', color="orange", label=labels[0])
    #     plt.plot(valid_years, ce_b, marker='o', color="blue", label=labels[1])
    #     plt.xlabel("Year")
    #     plt.ylabel("Cross-Entropy Loss against Gold Distribution")
    #     plt.title(f"Cross-Entropy Loss per Year | Checkpoint {checkpoint}")
    #     plt.legend()
    #     plt.grid(True)

    #     Path(output_dir).mkdir(parents=True, exist_ok=True)
    #     plt.tight_layout()
    #     plt.savefig(f"{output_dir}/ce_per_year_ckpt{checkpoint}_{labels[0]}_{labels[1]}.png", dpi=300)
    #     plt.close()

    # def plot_binary_ce_vs_gold(self, dist_a, dist_b, gold_dist, checkpoint, cutoff_yr, output_dir="binary_ce_plots", labels=("A vs Gold", "B vs Gold")):
    #     """
    #     Plot binary CE loss per year for each tense vs non-tense for dist_a and dist_b against gold distribution.
    #     Creates separate plots for each binary classification: past vs non-past, present vs non-present, future vs non-future.
        
    #     dist_a, dist_b, gold_dist: dict[checkpoint][year] -> {"past":..., "present":..., "future":...}
    #     checkpoint: int
    #     cutoff_yr: int used in gold data generation
    #     labels: tuple of labels for the legend
    #     """
    #     # Get available tenses from the tense mapping
    #     available_tenses = sorted(set(self.tense_mapping.values()))
        
    #     for target_tense in available_tenses:
    #         ce_a_binary = []
    #         ce_b_binary = []
    #         valid_years = []

    #         for year in range(self.year_range[0], self.year_range[1] + 1):
    #             year_str = str(year)

    #             # Gold distribution - convert to binary (target vs non-target)
    #             d_gold = gold_dist[checkpoint].get(year_str, {tense: 0 for tense in available_tenses})
    #             target_gold = d_gold.get(target_tense, 0)
    #             non_target_gold = sum(d_gold.get(tense, 0) for tense in available_tenses if tense != target_tense)
                
    #             # Skip if no gold data
    #             if target_gold + non_target_gold == 0:
    #                 continue
                    
    #             gold_binary = torch.tensor([target_gold, non_target_gold], dtype=torch.float32)

    #             # Distribution A - convert to binary
    #             d_a = dist_a[checkpoint].get(year_str, {tense: 0 for tense in available_tenses})
            #     s_a = sum(d_a.values())
            #     if s_a == 0:
            #         continue
            #     target_a = d_a.get(target_tense, 0) / s_a
            #     non_target_a = sum(d_a.get(tense, 0) for tense in available_tenses if tense != target_tense) / s_a
            #     a_binary = torch.tensor([target_a, non_target_a], dtype=torch.float32)

            #     # Distribution B - convert to binary
            #     d_b = dist_b[checkpoint].get(year_str, {tense: 0 for tense in available_tenses})
            #     s_b = sum(d_b.values())
            #     if s_b == 0:
            #         continue
            #     target_b = d_b.get(target_tense, 0) / s_b
            #     non_target_b = sum(d_b.get(tense, 0) for tense in available_tenses if tense != target_tense) / s_b
            #     b_binary = torch.tensor([target_b, non_target_b], dtype=torch.float32)

            #     # Binary CE (gold as target)
            #     ce_a_binary.append(-(gold_binary * torch.log(a_binary + 1e-12)).sum().item())
            #     ce_b_binary.append(-(gold_binary * torch.log(b_binary + 1e-12)).sum().item())
            #     valid_years.append(year)

            # # Plot binary CE for this tense
            # if valid_years:
            #     plt.figure(figsize=(12, 5))
            #     plt.plot(valid_years, ce_a_binary, marker='o', color="orange", label=f"{labels[0]} ({target_tense} vs non-{target_tense})")
            #     plt.plot(valid_years, ce_b_binary, marker='o', color="blue", label=f"{labels[1]} ({target_tense} vs non-{target_tense})")
            #     plt.xlabel("Year")
            #     plt.ylabel(f"Binary Cross-Entropy Loss ({target_tense} vs non-{target_tense})")
            #     plt.title(f"Binary CE Loss per Year | {target_tense} vs non-{target_tense} | Checkpoint {checkpoint} | Cutoff {cutoff_yr}")
            #     plt.legend()
            #     plt.grid(True)

            #     Path(output_dir).mkdir(parents=True, exist_ok=True)
            #     plt.tight_layout()
            #     filename = f"{output_dir}/binary_ce_{target_tense}_vs_non_{target_tense}_ckpt{checkpoint}_cutoff{cutoff_yr}.png"
            #     plt.savefig(filename, dpi=300)
            #     plt.close()



#########################################################################################
# EXPERIMENTAL FUNCTIONS. Each one of these runs a specific experiment.
#########################################################################################

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
    

# def plot_distribution():

#     analyzer = AnalyzerClass(TENSE_MAPPING, year_range=(1950, 2050))
#     checkpoints = [10000]
#     # analyzer.plot_single_distribution_stacked(analyzer.relative_model_data, checkpoints, label="Model predictions") # plot_width=30)
#     # analyzer.plot_single_distribution_stacked(analyzer.relative_human_gold, checkpoints, label="Gold distribution") #, plot_width=30)
#     analyzer.plot_single_distribution_stacked(analyzer.relative_training_data["string_match_cooccur"], checkpoints, label="\'In [year]\' and [tense] cooccurence")
#     analyzer.plot_single_distribution_stacked(analyzer.relative_training_data["string_match_cooccur"], checkpoints, label="\'In [year]\' and [tense] cooccurence", plot_width=30)

# def run_spearman_over_years():

#     smanalyzer = AnalyzerClass(TENSE_MAPPING, year_range=(1940, 2060))

#     smanalyzer.plot_spearman_sliding_window(
#         training_dict=smanalyzer.relative_training_data["string_match_cooccur"],
#         tense="past",
#         checkpoint=10000,
#         window=20,
#         label="\'In [year]\' and [tense] cooccurence"
#     )

# def run_ce_loss():

#     analyser = AnalyzerClass(TENSE_MAPPING, year_range=(1950, 2050))

#     for cutoff_yr in [2015, 2016, 2017, 2023, 2024, 2025]:
#         print(f"cutoff_year = {cutoff_yr}")
#         analyser.populate_gold_data(cutoff_yr)

#         # print("Training snapshot, string cooccur:", list(analyser.relative_training_data["string_match_cooccur"][10000].items())[:5])
#         analyser.compute_ce_loss_single(analyser.relative_training_data["string_match_cooccur"], checkpoint=10000, label=f"\'In [year]\' and [tense] cooccurence, cutoff_year={cutoff_yr}")

#         # print("Training snapshot, string match:", list(analyser.relative_training_data["exact_str_matching"][10000].items())[:5])
#         # analyser.compute_ce_loss_single(analyser.relative_training_data["exact_str_matching"], checkpoint=10000, label="training exact_str_matching")

#         # print("Model snapshot:", list(analyser.relative_model_data[10000].items())[:5])
#         analyser.compute_ce_loss_single(analyser.relative_model_data, checkpoint=10000, label=f"model, cutoff_year={cutoff_yr}")
    
#     # example results for 1800-2200:
#     # Training snapshot: [('1800', {'past': 0.9709270433351618, 'future': 0.029072956664838178}), ('1801', {'past': 0.9863782051282052, 'future': 0.013621794871794872}), ('1802', {'past': 0.9871858058156727, 'future': 0.012814194184327254}), ('1803', {'past': 0.9840881272949816, 'future': 0.01591187270501836}), ('1804', {'past': 0.9807534807534808, 'future': 0.019246519246519246})]
#     # training string_match_cooccur | Checkpoint 10000 | CE Loss: 1.9600
#     # Model snapshot: [('1800', {'past': 0.9996292106130621, 'future': 0.0003707893869378904}), ('1801', {'past': 0.998573632179892, 'future': 0.0014263678201079208}), ('1802', {'past': 0.9988521106383594, 'future': 0.0011478893616405149}), ('1803', {'past': 0.9984965393050429, 'future': 0.0015034606949570943}), ('1804', {'past': 0.9987147562745495, 'future': 0.0012852437254504427})]
#     # model | Checkpoint 10000 | CE Loss: 1.0092

# def run_ce_loss_over_years():
#     # Create a complete tense mapping for binary analysis
#     complete_tense_mapping = {
#         "was": "past",
#         "were": "past",
#         "is": "present", 
#         "are": "present",
#         "will": "future",
#     }
        
#     analyser = AnalyzerClass(complete_tense_mapping, year_range=(1950, 2050))
    
#     cutoff_yr = 2024
#     print(f"cutoff_year = {cutoff_yr}")
#     analyser.populate_gold_data(cutoff_yr)
    
#     # Plot binary CE losses for each tense vs non-tense
#     analyser.plot_binary_ce_vs_gold(
#         analyser.relative_model_data, 
#         analyser.relative_training_data["string_match_cooccur"], 
#         analyser.relative_human_gold, 
#         10000, 
#         cutoff_yr,
#         labels=("Model predictions", "\'In [year]\' and [tense] cooccurence")
#     )



# def run_training_dynamics_spearman():

#     # be careful to specify start years within the year range the plotting just errors out and gives 0's instead if you dont.
#     year_range=(1950, 2050)
#     analyser = AnalyzerClass(TENSE_MAPPING, year_range=year_range)

#     # RANK CORRELATION
#     analyser.plot_spearman_over_checkpoints(
#         analyser.relative_model_data,
#         analyser.relative_human_gold,
#         window_size=20,
#         start_years=range(year_range[0]+10, year_range[1], 20), # every 20 yr increment in range, leaving 20 at the end for the window
#         label1="Model predictions",
#         label2="Gold distribution"
#     )  

#     analyser.plot_spearman_over_checkpoints(
#         analyser.relative_model_data,
#         analyser.relative_training_data["string_match_cooccur"],
#         window_size=20,
#         start_years=range(year_range[0]+10, year_range[1], 20), # every 20 yr increment in range, leaving 20 at the end for the window
#         label1="Model predictions",
#         label2="\'In [year]\' and [tense] cooccurence"
#     )  

# def run_training_dynamics_ce():
#     year_range=(1950, 2050)
#     analyser = AnalyzerClass(TENSE_MAPPING, year_range=year_range)

#     # CROSS ENTROPY
#     # data1 is my "TRUE" distr, the gold. and data2 is the "MEASURED distr like model or training data.

#     analyser.plot_avg_ce_over_checkpoints(
#         analyser.relative_human_gold,
#         analyser.relative_model_data,
#         window_size=1,
#         start_years=range(year_range[0], year_range[1], 1), # every 20 yr increment in range, leaving 20 at the end for the window
#         label1="Gold distribution",
#         label2="Model prediction",
#     )  

#     analyser.plot_avg_ce_over_checkpoints(
#         analyser.relative_human_gold,
#         analyser.relative_training_data["string_match_cooccur"],
#         window_size=1,
#         start_years=range(year_range[0], year_range[1]), # every 20 yr increment in range, leaving 20 at the end for the window
#         label1="Gold distribution",
#         label2="\'In [year]\' and [tense] cooccurence",
#     )  

#     # analyser.plot_avg_ce_past_future_checkpoints(
#     #     analyser.relative_human_gold,
#     #     analyser.relative_model_data,
#     #     window_size=1,
#     #     start_years=range(year_range[0], year_range[1]), # every 20 yr increment in range, leaving 20 at the end for the window
#     #     label1="Gold distribution",
#     #     label2="Model prediction",
#     # )  



# def run_training_dynamic_output():
#     year_range=(1950, 2050)
#     analyser = AnalyzerClass(TENSE_MAPPING, year_range=year_range)

#     cps = [i for i in range(250, 10001, 250)]
#     analyser.plot_stacked_grid_over_checkpoints(analyser.relative_model_data, cps, label="Model predictions")
#     analyser.plot_stacked_grid_over_checkpoints(analyser.relative_training_data["string_match_cooccur"], cps, label="\'In [year]\' and [tense] cooccurence")

####################################################################################################################################################################################



if __name__ == "__main__":
    # python kl_divergence_checkpoints.py
    plot_training_data()
    plot_model_predictions()