import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from scipy.stats import spearmanr
import torch.nn.functional as F


MODEL_PREDICTIONS_FILE = "../olmo_predictions/output_checkpoints/checkpoint_{cp}.json"

TRAINING_DATA_FILE_1 = "../olmo_training_data/1000-3000__was.were.is.are.will__allenai_OLMo-2-0425-1B/aggregated/steps0-{cp}/analytics/1000-3000__was.were.is.are.will__aggregated_results_steps0-{cp}.json"
TRAINING_DATA_FILE_2 = "../olmo_training_data/1000-3000__was.were.is.are.will__allenai_OLMo-2-0425-1B/aggregated/steps0-{cp}/extra_analytics/1000-3000__was.were.is.are.will__extra_aggregated_results_steps0-{cp}.json"

COUNT_TYPE_TO_FILE = {
    "exact_str_matching": (TRAINING_DATA_FILE_1, "in_year_there_word_counts"),
    "string_match_cooccur": (TRAINING_DATA_FILE_2, "in_year_tense_sentence_counts")
}

# --- Configurable mapping ---
TENSE_MAPPING = {
    "was": "past",
    "were": "past",
    # "is": "present",
    # "are": "present",
    "will": "future",
}


class SpearmanAnalyzer:
    def __init__(self, tense_mapping, checkpoint_dir=MODEL_PREDICTIONS_FILE,
                 data_source="in_year_there_word_counts",
                 year_range=(1000, 2999), model_name="OLMo"): # TODO change for pythia
        self.tense_mapping = tense_mapping
        self.checkpoint_dir = checkpoint_dir
        self.data_source = data_source
        self.year_range = year_range
        self.model_name = model_name
        self.absolute_training_data = {}
        self.relative_training_data = {}
        self.model_data = {}
        self.relative_model_data = {}
        self.relative_human_gold = {}

        print("loading all data")
        self.load_relative_training_data("exact_str_matching")
        self.load_avg_training_data("exact_str_matching")
        self.load_relative_training_data("string_match_cooccur")
        self.load_relative_model_data()
        self.populate_gold_data()


    # Add these class attributes for consistent plotting
    TENSE_ORDER = ["past", "present", "future"]  # strict plotting order
    TENSE_COLORS = {"past": "orange", "future": "green", "present": "#4b0082"}


    def populate_gold_data(self):
        self.relative_human_gold = {}

        # Get available tenses from the tense mapping
        available_tenses = set(self.tense_mapping.values())
        
        # Define the gold distribution per year based on available tenses
        gold_per_year = {}
        for year in range(self.year_range[0], self.year_range[1] + 1):
            year_dist = {}
            
            if available_tenses == {"past", "present", "future"}:
                # All three tenses: past before 2024, present in 2024, future after 2024
                if year < 2024:
                    year_dist = {"past": 1.0, "present": 0.0, "future": 0.0}
                elif year == 2024:
                    year_dist = {"past": 0.0, "present": 1.0, "future": 0.0}
                else:  # year > 2024
                    year_dist = {"past": 0.0, "present": 0.0, "future": 1.0}
            elif available_tenses == {"past", "future"}:
                # Only past and future: past up to 2024, future after 2024
                if year <= 2024:
                    year_dist = {"past": 1.0, "future": 0.0}
                else:
                    year_dist = {"past": 0.0, "future": 1.0}
            elif available_tenses == {"past", "present"}:
                # Only past and present: past up to 2024, present after 2024
                if year <= 2024:
                    year_dist = {"past": 1.0, "present": 0.0}
                else:
                    year_dist = {"past": 0.0, "present": 1.0}
            else:
                raise ValueError(f"Invalid tense combination: {available_tenses}")
                
            gold_per_year[str(year)] = year_dist

        # Default checkpoint list
        cps = range(250, 10001, 250)

        # Populate each checkpoint with the same gold distribution
        for cp in cps:
            self.relative_human_gold[cp] = gold_per_year.copy()

    def load_absolute_training_data(self, count_type):
        file_template, data_source = COUNT_TYPE_TO_FILE[count_type]
        results = {}

        for cp in range(250, 10001, 250):
            cp_index = cp
            if cp % 100 == 50:
                cp_index += 10
            filepath = file_template.format(cp=cp_index)
            # print(f"loading training file {filepath[-45:]}")
            if not os.path.exists(filepath):
                print(f"cound not find {filepath}")
                continue
            with open(filepath, "r") as f:
                data = json.load(f)

            year_to_counts = {}
            for year, counts in data[data_source].items():
                year_int = int(year)
                if not (self.year_range[0] <= year_int <= self.year_range[1]):
                    continue
                year_to_counts[year] = {
                    w: counts.get(w, 0) for w in self.tense_mapping if w in counts
                }

            results[cp] = year_to_counts

        self.absolute_training_data[count_type] = results

    def _counts_to_probs(self, counts):
        totals = {}
        for w, cat in self.tense_mapping.items():
            totals.setdefault(cat, 0)
            totals[cat] += counts.get(w, 0)
        grand_total = sum(totals.values())
        if grand_total == 0:
            return {cat: 0 for cat in set(self.tense_mapping.values())}
        return {cat: val / grand_total for cat, val in totals.items()}

    def load_relative_training_data(self, count_type):
        if count_type not in self.absolute_training_data:
            self.load_absolute_training_data(count_type)

        abs_data = self.absolute_training_data[count_type]
        results = {}
        for cp, year_dict in abs_data.items():
            year_to_probs = {}
            for year, counts in year_dict.items():
                year_to_probs[year] = self._counts_to_probs(counts)
            results[cp] = year_to_probs

        self.relative_training_data[count_type] = results

    def load_avg_training_data(self, count_type):
        if count_type not in self.absolute_training_data:
            self.load_absolute_training_data(count_type)

        abs_data = self.absolute_training_data[count_type]
        results = {}

        for cp, year_dict in abs_data.items():
            total_counts = {}
            for counts in year_dict.values():
                for w, cat in self.tense_mapping.items():
                    total_counts[cat] = total_counts.get(cat, 0) + counts.get(w, 0)

            grand_total = sum(total_counts.values())
            if grand_total == 0:
                avg_probs = {cat: 0 for cat in set(self.tense_mapping.values())}
            else:
                avg_probs = {cat: val / grand_total for cat, val in total_counts.items()}

            year_to_avg = {year: avg_probs for year in year_dict.keys()}
            results[cp] = year_to_avg

        avg_key = f"{count_type}_avg"
        self.relative_training_data[avg_key] = results

    def load_model_data(self):
        results = {}
        for cp in range(250, 10001, 250):
            filepath = MODEL_PREDICTIONS_FILE.format(cp=cp)
            # print(f"loading model data file {filepath[-20:]}")
            if not os.path.exists(filepath):
                continue
            with open(filepath, "r") as f:
                data = json.load(f)

            year_to_probs = {}
            for year, counts in data["data"].items():
                year_int = int(year)
                if not (self.year_range[0] <= year_int <= self.year_range[1]):
                    continue
                year_to_probs[year] = {
                    w: counts.get(w, 0.0) for w in self.tense_mapping if w in counts
                }

            results[cp] = year_to_probs
        self.model_data = results

    def load_relative_model_data(self):
        if not self.model_data:
            self.load_model_data()

        results = {}
        for cp, year_dict in self.model_data.items():
            year_to_rel = {}
            for year, counts in year_dict.items():
                year_to_rel[year] = self._counts_to_probs(counts)
            results[cp] = year_to_rel
        self.relative_model_data = results


    def plot_training_vs_model_stacked(self, training_dict, checkpoints, output_dir="checkpoints", label="training"):
        """
        Plot stacked bars for a given training dict (e.g., self.relative_training_data['exact_str_matching_avg']
        or self.relative_human_gold) against self.relative_model_data.
        """
        print("plotting stacked training vs model distributions")
        
        # Make folder using the label
        Path(f"{output_dir}/{label}").mkdir(parents=True, exist_ok=True)

        train_data = training_dict
        model_data = self.relative_model_data

        # Only include tenses that exist in tense_mapping
        tenses_to_plot = [t for t in self.TENSE_ORDER if t in set(self.tense_mapping.values())]

        for ckpt in checkpoints:
            if ckpt not in model_data:
                print(f"No model data for checkpoint {ckpt}")
                continue

            train_cp_data = train_data[ckpt]
            model_cp_data = model_data[ckpt]
            years = sorted(train_cp_data.keys())
            ind = np.arange(len(years))

            fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

            # ---- Training Data ----
            bottom = np.zeros(len(years))
            for tense in tenses_to_plot:
                vals = np.array([train_cp_data[y].get(tense, 0) for y in years])
                axes[0].bar(ind, vals, bottom=bottom, label=tense, color=self.TENSE_COLORS[tense], width=1.0)
                bottom += vals
            axes[0].set_title(f"{label} Data")
            axes[0].set_xticks(ind[::max(1, len(ind)//20)])
            axes[0].set_xticklabels(years[::max(1, len(ind)//20)], rotation=45)
            axes[0].set_ylabel("Probability")
            axes[0].set_xlim(-0.5, len(years) - 0.5)
            axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

            # ---- Model Data ----
            bottom = np.zeros(len(years))
            for tense in tenses_to_plot:
                vals = np.array([model_cp_data[y].get(tense, 0) for y in years])
                axes[1].bar(ind, vals, bottom=bottom, label=tense, color=self.TENSE_COLORS[tense], width=1.0)
                bottom += vals
            axes[1].set_title(f"Model Checkpoint {ckpt}")
            axes[1].set_xticks(ind[::max(1, len(ind)//20)])
            axes[1].set_xticklabels(years[::max(1, len(ind)//20)], rotation=45)
            axes[1].set_xlim(-0.5, len(years) - 0.5)

            fig.suptitle(f"Year Distributions | Stacked Bars | Checkpoint {ckpt}")
            plt.tight_layout()
            filename = f"{output_dir}/{label}/{self.model_name}_stacked_year_distribution_{label}_ckpt{ckpt}_years{self.year_range[0]}-{self.year_range[1]}.png"
            plt.savefig(filename, dpi=300)
            plt.close()


    def plot_single_distribution_stacked(self, dist_dict, checkpoints, output_dir="single_dist", label="distribution"):
        """
        Plot stacked bars for a single distribution dict over specified checkpoints.
        """
        print(f"plotting single distribution: {label}")
        
        Path(f"{output_dir}/{label}").mkdir(parents=True, exist_ok=True)

        tenses_to_plot = [t for t in self.TENSE_ORDER if t in set(self.tense_mapping.values())]

        for ckpt in checkpoints:
            if ckpt not in dist_dict:
                print(f"No data for checkpoint {ckpt}")
                continue

            cp_data = dist_dict[ckpt]
            years = sorted(cp_data.keys())
            ind = np.arange(len(years))

            fig, ax = plt.subplots(figsize=(12, 6))
            bottom = np.zeros(len(years))

            for tense in tenses_to_plot:
                vals = np.array([cp_data[y].get(tense, 0) for y in years])
                ax.bar(ind, vals, bottom=bottom, label=tense, color=self.TENSE_COLORS[tense], width=1.0)
                bottom += vals

            ax.set_title(f"{label} | Checkpoint {ckpt}")
            ax.set_xticks(ind[::max(1, len(ind)//20)])
            ax.set_xticklabels(years[::max(1, len(ind)//20)], rotation=45)
            ax.set_ylabel("Probability")
            ax.set_xlim(-0.5, len(years) - 0.5)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

            save_path = f"{output_dir}/{label}/{self.year_range[0]}_{self.year_range[1]}/{self.model_name}_single_distribution_{label}_ckpt{ckpt}_years{self.year_range[0]}-{self.year_range[1]}.png"
            plt.tight_layout()
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300)
            plt.close()


    def plot_spearman_sliding_window(self, training_dict, tense, checkpoint, window=20, output_dir="spearman", label="exact_str_matching"):
        """
        Compute Spearman rank correlation between a training dict and model predictions
        for a specific tense over a sliding window of years.

        training_dict: dict like self.relative_training_data[...] or self.relative_human_gold
        tense: "past", "present", or "future"
        checkpoint: int checkpoint to use from model predictions
        window: size of sliding window in years
        """
        if tense not in self.TENSE_ORDER:
            raise ValueError(f"Tense '{tense}' not in defined tenses: {self.TENSE_ORDER}")
        
        if checkpoint not in self.relative_model_data:
            raise ValueError(f"Checkpoint {checkpoint} not available in model data")

        model_cp_data = self.relative_model_data[checkpoint]
        train_cp_data = training_dict[checkpoint]

        years = sorted(train_cp_data.keys())
        spearman_vals = []
        window_starts = []


        # inside your loop over windows
        for i in range(len(years) - window + 1):
            win_years = years[i:i+window]

            train_vals = [train_cp_data[y].get(tense, 0) for y in win_years]
            model_vals = [model_cp_data[y].get(tense, 0) for y in win_years]

            rho, _ = spearmanr(train_vals, model_vals)
            spearman_vals.append(rho)

            # Use center year instead of starting year
            center_year = int((int(win_years[0]) + int(win_years[-1])) / 2)
            window_starts.append(center_year)

        # Then plot as before
        plt.figure(figsize=(12, 5))
        plt.plot(window_starts, spearman_vals, marker='o', color=self.TENSE_COLORS[tense])
        plt.xlabel(f"Center Year of {window}-Year Window")
        plt.ylabel("Spearman Rank Correlation")
        plt.title(f"Spearman Rank Correlation ({tense}) | Checkpoint {checkpoint} | Counting method {label}")
        plt.grid(True)

        # Path(f"{output_dir}/{label}").mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        filename = f"{output_dir}/{self.model_name}_spearman_{tense}_{label}_ckpt{checkpoint}_window{window}_years{self.year_range[0]}-{self.year_range[1]}.png"
        plt.savefig(filename, dpi=300)
        plt.close()


    def compute_ce_loss_single(self, input_dict, checkpoint, label="training", compute_binary_losses=False):
        """
        Compute cross-entropy loss between predicted distribution (input_dict) 
        and human gold distribution (self.relative_human_gold).
        input_dict: dict like self.relative_training_data[...] or self.relative_model_data
        checkpoint: int
        compute_binary_losses: if True, also compute binary losses for each tense vs non-tense
        """
        gold_cp_data = self.relative_human_gold[checkpoint]
        cp_data = input_dict[checkpoint]
        
        # Get active tenses from tense_mapping
        active_tenses = sorted(set(self.tense_mapping.values()))
        
        probs = []
        gold = []
        years_used = 0
        years_skipped = 0

        for year in range(self.year_range[0], self.year_range[1] + 1):
            year_str = str(year)

            # Predicted distribution (normalize if necessary)
            d = cp_data.get(year_str, {tense: 0 for tense in active_tenses})
            s = sum(d.get(tense, 0) for tense in active_tenses)
            if s == 0:
                years_skipped += 1
                continue  # Skip this year if no data
            else:
                pred_dist = [d.get(tense, 0)/s for tense in active_tenses]
            
            # Gold distribution (already normalized)
            g = gold_cp_data.get(year_str, {tense: 0 for tense in active_tenses})
            gold_dist = [g.get(tense, 0) for tense in active_tenses]
            
            probs.append(pred_dist)
            gold.append(gold_dist)
            years_used += 1
        
        # Convert to tensors and compute loss
        probs_tensor = torch.tensor(probs, dtype=torch.float32)
        gold_tensor = torch.tensor(gold, dtype=torch.float32)
        
        log_probs = torch.log(probs_tensor + 1e-12)
        loss = -(gold_tensor * log_probs).sum(dim=1).mean()

        print(f"{label} | Checkpoint {checkpoint} | CE Loss: {loss:.4f} | Years used: {years_used} | Years skipped: {years_skipped}")
        
        # Compute binary losses if requested
        if compute_binary_losses:
            for i, target_tense in enumerate(active_tenses):
                # Create binary distributions: target tense vs all others
                binary_probs = []
                binary_gold = []
                
                for pred_row, gold_row in zip(probs, gold):
                    # Binary predicted: [target_tense_prob, sum_of_other_tenses_prob]
                    target_prob = pred_row[i]
                    other_prob = sum(pred_row[j] for j in range(len(pred_row)) if j != i)
                    binary_pred = [target_prob, other_prob]
                    
                    # Binary gold: [target_tense_gold, sum_of_other_tenses_gold]
                    target_gold = gold_row[i]
                    other_gold = sum(gold_row[j] for j in range(len(gold_row)) if j != i)
                    binary_gold_dist = [target_gold, other_gold]
                    
                    binary_probs.append(binary_pred)
                    binary_gold.append(binary_gold_dist)
                
                # Compute binary CE loss
                binary_probs_tensor = torch.tensor(binary_probs, dtype=torch.float32)
                binary_gold_tensor = torch.tensor(binary_gold, dtype=torch.float32)
                
                binary_log_probs = torch.log(binary_probs_tensor + 1e-12)
                binary_loss = -(binary_gold_tensor * binary_log_probs).sum(dim=1).mean()
                
                print(f"{label} | Checkpoint {checkpoint} | Binary CE Loss ({target_tense} vs non-{target_tense}): {binary_loss:.4f}")
        
        return loss.item()


    def compute_spearman_window(self, dict1, dict2, checkpoint, start_year, window_size):
        """
        Compute Spearman correlation between two distributions over a sliding window.
        """
        years = [str(y) for y in range(start_year, start_year + window_size)]
        d1_vals, d2_vals = [], []

        for y in years:
            d1 = dict1[checkpoint].get(y, {"past": 0, "future": 0})
            d2 = dict2[checkpoint].get(y, {"past": 0, "future": 0})
            d1_vals.append([d1["past"], d1["future"]])
            d2_vals.append([d2["past"], d2["future"]])

        # flatten: (past + future) across window
        d1_flat = [v for pair in d1_vals for v in pair]
        d2_flat = [v for pair in d2_vals for v in pair]

        rho, _ = spearmanr(d1_flat, d2_flat)
        return rho if rho == rho else None  # return None if nan


    def plot_spearman_over_checkpoints(
        self, dict1, dict2, window_size, start_years,
        label1="data 1", label2="data 2", output_dir="spearman_cp"
    ):
        """
        Plot Spearman rank correlation over checkpoints for different start years (sliding windows).
        Skip points where rho is None.
        Use different marker shapes for start years before/after 2022.
        """
        checkpoints = sorted(dict1.keys())
        colors = plt.cm.tab10(np.linspace(0, 1, len(start_years)))

        plt.figure(figsize=(12, 6))

        for color, start_year in zip(colors, start_years):
            spearmans = []
            valid_cps = []
            for cp in checkpoints:
                rho = self.compute_spearman_window(dict1, dict2, cp, start_year, window_size)
                if rho is not None:
                    spearmans.append(rho)
                    valid_cps.append(cp)

            if spearmans:
                marker_shape = "s" if start_year >= 2022 else "o"  # square for future, circle for past
                plt.plot(valid_cps, spearmans, marker=marker_shape, color=color,
                        label=f"{start_year} - {start_year+window_size}", linestyle='-')

        plt.xlabel("Checkpoint")
        plt.ylabel("Spearman correlation")
        plt.title(f"Spearman correlation over checkpoints ({label1} vs {label2})")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.ylim(-1, 1)

        os.makedirs(f"{output_dir}", exist_ok=True)
        save_path = f"{output_dir}/{self.model_name}_spearman_checkpoints_comparing_{label1}_{label2}_window{window_size}_years{self.year_range[0]}-{self.year_range[1]}.png"
        plt.savefig(save_path, dpi=300)
        plt.close()




#########################################################################################
# EXPERIMENTAL FUNCTIONS. Each one of these runs a specific experiment.
#########################################################################################

def run_model_training_plots():
    analyzer = SpearmanAnalyzer(TENSE_MAPPING, year_range=(1950, 2050))

    checkpoints_to_plot = [2000, 5000, 8000, 10000]
    analyzer.plot_training_vs_model_stacked(analyzer.relative_training_data['exact_str_matching_avg'], checkpoints_to_plot, label="exact_str_matching_avg")
    analyzer.plot_training_vs_model_stacked(analyzer.relative_training_data['exact_str_matching'], checkpoints_to_plot, label="exact_str_matching")
    analyzer.plot_training_vs_model_stacked(analyzer.relative_training_data["string_match_cooccur"], checkpoints_to_plot, label="string_match_cooccur")
    analyzer.plot_training_vs_model_stacked(analyzer.relative_human_gold, checkpoints_to_plot, label="relative_human_gold")

def plot_distribution():

    analyzer = SpearmanAnalyzer(TENSE_MAPPING, year_range=(1950, 2050))
    checkpoints = [2000, 5000, 8000, 10000]
    analyzer.plot_single_distribution_stacked(analyzer.relative_training_data["string_match_cooccur"], checkpoints, label="string_match_cooccur")
    analyzer.plot_single_distribution_stacked(analyzer.relative_model_data, checkpoints, label="relative_model_data")

    analyzer_full= SpearmanAnalyzer(TENSE_MAPPING, year_range=(1600, 2200))
    checkpoints = [2000, 5000, 8000, 10000]
    analyzer_full.plot_single_distribution_stacked(analyzer_full.relative_training_data["string_match_cooccur"], checkpoints, label="string_match_cooccur")
    analyzer_full.plot_single_distribution_stacked(analyzer_full.relative_model_data, checkpoints, label="relative_model_data")


def run_spearman():

    # smanalyzer = SpearmanAnalyzer(TENSE_MAPPING, year_range=(1950, 2050))
    smanalyzer = SpearmanAnalyzer(TENSE_MAPPING, year_range=(1800, 2200))

    smanalyzer.plot_spearman_sliding_window(
        training_dict=smanalyzer.relative_training_data["string_match_cooccur"],
        tense="past",
        checkpoint=10000,
        window=50,
        label="string_match_cooccur"
    )

def run_ce_loss():

    analyser = SpearmanAnalyzer(TENSE_MAPPING, year_range=(1800, 1900))

    # Debug: Print some sample data to see what's happening
    print("=== DEBUG: Sample data inspection ===")
    checkpoint = 10000
    
    # Check gold data
    print("Gold data sample (first 3 years):")
    print(f"Gold data keys for checkpoint {checkpoint}: {list(analyser.relative_human_gold.keys())}")
    if checkpoint in analyser.relative_human_gold:
        gold_cp_data = analyser.relative_human_gold[checkpoint]
        print(f"Number of years in gold data: {len(gold_cp_data)}")
        gold_items = list(gold_cp_data.items())[:3]
        for year, dist in gold_items:
            print(f"  {year}: {dist}")
    else:
        print(f"No gold data found for checkpoint {checkpoint}")
    
    print(f"Available tenses: {set(analyser.tense_mapping.values())}")
    
    # Check training data
    print("\nTraining data sample (string_match_cooccur, first 3 years):")
    train_items = list(analyser.relative_training_data["string_match_cooccur"][checkpoint].items())[:3]
    for year, dist in train_items:
        print(f"  {year}: {dist}")
    
    # Check model data
    print("\nModel data sample (first 3 years):")
    model_items = list(analyser.relative_model_data[checkpoint].items())[:3]
    for year, dist in model_items:
        print(f"  {year}: {dist}")
    
    print("\n=== Multi-class CE Loss ===")
    # print("Training snapshot, string cooccur:", list(analyser.relative_training_data["string_match_cooccur"][10000].items())[:5])
    analyser.compute_ce_loss_single(analyser.relative_training_data["string_match_cooccur"], checkpoint=10000, label="training string_match_cooccur")

    # print("Training snapshot, exact string match:", list(analyser.relative_training_data["exact_str_matching"][10000].items())[:5])
    analyser.compute_ce_loss_single(analyser.relative_training_data["exact_str_matching"], checkpoint=10000, label="training exact_str_matching")

    # print("Model snapshot:", list(analyser.relative_model_data[10000].items())[:5])
    analyser.compute_ce_loss_single(analyser.relative_model_data, checkpoint=10000, label="model")

    print("\n=== Binary CE Losses (tense vs non-tense) ===")
    analyser.compute_ce_loss_single(analyser.relative_training_data["string_match_cooccur"], checkpoint=10000, label="training string_match_cooccur", compute_binary_losses=True)
    analyser.compute_ce_loss_single(analyser.relative_training_data["exact_str_matching"], checkpoint=10000, label="training exact_str_matching", compute_binary_losses=True)
    analyser.compute_ce_loss_single(analyser.relative_model_data, checkpoint=10000, label="model", compute_binary_losses=True)

    # example results for 1800-2200:
    # Training snapshot: [('1800', {'past': 0.9709270433351618, 'future': 0.029072956664838178}), ('1801', {'past': 0.9863782051282052, 'future': 0.013621794871794872}), ('1802', {'past': 0.9871858058156727, 'future': 0.012814194184327254}), ('1803', {'past': 0.9840881272949816, 'future': 0.01591187270501836}), ('1804', {'past': 0.9807534807534808, 'future': 0.019246519246519246})]
    # training string_match_cooccur | Checkpoint 10000 | CE Loss: 1.9600
    # Model snapshot: [('1800', {'past': 0.9996292106130621, 'future': 0.0003707893869378904}), ('1801', {'past': 0.998573632179892, 'future': 0.0014263678201079208}), ('1802', {'past': 0.9988521106383594, 'future': 0.0011478893616405149}), ('1803', {'past': 0.9984965393050429, 'future': 0.0015034606949570943}), ('1804', {'past': 0.9987147562745495, 'future': 0.0012852437254504427})]
    # model | Checkpoint 10000 | CE Loss: 1.0092


def run_training_dynamics():

    # be careful to specify start years within the year range the plotting just errors out and gives 0's instead if you dont.
    analyser = SpearmanAnalyzer(TENSE_MAPPING, year_range=(1950, 2050))

    analyser.plot_spearman_over_checkpoints(
        analyser.relative_model_data,
        analyser.relative_human_gold,
        window_size=20,
        start_years=[1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020, 2030],
        label1="Model",
        label2="relative_human_gold"
    )

    # analyser.plot_spearman_over_checkpoints(
    #     analyser.relative_model_data,
    #     analyser.relative_training_data["string_match_cooccur"],
    #     window_size=20,
    #     start_years=[1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020, 2030],
    #     label1="Model",
    #     label2="string_match_cooccur"
    # )  
    

    # for working on specific ranges ood
    # year_range=(2350, 2450)
    # oodanalyser = SpearmanAnalyzer(TENSE_MAPPING, year_range=year_range)

    # oodanalyser.plot_spearman_over_checkpoints(
    #     oodanalyser.relative_model_data,
    #     oodanalyser.relative_training_data["string_match_cooccur"],
    #     window_size=20,
    #     start_years=range(year_range[0], year_range[1]-10, 10), # every 10 yr increment in range, leaving 20 at the end for the window
    #     label1="Model",
    #     label2="string_match_cooccur"
    # )  


def years_missing():
    analyzer = SpearmanAnalyzer(TENSE_MAPPING, year_range=(1950, 2050))
    checkpoints = [10000]
    
    # Check years with 0 counts for exact_str_matching method
    checkpoint = 10000
    if checkpoint in analyzer.absolute_training_data["exact_str_matching"]:
        exact_str_data = analyzer.absolute_training_data["exact_str_matching"][checkpoint]
        total_years_in_range = analyzer.year_range[1] - analyzer.year_range[0] + 1
        zero_count_years = []
        
        for year in range(analyzer.year_range[0], analyzer.year_range[1] + 1):
            year_str = str(year)
            if year_str in exact_str_data:
                # Check if all tense counts are 0
                total_count = sum(exact_str_data[year_str].get(word, 0) for word in analyzer.tense_mapping.keys())
                if total_count == 0:
                    zero_count_years.append(year)
        
        print(f"Year range: {analyzer.year_range[0]}-{analyzer.year_range[1]}")
        print(f"Total years in range: {total_years_in_range}")
        print(f"Years with 0 counts across all tenses in exact_str_matching: {len(zero_count_years)}")
        if len(zero_count_years) > 0:
            print(f"Years with 0 counts: {sorted(zero_count_years)}")
        else:
            print("No years with 0 counts found!")
####################################################################################################################################################################################



if __name__ == "__main__":
    # python kl_divergence_checkpoints.py

    # run_model_training_plots()  # this plots 2 distributions side by side
    # plot_distribution()
    # run_spearman()
    run_ce_loss()
    # run_training_dynamics()

    
    # analyzer.plot_single_distribution_stacked(analyzer.relative_training_data["string_match_cooccur"], checkpoints, label="string_match_cooccur")
    # analyzer.plot_single_distribution_stacked(analyzer.relative_training_data["exact_str_matching"], checkpoints, label="exact_str_matching")
    # analyzer.plot_single_distribution_stacked(analyzer.relative_model_data, checkpoints, label="relative_model_data")

    
