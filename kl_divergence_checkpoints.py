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
# TRAINING_DATA_FILE_2 = "../olmo_training_data/1000-3000__was.were.is.are.will__allenai_OLMo-2-0425-1B/aggregated/steps0-{cp}/extra_analytics/1000-3000__was.were.is.are.will__extra_aggregated_results_steps0-{cp}.json"
TRAINING_DATA_FILE_2 = "../olmo_training_data/1000-3000__was.were.is.are.will__allenai_OLMo-2-0425-1B/aggregated/steps0-{cp}/extra_strict_analytics/1000-3000__was.were.is.are.will__extra_strict_aggregated_results_steps0-{cp}.json"

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

class AnalyzerClass:
    def __init__(self, checkpoint_dir=MODEL_PREDICTIONS_FILE,
                 data_source="in_year_there_word_counts",
                 year_range=(1000, 2999)):
        self.checkpoint_dir = checkpoint_dir
        self.data_source = data_source
        self.year_range = year_range
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
    TENSE_COLORS = {"past": "orange", "present": "purple", "future": "green"}


    def populate_gold_data(self):
        self.relative_human_gold = {}

        # Define the gold distribution per year
        gold_per_year = {}
        for year in range(self.year_range[0], self.year_range[1] + 1):
            if year <= 2022:
                gold_per_year[str(year)] = {"past": 1.0, "present": 0.0, "future": 0.0}
            else:
                gold_per_year[str(year)] = {"past": 0.0, "present": 0.0, "future": 1.0}

        # Default checkpoint list
        cps = range(250, 10001, 250)

        # Populate each checkpoint with the same gold distribution
        for cp in cps:
            self.relative_human_gold[cp] = gold_per_year.copy()

    def load_absolute_training_data(self, count_type):
        file_template, data_source = COUNT_TYPE_TO_FILE[count_type]
        results = {}
        total_count = 0  # track all co-occurrences

        for cp in range(250, 10001, 250):
            cp_index = cp
            if cp % 100 == 50:
                cp_index += 10
            filepath = file_template.format(cp=cp_index)
            if not os.path.exists(filepath):
                print(f"could not find {filepath}")
                continue
            with open(filepath, "r") as f:
                data = json.load(f)

            year_to_counts = {}
            for year, counts in data[data_source].items():
                year_int = int(year)
                if not (self.year_range[0] <= year_int <= self.year_range[1]):
                    continue
                word_counts = {w: counts.get(w, 0) for w in TENSE_MAPPING if w in counts}
                year_counts = self._counts_to_tense_counts(word_counts)
                year_to_counts[year] = year_counts
                if (cp == 10000):
                    total_count += sum(year_counts.values())  # add to running total

            results[cp] = year_to_counts

        self.absolute_training_data[count_type] = results
        print(f"[{count_type}] Total co-occurrences loaded: {total_count}")

    def _counts_to_tense_counts(self, counts):
        """Convert word counts to tense category counts (absolute counts, not probabilities)"""
        totals = {}
        for w, cat in TENSE_MAPPING.items():
            totals.setdefault(cat, 0)
            totals[cat] += counts.get(w, 0)
        return totals

    def _tense_counts_to_probs(self, tense_counts):
        """Convert tense category counts to probabilities"""
        grand_total = sum(tense_counts.values())
        if grand_total == 0:
            return {cat: 0 for cat in set(TENSE_MAPPING.values())}
        return {cat: val / grand_total for cat, val in tense_counts.items()}

    def load_relative_training_data(self, count_type):
        if count_type not in self.absolute_training_data:
            self.load_absolute_training_data(count_type)

        abs_data = self.absolute_training_data[count_type]
        results = {}
        for cp, year_dict in abs_data.items():
            year_to_probs = {}
            for year, counts in year_dict.items():
                year_to_probs[year] = self._tense_counts_to_probs(counts)
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
                for w, cat in TENSE_MAPPING.items():
                    total_counts[cat] = total_counts.get(cat, 0) + counts.get(w, 0)

            grand_total = sum(total_counts.values())
            if grand_total == 0:
                avg_probs = {cat: 0 for cat in set(TENSE_MAPPING.values())}
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
                    w: counts.get(w, 0.0) for w in TENSE_MAPPING if w in counts
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
                year_to_rel[year] = self._tense_counts_to_probs(counts)
            results[cp] = year_to_rel
        self.relative_model_data = results


    def plot_single_distribution_stacked(self, dist_dict, checkpoints, output_dir="single_dist", label="distribution", plot_width=12):
        """
        Plot stacked bars for a single distribution dict over specified checkpoints.
        """
        print(f"plotting single distribution: {label}")
        
        Path(f"{output_dir}/{label}").mkdir(parents=True, exist_ok=True)

        tenses_to_plot = [t for t in self.TENSE_ORDER if t in set(TENSE_MAPPING.values())]

        for ckpt in checkpoints:
            if ckpt not in dist_dict:
                print(f"No data for checkpoint {ckpt}")
                continue

            cp_data = dist_dict[ckpt]
            years = sorted(cp_data.keys())
            ind = np.arange(len(years))

            fig, ax = plt.subplots(figsize=(plot_width, 6))
            bottom = np.zeros(len(years))

            for tense in tenses_to_plot:
                vals = np.array([cp_data[y].get(tense, 0) for y in years])
                ax.bar(
                    ind, vals, bottom=bottom,
                    label=tense, color=self.TENSE_COLORS[tense],
                    linewidth=0, edgecolor="none"  # no white lines
                )
                bottom += vals

            ax.set_title(f"{label} | Checkpoint {ckpt}")
            ax.set_xticks(ind[::max(1, len(ind)//20)])
            ax.set_xticklabels(years[::max(1, len(ind)//20)], rotation=45)
            ax.set_ylabel("Probability")
            ax.legend()

            save_dir = Path(f"{output_dir}/{label}/{self.year_range[0]}_{self.year_range[1]}")
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"stacked_year_distribution_ckpt{ckpt}.png"
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=600)  # smoother, higher resolution
            plt.close()


    def plot_stacked_grid_over_checkpoints(
        self, dist_dict, checkpoints, output_dir="single_dist_grid",
        label="distribution", n_cols=5, n_rows=8
    ):
        """
        Plot stacked bars for a single distribution over multiple checkpoints
        in a grid (n_cols x n_rows). Only show overall y-axis label and title.
        """
        print(f"Plotting stacked distribution grid: {label}")

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        tenses_to_plot = [t for t in self.TENSE_ORDER if t in set(TENSE_MAPPING.values())]

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(n_cols*3, n_rows*2.5), sharey=True
        )
        axes = axes.flatten()

        for i, ckpt in enumerate(checkpoints):
            ax = axes[i]
            if ckpt not in dist_dict:
                print(f"No data for checkpoint {ckpt}")
                ax.axis('off')
                continue

            cp_data = dist_dict[ckpt]
            years = sorted(cp_data.keys())
            ind = np.arange(len(years))
            bottom = np.zeros(len(years))

            for tense in tenses_to_plot:
                vals = np.array([cp_data[y].get(tense, 0) for y in years])
                ax.bar(ind, vals, bottom=bottom, label=tense, color=self.TENSE_COLORS[tense])
                bottom += vals

            ax.set_title(f"Checkpoint {ckpt}", fontsize=8)
            ax.axis('off')  # hide x/y ticks

        # Hide any extra subplots
        for j in range(i+1, n_cols*n_rows):
            axes[j].axis('off')

        # Overall labels and title
        fig.suptitle(
            f"{label} | Checkpoints {checkpoints[0]}-{checkpoints[-1]} | Years {self.year_range[0]}-{self.year_range[1]}",
            fontsize=14
        )
        fig.text(0.03, 0.5, "Probability", va='center', rotation='vertical', fontsize=12)

        # Adjust spacing to reduce outer margins but keep subplots separated
        plt.subplots_adjust(
            left=0.06, right=0.99, top=0.93, bottom=0.05,
            hspace=0.3, wspace=0.25
        )

        save_path = f"{output_dir}/{label}_stacked_grid_{checkpoints[0]}_{checkpoints[-1]}.png"
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

        Path(f"{output_dir}").mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/spearman_{tense}_years_{self.year_range[0]}_{self.year_range[1]}_ckpt{checkpoint}_window{window}_{label}.png", dpi=300)
        plt.close()


    def compute_ce_loss_single(self, input_dict, checkpoint, label="training"):
        """
        Compute cross-entropy loss between predicted distribution (input_dict) 
        and human gold distribution (self.relative_human_gold).
        input_dict: dict like self.relative_training_data[...] or self.relative_model_data
        checkpoint: int
        """
        gold_cp_data = self.relative_human_gold[checkpoint]  # gold distribution for this cp
        cp_data = input_dict[checkpoint]                     # predicted distribution for this cp

        probs = []
        gold = []

        for year in range(self.year_range[0], self.year_range[1] + 1):
            year_str = str(year)

            # Predicted distribution (normalize if necessary)
            d = cp_data.get(year_str, {"past": 0, "present": 0, "future": 0})
            s = sum(d.values())
            if s == 0:
                pred_dist = [1.0, 0.0, 0.0]  # fallback if no data
            else:
                pred_dist = [d.get("past", 0)/s, d.get("present", 0)/s, d.get("future", 0)/s]
            probs.append(pred_dist)

            # Gold distribution (already normalized)
            g = gold_cp_data.get(year_str, {"past": 0, "present": 0, "future": 0})
            gold_dist = [g["past"], g["present"], g["future"]]
            gold.append(gold_dist)

        # Convert to tensors
        probs_tensor = torch.tensor(probs, dtype=torch.float32)
        gold_tensor = torch.tensor(gold, dtype=torch.float32)

        # CE loss (gold is full distribution, so use log-softmax + NLL trick)
        log_probs = torch.log(probs_tensor + 1e-12)
        loss = -(gold_tensor * log_probs).sum(dim=1).mean()

        print(f"{label} | Checkpoint {checkpoint} | CE Loss: {loss:.4f}")
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
        plt.legend()
        plt.ylim(-1, 1)

        os.makedirs(f"{output_dir}", exist_ok=True)
        save_path = f"{output_dir}/comparing_{label1}_{label2}_window_{window_size}_years_{self.year_range[0]}_{self.year_range[1]}.png"
        plt.savefig(save_path, dpi=300)
        plt.close()


    def plot_avg_ce_over_checkpoints(self, dict1, dict2, window_size, start_years, label1="data 1", label2="data 2", output_dir="ce_cp"):
        """
        Plot average cross-entropy over checkpoints for different start years (sliding windows).
        Skip points where CE cannot be computed (e.g., all zeros in window).
        Use different marker shapes for start years before/after 2022.
        """
        checkpoints = sorted(dict1.keys())

        cmap = plt.get_cmap("rainbow")
        colors = cmap(np.linspace(0, 1, len(start_years)))[::-1]  # inverted order


        plt.figure(figsize=(14, 6))  # wider figure

        for color, start_year in zip(colors, start_years):
            avg_ces = []
            valid_cps = []

            for cp in checkpoints:
                ce_values = []

                for year in range(start_year, start_year + window_size):
                    year_str = str(year)

                    # Dist 1
                    d1 = dict1[cp].get(year_str, {"past":0,"present":0,"future":0})
                    s1 = sum(d1.values())
                    if s1 == 0:
                        dist1_tensor = torch.tensor([1.0,0.0,0.0], dtype=torch.float32)
                    else:
                        dist1_tensor = torch.tensor([d1.get("past",0)/s1,
                                                    d1.get("present",0)/s1,
                                                    d1.get("future",0)/s1], dtype=torch.float32)

                    # Dist 2
                    d2 = dict2[cp].get(year_str, {"past":0,"present":0,"future":0})
                    s2 = sum(d2.values())
                    if s2 == 0:
                        dist2_tensor = torch.tensor([1.0,0.0,0.0], dtype=torch.float32)
                    else:
                        dist2_tensor = torch.tensor([d2.get("past",0)/s2,
                                                    d2.get("present",0)/s2,
                                                    d2.get("future",0)/s2], dtype=torch.float32)

                    ce = -(dist2_tensor * torch.log(dist1_tensor + 1e-12)).sum().item()
                    ce_values.append(ce)

                if ce_values:
                    avg_ces.append(np.mean(ce_values))
                    valid_cps.append(cp)

            if avg_ces:
                marker_shape = "s" if start_year >= 2022 else "o"
                plt.plot(valid_cps, avg_ces, marker=marker_shape, color=color,
                        label=f"{start_year} - {start_year+window_size}", linestyle='-')


        plt.legend(
            bbox_to_anchor=(1.02, 1), 
            loc="upper left", 
            borderaxespad=0.
        )

        plt.xlabel("Checkpoint")
        plt.ylabel("Average Cross-Entropy")
        plt.title(f"Average CE over checkpoints ({label1} vs {label2})")
        plt.ylim(0, 30)
        plt.grid(True)

        os.makedirs(f"{output_dir}", exist_ok=True)
        save_path = f"{output_dir}/avg_ce_{label1}_{label2}_window_{window_size}_years_{self.year_range[0]}_{self.year_range[1]}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")  # ensures legend not cut off
        plt.close()


    def plot_ce_vs_gold(self, dist_a, dist_b, gold_dist, checkpoint, output_dir="ce_plots", labels=("A vs Gold", "B vs Gold")):
        """
        Plot CE loss per year for dist_a and dist_b against a gold distribution on the same plot.

        dist_a, dist_b, gold_dist: dict[checkpoint][year] -> {"past":..., "present":..., "future":...}
        checkpoint: int
        labels: tuple of labels for the legend
        """
        years = range(self.year_range[0], self.year_range[1] + 1)
        ce_a = []
        ce_b = []

        for year in years:
            year_str = str(year)

            # Gold distribution (normalize)
            d_gold = gold_dist[checkpoint].get(year_str, {"past":0, "present":0, "future":0})
            s_gold = sum(d_gold.values())
            if s_gold == 0:
                gold_tensor = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
            else:
                gold_tensor = torch.tensor([
                    d_gold.get("past", 0)/s_gold,
                    d_gold.get("present", 0)/s_gold,
                    d_gold.get("future", 0)/s_gold
                ], dtype=torch.float32)

            # Distribution A (normalize)
            d_a = dist_a[checkpoint].get(year_str, {"past":0, "present":0, "future":0})
            s_a = sum(d_a.values())
            if s_a == 0:
                a_tensor = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
            else:
                a_tensor = torch.tensor([
                    d_a.get("past", 0)/s_a,
                    d_a.get("present", 0)/s_a,
                    d_a.get("future", 0)/s_a
                ], dtype=torch.float32)

            # Distribution B (normalize)
            d_b = dist_b[checkpoint].get(year_str, {"past":0, "present":0, "future":0})
            s_b = sum(d_b.values())
            if s_b == 0:
                b_tensor = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
            else:
                b_tensor = torch.tensor([
                    d_b.get("past", 0)/s_b,
                    d_b.get("present", 0)/s_b,
                    d_b.get("future", 0)/s_b
                ], dtype=torch.float32)

            # CE (gold as target)
            ce_a.append(-(gold_tensor * torch.log(a_tensor + 1e-12)).sum().item())
            ce_b.append(-(gold_tensor * torch.log(b_tensor + 1e-12)).sum().item())

        # Plot
        plt.figure(figsize=(12,5))
        plt.plot(list(years), ce_a, marker='o', color="orange", label=labels[0])
        plt.plot(list(years), ce_b, marker='o', color="blue", label=labels[1])
        plt.xlabel("Year")
        plt.ylabel("Cross-Entropy Loss against Gold Distribution")
        plt.title(f"Cross-Entropy Loss per Year | Checkpoint {checkpoint}")
        plt.legend()
        plt.grid(True)

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/ce_per_year_ckpt{checkpoint}.png", dpi=300)
        plt.close()




#########################################################################################
# EXPERIMENTAL FUNCTIONS. Each one of these runs a specific experiment.
#########################################################################################

def plot_distribution():

    analyzer1 = AnalyzerClass(year_range=(1000, 3000))
    checkpoints = [10000]
    # analyzer1.plot_single_distribution_stacked(analyzer1.relative_model_data, checkpoints, label="Model predictions") # plot_width=30)
    # analyzer1.plot_single_distribution_stacked(analyzer1.relative_training_data["string_match_cooccur"], checkpoints, label="\'In [year]\' and [tense] cooccurence")
    analyzer1.plot_single_distribution_stacked(analyzer1.absolute_training_data["string_match_cooccur"], checkpoints, label="\'In [year]\' and [tense] cooccurence (absolute)")
    analyzer1.plot_single_distribution_stacked(analyzer1.relative_training_data["string_match_cooccur"], checkpoints, label="\'In [year]\' and [tense] cooccurence", plot_width=30)

def run_spearman_over_years():

    smanalyzer = AnalyzerClass(year_range=(1940, 2060))

    smanalyzer.plot_spearman_sliding_window(
        training_dict=smanalyzer.relative_training_data["string_match_cooccur"],
        tense="past",
        checkpoint=10000,
        window=20,
        label="\'In [year]\' and [tense] cooccurence"
    )

def run_ce_loss():

    analyser = AnalyzerClass(year_range=(1950, 2050))

    print("Training snapshot, string cooccur:", list(analyser.relative_training_data["string_match_cooccur"][10000].items())[:5])
    analyser.compute_ce_loss_single(analyser.relative_training_data["string_match_cooccur"], checkpoint=10000, label="\'In [year]\' and [tense] cooccurence")

    print("Training snapshot, string match:", list(analyser.relative_training_data["exact_str_matching"][10000].items())[:5])
    analyser.compute_ce_loss_single(analyser.relative_training_data["exact_str_matching"], checkpoint=10000, label="training exact_str_matching")

    print("Model snapshot:", list(analyser.relative_model_data[10000].items())[:5])
    analyser.compute_ce_loss_single(analyser.relative_model_data, checkpoint=10000, label="model")

    # example results for 1800-2200:
    # Training snapshot: [('1800', {'past': 0.9709270433351618, 'future': 0.029072956664838178}), ('1801', {'past': 0.9863782051282052, 'future': 0.013621794871794872}), ('1802', {'past': 0.9871858058156727, 'future': 0.012814194184327254}), ('1803', {'past': 0.9840881272949816, 'future': 0.01591187270501836}), ('1804', {'past': 0.9807534807534808, 'future': 0.019246519246519246})]
    # training string_match_cooccur | Checkpoint 10000 | CE Loss: 1.9600
    # Model snapshot: [('1800', {'past': 0.9996292106130621, 'future': 0.0003707893869378904}), ('1801', {'past': 0.998573632179892, 'future': 0.0014263678201079208}), ('1802', {'past': 0.9988521106383594, 'future': 0.0011478893616405149}), ('1803', {'past': 0.9984965393050429, 'future': 0.0015034606949570943}), ('1804', {'past': 0.9987147562745495, 'future': 0.0012852437254504427})]
    # model | Checkpoint 10000 | CE Loss: 1.0092

def run_ce_loss_over_years():
        
    analyser = AnalyzerClass(year_range=(1950, 2050))
    # the ground truth distribution MUST be the third distribution arg
    
    analyser.plot_ce_vs_gold(analyser.relative_model_data, analyser.relative_training_data["string_match_cooccur"], analyser.relative_human_gold, 10000, labels=("Model predictions", "\'In [year]\' and [tense] cooccurence\'"))


def run_training_dynamics_spearman():

    # be careful to specify start years within the year range the plotting just errors out and gives 0's instead if you dont.
    year_range=(1950, 2050)
    analyser = AnalyzerClass(year_range=year_range)

    # RANK CORRELATION
    analyser.plot_spearman_over_checkpoints(
        analyser.relative_model_data,
        analyser.relative_human_gold,
        window_size=20,
        start_years=range(year_range[0]+10, year_range[1], 20), # every 20 yr increment in range, leaving 20 at the end for the window
        label1="Model predictions",
        label2="Gold distribution"
    )  

    analyser.plot_spearman_over_checkpoints(
        analyser.relative_model_data,
        analyser.relative_training_data["string_match_cooccur"],
        window_size=20,
        start_years=range(year_range[0]+10, year_range[1], 20), # every 20 yr increment in range, leaving 20 at the end for the window
        label1="Model predictions",
        label2="\'In [year]\' and [tense] cooccurence"
    )  

def run_training_dynamics_ce():
    year_range=(2100, 2200)
    analyser = AnalyzerClass(year_range=year_range)

    # CROSS ENTROPY
    # data1 is my "TRUE" distr, the gold. and data2 is the "MEASURED distr like model or training data.

    analyser.plot_avg_ce_over_checkpoints(
        analyser.relative_human_gold,
        analyser.relative_model_data,
        window_size=5,
        start_years=range(year_range[0], year_range[1], 5), # every 20 yr increment in range, leaving 20 at the end for the window
        label1="Gold distribution",
        label2="Model prediction",
    )  

    analyser.plot_avg_ce_over_checkpoints(
        analyser.relative_human_gold,
        analyser.relative_training_data["string_match_cooccur"],
        window_size=5,
        start_years=range(year_range[0], year_range[1], 5), # every 20 yr increment in range, leaving 20 at the end for the window
        label1="Gold distribution",
        label2="\'In [year]\' and [tense] cooccurence",
    )  

def run_training_dynamic_output():
    year_range=(1950, 2050)
    analyser = AnalyzerClass(year_range=year_range)

    cps = [i for i in range(250, 10001, 250)]
    analyser.plot_stacked_grid_over_checkpoints(analyser.relative_model_data, cps, label="Model predictions")
    analyser.plot_stacked_grid_over_checkpoints(analyser.relative_training_data["string_match_cooccur"], cps, label="\'In [year]\' and [tense] cooccurence")

####################################################################################################################################################################################


if __name__ == "__main__":
    # python kl_divergence_checkpoints.py

    plot_distribution()
    # run_training_dynamic_output()    # this just plots the model output over cps in a big grid

    # run_training_dynamics_ce()
    # run_training_dynamics_ce()
    # run_training_dynamics_spearman()

    run_ce_loss_over_years()
    # run_spearman_over_years()

    # run_ce_loss()
