import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

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
    TENSE_COLORS = {"past": "orange", "present": "green", "future": "purple"}


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

        for cp in range(250, 10001, 250):
            cp_index = cp
            if cp % 100 == 50:
                cp_index += 10
            filepath = file_template.format(cp=cp_index)
            print(f"loading training file {filepath[-45:]}")
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
                    w: counts.get(w, 0) for w in TENSE_MAPPING if w in counts
                }

            results[cp] = year_to_counts

        self.absolute_training_data[count_type] = results

    def _counts_to_probs(self, counts):
        totals = {}
        for w, cat in TENSE_MAPPING.items():
            totals.setdefault(cat, 0)
            totals[cat] += counts.get(w, 0)
        grand_total = sum(totals.values())
        if grand_total == 0:
            return {cat: 0 for cat in set(TENSE_MAPPING.values())}
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
            print(f"loading model data file {filepath[-20:]}")
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

        # Only include tenses that exist in TENSE_MAPPING
        tenses_to_plot = [t for t in self.TENSE_ORDER if t in set(TENSE_MAPPING.values())]

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
                axes[0].bar(ind, vals, bottom=bottom, label=tense, color=self.TENSE_COLORS[tense])
                bottom += vals
            axes[0].set_title(f"{label} Data")
            axes[0].set_xticks(ind[::max(1, len(ind)//20)])
            axes[0].set_xticklabels(years[::max(1, len(ind)//20)], rotation=45)
            axes[0].set_ylabel("Probability")
            axes[0].legend()

            # ---- Model Data ----
            bottom = np.zeros(len(years))
            for tense in tenses_to_plot:
                vals = np.array([model_cp_data[y].get(tense, 0) for y in years])
                axes[1].bar(ind, vals, bottom=bottom, label=tense, color=self.TENSE_COLORS[tense])
                bottom += vals
            axes[1].set_title(f"Model Checkpoint {ckpt}")
            axes[1].set_xticks(ind[::max(1, len(ind)//20)])
            axes[1].set_xticklabels(years[::max(1, len(ind)//20)], rotation=45)

            fig.suptitle(f"Year Distributions | Stacked Bars | Checkpoint {ckpt}")
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{label}/stacked_year_distribution_ckpt{ckpt}_.png", dpi=300)
            plt.close()



if __name__ == "__main__":
    YEAR_RANGE = (1950, 2200)
    analyzer = SpearmanAnalyzer(year_range=YEAR_RANGE)

    checkpoints_to_plot = [2000, 5000, 8000, 10000]
    analyzer.plot_training_vs_model_stacked(analyzer.relative_training_data['exact_str_matching_avg'], checkpoints_to_plot, label="exact_str_matching_avg")
    analyzer.plot_training_vs_model_stacked(analyzer.relative_training_data['exact_str_matching'], checkpoints_to_plot, label="exact_str_matching")
    analyzer.plot_training_vs_model_stacked(analyzer.relative_training_data["string_match_cooccur"], checkpoints_to_plot, label="string_match_cooccur")
    analyzer.plot_training_vs_model_stacked(analyzer.relative_human_gold, checkpoints_to_plot, label="relative_human_gold")
