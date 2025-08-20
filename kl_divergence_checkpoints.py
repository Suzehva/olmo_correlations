import os
import json
import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt
import matplotlib.cm as cm


MODEL_PREDICTIONS_FILE = "../olmo_predictions/output_checkpoints/checkpoint_{cp}.json"

TRAINING_DATA_FILE_1 = "../olmo_training_data/1000-3000__was.were.is.are.will__allenai_OLMo-2-0425-1B/aggregated/steps0-{cp}/analytics/1000-3000__was.were.is.are.will__aggregated_results_steps0-{cp}.json"
TRAINING_DATA_FILE_2 = "../olmo_training_data/1000-3000__was.were.is.are.will__allenai_OLMo-2-0425-1B/aggregated/steps0-{cp}/extra_analytics/1000-3000__was.were.is.are.will__extra_aggregated_results_steps0-{cp}.json"


COUNT_TYPE_TO_FILE = {
    "exact_str_matching": (TRAINING_DATA_FILE_1, "in_year_there_word_counts"),
    "string_match_cooccur": (TRAINING_DATA_FILE_2, "in_year_tense_sentence_counts")
}

TENSE_WORDS = ["was", "were", "is", "are", "will"]

ID_OOD_FILE = "id_ood/in_ood_years_in_year_there_word_counts.json"  # keys: "in_distribution" and "ood"

class KLAnalyzer:
    def __init__(self, checkpoint_dir, data_source="in_year_there_word_counts", year_range=(1000,2999)):
        self.checkpoint_dir = checkpoint_dir
        self.data_source = data_source
        self.year_range = year_range
        self.year_range = year_range
        self.absolute_training_data = {}
        self.relative_training_data = {}
        self.model_data = {}
        self.relative_model_data = {}

        # loads absolute first, then relative
        print("loading all data")
        self.load_relative_training_data("exact_str_matching")
        self.load_avg_training_data("exact_str_matching") # self.relative_training_data[count_type_avg][cp][year]["was"]
        self.load_relative_training_data("string_match_cooccur") # self.relative_training_data[count_type][cp][year]["was"]
        self.load_relative_model_data() # self.relative_model_data[cp][year]["was"]

        with open(ID_OOD_FILE, "r") as f:
            id_ood = json.load(f)
            self.in_years = [str(y) for y in id_ood["in_distribution"]]
            self.ood_years = [str(y) for y in id_ood["ood"]]

    def load_absolute_training_data(self, count_type):
        file_template, data_source = COUNT_TYPE_TO_FILE[count_type]
        results = {}

        for cp in range(250, 10001, 250):
            cp_index = cp
            if cp % 100 == 50:
                cp_index += 10 # for the weird 60-issue
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
                year_to_counts[year] = {w: counts.get(w, 0) for w in TENSE_WORDS}

            results[cp] = year_to_counts

        self.absolute_training_data[count_type] = results

    def load_relative_training_data(self, count_type):
        if count_type not in self.absolute_training_data:
            self.load_absolute_training_data(count_type)

        abs_data = self.absolute_training_data[count_type]
        results = {}

        for cp, year_dict in abs_data.items():
            year_to_probs = {}
            for year, counts in year_dict.items():
                total = sum(counts.values())
                if total == 0:
                    year_to_probs[year] = {"past": 0, "present": 0, "future": 0}
                else:
                    year_to_probs[year] = {
                        "past": (counts["was"] + counts["were"]) / total,
                        "present": (counts["is"] + counts["are"]) / total,
                        "future": counts["will"] / total
                    }
            results[cp] = year_to_probs

        self.relative_training_data[count_type] = results

    def load_avg_training_data(self, count_type):
        if count_type not in self.absolute_training_data:
            self.load_absolute_training_data(count_type)

        abs_data = self.absolute_training_data[count_type]
        results = {}

        for cp, year_dict in abs_data.items():
            # sum all words across the entire year range
            total_counts = {"was": 0, "were": 0, "is": 0, "are": 0, "will": 0}
            for counts in year_dict.values():
                for w in TENSE_WORDS:
                    total_counts[w] += counts.get(w, 0)

            grand_total = sum(total_counts.values())
            if grand_total == 0:
                avg_probs = {"past": 0, "present": 0, "future": 0}
            else:
                avg_probs = {
                    "past": (total_counts["was"] + total_counts["were"]) / grand_total,
                    "present": (total_counts["is"] + total_counts["are"]) / grand_total,
                    "future": total_counts["will"] / grand_total,
                }

            # put the same value for every year in this checkpoint
            year_to_avg = {year: avg_probs for year in year_dict.keys()}
            results[cp] = year_to_avg

        # store under new key, e.g. "exact_str_matching_avg"
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
                year_to_probs[year] = {w: counts.get(w, 0.0) for w in TENSE_WORDS}

            results[cp] = year_to_probs

        self.model_data = results

    def load_relative_model_data(self):
        if not self.model_data:
            self.load_model_data()

        results = {}
        for cp, year_dict in self.model_data.items():
            year_to_rel = {}
            for year, counts in year_dict.items():
                total = sum(counts.values())
                if total == 0:
                    year_to_rel[year] = {"past": 0, "present": 0, "future": 0}
                else:
                    year_to_rel[year] = {
                        "past": (counts["was"] + counts["were"]) / total,
                        "present": (counts["is"] + counts["are"]) / total,
                        "future": counts["will"] / total
                    }
            results[cp] = year_to_rel

        self.relative_model_data = results


    @staticmethod
    def kl_divergence_from_dicts(dist_p, dist_q, epsilon=1e-12):
        all_categories = set(dist_p.keys()) | set(dist_q.keys())
        categories = sorted(all_categories)
        p = np.array([dist_p.get(c,0) for c in categories])
        q = np.array([dist_q.get(c,0) for c in categories])
        p = np.maximum(p, epsilon) / np.sum(np.maximum(p, epsilon))
        q = np.maximum(q, epsilon) / np.sum(np.maximum(q, epsilon))
        return entropy(p, q)

    def plot_kl_over_checkpoints_for_count_type(self, count_type):
        """
        Compute and plot KL divergence over checkpoints for a given relative training count type.
        Args:
            checkpoints (list[int]): checkpoints to evaluate
            count_type (str): key in self.relative_training_data 
                              (e.g. 'exact_str_matching', 'exact_str_matching_avg', 'string_match_cooccur')
        """
        if count_type not in self.relative_training_data:
            raise ValueError(f"{count_type} not found in relative_training_data. "
                             f"Available: {list(self.relative_training_data.keys())}")

        checkpoints = range(250, 10001, 250)
        print(f"plotting kl over checkpoints for {count_type}")

        kl_in_over_ckpt = []
        kl_ood_over_ckpt = []

        for ckpt in checkpoints:

            if ckpt not in self.relative_model_data:
                print(f"no model data at {ckpt}")
                continue
            if ckpt not in self.relative_training_data[count_type]:
                print(f"no train data at {ckpt}")
                continue

            train = self.relative_training_data[count_type][ckpt]
            model = self.relative_model_data[ckpt]

            # debug : Dump some data to a file for inspection
            # dump = {"train": train, "model": model}
            # with open(f"debug_counts_cp{ckpt}.json", "w") as f:
            #     json.dump(dump, f, indent=2)

            kl_in = []
            kl_ood = []

            for year, probs_model in model.items():
                if year not in train:
                    continue
                kl_val = self.kl_divergence_from_dicts(train[year], probs_model)
                if year in self.in_years:
                    kl_in.append(kl_val)
                elif year in self.ood_years:
                    kl_ood.append(kl_val)

            kl_in_over_ckpt.append(np.mean(kl_in) if kl_in else None)
            kl_ood_over_ckpt.append(np.mean(kl_ood) if kl_ood else None)

        # --- Plot ---
        plt.figure(figsize=(12,5))
        plt.plot(checkpoints, kl_in_over_ckpt, marker='o', color='blue', label='In-distribution')
        plt.plot(checkpoints, kl_ood_over_ckpt, marker='o', color='red', label='Out-of-distribution')
        plt.xlabel("Checkpoint")
        plt.ylabel("Average KL Divergence")
        plt.title(f"KL Divergence Over Checkpoints | {count_type}")
        plt.ylim(0, 1.2)
        plt.grid(True)
        plt.legend(loc='upper right')
        os.makedirs("checkpoints", exist_ok=True)
        plt.tight_layout()
        plt.savefig(f"checkpoints/kl_in_vs_ood_over_checkpoints_{count_type}.png")
        plt.show()
        plt.close()


if __name__ == "__main__":

    analyzer = KLAnalyzer(MODEL_PREDICTIONS_FILE)

    # Use averaged exact string matching
    analyzer.plot_kl_over_checkpoints_for_count_type("exact_str_matching_avg")

    # Use string match cooccurrence
    analyzer.plot_kl_over_checkpoints_for_count_type("string_match_cooccur")

