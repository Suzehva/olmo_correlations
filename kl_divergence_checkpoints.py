import os
import json
import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt

TRAINING_DATA_FILE = "../olmo_training_data/1000-3000__was.were.is.are.will__allenai_OLMo-2-0425-1B/aggregated/steps0-10000/analytics/1000-3000__was.were.is.are.will__aggregated_results_steps0-10000.json" 
CHECKPOINT_DIR = "../olmo_predictions/output_checkpoints"

class KLAnalyzer:
    def __init__(self, training_data_file, checkpoint_dir, data_source, year_range=(1000,2999)):
        self.training_data_file = training_data_file
        self.checkpoint_dir = checkpoint_dir
        self.data_source = data_source
        self.year_range = year_range

        with open(training_data_file, "r") as f:
            self.training_data = json.load(f)

    @staticmethod
    def kl_divergence_from_dicts(dist_p, dist_q, epsilon=1e-12):
        all_categories = set(dist_p.keys()) | set(dist_q.keys())
        categories = sorted(all_categories)
        p = np.array([dist_p.get(c,0) for c in categories])
        q = np.array([dist_q.get(c,0) for c in categories])
        p = np.maximum(p, epsilon) / np.sum(np.maximum(p, epsilon))
        q = np.maximum(q, epsilon) / np.sum(np.maximum(q, epsilon))
        return entropy(p, q)

    def get_relative_probabilities(self, predictions, prior_method="per_year"):
        year_to_model = {}
        normalized = predictions.get("normalized_data", predictions)
        for year, rel in normalized.items():
            year_int = int(year)
            if not (self.year_range[0] <= year_int <= self.year_range[1]):
                continue
            year_to_model[year] = {
                "past": rel.get("was", 0) + rel.get("were", 0),
                "present": rel.get("is", 0) + rel.get("are", 0),
                "future": rel.get("will", 0)
            }

        if prior_method == "per_year":
            year_to_train = {}
            for year, counts in self.training_data[self.data_source].items():
                year_int = int(year)
                if not (self.year_range[0] <= year_int <= self.year_range[1]):
                    continue
                total = sum(counts.get(k,0) for k in ["was","were","is","are","will"])
                if total == 0:
                    year_to_train[year] = {"past":0,"present":0,"future":0}
                else:
                    year_to_train[year] = {
                        "past": (counts.get("was",0)+counts.get("were",0))/total,
                        "present": (counts.get("is",0)+counts.get("are",0))/total,
                        "future": counts.get("will",0)/total
                    }
        elif prior_method == "global":
            total_counts = {"was":0,"were":0,"is":0,"are":0,"will":0}
            for year, counts in self.training_data[self.data_source].items():
                year_int = int(year)
                if not (self.year_range[0] <= year_int <= self.year_range[1]):
                    continue
                for k in total_counts:
                    total_counts[k] += counts.get(k,0)
            total_sum = sum(total_counts.values())
            global_probs = {
                "past": (total_counts["was"] + total_counts["were"])/total_sum,
                "present": (total_counts["is"] + total_counts["are"])/total_sum,
                "future": total_counts["will"]/total_sum
            }
            year_to_train = {year: global_probs for year in year_to_model.keys()}
        else:
            raise ValueError(f"Unknown prior_method: {prior_method}")

        return year_to_model, year_to_train

    def compute_avg_kl_for_checkpoint(self, ckpt, count_range=(1,10), prior_method="per_year"):
        pred_file = os.path.join(self.checkpoint_dir, f"checkpoint_{ckpt}.json")
        if not os.path.exists(pred_file):
            return None
        with open(pred_file, "r") as f:
            predictions = json.load(f)

        model, train = self.get_relative_probabilities(predictions, prior_method=prior_method)

        kl_list = []
        for y in model.keys():
            if y not in train:
                continue
            total_count = sum(self.training_data[self.data_source][y].get(k,0) for k in ["was","were","is","are","will"])
            if count_range[0] <= total_count <= count_range[1]:
                kl_list.append(self.kl_divergence_from_dicts(train[y], model[y]))
        return np.mean(kl_list) if kl_list else None

    def plot_avg_kl_over_checkpoints(self, checkpoints, avg_kl_values, count_range=(1,10)):
        plt.figure(figsize=(12,5))
        ckpts_filtered = [c for c, kl in zip(checkpoints, avg_kl_values) if kl is not None]
        kl_filtered = [kl for kl in avg_kl_values if kl is not None]

        plt.plot(ckpts_filtered, kl_filtered, marker='o')
        plt.xlabel("Checkpoint")
        plt.ylabel("Average KL Divergence")
        plt.title(f"Avg KL Divergence Trend | {self.data_source} | Count range {count_range}")
        plt.ylim(0,3)
        plt.grid(True)
        plt.tight_layout()
        os.makedirs("checkpoints", exist_ok=True)
        plt.savefig(f"checkpoints/avg_kl_checkpoints_{count_range[0]}_{count_range[1]}.png")
        plt.show()
        plt.close()


    def compute_avg_kl_by_bins(self, ckpt, cutoffs=[4,10], prior_method="per_year"):
        pred_file = os.path.join(self.checkpoint_dir, f"checkpoint_{ckpt}.json")
        if not os.path.exists(pred_file):
            return {self._bin_label(low, high, cutoffs): (None, 0) for low, high in self._bin_edges(cutoffs)}

        with open(pred_file, "r") as f:
            predictions = json.load(f)
        model, train = self.get_relative_probabilities(predictions, prior_method=prior_method)

        bin_edges = self._bin_edges(cutoffs)
        bin_labels = [self._bin_label(low, high, cutoffs) for low, high in bin_edges]
        bin_kl = {label: [] for label in bin_labels}
        bin_counts = {label: 0 for label in bin_labels}

        for y in model.keys():
            if y not in train:
                continue
            total_count = sum(self.training_data[self.data_source].get(y, {}).get(k,0) for k in ["was","were","is","are","will"])
            for (low, high), label in zip(bin_edges, bin_labels):
                if low <= total_count < high:
                    bin_kl[label].append(self.kl_divergence_from_dicts(train[y], model[y]))
                    bin_counts[label] += 1
                    break

        return {label: (np.mean(vals) if vals else None, bin_counts[label]) for label, vals in bin_kl.items()}

    def plot_avg_kl_by_bins(self, checkpoints, cutoffs=[4,10], prior_method="per_year"):
        plt.figure(figsize=(12,5))
        bin_edges = self._bin_edges(cutoffs)
        bin_labels = [self._bin_label(low, high, cutoffs) for low, high in bin_edges]
        bin_kl_over_ckpts = {label: [] for label in bin_labels}
        bin_counts_per_label = {label: 0 for label in bin_labels}

        for ckpt in checkpoints:
            kl_by_bin = self.compute_avg_kl_by_bins(ckpt, cutoffs=cutoffs, prior_method=prior_method)
            for label in bin_labels:
                kl_val, count = kl_by_bin.get(label, (None, 0))
                bin_kl_over_ckpts[label].append(kl_val)
                bin_counts_per_label[label] = max(bin_counts_per_label[label], count)

        # plot each bin
        for label in bin_labels:
            kl_vals = bin_kl_over_ckpts[label]
            ckpts_filtered = [c for c, kl in zip(checkpoints, kl_vals) if kl is not None]
            kl_filtered = [kl for kl in kl_vals if kl is not None]
            label_with_count = f"{label} ({bin_counts_per_label[label]})" if bin_counts_per_label[label] > 0 else None
            if label_with_count:
                plt.plot(ckpts_filtered, kl_filtered, marker='o', label=label_with_count)

        plt.xlabel("Checkpoint")
        plt.ylabel("Average KL Divergence")
        plt.title(f"Avg KL Divergence Trend | {self.data_source} | {prior_method}")
        plt.ylim(0,3)
        plt.grid(True)
        plt.legend(loc='upper right')
        plt.tight_layout()
        os.makedirs("checkpoints", exist_ok=True)
        plt.savefig(f"checkpoints/avg_kl_by_bins_{cutoffs[0]}_{cutoffs[-1]}_{prior_method}.png")
        plt.show()
        plt.close()


    @staticmethod
    def _bin_edges(cutoffs):
        bins = []
        low = 1
        for c in cutoffs:
            high = c-1
            if high >= low:
                bins.append((low, high))
            low = c
        bins.append((low, int(1e9)))  # last bin always ≥1
        return bins

    @staticmethod
    def _bin_label(low, high, cutoffs):
        if high >= 1e9:
            return f"freq ≥ {low}"
        else:
            return f"freq {low}-{high}"


if __name__ == "__main__":
    analyzer = KLAnalyzer(TRAINING_DATA_FILE, CHECKPOINT_DIR, "in_year_there_word_counts")
    CHECKPOINTS = list(range(250, 10001, 250))
    PRIOR_METHOD =  "global"

    # One plot at a time:
    # COUNT_RANGES = [(1,10), (1,2), (2,3), (3, 4), (5, 1000)]
    # for count_range in COUNT_RANGES:
    #     avg_kl_values = [analyzer.compute_avg_kl_for_checkpoint(ckpt, count_range=count_range, prior_method=PRIOR_METHOD)
    #                      for ckpt in CHECKPOINTS]
    #     analyzer.plot_avg_kl_over_checkpoints(CHECKPOINTS, avg_kl_values, count_range=count_range)

    # Many ranges at once, with cutoffs:
    CUTOFFS = [2, 4, 6, 10, 50, 100] # always pass cutoffs >1
    analyzer.plot_avg_kl_by_bins(CHECKPOINTS, cutoffs=CUTOFFS, prior_method=PRIOR_METHOD)
   
    PRIOR_METHOD =  "per_year"
    CUTOFFS = [2, 4, 6, 10, 50, 100] # always pass cutoffs >1
    analyzer.plot_avg_kl_by_bins(CHECKPOINTS, cutoffs=CUTOFFS, prior_method=PRIOR_METHOD)
   