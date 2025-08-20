import os
import json
import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt
import matplotlib.cm as cm

TRAINING_DATA_FILE = "../olmo_training_data/1000-3000__was.were.is.are.will__allenai_OLMo-2-0425-1B/aggregated/steps0-10000/analytics/1000-3000__was.were.is.are.will__aggregated_results_steps0-10000.json" 
CHECKPOINT_DIR = "../olmo_predictions/output_checkpoints"
ID_OOD_FILE = "id_ood/in_ood_years_in_year_there_word_counts.json"  # keys: "in_distribution" and "ood"


class KLAnalyzer:
    def __init__(self, training_data_file, checkpoint_dir, data_source="in_year_there_word_counts", year_range=(1000,2999)):
        self.training_data_file = training_data_file
        self.checkpoint_dir = checkpoint_dir
        self.data_source = data_source
        self.year_range = year_range

        with open(training_data_file, "r") as f:
            self.training_data = json.load(f)

        with open(ID_OOD_FILE, "r") as f:
            id_ood = json.load(f)
            self.in_years = [str(y) for y in id_ood["in_distribution"]]
            self.ood_years = [str(y) for y in id_ood["ood"]]

    @staticmethod
    def kl_divergence_from_dicts(dist_p, dist_q, epsilon=1e-12):
        all_categories = set(dist_p.keys()) | set(dist_q.keys())
        categories = sorted(all_categories)
        p = np.array([dist_p.get(c,0) for c in categories])
        q = np.array([dist_q.get(c,0) for c in categories])
        p = np.maximum(p, epsilon) / np.sum(np.maximum(p, epsilon))
        q = np.maximum(q, epsilon) / np.sum(np.maximum(q, epsilon))
        return entropy(p, q)

    def get_relative_probabilities(self, predictions):
        if self.data_source == "in_year_there_word_counts":
            return self.get_relative_probabilities_global(predictions)
        else:
            return self.get_relative_probabilities_per_year(predictions)

    def get_relative_probabilities_global(self, predictions):
        # compute model probabilities
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

        # global training distribution
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

        return year_to_model, year_to_train

    def get_relative_probabilities_per_year(self, predictions):
        # compute model probabilities
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

        # per-year training distribution
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

        return year_to_model, year_to_train

    def compute_kl_in_vs_ood(self, ckpt=10000):
        """
        Compute average KL divergence for two hardcoded buckets:
        in-distribution vs out-of-distribution, using checkpoint predictions.
        """

        # this helps me make the json file for in vs out of distr

        pred_file = os.path.join(self.checkpoint_dir, f"checkpoint_{ckpt}.json")
        if not os.path.exists(pred_file):
            raise FileNotFoundError(f"Checkpoint file not found: {pred_file}")

        with open(pred_file, "r") as f:
            predictions = json.load(f)

        model, train = self.get_relative_probabilities(predictions)

        kl_in = []
        kl_ood = []

        for year, probs in model.items():
            kl_val = self.kl_divergence_from_dicts(train[year], probs)
            if year in self.in_years:
                kl_in.append(kl_val)
            elif year in self.ood_years:
                kl_ood.append(kl_val)

        avg_kl_in = np.mean(kl_in) if kl_in else None
        avg_kl_ood = np.mean(kl_ood) if kl_ood else None

        # print(f"Checkpoint {ckpt} KL divergence:")
        # print(f"  ID years: {avg_kl_in} (n={len(kl_in)})")
        # print(f"  OOD years: {avg_kl_ood} (n={len(kl_ood)})")

        return {"in_distribution": (avg_kl_in, len(kl_in)),
                "ood": (avg_kl_ood, len(kl_ood))}


    def plot_kl_over_checkpoints(self, checkpoints):
        """
        Compute and plot KL divergence over all checkpoints for:
        - In-distribution (blue)
        - Out-of-distribution (red)
        """
        with open(ID_OOD_FILE, "r") as f:
            id_ood = json.load(f)
            in_years = [str(y) for y in id_ood["in_distribution"]]
            ood_years = [str(y) for y in id_ood["ood"]]

        kl_in_over_ckpt = []
        kl_ood_over_ckpt = []

        for ckpt in checkpoints:
            kl_vals = self.compute_kl_in_vs_ood(ckpt)
            kl_in_over_ckpt.append(kl_vals["in_distribution"][0])  # avg KL for in-distribution
            kl_ood_over_ckpt.append(kl_vals["ood"][0])             # avg KL for OOD

        # update data count type name for the plots
        if self.data_source=="in_year_there_word_counts":
            ds_label = "Exact string match method (avg.)"
        elif self.data_source=="in_year_cooccurence":  
            ds_label = "String match + cooccurence method"
        else:
            ds_label = self.data_source

        plt.figure(figsize=(12,5))
        plt.plot(checkpoints, kl_in_over_ckpt, marker='o', color='blue', label='In-distribution')
        plt.plot(checkpoints, kl_ood_over_ckpt, marker='o', color='red', label='Out-of-distribution')
        plt.xlabel("Checkpoint")
        plt.ylabel("Average KL Divergence")
        plt.title(f"KL Divergence Over Checkpoints | {ds_label}")
        plt.ylim(0, 1.2)
        plt.grid(True)
        plt.legend(loc='upper right')
        os.makedirs("checkpoints", exist_ok=True)
        plt.tight_layout()
        plt.savefig(f"checkpoints/kl_in_vs_ood_over_checkpoints_{ds_label}.png")
        plt.show()
        plt.close()


if __name__ == "__main__":
    CHECKPOINTS = list(range(250, 10001, 250)) 

    analyzer = KLAnalyzer(TRAINING_DATA_FILE, CHECKPOINT_DIR, "in_year_there_word_counts")
    analyzer.plot_kl_over_checkpoints(CHECKPOINTS)

    # need the new key for this!!
    # analyzer2 = KLAnalyzer(TRAINING_DATA_FILE, CHECKPOINT_DIR, "in_year_coocur_word_count")
    # analyzer2.plot_kl_over_checkpoints(CHECKPOINTS)
