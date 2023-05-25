import os
import json
import argparse
from scipy.stats.stats import spearmanr, pearsonr, kendalltau

def aggregateMetaEvaluation(aspect, results_dir):
    completion_filepath_list = [os.path.join(results_dir, filename) for filename in os.listdir(results_dir) if filename.startswith("completions")]
    
    expert_annotations = []
    predictions = []

    annotation_key = "expert_{}_mean".format(aspect)
    prediction_key = "score_{}".format(aspect)

    for filepath in completion_filepath_list:
        with open(filepath) as f_comp:
            fold_data = json.load(f_comp)
        fold_expert_annotations = [example[annotation_key] for example in fold_data]
        fold_predictions = [example[prediction_key] for example in fold_data]
        expert_annotations.extend(fold_expert_annotations)
        predictions.extend(fold_predictions)
    
    expert_correlations = {}
    expert_correlations["pearson"] = pearsonr(predictions, expert_annotations)[0]
    expert_correlations["spearman"] = spearmanr(predictions, expert_annotations)[0]
    expert_correlations["kendall"] = kendalltau(predictions, expert_annotations)[0]

    aggregate_filepath = os.path.join(results_dir, "aggregate_correlations")
    with open(aggregate_filepath, mode="w") as f_aggregate:
        json.dump(expert_correlations, f_aggregate, indent=1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--aspect', required=True)
    parser.add_argument('--results_dir', required=True)

    args = parser.parse_args()
    aggregateMetaEvaluation(
        aspect=args.aspect,
        results_dir=args.results_dir
    )
    