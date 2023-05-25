import random
import json
import argparse
import os
from scipy.optimize import linear_sum_assignment
import numpy as np
import itertools

from bucketizationUtils import bucketize_data

class CrossValidator:
    def __init__(self, data_path, results_dir, fold_group_seed, test_sample_seed, mode, setting, sampling_procedure):
        self.data_path = data_path
        self.results_dir = results_dir
        self.fold_group_seed = fold_group_seed
        self.test_sample_seed = test_sample_seed
        self.mode = mode
        self.setting = setting
        self.sampling_procedure = sampling_procedure

    def create_dict(self):
        """
        Returns a dictionary of the form
        {textid_1: [16 examples], textid_2: [16 examples],..., textid_100: [16 examples]}
        """
        data = json.load(open(self.data_path))
        
        self.text_dict = {}
        for example in data:
            if example["id"] not in self.text_dict:
                self.text_dict[example["id"]] = []
            self.text_dict[example["id"]].append(example)
        
    def split_dict(self):
        """
        Returns a list of the form
        [
          {textid_1: [16 examples],..., textid_4: [16 examples]}, 
          {textid_5: [16 examples],..., textid_8: [16 examples]},
          ...
          {textid_97: [16 examples],..., textid_100: [16 examples]}
        ]
        Grouping is determined by  fold_group_seed
        """

        random.seed(self.fold_group_seed)   # governs which texts get grouped together to form the folds
        l = list(self.text_dict.items())
        random.shuffle(l)
        text_dict_shuffled = dict(l)

        n = 4 # number of texts in each fold
        items = list(text_dict_shuffled.items())
        folds = [dict(items[x : x + n]) for x in range(0, len(text_dict_shuffled), n)]

        self.folds = folds
        return folds
    
    def get_aspect_ice_single_bucketed(self, text_examples, aspect):
        """
        Arguments:
            text_examples:  This is a dictionary corresponding to a fold. For instance, 
                            text_examples can contain {textid_1: [16 examples],..., textid_4: [16 examples]}
            aspect:         Aspect for which the in-context examples will be chosen
        Returns a list of size 4, where each entry is a selected in-context example)
        """
        bucket_scores=[0.25, 0.5, 0.75, 1]
        buckets = [bucketize_data(text_examples[example_id], aspect, bucket_scores) for example_id in text_examples]
        cost = -np.array([[len(b) for b in bucket] for bucket in buckets])

        row_ind, col_ind = linear_sum_assignment(cost)

        # col_ind are the bucket id from which we need to sample from the text_examples
        incontext_examples_aspect = []
        for idx, bucket in enumerate(buckets):
            score_bucket_of_interest = bucket[col_ind[idx]]
            if len(score_bucket_of_interest) == 0:
                score_bucket_of_interest = list(itertools.chain(*bucket))
            random.seed(self.fold_group_seed + idx)
            random.shuffle(score_bucket_of_interest)
            sampled_example = score_bucket_of_interest[0]
            incontext_examples_aspect.append(sampled_example)

        return incontext_examples_aspect

    def get_aspect_ice_single_uniform(self, text_examples, aspect):
        """
        Arguments:
            text_examples:  This is a dictionary corresponding to a fold. For instance, 
                            text_examples can contain {textid_1: [16 examples],..., textid_4: [16 examples]}
            aspect:         Aspect for which the in-context examples will be chosen
        Returns a list of size 4, where each entry is a selected in-context example)
        """
        incontext_examples_aspect = []

        for example_list in text_examples.values():
            sampled_example = random.sample(example_list, 1)[0]
            incontext_examples_aspect.append(sampled_example)

        return incontext_examples_aspect

    def get_aspect_ice_multi_bucketed(self, text_examples, aspect):
        """
        Arguments:
            text_examples:  This is a dictionary corresponding to a fold. For instance, 
                            text_examples can contain {textid_1: [16 examples],..., textid_4: [16 examples]}
            aspect:         Aspect for which the in-context examples will be chosen
        Returns a list of size 4 (one corresponding to each fold), where each entry is again a list of size 4 (each is an in-context example))
        """
        buckets_list = [bucketize_data(text_examples[example_id], aspect) for example_id in text_examples]

        selected_examples = []

        for idx, buckets in enumerate(buckets_list):
            current_text_examples = []
            random.seed(self.fold_group_seed + idx)
            missing_buckets = 0
            for bucket in buckets:
                if len(bucket) > 0:
                    random.shuffle(bucket)
                    sampled_example = bucket[0]
                    current_text_examples.append(sampled_example)
                    bucket.remove(sampled_example)
                else:
                    missing_buckets += 1
                
            combined = list(itertools.chain(*buckets))
            random.shuffle(combined)

            current_text_examples.extend(combined[:missing_buckets])
            selected_examples.append(current_text_examples)
        with open("temporary", "w") as trainfile:
            json.dump(selected_examples, trainfile, indent=1) 
        return selected_examples  

    def get_aspect_ice_multi_uniform(self, text_examples, aspect):
        """
        Arguments:
            text_examples:  This is a dictionary corresponding to a fold. For instance, 
                            text_examples can contain {textid_1: [16 examples],..., textid_4: [16 examples]}
            aspect:         Aspect for which the in-context examples will be chosen
        Returns a list of size 4 (one corresponding to each fold), where each entry is again a list of size 4 (each is an in-context example))
        """
        incontext_examples_aspect = []

        for example_list in text_examples.values():
            sampled_example = random.sample(example_list, 4)
            incontext_examples_aspect.append(sampled_example)

        return incontext_examples_aspect      

    def get_all_incontext_examples_bucketed(self):
        """
        Calls get_aspect_ice_<>_bucketed for each of the 25 folds and with each of the 4 aspects with bucket sampling
        Returns the resulting list of examples for all folds and aspects
        """
        aspects = ["coherence", "consistency", "relevance", "fluency"]
        in_context_examples_eachfold = dict()
        for aspect in aspects:
            if self.mode == "single":
                in_context_examples_eachfold[aspect] = [self.get_aspect_ice_single_bucketed(fold, aspect) for fold in self.folds]  
            if self.mode == "multi":   
                in_context_examples_eachfold[aspect] = [self.get_aspect_ice_multi_bucketed(fold, aspect) for fold in self.folds]     

        ice_filename = "ic_examples_{}_{}.json".format(self.fold_group_seed, self.mode)
        ice_filepath = os.path.join(self.results_dir, ice_filename)

        with open(ice_filepath, "w") as icefile:
            json.dump(in_context_examples_eachfold, icefile, indent=1) 
        self.incontext_examples = in_context_examples_eachfold
    
    def get_all_incontext_examples_uniform(self):
        """
        Calls get_aspect_ice_<>_uniform for each of the 25 folds and with each of the 4 aspects with uniform sampling
        Returns the resulting list of examples for all folds and aspects
        """
        aspects = ["coherence", "consistency", "relevance", "fluency"]
        in_context_examples_eachfold = dict()
        for aspect in aspects:
            if self.mode == "single":
                in_context_examples_eachfold[aspect] = [self.get_aspect_ice_single_uniform(fold, aspect) for fold in self.folds]  
            if self.mode == "multi":   
                in_context_examples_eachfold[aspect] = [self.get_aspect_ice_multi_uniform(fold, aspect) for fold in self.folds]     

        ice_filename = "ic_examples_{}_{}.json".format(self.fold_group_seed, self.mode)
        ice_filepath = os.path.join(self.results_dir, ice_filename)

        with open(ice_filepath, "w") as icefile:
            json.dump(in_context_examples_eachfold, icefile, indent=1) 
        self.incontext_examples = in_context_examples_eachfold

    def merge_folds(self, folds_idx_list):
        merged = []
        for idx in folds_idx_list:
            fold = self.folds[idx]
            for grouped_example_list in fold.values():
                merged.extend(grouped_example_list)
        return merged

    def create_train_file(self, train_fold_idx, aspect):
        train_sample = self.incontext_examples[aspect][train_fold_idx]
        train_filename = "train_examples_{}_{}_{}.json".format(train_fold_idx, aspect, self.mode)
        train_filepath = os.path.join(self.results_dir, train_filename)
        with open(train_filepath, "w") as trainfile:
            json.dump(train_sample, trainfile, indent=1) 
        return train_filepath

    def create_test_file(self, test_folds_idx_list, aspect):
        # If setting is ice, then we are in a multifold setting and the test file will have 100 examples
        # If setting is crossval, then we are in a single fold setting, and the test file will have 64 examples

        if self.setting == "ice":
            test_examples_list = self.merge_folds(test_folds_idx_list)
            random.seed(self.test_sample_seed)
            test_sample = random.sample(test_examples_list, 100)

            # Chunting changes to get alternate sample:
            test_filename = "test_examples.json"
            # test_examples_list_1 = [example for example in test_examples_list if example not in test_sample]
            # test_sample_1 = random.sample(test_examples_list_1, 100)
            # test_filename = "test_examples_alt.json"
            # end

        elif self.setting == "crossval":
            test_fold_idx = test_folds_idx_list[0]
            test_sample = []
            for text_example_list in self.folds[test_fold_idx].values():
                test_sample.extend(text_example_list)
            test_filename = "test_examples_{}.json".format(test_fold_idx)
        else:
            pass

        
        test_filepath = os.path.join(self.results_dir, test_filename)
        with open(test_filepath, "w") as testfile:
            # chunting change:
            json.dump(test_sample, testfile, indent=1)
            # json.dump(test_sample_1, testfile, indent=1)
        return test_filepath
    
    def prepare_data(self):
        # Possible settings:
        #   crossval: run cross-validation on all folds
        #   ice: pick different folds for ICE

        self.create_dict()
        self.split_dict()

        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

        if self.sampling_procedure == "uniform":
            self.get_all_incontext_examples_uniform()
        else:
            self.get_all_incontext_examples_bucketed()

        aspects = ["coherence", "consistency", "relevance", "fluency"]
        for aspect in aspects:
            if self.setting == "ice":
                n = 3 # hardcode number of ICE combinations to try
                test_folds_list = list(range(5, len(self.folds)))   # hardcode test folds to be 5 or above, 
                                                                    # in case n is increased to 4 or 5 later
                for train_fold_idx in range(n):
                    train_filepath = self.create_train_file(train_fold_idx, aspect)
                test_filepath = self.create_test_file(test_folds_list, aspect)

            elif self.setting == "crossval":
                for i in range(25):
                    train_fold_idx = i
                    test_folds_list = [i]

                    train_filepath = self.create_train_file(train_fold_idx, aspect)
                    test_filepath = self.create_test_file(test_folds_list, aspect)
                
            else:
                pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_filepath', required=True)
    parser.add_argument('--results_dir', required=True)
    parser.add_argument('--fold_group_seed', required=True, type=int)
    parser.add_argument('--test_sample_seed', required=True, type=int)
    parser.add_argument('--mode', required=True)
    parser.add_argument('--setting', required=True)
    parser.add_argument('--sampling_procedure', required=True)

    args = parser.parse_args()

    cv = CrossValidator(
        data_path=args.data_filepath,
        results_dir = args.results_dir,
        fold_group_seed=args.fold_group_seed,
        test_sample_seed=args.test_sample_seed,
        mode=args.mode,
        setting=args.setting,
        sampling_procedure=args.sampling_procedure
    )
    cv.prepare_data()