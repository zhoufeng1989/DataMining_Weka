import itertools
resample_filter = "weka.filters.supervised.instance.Resample"
gain_ratio = "weka.attributeSelection.GainRatioAttributeEval"
ranker = "weka.attributeSelection.Ranker"

Naive_Bayes_optimal_config = {
    "balance_config": {
        "classname": resample_filter,
        "option": ["-B", "1.0", "-Z", "50"],
    },
    "fs_config": {
        "option": [
            "-E", gain_ratio,
            "-S", f'{ranker} "-T" "-1.0E308" "-N" "5"'
        ]
    },
    "alg_config": {
        "classname": "weka.classifiers.bayes.NaiveBayes",
        "option": ["-K"]
    }
}
J48_optimal_config = {
    "balance_config": {
        "classname": resample_filter,
        "option": ["-B", "1.0", "-Z", "50"],
    },
    "fs_config": {
        "option": [
            "-E", gain_ratio,
            "-S", f'{ranker} "-T" "-1.0E308" "-N" "5"'
        ]
    },
    "alg_config": {
        "classname": "weka.classifiers.trees.J48",
        "option": ["-C", "0.2", "-M", "15"]
    }
}


IBk_optimal_config = {
    "balance_config": {
        "classname": resample_filter,
        "option": ["-B", "1.0", "-Z", "50"],
    },
    "fs_config": {
        "option": [
            "-E", gain_ratio,
            "-S", f'{ranker} "-T" "-1.0E308" "-N" "5"'
        ]
    },
    "alg_config": {
        "classname": "weka.classifiers.lazy.IBk",
        "option": [
            "-K", "20",
            "-W", "0",
            "-A", f"\"weka.core.neighboursearch.LinearNNSearch -A \\\"weka.core.ManhattanDistance -R first-last\\\"\"",
        ]
    }
}

IBk_optimal_config2 = {
    "balance_config": {
        "classname": resample_filter,
        "option": ["-B", "1.0", "-Z", "50"],
    },
    "fs_config": {
        "option": [
            "-E", gain_ratio,
            "-S", f'{ranker} "-T" "-1.0E308" "-N" "5"'
        ]
    },
    "alg_config": {
        "classname": "weka.classifiers.lazy.IBk",
        "option": [
            "-K", "20",
            "-W", "0",
            "-A", f"weka.core.neighboursearch.LinearNNSearch -A \"weka.core.ManhattanDistance -R first-last\"",
        ]
    }
}

# SMO_optimal_config = {
#     "balance_config": {
#         "classname": resample_filter,
#         "option": ["-B", "0.7", "-Z", "50"],
#     },
#     "fs_config": {
#         "option": [
#             "-E", gain_ratio,
#             "-S", f'{ranker} "-T" "-1.0E308" "-N" "5"'
#         ]
#     },
#     "alg_config": {
#         "classname": "weka.classifiers.functions.SMO",
#         "option": [
#             "-C", "6",
#             "-L", "0.001",
#             "-P", "1.0E-12",
#             # Normalize data
#             "-N", "0",
#             "-V", "-1",
#             "-W", "1",
#             "-K", "\"weka.classifiers.functions.supportVector.RBFKernel -G 0.4 -C 250007\"",
#             "-calibrator", "\"weka.classifiers.functions.Logistic -R 1.0E-8 -M -1 -num-decimal-places 4\""
#         ]
#     }
# }
#
# SMO_optimal_config2 = {
#     "balance_config": {
#         "classname": resample_filter,
#         "option": ["-B", "0.7", "-Z", "50"],
#     },
#     "fs_config": {
#         "option": [
#             "-E", gain_ratio,
#             "-S", f'{ranker} "-T" "-1.0E308" "-N" "5"'
#         ]
#     },
#     "alg_config": {
#         "classname": "weka.classifiers.functions.SMO",
#         "option": [
#             "-C", "6",
#             "-L", "0.001",
#             "-P", "1.0E-12",
#             # Normalize data
#             "-N", "0",
#             "-V", "-1",
#             "-W", "1",
#             "-K", "weka.classifiers.functions.supportVector.RBFKernel -G 0.4 -C 250007",
#             "-calibrator", "weka.classifiers.functions.Logistic -R 1.0E-8 -M -1 -num-decimal-places 4"
#         ]
#     }
# }

RF_optimal_config = {
    "balance_config": {
        "classname": resample_filter,
        "option": ["-B", "1.0", "-Z", "50"],
    },
    "fs_config": {
        "option": [
            "-E", gain_ratio,
            "-S", f'{ranker} "-T" "-1.0E308" "-N" "5"'
        ]
    },
    "alg_config": {
        "classname": "weka.classifiers.trees.RandomForest",
        "option": [
            "-P", "30",
            "-I", "80",
            "-num-slots", "1",
            "-K", "0",
            "-M", "1.0",
            "-V", "0.001",
            "-S", "1"
        ]
    }
}

optimal_configs = [J48_optimal_config, Naive_Bayes_optimal_config, IBk_optimal_config, RF_optimal_config]
optimal_configs2 = [J48_optimal_config, Naive_Bayes_optimal_config, IBk_optimal_config2, RF_optimal_config]


def generate_bagging_config():
    config = []
    model_name = "weka.classifiers.meta.Bagging"
    percentages = [30, 50, 80, 100]
    iters = [5, 10, 20, 40, 80, 100, 160]
    for percentage in percentages:
        for iter_count in iters:
            for optimal_config in optimal_configs2:
                alg_config = optimal_config["alg_config"]
                alg_name = alg_config["classname"]
                option = [
                    "-P", str(percentage),
                    "-S", "1",
                    "-num-slots", "1",
                    "-I", str(iter_count),
                    # "-W", " ".join([alg_name, "--", " ".join(alg_config["option"])])
                    "-W", alg_name,
                    "--"
                ] + alg_config["option"]
                key = f"Bagging-P-{percentage}-I-{iter_count}-W-{alg_name}"
                config.append({
                    "balance_config": optimal_config["balance_config"],
                    "fs_config": optimal_config["fs_config"],
                    "option": option,
                    "key": key,
                    "model_name": model_name
                })
    return config



def generate_adaboost_config():
    config = []
    model_name = "weka.classifiers.meta.AdaBoostM1"
    iters = [5, 10, 20, 40, 80, 100, 160]
    for iter_count in iters:
        for optimal_config in optimal_configs2:
            alg_config = optimal_config["alg_config"]
            alg_name = alg_config["classname"]
            option = [
                "-P", "100",
                "-S", "1",
                "-I", str(iter_count),
                #"-W", " ".join([alg_name, "--", " ".join(alg_config["option"])])
                "-W", alg_name,
                "--"
            ] + alg_config["option"]
            key = f"AdaBoost-I-{iter_count}-W-{alg_name}"
            config.append({
                "balance_config": optimal_config["balance_config"],
                "fs_config": optimal_config["fs_config"],
                "option": option,
                "key": key,
                "model_name": model_name
            })
    return config




def generate_stack_config():
    config = []
    preprocess_config = {
        "balance_config": {
            "classname": resample_filter,
            "option": ["-B", "1.0", "-Z", "50"],
        },
        "fs_config": {
            "option": [
                "-E", gain_ratio,
                "-S", f'{ranker} "-T" "-1.0E308" "-N" "5"'
            ]
        }
    }
    model_name = "weka.classifiers.meta.Stacking"
    meta_classifiers = {
        "weka.classifiers.trees.J48": ["-C", "0.25", "-M", "2"],
        "weka.classifiers.lazy.IBk": [
            "-K", "3",
            "-W", "0",
            "-A", f"\"weka.core.neighboursearch.LinearNNSearch -A \\\"weka.core.EuclideanDistance -R first-last\\\"\"",
        ]
    }
    for meta_alg, meta_options in meta_classifiers.items():
        stacks = []
        alg_count = len(optimal_configs)
        counts = [i + 1 for i in range(alg_count)][1:]
        for i in counts:
            stacks.extend(list(itertools.combinations(range(alg_count), i)))
        for stack in stacks:
            key = f"Stack-M-{meta_alg.split('.')[-1]}-"
            option = [
                "-X", "10",
                "-M", " ".join([meta_alg, " ".join(meta_options)]),
                "-S", "1",
                "-num-slots", "1",
            ]
            for i in stack:
                alg_config = optimal_configs[i]["alg_config"]
                alg_name = alg_config["classname"]
                key += f"-{alg_name.split('.')[-1]}"
                option.extend([
                    "-B",
                    f"{' '.join([alg_name] + alg_config['option'])}"
                ])
            config.append({
                "balance_config": preprocess_config["balance_config"],
                "fs_config": preprocess_config["fs_config"],
                "option": option,
                "key": key,
                "model_name": model_name
            })
    for i in config:
        print(i["key"])
    return config


bagging_config = generate_bagging_config()
adaboost_config = generate_adaboost_config()
stack_config = generate_stack_config()
ensemble_config = bagging_config + adaboost_config + stack_config
total_pipelines = len(ensemble_config)


if __name__ == "__main__":
    print(f"bagging config count is {len(bagging_config)}")
    print(f"adaboost config count is {len(adaboost_config)}")
    print(f"stack config count is {len(stack_config)}")
