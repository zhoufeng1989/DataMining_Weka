# Data Balance
def generate_balance_config():
    ratios = [0.7, 1.0]
    resample_percentages = [50, 100]
    config = []
    for ratio in ratios:
        for percentage in resample_percentages:
            classname = "weka.filters.supervised.instance.Resample"
            options = ["-B", str(ratio), "-S", "1", "-Z", str(percentage)]
            key = f"Resample-B-{ratio}-Z-{percentage}"
            config.append({"classname": classname, "option": options, "key": key})

    # SMOTE is slow.
    # for percentage in resample_percentages:
    #     classname = "weka.filters.supervised.instance.SMOTE"
    #     options = ["-C", "last", "-K", "5", "-P", str(percentage), "-S", "1"]
    #     key = f"SMOTE-P-{percentage}"
    #     config.append({"classname": classname, "option": options, "key": key})

    return config


# Feature selection
def generate_fs_config():
    attr_cnt = [5, 10, 15]
    search_alg = "weka.attributeSelection.Ranker"
    config = []
    # GainRatioAttributeEval
    for cnt in attr_cnt:
        search_options = ["-T", "-1.0E308", "-N", str(cnt)]
        option = [
            "-E", "weka.attributeSelection.GainRatioAttributeEval",
            "-S", f"{search_alg} {' '.join(search_options)}"
        ]
        key = f"GainRatio-Ranker-N-{cnt}"
        config.append({"option": option, "key": key})
    # CfsSubsetEval
    option = [
        "-E", "weka.attributeSelection.CfsSubsetEval -P 1 -E 1",
        "-S", "weka.attributeSelection.GreedyStepwise -T -Infinity -N -1 -num-slots 1"
    ]
    key = "Cfs-GreedyStepwise"
    config.append({"option": option, "key": key})

    # Wrapper
    option = [
        "-E", "weka.attributeSelection.WrapperSubsetEval -B weka.classifiers.bayes.NaiveBayes -F 5 -T 0.01 -R 1 -E DEFAULT --",
        "-S", "weka.attributeSelection.GreedyStepwise -T -Infinity -N -1 -num-slots 1"
    ]
    key = "Wrapper-GreedyStepwise"
    config.append({"option": option, "key": key})

    return config


# Bayes
def generate_bayes_config():
    options = [None, "-K", "-D"]
    config = []
    for o in options:
        key = "NaiveBayes"
        option = []
        if o:
            key += o
            option = [o]
        config.append({
            "option": option,
            "key": key,
            "model_name": "weka.classifiers.bayes.NaiveBayes"
        })
    return config


# Decision Tree
def generate_J48_config():
    options = {
        "confidence_factor": [0.1, 0.2, 0.3, 0.4],
        "min_num_obj": [2, 3, 4, 5]
    }
    config = []

    for confidence_factor in options["confidence_factor"]:
        for min_num_obj in options["min_num_obj"]:
            key = f"J48-C-{confidence_factor}-M-{min_num_obj}"
            option = ["-C", str(confidence_factor), "-M", str(min_num_obj)]
            config.append({
                "option": option,
                "key": key,
                "model_name": "weka.classifiers.trees.J48"
            })
    return config


# MLP
def generate_mlp_config():
    options = {
        "learning_rate": [0.03, 0.3, 3],
        "training_time": [50, 400]
    }
    config = []
    for learning_rate in options["learning_rate"]:
        for training_time in options["training_time"]:
            option = [
                "-L", str(learning_rate),
                "-M", "0.2",
                "-N", str(training_time),
                "-V", "0",
                "-S", "0",
                "-E", "20",
                "-H", "a"]
            key = f"MLP-L-{learning_rate}-N-{training_time}"
            config.append({
                "option": option,
                "key": key,
                "model_name": "weka.classifiers.functions.MultilayerPerceptron"
            })
    return config


# SVM
def generate_smv_config():
    options = {
        "c": [0.1, 0.5, 2.5],
        "exponent": [2, 4, 6],
        "gamma": [0.1, 0.5, 2.5, 10]
    }
    config = []
    for c in options["c"]:
        for exponent in options["exponent"]:
            kernel_name = "weka.classifiers.functions.supportVector.PolyKernel"
            kernel_option = " ".join([kernel_name, "-E", str(exponent), "-C", "250007"])
            option = [
                "-C", str(c),
                "-L", "0.001",
                "-P", "1.0E-12",
                # Normalize data
                "-N", "0",
                "-V", "-1",
                "-W", "1",
                "-K", kernel_option,
                "-calibrator", "weka.classifiers.functions.Logistic -R 1.0E-8 -M -1 -num-decimal-places 4"
            ]
            key = f"SMO-C-{c}-PolyKernel-E-{exponent}"
            config.append({
                "option": option,
                "key": key,
                "model_name": "weka.classifiers.functions.SMO"
            })

        for gamma in options["gamma"]:
            kernel_name = "weka.classifiers.functions.supportVector.RBFKernel"
            kernel_option = " ".join([kernel_name, "-G", str(gamma), "-C", "250007"])
            option = [
                "-C", str(c),
                "-L", "0.001",
                "-P", "1.0E-12",
                # Normalize data
                "-N", "0",
                "-V", "-1",
                "-W", "1",
                "-K", kernel_option,
                "-calibrator", "weka.classifiers.functions.Logistic -R 1.0E-8 -M -1 -num-decimal-places 4"
            ]
            key = f"SMO-C-{c}-RBFKernel-G-{exponent}"
            config.append({
                "option": option,
                "key": key,
                "model_name": "weka.classifiers.functions.SMO"
            })
    return config


def generate_knn_config():
    options = {
        "k": [1, 3, 5],
        "distance_weight": [None, "-I", "-F"]
    }
    config = []

    for k in options["k"]:
        for distance_weight in options["distance_weight"]:
            option = [
                "-K", str(k),
                "-W", "0",
                "-A", "weka.core.neighboursearch.LinearNNSearch -A \"weka.core.EuclideanDistance -R first-last\""
            ]
            key = f"IBk-K-{k}"
            if distance_weight:
                option.append(distance_weight)
                key += f"-DW{distance_weight}"
            config.append({
                "option": option,
                "key": key,
                "model_name": "weka.classifiers.lazy.IBk"})
    return config


balance_configs = generate_balance_config()
fs_configs = generate_fs_config()
bayes_configs = generate_bayes_config()
J48_configs = generate_J48_config()
mlp_configs = generate_mlp_config()
smv_configs = generate_smv_config()
knn_configs = generate_knn_config()

model_configs = (
    bayes_configs + J48_configs +
    mlp_configs + smv_configs + knn_configs)

total_pipelines = len(balance_configs) * len(fs_configs) * len(model_configs)
print(total_pipelines)
