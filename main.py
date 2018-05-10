import time
from missing_values import replace_missing_values, remove_missing_values
from balance import balance
from feature_selection import feature_selection
from train import train_model
from evaluate import evaluate_model, summary
from utils import load_arff, data_distribution


def pipeline(
        train_data, test_data, balance_config,
        fs_config, model_config):

    keys = []
    data_description = {}
    # Handle Missing Values.
    train_data, test_data = remove_missing_values(train_data, test_data, 0.2)
    train_data = replace_missing_values(train_data)

    # Rebalance Data
    keys.append(balance_config["key"])
    train_data = balance(
        train_data, balance_config["classname"],
        balance_config["option"])

    # Feature Selection
    keys.append(fs_config["key"])
    train_data, test_data = feature_selection(
        train_data, test_data, fs_config["option"])

    data_description["train"] = data_distribution(train_data)
    data_description["test"] = data_distribution(test_data)

    # Parameter tunning
    keys.append(model_config["key"])
    model = train_model(
        train_data,
        model_config["model_name"],
        model_config["option"])

    evaluator = evaluate_model(model, train_data, test_data)
    return evaluator, data_description, "-----".join(keys)


if __name__ == "__main__":
    import sys
    import config
    from weka.core import jvm

    jvm.start(packages=True, max_heap_size="8g")
    train_file, test_file = sys.argv[1:]
    train_data = load_arff(train_file)
    test_data = load_arff(test_file)
    evaluations = {}
    evaluation_measure_output = open("evaluation.csv", "w")
    evaluation_detail_output = open("evaluation.txt", "w")
    progress_output = open("progress.txt", "w")
    count = 0
    for balance_config in config.balance_configs:
        for fs_config in config.fs_configs:
            for model_config in config.model_configs:
                start_time = time.time()
                evaluator, data_description, key = pipeline(
                    train_data, test_data, balance_config,
                    fs_config, model_config)
                end_time = time.time()
                duration = "{:.3f}".format(end_time - start_time)
                count += 1
                progress_output.write(f"completed {count}/{config.total_pipelines} pipelines, current is {key}, cost {duration} seconds\n")
                #except Exception as error:
                #    print("error happens", error)
                eval_measures = [key]
                for index in [0, 1]:
                    label = train_data.class_attribute.values[index]
                    eval_measures += [
                        "f_measure " + label,
                        str("{:.3f}".format(evaluator.f_measure(index))),
                        "f_prc " + label,
                        str("{:.3f}".format(evaluator.area_under_prc(index))),
                        "f_roc " + label,
                        str("{:.3f}".format(evaluator.area_under_roc(index)))
                    ]


                evaluation_measure_output.write(",".join(eval_measures) + "\n")
                evaluation_detail_output.write("-" * 120 + "\n")
                evaluation_detail_output.write("-" * 120 + "\n")
                evaluation_detail_output.write("pipeline: " + key + "\n")
                evaluation_detail_output.write("data description: \n")
                evaluation_detail_output.write("train data summary: \n")
                evaluation_detail_output.write(data_description["train"] + "\n")
                evaluation_detail_output.write("test data summary: \n")
                evaluation_detail_output.write(data_description["test"] + "\n")
                evaluation_detail_output.write(summary(evaluator) + "\n")
                evaluation_measure_output.flush()
                evaluation_detail_output.flush()
                progress_output.flush()
    jvm.stop()
