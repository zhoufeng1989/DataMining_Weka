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

    data_description = {}
    # Handle Missing Values.
    train_data, test_data = remove_missing_values(train_data, test_data, 0.8)
    train_data = replace_missing_values(train_data)

    # Rebalance Data
    train_data = balance(
        train_data, balance_config["classname"],
        balance_config["option"])

    # Feature Selection
    train_data, test_data = feature_selection(
        train_data, test_data, fs_config["option"])

    data_description["train"] = data_distribution(train_data)
    data_description["test"] = data_distribution(test_data)

    # Parameter tunning
    model = train_model(
        train_data,
        model_config["model_name"],
        model_config["option"])

    evaluator = evaluate_model(model, train_data, test_data)
    return evaluator, data_description


if __name__ == "__main__":
    import sys
    from config_ensemble import ensemble_config
    from weka.core import jvm

    jvm.start(packages=True, max_heap_size="8g")
    train_file, test_file = sys.argv[1:]
    train_data = load_arff(train_file)
    test_data = load_arff(test_file)
    evaluations = {}
    evaluation_measure_output = open("evaluation_ensemble.csv", "a")
    evaluation_detail_output = open("evaluation_ensemble.txt", "a")
    progress_output = open("progress.txt", "a")
    count = 0
    total_pipelines = len(ensemble_config)
    for config in ensemble_config:
        balance_config = config["balance_config"]
        fs_config = config["fs_config"]
        model_config = {
            "option": config["option"],
            "model_name": config["model_name"]
        }
        key = config["key"]
        start_time = time.time()
        evaluator, data_description = pipeline(
            train_data, test_data, balance_config,
            fs_config, model_config)
        end_time = time.time()
        duration = "{:.3f}".format(end_time - start_time)
        count += 1
        progress_output.write(f"completed {count}/{total_pipelines} pipelines, current is {key}, cost {duration} seconds\n")
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
