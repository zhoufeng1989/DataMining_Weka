from weka.filters import Filter
from utils import load_arff, class_distribution


def balance(data, classname, options):
    data_filter = Filter(
        classname=classname,
        options=options)
    data_filter.inputformat(data)
    return data_filter.filter(data)


if __name__ == "__main__":
    import sys
    from weka.core import jvm
    from config import rebalance_configs

    train_file = sys.argv[1]

    jvm.start(packages=True, max_heap_size="8g")

    train_data = load_arff(train_file)
    dist = class_distribution(train_data)
    print(dist)

    for config in rebalance_configs:
        classname = config["classname"]
        key = config["key"]
        options = config["option"]
        print(options)
        new_train_data = balance(train_data, classname, options)
        dist = class_distribution(new_train_data)
        print(key)
        print(dist)

    jvm.stop()
