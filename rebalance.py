from weka.filters import Filter
from utils import load_arff, class_distribution


def resample(data, sample_size_percentage):
    data_filter = Filter(
        classname="weka.filters.supervised.instance.Resample",
        options=["-B", "1.0", "-S", "1", "-Z", str(sample_size_percentage)])
    data_filter.inputformat(data)
    return data_filter.filter(data)


def smote(data, percentage):
    data_filter = Filter(
        classname="weka.filters.supervised.instance.SMOTE",
        options=["-C", "last", "-K", "5", "-P", str(percentage), "-S", "1"])
    data_filter.inputformat(data)
    return data_filter.filter(data)


if __name__ == "__main__":
    import sys
    from weka.core import jvm

    train_file = sys.argv[1]

    jvm.start(packages=True, max_heap_size="8g")

    train_data = load_arff(train_file)
    dist = class_distribution(train_data)
    print(dist)

    new_train_data = resample(train_data, 50)
    dist = class_distribution(new_train_data)
    print(dist)

    new_train_data = smote(train_data, 50)
    dist = class_distribution(new_train_data)
    print(dist)

    jvm.stop()
