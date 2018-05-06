from weka.filters import Filter
from utils import load_arff
import numpy


def replace_missing_values(data):
    data_filter = Filter(
        classname="weka.filters.unsupervised.attribute.ReplaceMissingValues")
    data_filter.inputformat(data)
    return data_filter.filter(data)


def _get_missing_ratios(data):
    missing_ratios = []
    attribute_cnt = data.num_attributes
    for index in range(attribute_cnt):
        attribute_stat = data.attribute_stats(index)
        missing_count = attribute_stat.missing_count
        total_count = attribute_stat.total_count
        missing_ratio = missing_count / total_count
        missing_ratios.append(missing_ratio)
    return missing_ratios


def remove_missing_values(train_data, test_data, ratio):
    """Remove attributes (features) which contain missing values over the given ratio"""
    missing_ratios = _get_missing_ratios(train_data)
    removed_attribute_indexes = list(
        map(
            lambda item: str(item[0] + 1),
            filter(lambda item: item[1] > ratio, enumerate(missing_ratios))))
    data_filter = Filter(
        classname="weka.filters.unsupervised.attribute.Remove",
        options=["-R", ",".join(removed_attribute_indexes)])
    data_filter.inputformat(test_data)
    return data_filter.filter(train_data), data_filter.filter(test_data)


def missing_values_stats(train_data):
    """Return missing values stats"""
    bins = numpy.linspace(0, 1, 11)
    missing_ratios = _get_missing_ratios(train_data)
    digitized = numpy.digitize(missing_ratios, bins, right=True)
    missing_stats = [len(digitized[digitized == i]) for i in range(len(bins))]
    bins = numpy.insert(bins, 0, -1)
    stats = []
    stats.append("missing values stats:")
    for (index, stat) in enumerate(missing_stats):
        start = "{:.1f}".format(bins[index])
        end = "{:.1f}".format(bins[index + 1])
        stats.append(f"missing ratio: ({start}, {end}] {stat}")
    return stats


if __name__ == "__main__":
    import sys
    from weka.core import jvm
    from weka.core.dataset import Instances

    jvm.start(max_heap_size="4g")

    train_file, test_file = sys.argv[1:]
    train_data = load_arff(train_file)
    test_data = load_arff(test_file)

    stats = missing_values_stats(train_data)
    print("\n".join(stats))

    train_data, test_data = remove_missing_values(train_data, test_data, 0.2)
    print("train data summary:")
    print(Instances.summary(train_data))
    print("test data summary:")
    print(Instances.summary(test_data))

    train_data = replace_missing_values(train_data)
    test_data = replace_missing_values(test_data)
    print("train data summary:")
    print(Instances.summary(train_data))
    print("test data summary:")
    print(Instances.summary(test_data))

    jvm.stop()
