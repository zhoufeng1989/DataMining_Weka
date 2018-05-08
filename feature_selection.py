from weka.filters import Filter


def feature_selection(train_data, test_data, options):
    data_filter = Filter(
        classname="weka.filters.supervised.attribute.AttributeSelection",
        options=options)
    data_filter.inputformat(train_data)
    filtered_train_data = data_filter.filter(train_data)
    filtered_test_data = _remove_features(filtered_train_data, test_data)
    return filtered_train_data, filtered_test_data


def _remove_features(train_data, test_data):
    train_attr_names = list(map(
        lambda attr: attr.name,
        train_data.attributes()))
    test_attr_names = list(map(
        lambda attr: attr.name,
        test_data.attributes()))
    extra_attr_names = set(test_attr_names) - set(train_attr_names)
    extra_indexes = list(map(
        lambda attr_name: str(test_data.attribute_by_name(attr_name).index + 1),
        extra_attr_names
        ))
    data_filter = Filter(
        classname="weka.filters.unsupervised.attribute.Remove",
        options=["-R", ",".join(extra_indexes)])
    data_filter.inputformat(test_data)
    filtered_data = data_filter.filter(test_data)
    filtered_attr_names = list(map(
        lambda attr: attr.name,
        filtered_data.attributes()))
    order = list(map(
        lambda attr_name: str(filtered_attr_names.index(attr_name) + 1),
        train_attr_names))
    data_filter = Filter(
        classname="weka.filters.unsupervised.attribute.Reorder",
        options=["-R", ",".join(order)])
    data_filter.inputformat(filtered_data)
    filtered_data = data_filter.filter(filtered_data)
    return filtered_data


if __name__ == "__main__":
    import sys
    from weka.core import jvm
    from utils import load_arff
    from weka.core.dataset import Instances
    from config import fs_configs

    jvm.start(max_heap_size="8g")
    train_file, test_file = sys.argv[1:]
    train_data = load_arff(train_file)
    test_data = load_arff(test_file)

    new_train_data, new_test_data = feature_selection(
        train_data, test_data, fs_configs[0]["option"])

    print("train data summary:")
    print(Instances.summary(new_train_data))
    print("test data summary:")
    print(Instances.summary(new_test_data))

    jvm.stop()
