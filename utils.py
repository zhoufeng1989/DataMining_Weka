from weka.core.converters import Loader
from weka.core.dataset import Instances


def load_arff(file_name):
    loader = Loader(classname="weka.core.converters.ArffLoader")
    data = loader.load_file(file_name)
    data.class_is_last()
    return data


def install_smote():
    from weka.core import jvm
    from weka.core import packages
    jvm.start()
    if packages.install_package("SMOTE"):
        print("install SMOTE sucessfully")
    else:
        print("install SMOTE failed")
    jvm.stop()


def data_distribution(data):
    data_summary = Instances.summary(data)
    class_stats = data.attribute_stats(data.class_index)
    stats = ["data summary", data_summary, "class distribution"]
    for (label, count) in zip(data.class_attribute.values, class_stats.nominal_counts):
        stats.append(f"{label}: {count}")
    stats.append(f"total: {data.num_instances}")
    return "\n".join(stats)
