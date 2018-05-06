from weka.core.converters import Loader


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


def class_distribution(data):
    class_stats = data.attribute_stats(data.class_index)
    stats = ["class distribution"]
    for (label, count) in zip(data.class_attribute.values, class_stats.nominal_counts):
        stats.append(f"{label}: {count}")
    stats.append(f"total: {data.num_instances}")
    return "\n".join(stats)
