from weka.classifiers import Classifier


def train_model(train_data, model_name, options=[]):
    classifier = Classifier(classname=model_name, options=options)
    classifier.build_classifier(train_data)
    return classifier
