from weka.classifiers import Evaluation


def evaluate_model(model, train_data, test_data):
    evaluator = Evaluation(train_data)
    evaluator.test_model(model, test_data)
    return evaluator
    # out = PredictionOutput(classname="weka.classifiers.evaluation.output.prediction.PlainText")
    # evaluator.test_model(model, test_data, out) # model_commandline = to_commandline(model)
    # serialization.write(f"{model_commandline}.model", model)


def summary(evaluator):
    summary = evaluator.summary("\nSummary\n======\n", False)
    class_details = evaluator.class_details()
    confusion_matrix = evaluator.matrix()
    return "\n".join([summary, class_details, confusion_matrix])
