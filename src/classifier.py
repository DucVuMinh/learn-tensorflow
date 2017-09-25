class Classifier(object):

    def __init__(self):
        pass

    def get_name(self):
        raise NotImplementedError()

    def fit(self):
        raise NotImplementedError()

    def predict(self):
        raise NotImplementedError()
