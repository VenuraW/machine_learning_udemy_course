from sklearn.model_selection import GridSearchCV

class Classifier():
    def __init__(self, name, classifier, x_train, x_test, y_train, y_test):
        self._name = name
        self._default_classifier = classifier
        self._grid_classifier = classifier
        self._x_train = x_train
        self._x_test = x_test
        self._y_train = y_train
        self._y_test = y_test

    def train_classifier(self):
        self._default_classifier.fit(self._x_train, self._y_train)

    def get_score(self):
        return self._default_classifier.score(self._x_test, self._y_test)

    def train_grid_search(self, params):
        grid_search = GridSearchCV(estimator=self._grid_classifier,
                                   param_grid=params,
                                   scoring='accuracy',
                                   cv=10)
        self._grid_classifier = grid_search.fit(self._x_train, self._y_train)

    def get_grid_score(self):
        return self._grid_classifier.score(self._x_test, self._y_test)

    def set_best_classifer(self):
        default_score = self.get_score()
        grid_score = self.get_grid_score()

        if (grid_score <= default_score):
            self._name += " (Default)"
            self.best_classifier = self._default_classifier
            self.is_best_grid = 0
            self._best_score = default_score
        else:
            self._name += " (Grid)"
            self.best_classifier = self._grid_classifier
            self.is_best_grid = 1
            self._best_score = grid_score

    def get_best_score(self):
        return self._best_score

    def __str__(self):
        return "{} Score: {:.2f}%".format(self._name, self._best_score*100)
