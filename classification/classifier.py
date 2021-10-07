from sklearn.model_selection import GridSearchCV

class Classifier():
    """
    Classifier class to train and compare against other parameters
    """
    def __init__(self, name, classifier, x_train, x_test, y_train, y_test):
        self._name = name
        self._default_classifier = classifier
        self._grid_classifier = classifier
        self._x_train = x_train
        self._x_test = x_test
        self._y_train = y_train
        self._y_test = y_test

    def train_classifier(self):
        """
        Training the classifier
        """
        self._default_classifier.fit(self._x_train, self._y_train)

    def get_score(self):
        """
        Getting the score of the classifier
        :return: integer value of the training score out of 1
        """
        return self._default_classifier.score(self._x_test, self._y_test)

    def train_grid_search(self, params):
        """
        Determines the best paramters based on the parameters provided
        :param params: array of different parameters
        """
        grid_search = GridSearchCV(estimator=self._grid_classifier,
                                   param_grid=params,
                                   scoring='accuracy',
                                   cv=10)
        self._grid_classifier = grid_search.fit(self._x_train, self._y_train)

    def get_grid_score(self):
        """
        Gets the score for the classifier with hyperparameter tuning through grid search
        :return: integer value of the classifier score
        """
        return self._grid_classifier.score(self._x_test, self._y_test)

    def set_best_classifer(self):
        """
        Determines the best classifier against the training dataset between the default parameters and the hyperparameter tuning through grid search
        :return:
        """
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
        """
        Gets the best score between default classifier and hyperparameter tuned grid search classifier
        :return:
        """
        return self._best_score

    def __str__(self):
        """
        Prints the name followed by the best score
        :return: string the name and best score
        """
        return "{} Score: {:.2f}%".format(self._name, self._best_score*100)
