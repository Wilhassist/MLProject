from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

def grid_search(model, param_grid, X, y, scoring="accuracy", cv=5):
    """
    Perform Grid Search to find the best hyperparameters.
    
    :param model: estimator, the model to tune
    :param param_grid: dict, hyperparameter grid
    :param X: DataFrame, feature matrix
    :param y: Series, target variable
    :param scoring: str, scoring metric
    :param cv: int, number of cross-validation folds
    :return: GridSearchCV object, with the best parameters and model
    """
    grid = GridSearchCV(model, param_grid, scoring=scoring, cv=cv, n_jobs=-1)
    grid.fit(X, y)
    return grid

def random_search(model, param_dist, X, y, scoring="accuracy", n_iter=50, cv=5):
    """
    Perform Randomized Search to find the best hyperparameters.
    
    :param model: estimator, the model to tune
    :param param_dist: dict, hyperparameter distributions
    :param X: DataFrame, feature matrix
    :param y: Series, target variable
    :param scoring: str, scoring metric
    :param n_iter: int, number of iterations for the search
    :param cv: int, number of cross-validation folds
    :return: RandomizedSearchCV object, with the best parameters and model
    """
    random_search = RandomizedSearchCV(model, param_distributions=param_dist, 
                                       scoring=scoring, n_iter=n_iter, cv=cv, n_jobs=-1, random_state=42)
    random_search.fit(X, y)
    return random_search
