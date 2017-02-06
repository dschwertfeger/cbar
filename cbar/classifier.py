import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MultiLabelBinarizer,
    StandardScaler,
)
from .util.io import (
    load_dataset,
    load_train_data,
    save_model
)


estimators = dict(gd=(SGDClassifier(), 'stochastic-gd'),
                  gb=(GradientBoostingClassifier(), 'gradient-boosting'),
                  rf=(RandomForestClassifier(), 'random-forest'),
                  lr=(LogisticRegression(), 'log-regression'))


def params_for(estimator, dual):
    # TODO: should eventually load the BEST parameters found during grid search
    # sgd: n_iter=n_iter = np.ceil(10**6 / n_samples_train)
    params = dict(gd=dict(loss='log', alpha=0.03, verbose=5, n_jobs=-1,
                          warm_start=True, class_weight='balanced', n_iter=11),
                  gb=dict(learning_rate=0.03, verbose=5),
                  rf=dict(n_estimators=200, n_jobs=1, verbose=5,
                          class_weight='balanced', oob_score=True),
                  lr=dict(dual=dual, n_jobs=-1, solver='liblinear', verbose=1,
                          warm_start=True, class_weight='balanced'))
    return params[estimator]


def param_grid_for(estimator, dual):
    param_grid = dict(gd=[dict(clf__estimator__alpha=10.0**-np.arange(1, 7),
                               clf__estimator__n_iter=[5, 11, 15])],
                      # TODO: update these
                      # gb=dict(learning_rate=0.03, verbose=5),
                      # rf=dict(n_estimators=30, n_jobs=-1, verbose=5, class_weight='balanced'),
                      lr=[dict(clf__estimator__penalty=('l1', 'l2'),
                               clf__estimator__C=10.0**-np.arange(1, 7),
                               clf__estimator__max_iter=(20, 50))])
    return param_grid[estimator]


def grid_search(args):
    X, Y = load_dataset(args.dataset)
    n_samples, n_features = X.shape
    dual = n_samples < n_features
    param_grid = param_grid_for(args.estimator, dual)
    pipeline = make_pipeline(args.estimator, dual)
    clf = GridSearchCV(pipeline, param_grid, n_jobs=-1, verbose=5)
    clf.fit(X, Y)


def make_pipeline(est, dual):
    estimator, name = estimators[est]
    estimator.set_params(**params_for(est, dual))
    # classifier = OneVsRestClassifier(CalibratedClassifierCV(estimator, cv=2, method='sigmoid'), n_jobs=-1)
    classifier = OneVsRestClassifier(estimator, n_jobs=-1)
    return Pipeline([
        ('scaler', StandardScaler()),
        ('clf', classifier)
    ])


def train(args):
    X_train, Y_train = load_train_data()
    n_samples, n_features = X_train.shape
    dual = n_samples < n_features

    mlb = MultiLabelBinarizer()
    Y_train_bin = mlb.fit_transform(Y_train)

    pipeline = make_pipeline(args.estimator, dual)
    pipeline.fit(X_train, Y_train_bin)

    _, name = estimators[args.estimator]
    clf = pipeline.named_steps['clf']
    save_model(clf, name)
