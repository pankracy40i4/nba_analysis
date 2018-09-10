import logging
import warnings
import pandas as pd
from utils.timer import timed
from xgboost.sklearn import XGBRegressor
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
from utils.xgb_loss_objectives import huber_approx_obj, fair_obj
from config import track_file_name, odd_file_name, boxscores_file_name

from data_loading import get_track_data, get_nba_data, get_odd_data, team_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@timed(logger)
def model_cv(train):
    cols_feats = train.columns.difference(['GAME_ID', 'output', 'tot'])
    feats = train[cols_feats]
    labels = train.tot

    clf = XGBRegressor(max_depth=4, learning_rate=0.01, n_estimators=50, objective=huber_approx_obj, \
                       gamma=0.1, min_child_weight=1, max_delta_step=0, subsample=0.9, \
                       colsample_bytree=0.2, colsample_bylevel=1, reg_alpha=0, reg_lambda=1)

    param_grid = {'max_depth': [3, 4], 'learning_rate': [0.05, 0.1, 0.3], 'reg_alpha': [0, 0.1, 1, 10, 20],
                  'n_estimators':
                      [25, 50, 100], 'reg_lambda': [0, 0.1, 0.5], 'colsample_bytree': [0.1, 0.25]}

    # param_grid = {'max_depth': [2, 3, 4]}

    scoring = dict(mae=make_scorer(mean_absolute_error, greater_is_better=False),
                   mse=make_scorer(mean_squared_error, greater_is_better=False))

    grid_cv = GridSearchCV(clf, param_grid, scoring=scoring, fit_params=None, n_jobs=-1, iid=False,
                           cv=7, refit='mae',
                           verbose=2, return_train_score=True)

    grid_cv.fit(feats, labels)
    results = grid_cv.cv_results_
    pd.DataFrame.from_dict(results).to_csv('cv_res.csv')
    best_clf = grid_cv.best_estimator_

    return best_clf, cols_feats


def main():
    warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)

    window = 5
    track_data = get_track_data(track_file_name)
    box_scores = get_nba_data(boxscores_file_name)
    odds = get_odd_data(odd_file_name, boxscores_file_name)

    match_data = track_data.merge(box_scores, on=['TEAM', 'GAME_ID'])

    df_out = team_features(match_data, window)

    df = df_out.merge(odds, on='GAME_ID')

    train = df.loc[df.GAME_ID < 21600001].dropna()

    test = df.loc[(df.GAME_ID > 21600001)]

    clf, cols_feats = model_cv(train)

    feats = test[cols_feats]
    labels = test.tot
    pred = clf.predict(feats)

    print('MSE: {}'.format(mean_absolute_error(pred, labels)))
    print('BKID MSE: {}'.format(mean_absolute_error(test.prematchLinePIN, labels)))


if __name__ == "__main__":
    main()
