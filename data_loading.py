import logging
import time
import datetime
import warnings
import numpy as np
import pandas as pd
from utils.timer import timed
from xgboost.sklearn import XGBClassifier, XGBRegressor
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
from config import FILES_PATH, OUTPUT_PATH
from utils.xgb_loss_objectives import huber_approx_obj, fair_obj

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

track_file_name = 'NBASpecialFeaturesData.csv'
boxscores_file_name = 'NBABoxScoreData.csv'
odd_file_name = 'NBAOddsHistorical_spread.csv'


@timed(logger)
def get_track_data(file_name):
    th = time.time()

    track = pd.read_csv(FILES_PATH + file_name)
    track = track.drop_duplicates()
    track.GAME_ID = pd.to_numeric(track.GAME_ID)
    track = track.sort_values('GAME_ID')
    track.drop(['TEAM_ID', 'TEAM_NAME', 'TEAM_NICKNAME',
                'TEAM_CITY', 'MIN'], axis=1, inplace=True)
    track = track.loc[track.GAME_ID > 21202000].reset_index() \
        .drop('index', axis=1)
    track.rename(columns={'TEAM_ABBREVIATION': 'TEAM'}, inplace=True)
    logger.info('Reading NBASpecialFeaturesData with %d game \
                took %0.3f secs.' % (track.shape[0], time.time() - th))

    return track


@timed(logger)
def get_odd_data(file_name, boxscores_file_name):
    th = time.time()

    odd = pd.read_csv(FILES_PATH + file_name)

    odd['date'] = [datetime.datetime(int(x[:4]), int(x[4:6]), int(x[6:]))
                   for x in odd.key.astype(str)]
    team_codes = {'Atlanta Hawks': 'ATL', 'Boston Celtics':
        'BOS', 'Brooklyn Nets': 'BKN', 'Chicago Bulls': 'CHI',
                  'Charlotte Hornets': 'CHA', 'Charlotte Bobcats': 'CHA',
                  'Cleveland Cavaliers': 'CLE',
                  'Dallas Mavericks': 'DAL', 'Denver Nuggets': 'DEN',
                  'Detroit Pistons': 'DET',
                  'Golden State Warriors': 'GSW', 'Houston Rockets': 'HOU',
                  'Indiana Pacers': 'IND',
                  'Los Angeles Clippers': 'LAC', 'Los Angeles Lakers': 'LAL',
                  'L.A. Lakers': 'LAL', 'L.A. Clippers': 'LAC',
                  'Memphis Grizzlies': 'MEM', 'Miami Heat': 'MIA',
                  'Milwaukee Bucks': 'MIL', 'Minnesota Timberwolves': 'MIN',
                  'New Orleans Pelicans': 'NOP', 'New York Knicks': 'NYK',
                  'Oklahoma City Thunder': 'OKC', 'Orlando Magic': 'ORL',
                  'Philadelphia 76ers': 'PHI', 'Phoenix Suns': 'PHX',
                  'Portland Trail Blazers': 'POR', 'Sacramento Kings': 'SAC',
                  'San Antonio Spurs': 'SAS', 'Toronto Raptors': 'TOR',
                  'Utah Jazz': 'UTA', 'Washington Wizards': 'WAS',
                  'New Jersey Nets': 'NJN', 'New Orleans Hornets': 'NOH'}

    home_team_abb = []
    away_team_abb = []

    home_teams = odd.team.values
    away_teams = odd.opp_team.values

    for home_team, away_team in zip(home_teams, away_teams):
        found_home = False
        found_away = False
        for team_code in team_codes:
            if (home_team.upper() in team_code.upper()) and (found_home == False):
                home_team_abb.append(team_codes[team_code])
                found_home = True
            if (away_team.upper() in team_code.upper()) and (found_away == False):
                away_team_abb.append(team_codes[team_code])
                found_away = True
        if found_home == False:
            home_team_abb.append(None)
        if found_away == False:
            away_team_abb.append(None)

        found_away = False
        found_home = False

    odd['home_team_abb'] = home_team_abb
    odd['away_team_abb'] = away_team_abb

    odd.loc[(odd.home_team_abb.isnull()) | (odd.away_team_abb.isnull())]

    odd.rename(columns={'rl_PIN_line': 'prematchLinePIN',
                        'rl_BOL_line': 'prematchLineBOL'}, inplace=True)
    odd = odd[['date', 'home_team_abb', 'away_team_abb',
               'prematchLinePIN', 'prematchLineBOL']]

    logger.info('Reading NBAOddsHistorical with %d game took %0.3f secs.'
                % (odd.shape[0], time.time() - th))

    odd = odd.drop_duplicates()

    data = pd.read_csv(FILES_PATH + boxscores_file_name)
    data = data.drop_duplicates()

    data = data.loc[data.GAME_ID >= 21300000]
    data.index = [i for i in range(len(data))]

    data.GAME_ID = pd.to_numeric(data.GAME_ID)
    data = data.sort_values('GAME_ID')

    a = [datetime.datetime.strptime(x.split(',', 1)[1].replace(',', '').strip(),
                                    '%B %d %Y') for x in data.date]
    data['DATE'] = a

    data = data.merge(odd, how='left', left_on=['DATE', 'HOME', 'AWAY'],
                      right_on=['date', 'home_team_abb', 'away_team_abb'])

    data['prematchLinePIN'] = pd.to_numeric(data['prematchLinePIN'])
    data['prematchLineBOL'] = pd.to_numeric(data['prematchLineBOL'], errors='coerce')

    data['output'] = list(map(int, - data.HOME_FT_PTS + data.AWAY_FT_PTS > data.prematchLinePIN))

    data['tot'] = - data.HOME_FT_PTS + data.AWAY_FT_PTS

    data.fillna(np.median(data.prematchLineBOL))
    data.index = [i for i in range(len(data))]

    return data[['prematchLinePIN', 'GAME_ID', 'output']]


@timed(logger)
def modify_track_data(df, home_away):
    cols_game_info = ['DATE', 'GAME_ID', 'tot']
    cols_home = [col for col in list(df) if home_away in col] + cols_game_info
    home = df[cols_home].reset_index().rename(columns=lambda x: x[5:] if home_away + '_' in x else x)
    home['TEAM'] = home[home_away]
    if home_away == 'HOME':

        home['HOME'] = 1
        home['AWAY'] = 0
    else:
        home['HOME'] = 0
        home['AWAY'] = 1

    return home.reset_index(drop=True)


@timed(logger)
def get_nba_data(boxscores_file_name):
    th = time.time()

    data = pd.read_csv(FILES_PATH + boxscores_file_name)

    data['tot'] = - data.HOME_FT_PTS + data.AWAY_FT_PTS

    data.GAME_ID = pd.to_numeric(data.GAME_ID)
    data = data.sort_values('GAME_ID')

    a = [datetime.datetime.strptime(x.split(',', 1)[1].replace(',', '').strip(),
                                    '%B %d %Y') for x in data.date]
    data['DATE'] = a

    home = modify_track_data(data, 'HOME')
    away = modify_track_data(data, 'AWAY')

    data = pd.concat([home, away])
    logger.info('Reading NBABoxScoreData with %d game took %0.3f secs.'
                % (data.shape[0], time.time() - th))
    return data.drop('index', axis=1)


def team_features(match_data, window):
    cols = list(match_data.select_dtypes(include=[np.number]).columns.difference(['GAME_ID', 'HOME', 'AWAY', 'tot']))
    grouped = match_data.sort_values('GAME_ID').groupby('TEAM')[cols]
    result = []
    for _, g in grouped:
        opponent_ = match_data.loc[(match_data.GAME_ID.isin(g['GAME_ID'].values)) & (match_data.TEAM != _)]. \
            sort_values('GAME_ID')

        # opponent = opponent_.set_index(opponent_.GAME_ID)[cols].rolling(window).mean().shift(1).\
            # rename(columns=lambda x: x + '_opp').reset_index()
        opponent = opponent_.set_index(opponent_.GAME_ID)[cols].rename(columns=lambda x: x + '_opp').reset_index()

        g_hist = g[cols]#.rolling(window).mean().shift(1)
        g_hist['GAME_ID'] = g['GAME_ID']
        g_hist['HOME'] = g['HOME']
        g_hist['tot'] = g['tot']
        result.append(g_hist.merge(opponent, on='GAME_ID'))

    df_hist = pd.concat(result)
    df_hist_home = df_hist.loc[df_hist.HOME == 1].sort_values('GAME_ID')
    df_hist_away = df_hist.loc[df_hist.HOME == 0].sort_values('GAME_ID')

    df_out = df_hist_home.merge(df_hist_away, on=['GAME_ID', 'tot'], suffixes=('_home', '_away'))



    return df_out


def model_cv(train):
    cols_feats = train.columns.difference(['GAME_ID', 'output', 'tot'])
    feats = train[cols_feats]
    labels = train.tot



    clf = XGBRegressor(max_depth=4, learning_rate=0.01, n_estimators=50, objective=huber_approx_obj, \
                        gamma=0.1, min_child_weight=1, max_delta_step=0, subsample=0.9, \
                        colsample_bytree=0.2, colsample_bylevel=1, reg_alpha=0, reg_lambda=1)

    param_grid = {'max_depth': [3, 4], 'learning_rate': [0.05, 0.1, 0.3], 'reg_alpha': [0, 0.1,1,10,20], 'n_estimators':
        [25, 50, 100], 'reg_lambda': [0, 0.1, 0.5], 'colsample_bytree': [0.1, 0.25]}

    # param_grid = {'max_depth': [2, 3, 4]}

    scoring = {'mae': make_scorer(mean_absolute_error, greater_is_better=False), 'mse':
        make_scorer(mean_squared_error, greater_is_better=False)}

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
