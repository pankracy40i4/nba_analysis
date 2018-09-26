import logging
import warnings
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from utils.timer import timed
from utils.utils_functions import all_possible_pairs
from config import FILES_PATH
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data():
    data = pd.read_csv(FILES_PATH + 'teams.csv', sep=';')
    data['season'] = (data.GAME_ID / 100000).map(int)

    return data


def scale_and_split(data):

    sc = StandardScaler()
    df = data.copy()
    cols = df.columns.difference(['TEAM', 'TEAM_opp', 'season', 'HOME', 'tot'])

    sc.fit(df.loc[df.season < 217][cols])
    df[cols] = sc.transform(df[cols])

    train = df.loc[df.season < 217]
    test = df.loc[df.season >= 217]

    return train, test, sc, cols


def plot_grouped(grouped, column='FT_PTS'):

    fig, ax = plt.subplots(figsize=(8, 6))

    for name, group in grouped:
        group.plot(y=column, kind='kde', ax=ax, label=str(name))

    plt.xlabel(column)
    plt.ylabel('density')
    plt.show()


def data_scores_analysis(df):

    sample_std = df.tot.std()
    logger.info('The full time scores sample deviation is %0.2f ' % sample_std)

    # Are there any differences over seasons?
    grouped = df.groupby('season')
    logger.info(' Full time scores over seasons \n {}'.format(grouped['tot'].describe()))
    plot_grouped(grouped)
    # How does it look like for a team over season? are the FT scores similar?

    col_groupby = ['TEAM', 'season']
    grouped = df.groupby(col_groupby)['tot'].std()
    logger.info(' Full time scores over seasons \n {}'.format(grouped.describe()))
    # fig, ax = plt.plot(figsize=(8, 6))
    # TODO make this work
    # grouped.plot(kind='kde', ax=ax)

    # How does it look like for a (team, team_opp pairs) over season? are the FT scores similar?

    col_groupby = ['TEAM', 'TEAM_opp', 'season']
    grouped = df.groupby(col_groupby)['tot'].std()
    logger.info(' Full time scores over seasons \n {}'.format(grouped.describe()))


@timed(logger)
def data_transofrmation_for_model(df):
    col_groupby = ['TEAM', 'TEAM_opp', 'season']
    grouped = df.groupby(col_groupby)
    left = []
    right = []
    label = []
    df_out = []
    for info, g_ in grouped:
        indexes = list(range(g_.shape[0]))
        team = info[0]
        team_opp = info[1]
        season = info[2]
        this_teams = [team, team_opp]
        if len(indexes) < 2:
            continue

        g = g_.copy().reset_index(drop=True)
        combinations = all_possible_pairs(indexes)
        n_comb = len(combinations)
        df_counter_examples_right = df.loc[
            ((-df.TEAM.isin(this_teams)) | (-df.TEAM_opp.isin(this_teams))) & (df.season == season)].sample(n_comb)
        df_counter_examples_left = g.sample(n_comb, replace=True).reset_index(drop=True)

        counter = 0
        for pair in combinations:
            left.append(g.iloc[pair[0]][cols])
            right.append(g.iloc[pair[1]][cols])
            label.append(1)
            df_out.append({'team_left': team, 'team_opp_left': team_opp, 'team_right': team,
                           'team_opp_right': team_opp, 'season': season, 'label': 1})
            #### False examples ####
            left.append(df_counter_examples_left.iloc[counter][cols])
            right.append(df_counter_examples_right.iloc[counter][cols])
            label.append(0)
            df_out.append({'team_left': team, 'team_opp_left': team_opp,
                           'team_right': df_counter_examples_right.iloc[counter].TEAM,
                           'team_opp_right': df_counter_examples_right.iloc[counter].TEAM_opp, 'season': season,
                           'label': 0})
            #### False examples ####

            counter += 1