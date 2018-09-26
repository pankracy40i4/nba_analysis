import itertools


def all_possible_pairs(input_list):

    return list(itertools.combinations(input_list, 2))


def get_df_negative_examples(df, this_teams, season, n):
    """
    Select counter-examples from data frame, where the teams are different but from the same season.

    :param df: DF, data frame with all games
    :param this_teams: list, list of the teams
    :param season: int, season to sample the games from
    :param n: int, sample size
    :return: DF, data frame with n rows (games)
    """

    return df.loc[((-df.TEAM.isin(this_teams)) & (-df.TEAM_opp.isin(this_teams))) & (df.season == season)].sample(n)

