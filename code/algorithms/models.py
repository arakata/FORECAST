import pandas as pd
import numpy as np
import math
import config
import pprint
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import poisson, skellam

logger = config.config_logger(__name__, 10)


class Fixture:
    def __init__(self, my_fixture, name, local_fixture=True):
        self.fixture = my_fixture
        self.name = name
        self.seasons = set(my_fixture['season_id'])
        self.local_fixture = local_fixture
        if local_fixture:
            self.team_ids = set(my_fixture['localteam_id'])
            self.team_names = set(my_fixture['localTeam.data.name'])
        else:
            self.team_ids = set(my_fixture['team_id'])
            self.team_names = set(my_fixture['Team.data.name'])

    def __str__(self):
        return 'League: {0} - Shape: {1}'.format(self.name, self.fixture.shape)

    def get_team_games(self, team_id, home):
        df = self.fixture
        local_fixture = self.local_fixture
        if local_fixture:
            team_name = df.loc[df['localteam_id'] == team_id]['localTeam.data.name'].iloc[0]
            name = self.name + ' - {0}'.format(team_name)
            if home == 0:
                output = df.loc[df['visitorteam_id'] == team_id]
            elif home == 1:
                output = df.loc[df['localteam_id'] == team_id]
            elif home == 2:
                output = df.loc[(df['localteam_id'] == team_id) | (df['visitorteam_id'] == team_id)]
            else:
                raise ValueError('home must be an integer between 0 and 2')
        else:
            if home == 0:
                output = df.loc[(df['team_id'] == team_id) & (df['is_home'] == 0)]
                output = output.append(df.loc[(df['op_team_id'] == team_id) & (df['is_home'] == 1)])
            elif home == 1:
                output = df.loc[(df['team_id'] == team_id) & (df['is_home'] == 1)]
                output = output.append(df.loc[(df['op_team_id'] == team_id) & (df['is_home'] == 0)])
            elif home == 2:
                output = df.loc[(df['team_id'] == team_id) | (df['op_team_id'] == team_id)]
            else:
                raise ValueError('home must be an integer between 0 and 2')
            team_name = df.loc[df['team_id'] == team_id]['Team.data.name'].iloc[0]
            name = self.name + ' - {0}'.format(team_name)
        return Fixture(output, name, local_fixture=local_fixture)

    def get_team_scores(self, team_id):
        local = self.get_team_games(team_id=team_id, home=1)
        local_fixture = local.fixture.copy()
        local_fixture['is_home'] = 1
        local_fixture = local_fixture.rename(columns={'scores.localteam_score': 'score',
                                                      'scores.visitorteam_score': 'op_score',
                                                      'localteam_id': 'team_id',
                                                      'localTeam.data.name': 'Team.data.name',
                                                      'visitorteam_id': 'op_team_id',
                                                      'visitorTeam.data.name': 'op_Team.data.name'})
        visitor = self.get_team_games(team_id=team_id, home=0)
        visitor_fixture = visitor.fixture.copy()
        visitor_fixture['is_home'] = 0
        visitor_fixture = visitor_fixture.rename(columns={'scores.visitorteam_score': 'score',
                                                          'scores.localteam_score': 'op_score',
                                                          'visitorteam_id': 'team_id',
                                                          'visitorTeam.data.name': 'Team.data.name',
                                                          'localteam_id': 'op_team_id',
                                                          'localTeam.data.name': 'op_Team.data.name'})
        output_fixture = local_fixture
        output_fixture = output_fixture.append(visitor_fixture)
        output_fixture = output_fixture.sort_values('time.starting_at.date')
        output = Fixture(name=team_id, my_fixture=output_fixture, local_fixture=False)
        return output

    def clean_fixture(self):
        output = self
        vars_to_keep = output.variables_to_keep()
        output.fixture = output.fixture[vars_to_keep]
        output.fixture = output.fixture.drop_duplicates().dropna()
        return output

    def get_season(self, season_id):
        df = self.fixture
        local_fixture = self.local_fixture
        output = df.loc[df['season_id'] == season_id]
        name = self.name + ' - season {0}'.format(season_id)
        return Fixture(my_fixture=output, name=name, local_fixture=local_fixture)

    def drop_x_games_first_last(self, x):
        """ Drop the first and last 4 matches for each season """
        my_fixture = self.fixture
        my_fixture = my_fixture.sort_values('time.starting_at.date')
        if len(my_fixture) > 15:
            output = my_fixture.iloc[x:-x]
        else:
            output = my_fixture.iloc[x:]
            logger.warning('Team has less than 15 matches: {0}'.format(self.name))
        name = self.name + ' - {0} dropped'.format(x)
        return Fixture(output, name, local_fixture=self.local_fixture)

    def remove_x_games(self, n):
        """" Drop the first and last 4 matches for each season in a Fixture """
        output = pd.DataFrame([])
        seasons = self.seasons
        original_name = self.name
        logger.info('Main Fixture original size: {0}'.format(self.fixture.shape))
        for season in seasons:
            temp_season = self.get_season(season)
            teams = temp_season.team_ids
            logger.info('Season {1} original size: {0}'.format(temp_season.fixture.shape, season))
            for team_id in teams:
                temp_team = temp_season.get_team_games(team_id, home=1)
                temp_clean = temp_team.drop_x_games_first_last(n)
                output = output.append(temp_clean.fixture)
        output = output.sort_values('time.starting_at.date')
        name = '{0} - {1} games dropped'.format(original_name, n)
        return Fixture(output, name=name, local_fixture=self.local_fixture)

    def get_score_rolling_mean(self, window_scored, window_conceded):
        original_name = self.name
        team_ids = self.team_ids
        output = pd.DataFrame([])
        for team_id in team_ids:
            team_fixture = self.get_team_scores(team_id=team_id).fixture
            team_fixture['roll_score'] = team_fixture['score'].rolling(window=window_scored).sum().shift(1)
            team_fixture['roll_op_score'] = team_fixture['op_score'].rolling(window=window_conceded).sum().shift(1)
            output = output.append(team_fixture)
        only_main_team = output[['team_id', 'Team.data.name', 'time.starting_at.date', 'roll_op_score']]
        only_main_team = only_main_team.sort_values(['time.starting_at.date', 'team_id'])
        only_main_team = only_main_team['roll_op_score']
        output = output.sort_values(['time.starting_at.date', 'op_team_id'])
        output['roll_op_score'] = only_main_team
        output = output.sort_values('time.starting_at.date')
        name = original_name + ' - roll sum'
        return Fixture(name=name, my_fixture=output, local_fixture=False)

    def generate_dataset(self):
        output = self.get_score_rolling_mean(window_scored=10, window_conceded=2)
        output = output.remove_x_games(4)
        output.fixture = output.fixture.dropna()
        return output

    def get_team_names_and_ids(self):
        if self.local_fixture:
            names_and_ids = set(zip(self.fixture['localteam_id'], self.fixture['localTeam.data.name']))
        else:
            names_and_ids = set(zip(self.fixture['team_id'], self.fixture['Team.data.name']))
        output = {}
        for team_id, team_name in names_and_ids:
            output[team_name] = team_id
        return output

    def add_champion_dummy(self, champions_df):
        origin_name = self.name
        teams_dict = self.get_team_names_and_ids()
        champions_df = champions_df.replace(teams_dict)
        output = pd.DataFrame([])
        for season in self.seasons:
            season_dict = self.get_seasons_dict()
            target_year = season_dict[season]
            champions = self.get_champions_in_period(champions_df, target_year, 4)
            temp_fixture = self.get_season(season).fixture.copy()
            if self.local_fixture:
                temp_fixture['champion'] = temp_fixture['localteam_id'].apply(
                    lambda x: create_dummy_for_champions(x, champions))
            else:
                temp_fixture['champion'] = temp_fixture['team_id'].apply(
                    lambda x: create_dummy_for_champions(x, champions))
            output = output.append(temp_fixture)
        output = output.sort_values('time.starting_at.date')
        name = origin_name + ' - add champion'
        return Fixture(name=name, my_fixture=output, local_fixture=self.local_fixture)

    def get_seasons_dict(self):
        my_fixture = self.fixture
        my_fixture['year'] = my_fixture['time.starting_at.date'].apply(lambda x: x[:4]).astype('int')
        output = {}
        for season in self.seasons:
            temp = my_fixture.loc[my_fixture['season_id'] == season, :]
            year_mean = math.ceil(np.mean(temp['year']))
            output.update({season: year_mean})
        return output

    def train_model(self):
        model_variables = self.variables_in_model()
        my_fixture = self.fixture[model_variables]
        model = smf.glm(formula='score ~ is_home + roll_score + roll_op_score + champion', data=my_fixture,
                        family=sm.families.Poisson()).fit()
        return model

    def exclude_last_x_seasons(self, n):
        seasons_dict = invert_dictionary(self.get_seasons_dict())
        max_year = max(list(seasons_dict.keys()))
        drop_range = range(max_year+1-n, max_year+1)
        my_fixture = self.fixture.copy()
        keep, drop = my_fixture, pd.DataFrame([])
        for year in drop_range:
            target = seasons_dict[year]
            drop = drop.append(my_fixture.loc[my_fixture['season_id'] == target, :])
            keep = keep.loc[keep['season_id'] != target, :]
        drop = drop_single_matches(drop)
        name_keep = self.name + ' - train'
        name_drop = self.name + ' - test'
        keep = Fixture(name=name_keep, my_fixture=keep, local_fixture=self.local_fixture)
        drop = Fixture(name=name_drop, my_fixture=drop, local_fixture=self.local_fixture)
        return keep, drop

    def add_predictions(self, predictions):
        my_fixture = self.fixture.copy()
        my_fixture['expected_score'] = predictions
        name = self.name + ' - predicted'
        return Fixture(my_fixture=my_fixture, name=name, local_fixture=self.local_fixture)

    def convert_to_matches(self):
        my_fixture = self.fixture.copy()
        my_fixture = my_fixture.sort_values(['time.starting_at.date', 'fixture_id'])
        my_fixture = drop_single_matches(my_fixture)
        output = my_fixture.loc[my_fixture['is_home'] == 1].copy()
        temp = my_fixture.loc[my_fixture['is_home'] == 0].copy()
        output['op_expected_score'] = temp['expected_score']
        return Fixture(name='match predicitons', my_fixture=output, local_fixture=self.local_fixture)

    def get_matches_prediction(self, model):
        prediction = model.predict(self.fixture)
        my_fixture = self.add_predictions(prediction)
        output = my_fixture.convert_to_matches()
        output = output.get_match_probabilities()
        output.fixture['winner'] = output.fixture.apply(get_winner, axis=1)
        return output

    def get_match_probabilities(self):
        my_fixture = self.fixture.copy()
        local_score = self.fixture['expected_score']
        visitor_score = self.fixture['op_expected_score']
        local_prob_list, visitor_prob_list, tie_prob_list, winner = [], [], [], []
        for i in range(self.fixture.shape[0]):
            match_prob = simulate_match(local_score.iloc[i], visitor_score.iloc[i], max_goals=10)
            local_prob, tie_prob, visitor_prob = sum_triangle_and_diagonal_from_matrix(match_prob)
            local_prob_list.append(local_prob)
            tie_prob_list.append(tie_prob)
            visitor_prob_list.append(visitor_prob)
            if local_prob == max(local_prob, tie_prob, visitor_prob):
                winner.append(self.fixture['Team.data.name'].iloc[i])
            elif visitor_prob == max(local_prob, tie_prob, visitor_prob):
                winner.append(self.fixture['op_Team.data.name'].iloc[i])
            else:
                winner.append('tie')
        my_fixture['local_prob'] = pd.Series(local_prob_list, index=my_fixture.index)
        my_fixture['tie_prob'] = pd.Series(tie_prob_list, index=my_fixture.index)
        my_fixture['visitor_prob'] = pd.Series(visitor_prob_list, index=my_fixture.index)
        my_fixture['expected_winner'] = pd.Series(winner, index=my_fixture.index)
        return Fixture(my_fixture=my_fixture, name=self.name, local_fixture=self.local_fixture)

    def clean_results(self):
        self.fixture = self.fixture[self.result_variables()]
        return

    def get_accuracy(self):
        my_fixture = self.fixture.copy()
        total = my_fixture.shape[0]
        good = np.sum([my_fixture['expected_winner'] == my_fixture['winner']])
        return good/total


    @staticmethod
    def get_champions_in_period(champions_df, year, window):
        output = set()
        for i in range(year-window, year):
            output.update(champions_df[str(i)].tolist())
        return output

    @staticmethod
    def variables_to_keep():
        output = ['league_id', 'season_id', 'fixture_id', 'localteam_id', 'visitorteam_id', 'time.starting_at.date',
                  'localTeam.data.name', 'visitorTeam.data.name', 'scores.localteam_score', 'scores.visitorteam_score']
        return output

    @staticmethod
    def variables_in_model():
        output = ['score', 'roll_score', 'roll_op_score', 'champion', 'is_home']
        return output

    @staticmethod
    def result_variables():
        output = ['fixture_id', 'time.starting_at.date', 'Team.data.name', 'op_Team.data.name', 'expected_score',
                  'op_expected_score', 'local_prob', 'tie_prob', 'visitor_prob', 'expected_winner', 'winner']
        return output


def create_dummy_for_champions(team_id, champion_set):
        if team_id in champion_set:
            return 1
        else:
            return 0


def invert_dictionary(my_dict):
    return {v: k for k, v in my_dict.items()}


def simulate_match(home_goals_avg, away_goals_avg, max_goals=10):
    team_pred = [[poisson.pmf(i, team_avg) for i in range(0, max_goals+1)] for
                 team_avg in [home_goals_avg, away_goals_avg]]
    return np.outer(np.array(team_pred[0]), np.array(team_pred[1]))


def drop_single_matches(df):
    df = df.sort_values('fixture_id')
    i = 0
    output = pd.DataFrame([])
    fixture_id = df['fixture_id']
    while i < df.shape[0]:
        if fixture_id.iloc[i] == fixture_id.iloc[i+1]:
            output = output.append(df.iloc[i])
            output = output.append(df.iloc[i+1])
            i += 2
        else:
            i += 1
    return output


def sum_triangle_and_diagonal_from_matrix(my_matrix):
    upper, lower, diagonal = 0, 0, 0
    for i in range(my_matrix.shape[0]):
        for j in range(my_matrix.shape[1]):
            if i > j:
                lower += my_matrix[i, j]
            if i < j:
                upper += my_matrix[i, j]
            if i == j:
                diagonal += my_matrix[i, j]
    return lower, diagonal, upper


def get_winner(row):
    local = row['score']
    visitor = row['op_score']
    if local > visitor:
        return row['Team.data.name']
    elif local < visitor:
        return row['op_Team.data.name']
    else:
        return 'tie'


