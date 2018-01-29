import pandas as pd
import numpy as np
import config

logger = config.config_logger(__name__,10)

class Fixture:
    def __init__(self, my_fixture, name):
        self.fixture = my_fixture
        self.name = name
        self.seasons = set(my_fixture['season_id'])
        self.team_ids = set(my_fixture['localteam_id'])
        self.team_names = set(my_fixture['localTeam.data.name'])

    def __str__(self):
        return 'League: {0} - Shape: {1}'.format(self.name, self.fixture.shape)

    def get_team_games(self, team_id, home):
        df = self.fixture
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
        return Fixture(output, name)

    def get_team_scores(self, team_id):
        local = self.get_team_games(team_id=team_id, home=1)
        del(local.fixture['visitorTeam.data.name'])
        del(local.fixture['visitorteam_id'])
        local.fixture = local.fixture.rename(columns={'scores.localteam_score': 'score',
                                                      'scores.visitorteam_score': 'op_score'})
        visitor = self.get_team_games(team_id=team_id, home=0)
        del(visitor.fixture['localTeam.data.name'])
        del(visitor.fixture['localteam_id'])
        visitor.fixture = visitor.fixture.rename(columns={'scores.visitorteam_score': 'score',
                                                          'visitorteam_id': 'localteam_id',
                                                          'visitorTeam.data.name': 'localTeam.data.name',
                                                          'scores.localteam_score': 'op_score'})
        output_fixture = local.fixture
        output_fixture = output_fixture.append(visitor.fixture)
        output_fixture.sort_values('time.starting_at.date', inplace=True)
        output = Fixture(name=team_id, my_fixture=output_fixture)
        return output

    def clean_fixture(self):
        output = self
        vars_to_keep = output.variables_to_keep()
        output.fixture = output.fixture[vars_to_keep]
        output.fixture = output.fixture.drop_duplicates().dropna()
        return output

    def variables_to_keep(self):
        output = ['league_id', 'season_id', 'fixture_id', 'localteam_id', 'visitorteam_id', 'time.starting_at.date',
                  'localTeam.data.name', 'visitorTeam.data.name', 'scores.localteam_score', 'scores.visitorteam_score']
        return output

    def get_season(self, season_id):
        df = self.fixture
        output = df.loc[df['season_id'] == season_id]
        name = self.name + ' - season {0}'.format(season_id)
        return Fixture(output, name)

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
        return Fixture(output, name)

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
        return Fixture(output, name='{0} - {1} games dropped'.format(original_name, n))

    def get_mean(self, window_scored, window_conceded):
        original_name = self.name
        team_ids = self.team_ids
        output = pd.DataFrame([])
        for team_id in team_ids:
            team_fixture = self.get_team_scores(team_id=team_id).fixture
            team_fixture['roll_score'] = team_fixture['score'].rolling(window=window_scored).sum()
            team_fixture['roll_op_score'] = team_fixture['score'].rolling(window=window_conceded).sum()
            output = output.append(team_fixture)
        output = output.sort_values('time.starting_at.date')
        name = original_name + ' - roll sum'
        return Fixture(name=name, my_fixture=output)

    def generate_dataset(self):
        output = self.get_mean(window_scored=10, window_conceded=2)
        output = output.remove_x_games(4)
        output.fixture = output.fixture.dropna()
        return output






