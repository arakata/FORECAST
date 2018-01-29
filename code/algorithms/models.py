import pandas as pd
import numpy as np


class Fixture:
    def __init__(self, my_fixture, name):
        self.fixture = my_fixture
        self.name = name
        self.seasons = set(my_fixture['season_id'])
        self.team_ids = set(my_fixture['localteam_id'])
        self.team_names = set(my_fixture['localTeam.data.name'])

    def __str__(self):
        return 'League: {0} - Shape: {1}'.format(self.name, self.fixture.shape)

    def get_team(self, team_id, home):
        df = self.fixture
        #CHECK THIS!!
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

    def clean_fixture(self):
        vars_to_keep = self.variables_to_keep()
        my_fixture = self.fixture[vars_to_keep]
        my_fixture = my_fixture.drop_duplicates().dropna()
        self.fixture = my_fixture

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
            print(self.name)
        name = self.name + ' - {0} dropped'.format(x)
        return Fixture(output, name)

    #def get_mean(self, window):


    def generate_dataset(self):
        """ Generate a dataset ready for the poisson regression """
        output = pd.DataFrame([])
        seasons = self.seasons
        print('original size: {0}'.format(self.fixture.shape))
        for season in seasons:
            temp_season = self.get_season(season)
            teams = temp_season.team_ids
            print('original size: {0}'.format(temp_season.fixture.shape))
            for team_id in teams:
                temp_team = temp_season.get_team(team_id, home=1)
                temp_clean = temp_team.drop_x_games_first_last(4)
                output = output.append(temp_clean.fixture)
        output = output.sort_values('time.starting_at.date')
        print(output.shape)
        print(output.tail().to_string())

        # TODO: create the dataset with the rolling mean.







