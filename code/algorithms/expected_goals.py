import pandas as pd
import time

import code.algorithms.config as config
import code.algorithms.models as models


def main():
    t0 = time.time()
    logger = config.config_logger(__name__, 10)
    data_google_path = './data/google/'
    output_path = './output/goldman_sachs/'
    target_league = 'premier'

    comp_dict = {'MLS': 98,
                 'la_liga': 23,
                 'premier': 8}

    last_year_dict = {'MLS': 2014,
                      'la_liga': 2013,
                      'premier': 2012}

    competitionid = comp_dict[target_league]

    logger.info('Begin execution')
    logger.info('Open OPTA database')
    raw_df = pd.read_csv('{0}raw_data_ready.csv'.format(data_google_path), header=0, index_col=0)
    logger.info('Open league winners database')
    champions_df = pd.read_csv('{0}league_winners/{1}.csv'.format(data_google_path, target_league),
                               header=0, index_col=None)
    print(raw_df.head().to_string())

    logger.info('Preprocess data')
    raw_df = raw_df.loc[raw_df['competitionid'] == competitionid]
    raw_df = raw_df.rename(columns={'competitionid': 'league_id',
                                    'seasonid': 'season_id',
                                    'matchid': 'fixture_id',
                                    'teamid': 'localteam_id',
                                    'op_teamid': 'visitorteam_id',
                                    'timestamp': 'time.starting_at.date',
                                    'team_name': 'localTeam.data.name',
                                    'op_team_name': 'visitorTeam.data.name',
                                    'goals': 'scores.localteam_score',
                                    'op_goals': 'scores.visitorteam_score'})
    opta_df = raw_df[['fixture_id', 'expected_goals', 'is_home']]
    opta_df = models.convert_2match_to_1match(opta_df)

    raw_df = raw_df.loc[raw_df['is_home'] == 1]

    my_league = models.Fixture(raw_df, target_league, last_year=last_year_dict[target_league],
                               local_fixture=True)
    my_league = my_league.clean_fixture(is_sportmonks=False)

    my_league = my_league.generate_dataset()
    my_league = my_league.add_champion_dummy(champions_df)

    train, test = my_league.exclude_last_x_seasons(1)
    logger.info('Train model: poisson regression - {0} obs'.format(train.fixture.shape[0]))
    model = train.train_model()
    print(my_league)
    print(my_league.fixture.head().to_string())

    logger.info('Extract predictions')
    results_gs = test.get_matches_prediction(model)
    print(results_gs.fixture.head().to_string())

    results_gs.clean_results()
    print(results_gs.fixture.head().to_string())

    accuracy = results_gs.get_accuracy()
    logger.info('GS - Accuracy obtained: {0}'.format(accuracy))

    results_gs = results_gs.fixture.rename(columns={'local_prob': 'gs_local_prob',
                                                    'tie_prob': 'gs_tie_prob',
                                                    'visitor_prob': 'gs_visitor_prob',
                                                    'expected_winner': 'gs_expected_winner',
                                                    'expected_score': 'gs_expected_score',
                                                    'op_expected_score': 'gs_op_expected_score'})

    logger.info('Merge results')
    results = pd.merge(results_gs, opta_df, how='inner', on=['fixture_id'])
    logger.info('Results predicted: {0}'.format(results.shape[0]))
    print(results.head().to_string())
    squared_errors = models.get_squared_error(results)
    logger.info('Squared errors - GS: {0} | Google: {1}'.format(squared_errors[0],
                                                                squared_errors[1]))

    logger.info('Save results')
    results.to_csv('{0}expected_goals_{1}.csv'.format(output_path, target_league))
    print(results.head().to_string())

    config.time_taken_display(t0)


if __name__ == '__main__':
    main()