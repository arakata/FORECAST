import pandas as pd
import time

import code.algorithms.config as config
import code.algorithms.models as models


def main():
    t0 = time.time()
    logger = config.config_logger(__name__, 10)
    fixtures_path = './data/sportmonks/with_data/'
    champions_path = './data/sportmonks/league_winners/'
    output_path = './output/goldman_sachs/'
    stats_path = './data/sportmonks/'

    logger.info('Begin execution')
    target_league = '8Premier_League'
    stats_file = 'sportmonks_final'

    logger.info('Open matches dataframe: {0}'.format(target_league))
    raw_df = pd.read_csv(fixtures_path + target_league + '.csv', index_col=0)
    logger.info('Open champions dataframe: {0}'.format(target_league))
    champions_df = pd.read_csv(champions_path + target_league + '.csv', index_col=None)
    logger.info('Open stats dataframe: {0}'.format(stats_file))
    stats_df = pd.read_csv(stats_path + stats_file + '.csv', index_col=0)
    print(stats_df.head().to_string())

    stats_df = stats_df.rename(columns={'fixture_id': 'matchid',
                                        'team_id': 'teamid',
                                        'op_team_id': 'op_teamid',
                                        'team_name': 'Team.data.name',
                                        'op_team_name': 'op_Team.data.name',
                                        'goals': 'score',
                                        'op_goals': 'op_score'})

    objective = stats_df.loc[stats_df['league_id'] == 8]
    stats_df = stats_df.loc[stats_df['league_id'] != 8]
    print(stats_df.head().to_string())
    (model, test) = models.train_model(
        stats_df, models.non_feature_cols())
    results = models.predict_model(model, objective, models.non_feature_cols())
    print(results.head().to_string())
    results = models.get_winners(results)
    accuracy = models.get_accuracy(results)
    logger.info('GOOGLE - Accuracy obtained: {0}'.format(accuracy))
    results.to_csv(output_path + 'google_predictions.csv', index=None)

    logger.info('Preprocess data')
    premier = models.Fixture(raw_df, 'premier')
    premier = premier.clean_fixture()
    premier = premier.generate_dataset()
    premier = premier.add_champion_dummy(champions_df)

    logger.info('Train model: poisson regression')
    train, test = premier.exclude_last_x_seasons(2)
    model = train.train_model()
    print(premier)
    print(premier.fixture.head().to_string())

    logger.info('Extract predictions')
    results = test.get_matches_prediction(model)
    print(results.fixture.head().to_string())

    results.clean_results()
    print(results.fixture.head().to_string())
    logger.info('Save results')
    results.fixture.to_csv(output_path + 'gs_predictions.csv', index=None)

    accuracy = results.get_accuracy()
    logger.info('GS - Accuracy obtained: {0}'.format(accuracy))

    config.time_taken_display(t0)


if __name__ == '__main__':
    main()
