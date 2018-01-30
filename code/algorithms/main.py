import pandas as pd
import numpy as np
import time

import code.algorithms.config as config
import code.algorithms.models as models


def main():
    t0 = time.time()
    logger = config.config_logger(__name__, 10)
    fixtures_path = './data/sportmonks/with_data/'
    champions_path = './data/sportmonks/league_winners/'

    logger.info('Begin execution')
    target_league = '8Premier_League'
    logger.info('Open matches dataframe: {0}'.format(target_league))
    raw_df = pd.read_csv(fixtures_path + target_league + '.csv', index_col=0)
    logger.info('Open champions dataframe: {0}'.format(target_league))
    champions_df = pd.read_csv(champions_path + target_league + '.csv', index_col=None)
    premier = models.Fixture(raw_df, 'premier')
    premier = premier.clean_fixture()
    premier = premier.generate_dataset()
    premier = premier.add_champion_dummy(champions_df)
    print(premier)
    print(premier.fixture.head().to_string())

    config.time_taken_display(t0)


if __name__ == '__main__':
    main()
