import pandas as pd
import numpy as np
import time

import code.algorithms.config as config
import code.algorithms.models as models


def main():
    t0 = time.time()
    logger = config.config_logger(__name__, 10)
    fixtures_path = './data/sportmonks/with_data/'

    logger.info('Begin execution')
    target_league = '8Premier_League'
    logger.info('Open dataframe: {0}'.format(target_league))
    raw_df = pd.read_csv(fixtures_path + target_league + '.csv', index_col=0)
    premier = models.Fixture(raw_df, 'premier')
    premier = premier.clean_fixture()
    premier = premier.generate_dataset()
    print(premier.fixture.head().to_string())

    config.time_taken_display(t0)


if __name__ == '__main__':
    main()
