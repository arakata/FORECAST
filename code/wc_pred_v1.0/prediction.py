#!/usr/bin/python3
#
# Author: Bruno Esposito*
# *Most of the code was written by the author of the following repository. 
#  I have modified it.
#  https://github.com/GoogleCloudPlatform/ipython-soccer-predictions
# 
# Objective: predict the outcome of football matches
#
# Style guide: https://google.github.io/styleguide/pyguide.html

import pandas as pd
import numpy as np
import math
import time
import pylab as pl
import matplotlib.pyplot as plt
from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)

import world_cup
import match_stats
import power
import odds
import code.algorithms.config as config


# ======== FUNCTIONS =========
# -------- PROCESSING --------
def print_params(model, limit=None):
    ''' Print the parameters of the model estimated '''
    # We do not print the estimated coefficients -bethas-, but the odds 
    # ratios minus 1. This means that one unit increase in the attribute 
    # A would increase in X the odds of winning. X is the number printed 
    # by this function associated with attribute A.
    params = model.params.copy()
    del params['intercept']

    if not limit:
        limit = len(params)

    print("Positive features")
    temp = params.sort_values(ascending=False)
    print(np.exp(temp[[param > 0.001 for param in temp]]).sub(1)[:limit])

    print("\nDropped features")
    print(temp[[param  == 0.0 for param in temp]][:limit])

    print("\nNegative features")
    temp = params.sort_values(ascending=True)
    print(np.exp(temp[[param < -0.001 for param in temp]]).sub(1)[:limit])


def points_to_sgn(p):
    if p > 0.1: return 1.0
#    elif p < -0.1: return -1.0
    elif p < -0.1: return 0.0
    else: return 0.0

def add_home_override(df, home_map):
    ''' Replace is_home variable for the World Cup prediction '''
    df = df.copy()
    for ii in range(len(df)):
        team = df.iloc[ii]['teamid']
        if team in home_map:
            df['is_home'].iloc[ii] = home_map[team]
        else:
            # If we don't know, assume not at home.
            df['is_home'].iloc[ii] = 0.0
    return df


# ---------- MAIN ------------
def main():

    INPUT1 = './data/raw_data_ready.csv'
    INPUT2 = './data/game_summaries_mod.csv'
    PERU_APP = './data/comebol_complete.csv'
    WC_INPUT1 = './data/wc_mod.csv'
    WC_INPUT2 = './data/wc_comp_mod.csv'
    WC_HOME = './data/wc_home.csv'
    ODDS_NAMES = './data/odds/team_names.csv'
    ODDS_PATH = './data/odds/pool/'

    OUTPUT_GRAPH_PATH = './output/graphs/'
    t0 = time.time()
    logger = config.config_logger(__name__, 10)


    # ---- Preprocessing ----
    # Import databases
    # raw_data database has information about each match before its 
    # realization and the outcome of the match. Past information is the 
    # average of each attribute among the last six games of each team. Each
    # observation is a team in a game. There will be 2 observations for 
    # each game played as long as the main database has information about 
    # the last 6 games of both teams. Only three leagues are considered 
    # from 2011 to 2014*:
    # - MLS (USA)
    # - Premier League (England)
    # - La Liga (Spain)
    #
    # *It also includes information about WCs: 2014 (only group stage),
    # 2010 and 2006. 
    logger.info('Importing CSV: {0}'.format(INPUT1))

    parser2 = lambda date: pd.datetime.strptime(date, '%Y-%m-%d %H:%M:%f')
    raw_data = pd.read_csv(
                   INPUT1, 
                   index_col = 0,
                   header = 0,
                   parse_dates = ['timestamp'],
                   date_parser = parser2,
                   encoding = 'utf-8')

    # game_summaries has information about every match played in the 
    # leagues included from 2011 to 2014 and WC data from 2014, 2010 and 
    # 2006.
    logger.info('Importing CSV: {0}'.format(INPUT2))
    game_summaries = pd.read_csv(
                         INPUT2, 
                         index_col = 0,
                         header = 0)

    logger.info('Import CSV: {0}'.format(PERU_APP))
    test_peru = pd.read_csv(
                         PERU_APP, 
                         index_col = 0,
                         header = 0)

    logger.info('Number of attributes: {0}'.format(raw_data.shape[1]))
    logger.info('Total observations: {0}'.format(len(raw_data)))

    # Partition the world cup data and the club data. We're only going to 
    # train our model using club data.
    club_data = raw_data[raw_data['competitionid'] != 4]
    logger.info('Club data observations: {0}'.format(len(club_data)))

    # Show the features latest game in competition id 4, which is the world
    # cup.
    temp_wc = raw_data[raw_data['competitionid'] == 4].iloc[0]
    
    # Generate a table with goals and points using club data.
    points = club_data.replace(
                 {'points': {
                     0: 'lose', 1: 'tie', 3: 'win'}})['points']
    goals_points =  pd.crosstab(
                        club_data['goals'],
                        points)

    logger.info('Getting descriptive stats:')
    print('Goals and points:\n{0}'.format(goals_points))
    print('\nPoints frequency:\n{0}'.format(points.value_counts()))
    print('\nGoals frequency:\n{0}'.format(
              club_data['goals'].value_counts()))
   
    # Don't train on games that ended in a draw, since they have less 
    # signal.
    # TODO We are giving up on predicting draws. Perhaps a better approach 
    # is to use an ordered logit? Or a neural network?
    train = club_data.loc[club_data['points'] != 1] 
    #train = club_data


    # ---- Processing ----
    logger.info('Beginning training')
    # The train_model function also does the following procedures:
    # - Drop observations that do not have a matching game. All matches 
    #   must have two observations.
    # - Standardized numberical varaibles: (x - mean(x))/sd(x)
    # - Pick 60% of the club_data randomly and use it as training set.
    # - Copy data from the opponent team in the same row as the first team.
    # Then, we estimate a regularized logit. 
    # The target variable is a dummy that is 1 if the first team won and 
    # 0 otherwise. The regularization parameter used is 8.
    (model, test) = world_cup.train_model(
         train, match_stats.get_non_feature_columns())
    print('\n{0}'.format(model.summary()))

    # We print the Pseudo-Rsquared and the odds ratio increase generated by
    # each attribute.
    logger.info('Rsquared: {0:.3f}'.format(model.prsquared))
    logger.info('Printing the five highest parameters from each category:')
    print_params(model)


    # ---- Postprocessing ----
    # Using the coefficients of the model, we predict the results of the 
    # test set. 
    results = world_cup.predict_model(model, test, 
        match_stats.get_non_feature_columns())
    logger.debug('Results predicted: {0}'.format(len(results)))
#    print(results.head())

    # PERU APPLICATION:
    results_peru = world_cup.predict_model(model, test_peru,
        match_stats.get_non_feature_columns())
    logger.debug('Results predicted: {0}'.format(len(results_peru)))

    pred_peru = world_cup.extract_predictions(
        results_peru.copy(), results_peru['predicted'], 50)
    logger.info('Peru predictions:\n{0}'.format(pred_peru))
    pred_peru.to_csv('./output/results_comebol.csv')
#    print(results_peru)

    # Brute force to find the threshold that maximizes the accuracy of 
    # the model.
    logger.info('Calculating optimal threshold')
    y = [yval == 3 for yval in test['points']]
    #optimal_threshold = world_cup.get_optimal_threshold(
    #                        y, results['predicted'])
    optimal_threshold = (0.5, 'NA')
    threshold = optimal_threshold[0]
    logger.info('Optimal threshold is {0} with an accuracy of {1}'.format(
                    optimal_threshold[0], optimal_threshold[1]))
    
    # Using the predictions from the test set, we check if we were right.
    # We do not asume that the probability of team A beating team B and 
    # the probability of team B beating team A add up to 1. In order assure
    # this, they normalize these probabilities. In this function, the 
    # threshold used to allocate a win to a team is 0.5. This means that, 
    # if the outcome predicted (y_hat) is higher than 0.5, the model 
    # predicts that the first team will be the winner. Also, the function 
    # multiplies y_hat by 100.
    predictions = world_cup.extract_predictions(
        results.copy(), results['predicted'], threshold*100)
    logger.info('First five predictions:\n{0}'.format(predictions.iloc[:5]))
    
    # Print True Positives and False Positives using the 0.5 -50- threshold.
    correct = predictions[(predictions['predicted'] > threshold*100) & 
                          (predictions['points'] == 3)][:5]
    print('\nCorrect predictions:')
    print(correct)

    incorrect = predictions[(predictions['predicted'] > threshold*100) & (
                             predictions['points'] < 3)][:5]
    print('\nIncorrect predictions:')
    print(incorrect)

    # Compute a baseline, which is the percentage of overall outcomes 
    # are actually wins (remember in soccer we can have draws too).
    baseline = (sum([yval == 3 for yval in club_data['points']]) 
                * 1.0 / len(club_data))
    y = [yval == 3 for yval in test['points']]
    logger.info('Proportion of wins in club data: {0:.3f}'.format(baseline))
    
    # Using the predictions from the test dataset, compute the following 
    # varaibles:
    # - False Positives -- (y_hat > threshold) != y
    # - True Positives  -- (y_hat > threshold) == y
    # - False Negatives -- (y_hat < threshold) != y
    # - True Negatives  -- (y_hat < threshold) == y
    # where, y_hat is the outcome predicted
    #        threshold is the threshold used to allocate a win
    #        y is the real outcome of the match (1 if first team won, 
    #        0 otherwise)
    # Then, compute the confusion matrix (a summary of these metrics), 
    # the lift metric, the ROC curve (Receiver Operating Characteristic 
    # curve) and the area under the ROC curve (AUC).
    #
    # It is important to notice that, in this case, the threshold used is 
    # not 0.5. Instead, the threshold is endogenous and determined by the 
    # amount of Positives of real y. If y has 40% Positives, we will pick 
    # the highest 40% of y_hat estimated and say that these predict a 
    # Positive outcome. In this scenario, the value of y_hat and the 
    # threshold does not have relevance. Instead, the most important rule 
    # to define predictions is to assure that the model predicts the same 
    # amount of Positive observations as the real y.
    #
    # TODO: perhaps there is a better approach for choosing the thresholds.
    # It might be a good idea to brute force it and maximize AUC metric
    # using the training dataset? Careful with overfitting. 
    logger.info('Prediction metrics:')
    world_cup.validate(3, y, results['predicted'], baseline,
                     compute_auc=True, quiet=False)
    pl.savefig(OUTPUT_GRAPH_PATH + '/ROC_initial.png')
    pl.close()

    # ---- Re-processing ----
    # Now, we focus on improving the prediction power of the model. The 
    # previous model lacks information about how tough were the opponents 
    # that each team faced. Therefore, we could have biased predictions if 
    # a team faced weak teams in their last matches. We might fix this 
    # issue by adding a 'power' measure as a new attribute. This new 
    # variable will try to capture the effect of the 'legacy' of a team. 
    logger.info('Adding power information')
    power_cols = [('points', points_to_sgn, 'points'), ]
    
    game_summaries = game_summaries.sort_values(
                         ['seasonid', 'matchid'], 
                         ascending = [False, True])
    logger.info('Seasons frequency:\n{0}'.format(
                    game_summaries['seasonid'].value_counts()))
    logger.info('Competitions frequency:\n{0}'.format(
                    game_summaries['competitionid'].value_counts()))


    # The power attribute tries to predict how likely is a team to win 
    # their matches, using as input only their name. 
    #
    # Add the power estimated for each team. The power calculations have 
    # been done within leagues. Since teams only face their league 
    # opponents, it would be difficult to assert if team A from league Z 
    # is better than tieam B from league W. We use game_summaries dataset 
    # to create the inputs for the power model because it contains all 
    # the matches played in the seasons selected.
    #
    # The power algorithm follows these steps for each league:
    # 1. Generate a matrix with rows representing games and columns 
    #    representing teams.
    # 2. For each element of the matrix, if the team 'i' participated 
    #    match 'j', the element [j,i] of the matrix should be filled 
    #    with a one. The value is zero otherwise. Here, teams the 
    #    attributes and games are the observations. 
    # 3. Add 0.25 to the element if the team is playing in home. Since 
    #    home advantage is important in football, the model should reflect 
    #    this fact. Adding 0.25 to the home team will reduce the 'power' 
    #    estimated of this team.
    # 4. Discount older seasons. Games from older seasons should have a 
    #    higher value. Therefore, their contribution to the power 
    #    estimation should be lower than recent seasons.
    # 5. The target variable is points obtained by the first team minus 
    #    points obtained by the second team. Therefore, the range of this 
    #    variables is {-3, 3}. The function points_to_sgn is used to 
    #    transform this variable into a binary one.
    # 6. The model is estimated using a regularized logit. The 
    #    regularization parameter starts at 0.5 and decreases each 
    #    iteration until at least one coefficient is different than zero.
    # 7. Extract the odds ratio of each attribute (team) and normalize it, 
    #    so the range of the power variable is bounded between {0,1}. 
    #
    # TODO I think we would get the same outcomes -or better- if we 
    #      just add dummy variables for each team.
    power_data = power.add_power(club_data, game_summaries, power_cols)
    
    # Like before, exclude draws from the training set.
    power_train = power_data.loc[power_data['points'] != 1] 
    # power_train = power_data

    # Estimate the model using the club data we had plus our new power 
    # variable.
    (power_model, power_test) = world_cup.train_model(
        power_train, match_stats.get_non_feature_columns())
    # Report new pseudo r-quared.
    logger.info('Rsquared: {0:.3f}, Power Coef {1:.3f}.'.format(
        power_model.prsquared, 
        math.exp(power_model.params['power_points'])))

    # Predict the outcomes of the test set. 
    power_results = world_cup.predict_model(power_model, power_test, 
        match_stats.get_non_feature_columns())
    logger.debug('Power results predicted: {0}'.format(len(power_results)))
    
    # Like before, extract metrics from the new model after predicting 
    # outcomes of the test set.
    power_y = [yval == 3 for yval in power_test['points']]
    world_cup.validate(3, power_y,
                       power_results['predicted'], baseline, 
                       compute_auc=True, quiet=False)

    # Print before and after ROC curve.
    pl.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
    # Add the old model to the graph
    world_cup.validate('old', y, results['predicted'], baseline, 
                       compute_auc=True, quiet=False)
    pl.legend(loc="lower right")
    pl.savefig(OUTPUT_GRAPH_PATH + '/ROC_power.png')
    pl.close()

    # Print estimated odds ratios.
    logger.info('Printing the five highest parameters from each category:')
    print_params(power_model, 5)


    # ---- ODDS ----
    # Generate a list with the headers related to the GAMBLE_HOUSE chosen. 
    # H stands for Home team wins, D for draw, A for Away team wins
    # TODO select the highest payoffs among home/tie/away
    gamble_house = 'B365'
    gambling_heads = [gamble_house + a for a in ['H', 'D', 'A']]

    # Importing odds data
    # The CSV file '{league}_{year1}_{year2}' contains information 
    # about the odds rate of multiple gambling houses. We will use 
    # this information to predict if our strategy is profitable. 
    # The CSV file 'team_names' contains the names of the teams that 
    # appear in the odds database and the game_summaries database.
    # Since I do not have a unique key to link the games between both
    # databases, I will generate an index using these names.
    selected_vars = ['Date', 'HomeTeam', 'AwayTeam'] + gambling_heads
 
    logger.info('Importing CSV: {0}'.format(ODDS_NAMES))
    odds_names = pd.read_csv(ODDS_NAMES, header=0, encoding='utf-8')
    odds_dict = odds.open_odds(ODDS_PATH, selected_vars) 
    
    # The funtion preprocessing does the following: 
    # 1. Add the correct names provided by odds_names dataset to the odds dataset.
    # 2. Generate an unique index that identifies observations thoughout datasets.
    # 3. Drop variables that do not appear in odds_vars (irrelevant variables).
    odds_vars = ['index'] + gambling_heads
    odds_dict = odds.preprocessing(odds_dict, odds_names, odds_vars)
    #print(odds_dict.values())

    # Add odds to the dataframe and generate an index
    logger.info('Adding odds from gambling houses to the results')
    power_results['index'] = odds.generate_index(power_results, 'timestamp', 'team_name',
                                                 'op_team_name', order='is_home')
    odds_results = odds.add_odds(power_results, odds_dict, 'index', print_list=False)

    # Keep only (i) variables relevant to the strategy and (ii) results 
    # with odds information
    strategy_vars = ['index', 'timestamp', 'team_name', 'op_team_name',
                     'competitionid', 'points', 'predicted']
    odds_results = odds.get_matches(odds_results, strategy_vars + gambling_heads)

    print(odds_results.head().to_string())
    print(odds_results.shape)
    hi

    odds_results = odds.get_graphs(odds_results)
    plt.savefig(OUTPUT_GRAPH_PATH + '/performance.png')
    plt.close()

    # Save results                                
    odds_results.to_csv('./output/temp1.csv')
    print(odds_results.head().to_string())
    logger.info('Matches with odds information: {0}'.format(
                    len(odds_results)))
 
    # The gamble function simulates a gambling exercise where the agent has
    # a fixed budget. He can bet in a window of games. The amount bet in
    # each match is chosen by the strategy. Currently, there are only two
    # strategies implemented:
    # 1. strat_all: bet the whole budget in equal parts on each match in
    #    the window.
    # 2. strat_kelly_naive: the percentage of the budget bet is chosen by
    #    a naive kelly's rule, which asumes that there is no 'track take'.
    #    This means that the user can still make cancelling bets.
    #
    # Then, we compute the cost of the bets and the income recieved. Thus,
    # we find a new budget for the next window. If the budget is, at any 
    # point, lower than 0.01, the exercise finishes.
    #
    # Obtain gambling results 
    HA_payouts = (gambling_heads[0], gambling_heads[2])
    logger.info('Beginning gamble')
    print(odds_results.head().to_string())
    hi

    gamble_results = odds.gamble(odds_results,
                                 threshold=0.5,
                                 strategy=odds.strat_kelly_naive,
                                 window=10,
                                 budget=1000,
                                 gamble_heads=HA_payouts)
    logger.info('Final budget: {0:.2f}'.format(gamble_results))
    hi
         
    # ---- WC ----
    # We begin with the World Cup matches.

    # Dataset with the WC games and their attributes as an average of the 
    # previous 6 matches. Includes games from older WCs. Does not include 
    # results of matches.
    wc_data = pd.read_csv(
                  WC_INPUT1,
                  index_col = 0,
                  header = 0)
    # Same database as game_summaries.
    wc_labeled = pd.read_csv(
                     WC_INPUT2,
                     index_col = 0,
                     header = 0)
    # Dataset with the home attibute of the national teams in the WC. The 
    # WC was played in Brazil, but Brazil was not the only one considered 
    # as home team. 
    wc_home = pd.read_csv(
                  WC_HOME,
                  index_col=0,
                  header=0)

    wc_labeled = wc_labeled[wc_labeled['competitionid'] == 4]
    wc_power_train = game_summaries[
                         game_summaries['competitionid'] == 4].copy()
 
    home_override = {}
    for ii in range(len(wc_home)):
        row = wc_home.iloc[ii]
        home_override[row['teamid']] = row['is_home']
    
    # Change is_home attribute of national teams in the WC.
    wc_data = add_home_override(wc_data, home_override) 

    # When training power data, since the games span multiple competitions,
    # just set is_home to 0.5. Otherwise when we looked at games from the 
    # 2010 world cup, we'd think Brazil was still at home instead of South 
    # Africa.
    wc_power_train['is_home'] = 0.5
    wc_power_data = power.add_power(wc_data, wc_power_train, power_cols)

    # Predict the WC using the model we had estimated.
    wc_results = world_cup.predict_model(power_model, wc_power_data, 
        match_stats.get_non_feature_columns())

    wc_with_points = wc_power_data.copy()
    wc_with_points.index = pd.Index(
        zip(wc_with_points['matchid'], wc_with_points['teamid']))
    wc_labeled.index = pd.Index(
        zip(wc_labeled['matchid'], wc_labeled['teamid']))
    wc_with_points['points'] = wc_labeled['points']

    # Extract WC predictions.
    wc_pred = world_cup.extract_predictions(wc_with_points, 
                                            wc_results['predicted'])

    # Reverse our predictions to show the most recent first.
    wc_pred.reindex(index=wc_pred.index[::-1])

    # Show our predictions for the games that have already happenned.
    print(wc_pred[wc_pred['points'] >= 0.0])
    print(wc_pred[~(wc_pred['points'] >= 0)])

    config.time_taken_display(t0)
    print(' ')


if __name__ == '__main__':
    main()
