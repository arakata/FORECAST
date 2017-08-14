#!/usr/bin/python3
#
"""
   Prepare odds data for analysis
"""

import pandas as pd
import numpy as np
import logging
import os 
import os.path
 
# TODO generate graphs of payouts and possible payouts(payout we would get
# if we predict matches with 100% accuracy).

logger = logging.getLogger(__name__)

def prediction_to_payout(row, threshold, gamble_heads):
    '''
    Get the payouts for betting on the 'predicted' team.
    '''
    predicted = row['predicted']
    result = row['points']
    home_payout = gamble_heads[0]
    away_payout = gamble_heads[1]
    if np.isnan(predicted):
       raise Exception('There is no prediction for certain matches')
    if predicted >= threshold:
        prediction = 3
        if prediction == result:
            output = row[home_payout]
            #print(row['index'], end = '')
        else:
            output = 0
    else:
        prediction = 0
        if prediction == result:
            output = row[away_payout]
            #print(row['index'], end = '')
        else: 
            output = 0
    return output 


def get_stats(real, pred):
    '''
    Get the number of True Positives, True Negatives, False Positives
    and False Negatives.
    '''
    tp, tn, fp, fn = (0,0,0,0)
    for i in range(len(real)):
        if pred[i] == 3:
            if real[i] == 3:
                tp += 1
            else:
                fp += 1
        else:
            if real[i] != 0:
                fn += 1
            else:
                tn += 1  
    #logger.info(('TP:{0} FP:{1} FN:{2} TN:{3}').format(tp, fp, fn, tn)) 
    return tp, fp, fn, tn
   


def gamble(odds_df, threshold, strategy, window, budget, gamble_heads):
    '''
    Predict the outcome of a certain gamble strategy for all the matches in 
    'odds_df'. Gambles will be allocated with a peridiocity of 'window'
    days. After losing the budget allocated, the gamble game is finished.
    Predictions are made using the threshold chosen. It must be lower than
    1 and higher than 0. 
    Gamble_heads is a tuple of length 2. The first one is the name of the
    column with the payout for HOME team; the second one, for AWAY team.
    The algorithm only needs those two because we are not predicting draws.
    '''
    logger.info(('Gamble -- Strat: {0} | Budget: {1}'
                 ' | Window: {2} days').format(
                     strategy.__name__, budget, window))
    df = odds_df.copy()
    n_matches = len(df)
    df['pred'] = df.apply(lambda row: int(row['predicted'] > threshold)*3, 
                     axis = 1)
    df['payout'] = df.apply(lambda row: 
                       prediction_to_payout(
                           row, threshold, gamble_heads),
                       axis = 1)

    n_correct = sum(df['payout'] > 0)
    logger.info(('Matches correctly predicted: {0} of {1} '
                 '-- {2:.2f}%').format(
                 n_correct, n_matches, n_correct/n_matches*100))
    i, gambles = 0, 0    
    while gambles < n_matches:
        #TODO store (i) matches bet, (ii) amount bet, (iii) profit received
        #    for each gambling window
        lower = i*window
        upper = window + i*window
        matches = df.iloc[lower:upper].reset_index()

        payout = matches['payout']
        bet = strategy(df, matches, budget)
        outcome = payout * bet

        cost = sum(bet)
        income = sum(outcome)
        profit = income - cost
        budget += profit
        
        print(('Week {0} -- Cost: {1:.2f} | Income: {2:.2f} | '
               'Profit: {3:.2f} | Budget: {4:.2f}').format(
               i, cost, income, profit, budget))
        
        if budget <= 0.1:
            final_outcome = 0
            logger.info('We lost all the money after {0} weeks'.format(
                i+1))
            return final_outcome
         
        i += 1
        gambles += len(matches)
    
    final_outcome = budget
    return final_outcome 

def strat_kelly_naive(df, matches, budget):
    '''
    Takes a dataframe with matches as input. Outputs a pd.Series 
    containing the amount bet in each match.
    This strat will bet a percentage of the budget equal 
    to the probability that the team wins given that we predict 
    a win: P(S=1, R=1)   
    '''
    n_matches = len(matches)
    real = df['points'].reset_index(drop = True)
    pred = df['pred'].reset_index(drop = True)
    prob_s = sum([y == 3 for y in real])
    prob_r = sum([y == 3 for y in pred])

    tp, fp, fn, tn = get_stats(real, pred)
    sensitivity = tp/(tp+fn)
    bet = (prob_s * sensitivity / prob_r) * (budget / n_matches)
    bet_list = pd.Series(bet, index = range(n_matches), dtype = 'float')
    return bet_list
   
def strat_all(df, matches, budget):
    '''
    Takes a dataframe with matches as input. Outputs a pd.Series 
    containing the amount bet in each match.
    This strat will bet in all matches. The amoun bet will be equal to
    the budget divided by the number of matches.
    '''
    n_matches = len(matches)
    bet = budget/n_matches
    bet_list = pd.Series(bet, index = range(n_matches), dtype = 'float')
    return bet_list 

def get_matches(my_df, selected_vars):
    '''
    Remove NAs, keep only home results and chosen variables in 
    'selected_vars'.
    '''
    df = my_df.copy()
    df = df.dropna()
    df = df[df['is_home'] == 1]
    df = df[selected_vars]
    return df

def preprocessing(odds_dict, odds_names, subset_vars):
    '''
    Fix names, generate index and keep subsert_vars for each odds 
    dataframe in odds_dict dictionary. 
    '''
    output = odds_dict.copy()
    for (df_name, df) in odds_dict.items():
        temp = output[df_name]
        temp = add_names(df, odds_names)
        temp['index'] = generate_index(
                            temp, 
                            'Date', 
                            'HomeTeamMain', 
                            'AwayTeamMain')
        temp = temp[subset_vars]
        output[df_name] = temp
    return output

def open_odds(my_path, subset_vars):
    '''
    Open all csv files in my_path, pick only the subset_vars varaibles 
    and store the output dataframes in  a dictionary.
    '''
    logger.info('Extracting odds information from {0}'.format(my_path))
    output = {}
    only_files = [f for f in os.listdir(my_path) if os.path.isfile(
                      os.path.join(my_path, f))]
    parser = lambda date: pd.datetime.strptime(date, '%d/%m/%y')
    for csv in only_files:
        logger.info('Importing CSV: {0}'.format(csv))
        temp  = pd.read_csv(
                    os.path.join(my_path, csv),
                    header = 0,
                    usecols = subset_vars,
                    parse_dates = ['Date'],
                    date_parser = parser)
        output[csv] = temp
    return output

def add_names(odds_df, names_df):
    '''
    Add names obtained from names_df database to the odds_df database.
    '''
    odds_names = ['HomeTeam', 'AwayTeam']
    main_names = ['HomeTeamMain', 'AwayTeamMain']
    df = odds_df.copy()
    for (nameA, nameB) in zip(odds_names, main_names):
        logger.debug('Appending names of {0}'.format(nameA))
        df = df.merge(
                 names_df, 
                 left_on = nameA, 
                 right_on = 'odds_name', 
                 indicator = True, 
                 how = 'left')
        tab_merge = df['_merge'].value_counts()
        logger.debug('Merged games: {0}'.format(tab_merge.loc['both']))
        if (tab_merge.loc['left_only'] != 0) |\
           (tab_merge.loc['right_only'] != 0):
            raise Exception('A team\'s name was not found in the ' +\
                            'odds_names database. ' +\
                            'Check if all names are there.')

        df = df[df['_merge'] == 'both']
        del(df['_merge'])
        del(df['odds_name'])
        df = df.rename(index = str, columns = {'main_name': nameB})
    return df

def generate_index(dataframe, date_name, *args, order = None):
    '''
    Generate a unique index for each observation of the database.
    The index is build as 
    {day_match}/{month_match}/{year_match}/{home_team_name}/{away_team_name}
    The name of the home team must be declared as argument before the name 
    of the away team. Order argument is used as idenfificatior for home 
    team if the names are mixed. Order variable must be 1 if first team is 
    home and 0 otherwise.
    '''
    logger.info('Creating index')
    df = dataframe.copy()
    index = df[date_name].map(lambda t: t.date().year).astype(str) +'/'+\
            df[date_name].map(lambda t: t.date().month).astype(str) +'/'+\
            df[date_name].map(lambda t: t.date().day).astype(str)

    if order == None:
        for var in args:
            index = index + '/' + df[var].astype(str)
    else:
        home = df[args[0]]
        away = df[args[1]]
        for i in list(home.index):
            obs_order = df[order][i]
            if obs_order:
                index[i] = index[i] + '/' + home[i] + '/' + away[i]
            else:
                index[i] = index[i] + '/' + away[i] + '/' + home[i]
    return index

def add_odds(main_df, odds_dict, index, info = True, print_list = False):
    '''
    Add the odds of the selected gambling house to the main_df. Use index 
    generated as key.
    '''
    output = main_df.copy().set_index(index)
    odds_dict = odds_dict.copy()
    sample = list(odds_dict.values())[0]
    sample = sample.set_index(index)
    sample_col = sample.columns.values[0]

    output = output.loc[:, output.columns.union(sample.columns)]

    for (odds_name, odds_df) in odds_dict.items():
        logger.info('Merging odds df and main database: {0}'.format(
            odds_name))
        odds_df = odds_df.set_index(index)
        output.update(odds_df)

        if info:
            temp = output.merge(
                       odds_df,
                       left_index = True,
                       right_index = True,
                       indicator = True,
                       how = 'outer')
            tab_merge = temp['_merge'].value_counts()
            logger.info('Merge results for {1}\n{0}'.format(
                tab_merge, odds_name))
            right_only = tab_merge.loc['right_only']
            logger.warning(('Matches in the odds data without match in the '
                            'main database: {0}').format(right_only))
            if (right_only > 0) & print_list:
                fails = pd.Series(
                            temp[temp['_merge'] == 'right_only'].index)
                print(fails)
    n_odds = sum(~np.isnan(output[sample_col]))
    n_obs = len(output)
    logger.info(('Odds information: {0} out of {1} observations '
                '-- {2:.2f}%').format(n_odds, n_obs, (n_odds/n_obs)*100))
    output = output.reset_index()
    return output

 
