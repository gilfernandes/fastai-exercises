import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Any results you write to the current directory are saved as output.
from time import time
from tqdm import tqdm_notebook as tqdm
from collections import Counter
from scipy import stats
import lightgbm as lgb
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import KFold, StratifiedKFold
from scipy.stats import kurtosis, skew
import gc
import json
pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

from pathlib import Path
import sys
import re
import json
import math

def remove_wrong_event_codes(df):
    return df[((df['title'] == 'Bird Measurer (Assessment)') & (df['event_code'] == 4100)) == False]

def remove_ids_with_no_assessment(df):
    # Remove `installation_id` without any assesments
    ids_with_subms = df[df.type == "Assessment"][['installation_id']].drop_duplicates()
    df = pd.merge(df, ids_with_subms, on="installation_id", how="inner")
    return df

def create_unique_list(field_name, train_df, test_df):
    return list(set(train_df[field_name].unique()) | (set(test_df[field_name].unique())))

def safe_div(dividend, divisor):
    return dividend / divisor if divisor != 0 else 0

def update_counters(counter, col, session):
    increment_counter = Counter(session[col])
    for x in increment_counter.keys():
        counter[x] += increment_counter[x]
    return counter

MAX_SESSION_LENGTH = 1800

def create_combined_col(df, col1, col2):
    df[f'{col1}_{col2}']= df[col1].map(str) + df[col2].map(str)
    
def create_combined_cols(dfs, col1, col2):
    for df in dfs:
        create_combined_col(df, col1, col2)
    return create_unique_list(f'{col1}_{col2}', dfs[0], dfs[1])

def create_structs(train_df, test_df):
    list_of_user_activities = create_unique_list('title', train_df, test_df)
    activities_labels = dict(zip(np.arange(len(list_of_user_activities)), list_of_user_activities))
    activities_map = dict(zip(list_of_user_activities, np.arange(len(list_of_user_activities))))
    win_code = dict(zip(activities_map.values(), (4100*np.ones(len(activities_map))).astype('int')))
    assess_titles = list(set(train_df[train_df['type'] == 'Assessment']['title'].value_counts().index) | (set(test_df[test_df['type'] == 'Assessment']['title'].value_counts().index)))
    # then, it set one element, the 'Bird Measurer (Assessment)' as 4110, 10 more than the rest
    win_code[activities_map['Bird Measurer (Assessment)']] = 4110
    list_of_event_code = create_unique_list('event_code', train_df, test_df)
    list_of_event_id = create_unique_list('event_id', train_df, test_df)
    list_of_worlds = create_unique_list('world', train_df, test_df)
    list_of_title = create_unique_list('title', train_df, test_df)
    train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])
    test_df['timestamp'] = pd.to_datetime(train_df['timestamp'])
    # Create extra columns
    list_of_event_code_world = create_combined_cols([train_df, test_df], 'event_code', 'world')
    list_of_event_code_title = create_combined_cols([train_df, test_df], 'event_code', 'title')
    list_of_event_id_world = create_combined_cols([train_df, test_df], 'event_id', 'world')
    return list_of_user_activities, activities_labels, activities_map, win_code, assess_titles, list_of_event_code, list_of_event_id, list_of_worlds, list_of_title, list_of_event_code_world, list_of_event_code_title, list_of_event_id_world

def create_counter(unique_list):
    return Counter({ev: 0 for ev in unique_list})

def convert_to_accuracy_group(accuracy):
    if accuracy == 0:
        return 0
    elif accuracy == 1:
        return 3
    elif accuracy == 0.5:
        return 2
    else:
        return 1

def feature_generation(samples, is_test=False, sample_slice=slice(0, sys.maxsize), extra_training=[], assess_titles=[],
                       list_of_event_code=[],
                       list_of_event_id=[], list_of_worlds=[],
                       list_of_title=[], win_code=[], 
                       activities_map=[], list_of_event_code_world=[],
                       list_of_event_code_title=[], list_of_event_id_world=[], include_all=False):

    all_features = [] # 21239
    split_counter = 0

    for (installation_id, user_sample) in tqdm(samples[sample_slice], total = len(samples[sample_slice])):
        last_accuracy = 0
        accumulated_accuracy = 0
        accumulated_actions = 0
        counter = 0
        accumulated_correct_attempts = 0
        accumulated_incorrect_attempts = 0
        accumulated_accuracy_group = 0
        last_accuracy_title = {'acc_' + title: -1 for title in assess_titles}
        title_durations = {title: [] for title in assess_titles}
        user_activities_count = {'Clip':0, 'Activity': 0, 'Assessment': 0, 'Game':0}
        user_activities_time = user_activities_count.copy()
        user_activities_game_time = {'Clip': [], 'Activity': [], 'Assessment': [], 'Game':[]}
        user_activities_session_len = user_activities_count.copy()
        event_code_count = create_counter(list_of_event_code)
        event_code_world_count = create_counter(list_of_event_code_world)
        event_code_title_count = create_counter(list_of_event_code_title)
        event_id_world_count = create_counter(list_of_event_id_world)
        event_code_count_previous = event_code_count.copy()
        event_code_count_hist = {ev: [] for ev in list_of_event_code}
        event_id_count = create_counter(list_of_event_id)
        event_id_count_previous = event_id_count.copy()
        world_count = create_counter(list_of_worlds)
        world_count_previous = world_count.copy()
        title_count = create_counter(list_of_title)
        title_count_previous = title_count.copy()
        train_features = None
        compiled_train = []
        durations = []
        
        for i, session in user_sample.groupby('game_session', sort=False):
            def process_session(session):
                nonlocal accumulated_accuracy, counter, accumulated_correct_attempts, accumulated_accuracy_group, last_accuracy_title,\
                user_activities_count, user_activities_time, user_activities_game_time, user_activities_session_len, event_code_count,\
                world_count, title_count, train_features, compiled_train, durations, event_id_count, accumulated_incorrect_attempts,\
                event_code_count_previous, event_id_count_previous, world_count_previous, title_count_previous, event_code_count_hist,\
                event_code_world_count, event_code_title_count, event_id_world_count, last_accuracy, accumulated_actions
                
                session_type = session['type'].iloc[0]
                session_title = session['title'].iloc[0]
                session_length = len(session)

                def increment_user_activities():
                    user_activities_count[session_type] += 1
                    last_time = session['game_time'].iloc[-1] # select last time
                    user_activities_time[session_type] += last_time
                    user_activities_game_time[session_type].append(last_time / 1000)
                    user_activities_session_len[session_type] += session_length
                    
                def increment_counters(event_code_count, event_id_count, world_count, title_count, event_code_world_count, event_code_title_count, event_id_world_count):
                    counters = [event_code_count, event_id_count, world_count, title_count, event_code_world_count, event_code_title_count, event_id_world_count]
                    count_fields = ['event_code', 'event_id', 'world', 'title', 'event_code_world', 'event_code_title', 'event_id_world']
                    return [update_counters(counter, count_fields[i], session) for i, counter in enumerate(counters)]
                
                if (session_type == 'Assessment') and (session_length > 1 or is_test):
                    all_attempts = session[session['event_code'] == win_code[activities_map[session_title]]]
                    all_attempts_str = all_attempts['event_data'].str
                    true_attempts = all_attempts_str.contains('true').sum()
                    false_attempts = all_attempts_str.contains('false').sum()

                    train_features = user_activities_count.copy()
                    train_features['game_session'] = i
                    train_features.update({f'{k}_time_mean': safe_div(v, user_activities_session_len[k]) for k,v in user_activities_time.items()})

                    # the time spent in the app so far
                    if durations == []:
                        train_features['duration_mean'] = 0
                    else:
                        train_features['duration_mean'] = np.mean(durations)
                        train_features['duration_std'] = np.std(durations)
                        train_features['duration_skew'] = skew(durations)
                        train_features['duration_kurtosis'] = kurtosis(durations)
                    duration = (session['timestamp'].iloc[-1] - session['timestamp'].iloc[0] ).seconds
                    train_features.update({f'title_duration_mean{k}': np.mean(v) for k,v in title_durations.items()})
                    train_features.update({f'activity_game_time_{k}': np.mean(v) for k,v in user_activities_game_time.items()})
                    
                    title_durations[session_title].append(duration)
                    durations.append(duration)

#                     train_features.update({f'{k}_diff':event_id_count[k] - v for k,v in event_id_count_previous.items()})
                    train_features.update(event_id_count.copy())
                    event_id_count_previous = event_id_count.copy()
#                     train_features.update({f'{k}_diff':event_code_count[k] - v for k,v in event_code_count_previous.items()})
                    event_code_count_previous = event_code_count.copy()
                    train_features.update(event_code_count.copy())
#                     train_features.update({f'{k}_mean': np.mean(v) for k,v in event_code_count_hist.items()})
                    event_code_count_hist.update({k: [*v, event_code_count[k]] if event_code_count[k] not in v else [*v] for k,v in event_code_count_hist.items()})
#                     train_features.update({f'{k}_diff':world_count[k] - v for k,v in world_count_previous.items()})
                    train_features.update(world_count.copy())
                    train_features.update(event_code_world_count.copy())
                    train_features.update(event_code_title_count.copy())
#                     train_features.update(event_id_world_count.copy())
                    world_count_previous = world_count.copy()
#                     train_features.update({f'{k}_diff':title_count[k] - v for k,v in title_count_previous.items()})
                    train_features.update(title_count.copy())
                    title_count_previous = title_count.copy()
                    train_features.update(last_accuracy_title.copy())
                    train_features['timestamp'] = session["timestamp"].iloc[0]
                    train_features['accumulated_accuracy'] = accumulated_accuracy / counter if counter > 0 else 0
                    train_features['session_title'] = activities_map[session['title'].iloc[0]]
                    train_features['accumulated_correct_attempts'] = accumulated_correct_attempts
                    train_features['accumulated_incorrect_attempts'] = accumulated_incorrect_attempts
                    train_features['installation_id'] = session['installation_id'].iloc[-1]
                    train_features['accumulated_accuracy_group'] = accumulated_accuracy_group/counter if counter > 0 else 0
                    train_features['last_accuracy'] = last_accuracy
                    train_features['accumulated_actions'] = accumulated_actions

                    accumulated_correct_attempts += true_attempts
                    accumulated_incorrect_attempts += false_attempts
                    accuracy = true_attempts/(true_attempts+false_attempts) if (true_attempts+false_attempts) != 0 else 0
                    accumulated_accuracy += accuracy

                    train_features['accuracy_group'] = convert_to_accuracy_group(accuracy)

                    accumulated_accuracy_group += train_features['accuracy_group']

                    if is_test:
                        compiled_train.append(train_features)
                    elif true_attempts + false_attempts > 0:
                        compiled_train.append(train_features)
                    counter += 1

                    last_accuracy_title['acc_' + session_title] = accuracy
                    last_accuracy = accuracy
                    train_features['accuracy'] = accuracy
                    
                if not is_test or not (session_type == 'Assessment' and session_length == 1):
                    accumulated_actions += len(session)
                    increment_user_activities()
                    event_code_count, event_id_count, world_count, title_count, event_code_world_count, event_code_title_count, event_id_world_count = \
                    increment_counters(event_code_count, event_id_count, world_count, title_count, event_code_world_count, event_code_title_count, event_id_world_count)
                    
            if (session["timestamp"].iloc[-1] - session["timestamp"].iloc[0]).seconds > MAX_SESSION_LENGTH:
                session["passed"] = session.loc[:, "timestamp"].diff().apply(lambda x: x.seconds)
                id_max = np.nanargmax(session["passed"].to_numpy())
                if session["passed"].max() > MAX_SESSION_LENGTH:
                    initial_session = session.iloc[:id_max]
                    continued_session = session.iloc[id_max:]
                    process_session(initial_session)
                    if(len(continued_session) > 0):
                        split_counter += 1
                        process_session(continued_session)
                    continue
            
            process_session(session)
            
        if not is_test or include_all:
            all_features += compiled_train 
        else:
            all_features.append(compiled_train[-1]) # just append the last one
            extra_training.extend(compiled_train[:-1])

    comp_df = pd.DataFrame(all_features)
    comp_df.fillna(0, inplace=True)
    return comp_df


def feature_generation_2(df, test_set=False, sample_slice=slice(0, sys.maxsize), 
                         assess_titles=[], 
                         list_of_event_code=[], 
                         list_of_event_id=[], 
                         activities_labels=[], 
                         all_title_event_code=[],
                         win_code=[],
                         activities_map={},
                         extra_training=[],
                         include_all=False):
    '''
    The user_sample is a DataFrame from train or test where the only one 
    installation_id is filtered
    And the test_set parameter is related with the labels processing, that is only required
    if test_set=False
    '''
    all_features = [] # 21239
    split_counter = 0

    for (installation_id, user_sample) in tqdm(df[sample_slice], total = len(df[sample_slice])):
        # Constants and parameters declaration
        last_activity = 0

        user_activities_count = {'Clip':0, 'Activity': 0, 'Assessment': 0, 'Game':0}

        # new features: time spent in each activity
        last_session_time_sec = 0
        accuracy_groups = {0:0, 1:0, 2:0, 3:0}
        all_assessments = []
        accumulated_accuracy_group = 0
        accumulated_accuracy = 0
        accumulated_correct_attempts = 0 
        accumulated_uncorrect_attempts = 0
        accumulated_actions = 0
        counter = 0
        durations = []
        last_accuracy_title = {'acc_' + title: -1 for title in assess_titles}
        event_code_count: Dict[str, int] = {ev: 0 for ev in list_of_event_code}
        event_id_count: Dict[str, int] = {eve: 0 for eve in list_of_event_id}
        title_count: Dict[str, int] = {eve: 0 for eve in activities_map.values()} 
        title_event_code_count: Dict[str, int] = {t_eve: 0 for t_eve in all_title_event_code}

        # itarates through each session of one instalation_id
        for i, session in user_sample.groupby('game_session', sort=False):
            # i = game_session_id
            # session is a DataFrame that contain only one game_session

            # get some sessions information
            session_type = session['type'].iloc[0]
            session_title = session['title'].iloc[0]
            session_title_text = session_title

            # for each assessment, and only this kind off session, the features below are processed
            # and a register are generated
            if (session_type == 'Assessment') & (test_set or len(session)>1):
                # search for event_code 4100, that represents the assessments trial
                all_attempts = session[session['event_code'] == win_code[activities_map[session_title]]]
                # then, check the numbers of wins and the number of losses
                true_attempts = all_attempts['event_data'].str.contains('true').sum()
                false_attempts = all_attempts['event_data'].str.contains('false').sum()
                # copy a dict to use as feature template, it's initialized with some itens: 
                # {'Clip':0, 'Activity': 0, 'Assessment': 0, 'Game':0}
                features = user_activities_count.copy()
                features.update(last_accuracy_title.copy())
                features.update(event_code_count.copy())
                features.update(event_id_count.copy())
                features.update(title_count.copy())
                features.update(title_event_code_count.copy())
                features.update(last_accuracy_title.copy())

                # get installation_id for aggregated features
                features['installation_id'] = session['installation_id'].iloc[-1]
                # add title as feature, remembering that title represents the name of the game
                features['session_title'] = activities_map[session_title_text]
                # the 4 lines below add the feature of the history of the trials of this player
                # this is based on the all time attempts so far, at the moment of this assessment
                features['accumulated_correct_attempts'] = accumulated_correct_attempts
                features['accumulated_incorrect_attempts'] = accumulated_uncorrect_attempts
                accumulated_correct_attempts += true_attempts 
                accumulated_uncorrect_attempts += false_attempts
                # the time spent in the app so far
                if durations == []:
                    features['duration_mean'] = 0
                else:
                    features['duration_mean'] = np.mean(durations)
                durations.append((session.iloc[-1, 2] - session.iloc[0, 2] ).seconds)
                # the accurace is the all time wins divided by the all time attempts
                features['accumulated_accuracy'] = accumulated_accuracy/counter if counter > 0 else 0
                accuracy = true_attempts/(true_attempts+false_attempts) if (true_attempts+false_attempts) != 0 else 0
                accumulated_accuracy += accuracy
                last_accuracy_title['acc_' + session_title_text] = accuracy
                # a feature of the current accuracy categorized
                # it is a counter of how many times this player was in each accuracy group
                if accuracy == 0:
                    features['accuracy_group'] = 0
                elif accuracy == 1:
                    features['accuracy_group'] = 3
                elif accuracy == 0.5:
                    features['accuracy_group'] = 2
                else:
                    features['accuracy_group'] = 1
                features.update(accuracy_groups)
                accuracy_groups[features['accuracy_group']] += 1
                # mean of the all accuracy groups of this player
                features['accumulated_accuracy_group'] = accumulated_accuracy_group/counter if counter > 0 else 0
                accumulated_accuracy_group += features['accuracy_group']
                # how many actions the player has done so far, it is initialized as 0 and updated some lines below
                features['accumulated_actions'] = accumulated_actions
                features['accuracy'] = accuracy

                # there are some conditions to allow this features to be inserted in the datasets
                # if it's a test set, all sessions belong to the final dataset
                # it it's a train, needs to be passed throught this clausule: session.query(f'event_code == {win_code[session_title]}')
                # that means, must exist an event_code 4100 or 4110
                if test_set:
                    all_assessments.append(features)
                elif true_attempts+false_attempts > 0:
                    all_assessments.append(features)

                counter += 1

            # this piece counts how many actions was made in each event_code so far
            def update_counters(counter: dict, col: str):
                    num_of_session_count = Counter(session[col])
                    for k in num_of_session_count.keys():
                        x = k
                        if col == 'title':
                            x = activities_map[k]
                        counter[x] += num_of_session_count[k]
                    return counter

            event_code_count = update_counters(event_code_count, "event_code")
            event_id_count = update_counters(event_id_count, "event_id")
            title_count = update_counters(title_count, 'title')
            title_event_code_count = update_counters(title_event_code_count, 'title_event_code')

            # counts how many actions the player has done so far, used in the feature of the same name
            accumulated_actions += len(session)
            if last_activity != session_type:
                user_activities_count[session_type] += 1
                last_activitiy = session_type 

        # if it't the test_set, only the last assessment must be predicted, the previous are scraped
        if not test_set or include_all:
            all_features += all_assessments
        # in the train_set, all assessments goes to the dataset
        else:
            if test_set:
                all_features.append(all_assessments[-1])
            else:
                all_features += all_assessments
        if test_set:
            extra_training.extend(all_assessments[:-1])
            
    comp_df = pd.DataFrame(all_features)
    comp_df.fillna(0, inplace=True)
    return comp_df

def accumulate_diffs(group):
    previous = 0
    diffs = []
    for v in group:
        diffs.append(v - previous)
        previous = v
    return diffs

def mean_diff(group):
    return np.mean(accumulate_diffs(group))

def median_diff(group):
    return np.median(accumulate_diffs(group))

def std_diff(group):
    return np.std(accumulate_diffs(group))

def max_diff(group):
    return np.max(accumulate_diffs(group))

def preprocess(reduce_train, reduce_test, activities_map=[]):
    
    for df in [reduce_train, reduce_test]:
        df['installation_session_count'] = df.groupby(['installation_id'])['Clip'].transform('count')
        df['installation_duration_mean'] = df.groupby(['installation_id'])['duration_mean'].transform('mean')
        df['accumulated_actions_diff_mean'] = df.groupby(['installation_id'])['accumulated_actions'].transform(mean_diff)
        df['accumulated_actions_diff_std'] = df.groupby(['installation_id'])['accumulated_actions'].transform(std_diff)
        df['accumulated_actions_diff_median'] = df.groupby(['installation_id'])['accumulated_actions'].transform(median_diff)
        selected_activities = ['Assessment', 'Clip', 'Activity', 'Game'] + [i for i in list(activities_map.values())]
        for selected_activity in selected_activities:
            df[f'{selected_activity}_diff_mean'] = df.groupby(['installation_id'])[selected_activity].transform(mean_diff)
        df['installation_title_nunique'] = df.groupby(['installation_id'])['session_title'].transform('nunique')
        
#         df['sum_event_code_count'] = df[[2050, 4100, 4230, 5000, 4235, 2060, 4110, 5010, 2070, 2075, 2080, 2081, 2083, 3110, 4010, 3120, 3121, 4020, 4021, 4022, 4025, 4030, 4031, 3010, 4035, 4040, 3020, 3021, 4045, 2000, 4050, 2010, 2020, 4070, 2025, 2030, 4080, 2035, 2040, 4090, 4220, 4095]].sum(axis = 1)
        
#         df['installation_event_code_count_mean'] = df.groupby(['installation_id'])['sum_event_code_count'].transform(mean_diff)
        
    return reduce_train, reduce_test

## Training related stuff

from functools import partial
import scipy as sp
from sklearn.metrics import cohen_kappa_score

class OptimizedRounder():
    """
    An optimizer for rounding thresholds
    to maximize Quadratic Weighted Kappa (QWK) score
    # https://www.kaggle.com/naveenasaithambi/optimizedrounder-improved
    """
    def __init__(self, initial_coef, labels):
        self.coef_ = 0
        self.initial_coef = initial_coef
        self.labels = labels

    def _kappa_loss(self, coef, X, y):
        """
        Get loss according to
        using current coefficients
        
        :param coef: A list of coefficients that will be used for rounding
        :param X: The raw predictions
        :param y: The ground truth labels
        """
        X_p = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = self.labels)
        return -cohen_kappa_score(X_p, y, weights="quadratic")

    def fit(self, X, y):
        """
        Optimize rounding thresholds
        
        :param X: The raw predictions
        :param y: The ground truth labels
        """
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        self.coef_ = sp.optimize.minimize(loss_partial, self.initial_coef, method='nelder-mead')

    def predict(self, X, coef):
        """
        Make predictions with specified thresholds
        
        :param X: The raw predictions
        :param coef: A list of coefficients that will be used for rounding
        """
        return pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = self.labels)

    def coefficients(self): return self.coef_['x']