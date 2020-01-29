import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from collections import defaultdict
from tqdm import tqdm_notebook as tqdm
from collections import Counter
import json

def get_accuracy(correct_data):
    # Rounding correct > 1 to 1 lowers the score. Why?
    correct = len(correct_data.loc[correct_data])
    wrong = len(correct_data.loc[~correct_data])
    accuracy = correct/(correct + wrong) if correct + wrong else 0
    return accuracy, correct, wrong

def get_group(accuracy):
    if not accuracy:
        return 0
    elif accuracy == 1:
        return 3
    elif accuracy >= 0.5:
        return 2
    return 1

# I prefer this over calculating average
def lin_comb(v1, v2, beta): return beta*v1 + (1-beta)*v2

def prepare(data: pd.DataFrame, one_hot, test=False) -> pd.DataFrame:
    one_hot_dict = defaultdict(int)

    prepared = []
    for id_, g in tqdm(data.groupby("installation_id", sort=False)):
        features = process_id(g, one_hot, one_hot_dict.copy(), test)
        if not features:
            continue
        if test:
            features[-1]["is_test"] = 1
        prepared.extend(features)
    return pd.DataFrame(prepared).fillna(0).sort_index(axis=1)

def process_id(id_data: pd.DataFrame, one_hot_cols, one_hot_dict, test: bool) -> pd.DataFrame:
    a_accuracy, a_group, a_correct, a_wrong, counter, accumulated_duration_mean = 0, 0, 0, 0, 0, 0
    a_groups = {"0":0, "1":0, "2":0, "3":0}
    a_durations = defaultdict(int)
    features = []

    for s, gs in id_data.groupby("game_session", sort=False):
        def update_counter(counter: dict, column: str):
            session_counter = Counter(gs.loc[:, column])
            for value in session_counter.keys():
                counter[f"{column}_{value}"] += session_counter[value]
            return counter

        def process_session(gs):
            # share state with parent process_id()
            nonlocal one_hot_dict, a_groups, a_durations, a_accuracy, a_group, a_correct, a_wrong, counter, accumulated_duration_mean
            # increment one hot columns for session, e.g. Bird Measurer: 50
            def accumulate():
                nonlocal accumulated_duration_mean
                # accumulated one_hot features per id for a given session, e.g. Bird Measurer: 50
                for c in one_hot_cols:
                    one_hot_dict.update(update_counter(one_hot_dict, c))
                duration = (gs["timestamp"].iloc[-1] - gs["timestamp"].iloc[0]).seconds
                # an accumulated session duration mean
                accumulated_duration_mean = lin_comb(accumulated_duration_mean or duration,
                                                     duration, beta=0.9)
                a_durations[f"duration_{gs.title.iloc[0]}"] = duration
                
            if gs["type"].iloc[0] != "Assessment":
                accumulate()
                return

            guess_mask = ((gs["event_data"].str.contains("correct")) & 
             (((gs["event_code"] == 4100) &(~gs["title"].str.startswith("Bird")) | 
               ((gs["event_code"] == 4110) & (gs["title"].str.startswith("Bird"))))))
            answers = gs.loc[guess_mask, "event_data"].apply(lambda x: json.loads(x).get("correct"))

            # skip assessments without attempts in train
            if answers.empty and not test:
                accumulate()
                return

            accuracy, correct, wrong = get_accuracy(answers)
            group = get_group(accuracy)
            processed = {"installation_id": id_data["installation_id"].iloc[0],
                         "title": gs["title"].iloc[0],
                         "timestamp": gs["timestamp"].iloc[0],
                         "accumulated_duration_mean": accumulated_duration_mean,
                         "accumulated_correct": a_correct, "accumulated_incorrect": a_wrong,
                         "accumulated_accuracy_mean": a_accuracy/counter if counter > 0 else 0,
                         "accumulated_accuracy_group_mean": a_group/counter if counter > 0 else 0, 
                         "accuracy_group": group,
                        }
            processed.update(a_groups)
            processed.update(one_hot_dict)
            processed.update(a_durations)
            counter += 1
            a_accuracy += accuracy
            a_correct += correct
            a_wrong += wrong
            a_group += group
            a_groups[str(group)] += 1
            accumulate()
            return processed
        
        # skip sessions with 1 row
        if len(gs) == 1 and not test:
            continue
        gs.reset_index(inplace=True, drop=True)
        if (gs["timestamp"].iloc[-1] - gs["timestamp"].iloc[0]).seconds > 1800:
            gs["passed"] = gs.loc[:, "timestamp"].diff().apply(lambda x: x.seconds)
            id_max = gs["passed"].idxmax()
            if gs["passed"].max() > 1800:
                session = gs.iloc[:id_max]
                continued_session = gs.iloc[id_max:]
                fs = process_session(session)
                c_fs = process_session(continued_session)
                if fs:
                    features.append(fs)
                if c_fs:
                    features.append(c_fs)
                continue

        session_features = process_session(gs)
        if session_features:
            features.append(session_features)
        
    return features