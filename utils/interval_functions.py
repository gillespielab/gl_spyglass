import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

import spyglass.common as sgc
import spyglass.position as sgp
from spyglass.common.common_interval import Interval

sys.path.append("../..")
sys.path.append("../../gl_spyglass/")
from gl_spyglass.custom_spyglass_tables.trial_info import TrialInfo8Arm

def get_bool_intervals(arr):
    '''
    Convert boolean array (of length timestamps) to sets of intervals (list of start and end indices)
    '''
    arr = np.asarray(arr, dtype=bool)

    # Pad to catch intervals at the edges
    padded = np.concatenate([[False], arr, [False]])

    diff = np.diff(padded.astype(int))

    starts = np.where(diff == 1)[0]
    ends   = np.where(diff == -1)[0]

    return np.column_stack((starts, ends))


def convert_to_times(intervals, times):
    '''
    Given a set of intervals (start and end indices of a boolean array of len(times)),
    and the times array, get the start and end times.
    '''
    return np.array([
        [times[s], times[e - 1]]
        for s, e in intervals
    ])


def insert_immobile_times_interval(
    nwb_file_name,
    interval_list_name,
    trodes_pos_params_name,
    speed_thresh=4,
    time_thresh=1,
    new_interval_list_name=None,
    plot_pos=True,
    trunc_time=None,
):
    '''
    Insert an immobile times interval for the interval list you're pulling in (if it doesn't already exist)
    Returns the name of the immobile times interval list
    
    :param nwb_file_name: nwb_file_name for this session
    :param interval_list_name: interval_list_name for this epoch that you want to parse into immobile times
    :param trodes_pos_params_name: whichever lets you parse out the position dataframe for this epoch (to get the speed)
    :param speed_thresh: (cm/s) the speed needs to be < this value to be considered in these intervals
    :param time_thresh: (s) minimum duration of the interval to be considered one of these intervals
    :param new_interval_list_name: (optional) if you have a specific interval list name in mind, otherwise will default to appending 'immobile times' to the existing interval list name
    :param plot_pos: (optional) if True will plot the position only within the immobile interval list (good for sanity checking)
    :param trunc_time: (optional) (s) if you want to truncate the intervals (say you want intervals when the animal is immobile for at least 60s but you want to discard (truncate) the first 30), this is where you'd set that
    '''

    if new_interval_list_name is None:
        if "valid times" in interval_list_name:
            new_interval_list_name = interval_list_name.replace(
                "valid times", "immobile times"
            )
        else:
            new_interval_list_name = f"{interval_list_name} immobile times"

    if time_thresh > 1:
        new_interval_list_name = new_interval_list_name + f" {time_thresh}s"
    
    if trunc_time != None:
        new_interval_list_name = new_interval_list_name + f" trunc {trunc_time}s"

    # check if this interval already exists
    if (
        len(
            sgc.IntervalList()
            & {
                "nwb_file_name": nwb_file_name,
                "interval_list_name": new_interval_list_name,
            }
        )
        != 0
    ):
        print(f"{new_interval_list_name} already exists, skipping")
        return new_interval_list_name

    # create selection key
    trodes_s_key = {
        "nwb_file_name": nwb_file_name,
        "interval_list_name": interval_list_name,
        "trodes_pos_params_name": trodes_pos_params_name,
    }

    # use trodes selection key to load pos_df
    trodes_key = (sgp.v1.TrodesPosSelection() & trodes_s_key).fetch1("KEY")
    merge_key = (sgp.PositionOutput.merge_get_part(trodes_key)).fetch1("KEY")
    pos_df = (sgp.PositionOutput & merge_key).fetch1_dataframe()

    # get boolean df with whether each timepoint is lower than speed_thresh
    speed_bools = pd.DataFrame(
        pos_df.loc[pos_df.index, "speed"] < speed_thresh
    ).reset_index()
    print(
        f'fraction immobile time = {len(pos_df[pos_df["speed"] < speed_thresh]) / len(pos_df)}'
    )

    if plot_pos:
        # plot position at immobile times as a sanity check
        n = len(pos_df)
        pos_immobile_df = pos_df[pos_df["speed"] < speed_thresh]
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.scatter(
            pos_immobile_df.iloc[:n]["position_x"], pos_immobile_df.iloc[:n]["position_y"]
        )
        ax.set_xlabel("x-position [cm]", fontsize=16)
        ax.set_ylabel("y-position [cm]", fontsize=16)
        ax.set_title("Position", fontsize=20)
        plt.show()

    immobile_intervals = convert_to_times(get_bool_intervals(speed_bools['speed'].values), speed_bools['time'].values)
    immobile_durations = immobile_intervals[:, 1] - immobile_intervals[:, 0]

    # Keep intervals longer than the threshold
    immobile_intervals = immobile_intervals[immobile_durations >= time_thresh, :]

    if trunc_time != None:
        immobile_intervals[:, 0] += trunc_time

    # insert immobile intervals into IntervalList
    sgc.IntervalList.insert1(
        {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": new_interval_list_name,
            "valid_times": np.asarray(immobile_intervals),
        },
        skip_duplicates=True,
    )
    print(f"Inserted new interval: {new_interval_list_name}")

    return new_interval_list_name


def insert_mobile_times_interval(
    nwb_file_name,
    interval_list_name,
    trodes_pos_params_name,
    speed_thresh=4,
    time_thresh=1,
    plot_pos=True,
    new_interval_list_name=None,
):
    '''
    Insert a mobile times interval for the interval list you're pulling in (if it doesn't already exist)
    Returns the name of the mobile times interval list
    
    :param nwb_file_name: nwb_file_name for this session
    :param interval_list_name: interval_list_name for this epoch that you want to parse into immobile times
    :param trodes_pos_params_name: whichever lets you parse out the position dataframe for this epoch (to get the speed)
    :param speed_thresh: (cm/s) the speed needs to be >= this value to be considered in these intervals
    :param time_thresh: (s) minimum duration of the interval to be considered one of these intervals
    :param new_interval_list_name: (optional) if you have a specific interval list name in mind, otherwise will default to appending 'mobile times' to the existing interval list name
    :param plot_pos: (optional) if True will plot the position only within the mobile interval list (good for sanity checking)
    '''

    if new_interval_list_name is None:
        if "valid times" in interval_list_name:
            new_interval_list_name = interval_list_name.replace(
                "valid times", "mobile times"
            )
        else:
            new_interval_list_name = f"{interval_list_name} mobile times"
    
    if time_thresh > 1:
        new_interval_list_name = new_interval_list_name + f" {time_thresh}s"

    # check if this interval already exists new_interval_list_name = f"{interval_list_name} mobile times"
    if (
        len(
            sgc.IntervalList()
            & {
                "nwb_file_name": nwb_file_name,
                "interval_list_name": new_interval_list_name,
            }
        )
        != 0
    ):
        print(f"{new_interval_list_name} already exists, skipping")
        return new_interval_list_name

    # create selection key
    trodes_s_key = {
        "nwb_file_name": nwb_file_name,
        "interval_list_name": interval_list_name,
        "trodes_pos_params_name": trodes_pos_params_name,
    }

    # use trodes selection key to load pos_df
    trodes_key = (sgp.v1.TrodesPosSelection() & trodes_s_key).fetch1("KEY")
    merge_key = (sgp.PositionOutput.merge_get_part(trodes_key)).fetch1("KEY")
    pos_df = (sgp.PositionOutput & merge_key).fetch1_dataframe()

    # get boolean df with whether each timepoint exceeds speed_thresh
    speed_bools = pd.DataFrame(
        pos_df.loc[pos_df.index, "speed"] >= speed_thresh
    ).reset_index()
    print(
        f'fraction mobile time = {len(pos_df[pos_df["speed"] >= speed_thresh]) / len(pos_df)}'
    )

    if plot_pos:
        # plot position at mobile times as a sanity check
        n = len(pos_df)
        pos_mobile_df = pos_df[pos_df["speed"] < speed_thresh]
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.scatter(
            pos_mobile_df.iloc[:n]["position_x"], pos_mobile_df.iloc[:n]["position_y"]
        )
        ax.set_xlabel("x-position [cm]", fontsize=16)
        ax.set_ylabel("y-position [cm]", fontsize=16)
        ax.set_title("Position", fontsize=20)
        plt.show()

    mobile_intervals = convert_to_times(get_bool_intervals(speed_bools['speed'].values), speed_bools['time'].values)
    mobile_durations = mobile_intervals[:, 1] - mobile_intervals[:, 0]

    # Keep intervals longer than the threshold
    mobile_intervals = mobile_intervals[mobile_durations >= time_thresh, :]

    # insert mobile intervals into IntervalList
    sgc.IntervalList.insert1(
        {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": new_interval_list_name,
            "valid_times": np.asarray(mobile_intervals),
        },
        skip_duplicates=True,
    )
    print(f"Inserted new interval: {new_interval_list_name}")

    return new_interval_list_name


def interval_list_during_trials(nwb_file_name, interval_list_name, epoch):
    '''
    For a given epoch and interval list name, create a trimmed interval that doesn't exceed the bounds of the first and last trial of the epoch (based on TrialInfo8Arm), 
    insert a trimmed interval into IntervalList(), and return the new trimmed interval list name (ends in 'during trial').

    NOTE: currently only works for TrialInfo8Arm()
    '''
    # load in the total interval you want to cut down
    valid_times = (sgc.IntervalList() & {'nwb_file_name': nwb_file_name, 'interval_list_name': interval_list_name}).fetch1('valid_times')

    # find the first trial start and last trial end times
    trial_df = (TrialInfo8Arm() & {'nwb_file_name': nwb_file_name, 'epoch': epoch}).fetch1_dataframe()
    first_trial_start = trial_df['start_time'].values[0]
    last_trial_end = trial_df['end_time'].values[-1]
    during_trials_int = np.array([[first_trial_start, last_trial_end]])

    # trim the original interval to start and end at the same starts and ends as the trials
    trimmed_valid_times = Interval(valid_times).intersect(during_trials_int).times

    # save the trimmed valid times as a new interval list
    trimmed_interval_list_name = f'{interval_list_name} (during trials)'
    print(f'inserting new interval: {trimmed_interval_list_name}')
    sgc.IntervalList().insert1({
        'nwb_file_name': nwb_file_name,
        'interval_list_name': trimmed_interval_list_name,
        'valid_times': trimmed_valid_times,
        'pipeline': 'during_trials',
    }, skip_duplicates=True)

    return trimmed_interval_list_name