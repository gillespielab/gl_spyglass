import numpy as np
import pandas as pd
import spyglass.common as sgc
from scipy.ndimage import label
from spyglass.common.common_interval import Interval

def detect_coincident_spikes(
    spike_times: list,
    spike_closeness_threshold: float = 0.00004,
    max_coincident_fraction: float = 0.33,
) -> tuple:
    '''
    Given the spike times from a particular epoch (across all relevant electrodes), find any instances of spikes that occur across at least `max_coincident_fraction`
    of the electrodes and are within `spike_closeness_threshold` of each other.

    spike_closeness threshold is in seconds
    '''
    # Concatenate all spike times
    concat_spike_times = np.concatenate(spike_times)
    # Create group IDs for each spike time
    sort_group_id = np.concatenate(
        [
            np.ones(len(spike_time), dtype=int) * i
            for i, spike_time in enumerate(spike_times)
        ]
    )
    time_bin_ind = np.concatenate(
        [np.arange(len(spike_time), dtype=int) for spike_time in spike_times]
    )

    # Sort spike times and group IDs based on the spike times
    sort_ind = np.argsort(concat_spike_times)
    sorted_spike_times = concat_spike_times[sort_ind]
    sort_group_id = sort_group_id[sort_ind]
    time_bin_ind = time_bin_ind[sort_ind]

    # Find differences and label close spikes
    is_close = np.diff(sorted_spike_times) < spike_closeness_threshold
    is_close = np.concatenate([[False], is_close])  # pad with False label at the front so that it's the right length
    labels, _ = label(is_close)  # label is labeling each different True group as a distinct event

    # Create a DataFrame for further analysis
    df = pd.DataFrame(
        {
            "labels": labels,
            "sort_group_id": sort_group_id,
            "time_bin_ind": time_bin_ind,
            "spike_times": sorted_spike_times,
        },
    )
    # Calculate the fraction of each group
    n_sort_groups = len(spike_times)
    frac = (
        df.loc[df.labels > 0].groupby("labels").sort_group_id.nunique() / n_sort_groups
    )
    frac = frac[frac > max_coincident_fraction]

    # Calculate the median spike time of each group that is excluded (based on the fraction above)
    med_coinc_spike_times = (
        df.loc[(df.labels > 0) & (df.labels.isin(frac.index))].groupby("labels").spike_times.median()
    )

    # Print number of excluded spikes
    n_coinc_spikes = len(df.loc[df.labels.isin(frac.index)])
    n_total_spikes = len(df)
    perc_coinc_spikes = (n_coinc_spikes / n_total_spikes) * 100
    print(f'{round(perc_coinc_spikes, 3)}% of the spikes were filtered out')

    # Filter out coincident spikes
    filt_df = df.loc[~df.labels.isin(frac.index)]

    filtered_spike_times = (
        filt_df.groupby("sort_group_id").spike_times.apply(np.array).tolist()
    )
    filtered_time_bin_ind = (
        filt_df.groupby("sort_group_id").time_bin_ind.apply(np.array).tolist()
    )

    return filtered_spike_times, filtered_time_bin_ind, med_coinc_spike_times

def insert_coincident_spike_interval_list(med_coinc_spike_times, removal_window_s, nwb_file_name, interval_list_name, spike_closeness_threshold, max_coincident_fraction):
    # Based on median coincident spike times, define a new interval list with these removed times

    # define the interval times
    coinc_intervals = [[coinc_spike_time - removal_window_s / 2, coinc_spike_time + removal_window_s / 2] for coinc_spike_time in med_coinc_spike_times]

    # subtract the coinc intervals from the input interval list
    orig_interval = Interval((sgc.IntervalList() & {'nwb_file_name': nwb_file_name, 'interval_list_name': interval_list_name}).fetch1('valid_times'))
    filt_interval = orig_interval.subtract(np.asarray(coinc_intervals)).times

    # interval labels
    # NOTE: including parameters in the interval list name for now but should also eventually add these to a custom spyglass table and selection table to better keep track of them somewhere
    filt_interval_list_name = f'{interval_list_name} coincident_spikes_removed_times rem_{removal_window_s} close_{spike_closeness_threshold} frac_{max_coincident_fraction}'
    pipeline = f'coincident spike detection'

    # insert interval
    sgc.IntervalList.insert1(
        {
            'nwb_file_name': nwb_file_name,
            'interval_list_name': filt_interval_list_name,
            'valid_times': filt_interval,
            'pipeline': pipeline,
        },
        skip_duplicates=True,
    )

    return filt_interval_list_name