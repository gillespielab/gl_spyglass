import datajoint as dj
import numpy as np
import pandas as pd
from scipy import signal
from tqdm import tqdm
from typing import List

from spyglass.utils import SpyglassMixin, logger
import spyglass.common as sgc
import spyglass.lfp as lfp
from spyglass.lfp.v1.lfp import LFPV1
from spyglass.lfp import LFPOutput
from spyglass.common.common_nwbfile import AnalysisNwbfile
from spyglass.lfp.analysis.v1 import LFPBandSelection, LFPBandV1
from spyglass.lfp.v1 import LFPArtifactRemovedIntervalList
from spyglass.common import IntervalList

from ripple_detection.core import gaussian_smooth, get_envelope

from gl_spyglass.utils.interval_functions import insert_immobile_times_interval

schema = dj.schema("gl_sleep_structure")

@schema
class ThetaDeltaRatioSelection(SpyglassMixin, dj.Manual):
    definition = """
    theta_delta_nwb_file_name : varchar(80)
    -> LFPOutput.proj(theta_delta_lfp_merge_id='merge_id')
    interval_list_name : varchar(80)
    smoothing_sigma : varchar(16)
    referenced: int
    """

    class LFPBandV1(SpyglassMixin, dj.Part):
       definition = """
       -> master
       -> LFPBandV1
       """
    
    def add_theta_delta(self, nwb_file_name, lfp_merge_id, lfp_sampling_rate, interval_list_name, lfp_band_sampling_rate, smoothing_sigma, referenced=True):

        if referenced:
            lfp_band_filter_names = ['Theta 5-11 Hz', 'Delta 0.5-4 Hz']
        else:
            lfp_band_filter_names = ['Theta 5-11 Hz (unreferenced)', 'Delta 0.5-4 Hz (unreferenced)']

        part_keys = []
        for lfp_band_filter_name in lfp_band_filter_names:
            lfp_band_s_key = {
                'lfp_merge_id': lfp_merge_id,
                'filter_name': lfp_band_filter_name,
                'filter_sampling_rate': lfp_sampling_rate,
                'nwb_file_name': nwb_file_name,
                'target_interval_list_name': interval_list_name,
                'lfp_band_sampling_rate': lfp_band_sampling_rate,
            }
            lfp_band_s_key = (LFPBandV1() & lfp_band_s_key).fetch1('KEY')
            part_keys.append(lfp_band_s_key)
        
        master_key = dict(
            theta_delta_nwb_file_name=nwb_file_name,
            theta_delta_lfp_merge_id=lfp_merge_id,
            interval_list_name=interval_list_name,
            smoothing_sigma=smoothing_sigma,
            referenced=int(referenced),
        )

        self.insert1(master_key)
        self.LFPBandV1().insert(
            [{**k, 'theta_delta_nwb_file_name': nwb_file_name, 'theta_delta_lfp_merge_id': lfp_merge_id, 'interval_list_name': interval_list_name, 'smoothing_sigma': smoothing_sigma, 'referenced': int(referenced)} for k in part_keys]
        )


@schema
class ThetaDeltaRatio(SpyglassMixin, dj.Computed):
    definition = """
    -> ThetaDeltaRatioSelection
    ---
    -> AnalysisNwbfile
    theta_delta_df_object_id: varchar(40)
    """

    def make(self, key):
        # key includes nwb_file_name, lfp_merge_id, interval_list_name, and smoothing sigma
        nwb_file_name = key['theta_delta_nwb_file_name']
        smoothing_sigma = float(key['smoothing_sigma'])
        referenced = key['referenced']

        # add processed theta, delta, and theta/delta ratio to a dataframe
        theta_delta_df = None
        if referenced:
            lfp_band_filter_names = ['Theta 5-11 Hz', 'Delta 0.5-4 Hz']
        else:
            lfp_band_filter_names = ['Theta 5-11 Hz (unreferenced)', 'Delta 0.5-4 Hz (unreferenced)']
        lfp_band_labels = ['theta', 'delta']
        for lfp_band_filter_name, lfp_band_label in zip(lfp_band_filter_names, lfp_band_labels):
            print(f'Processing {lfp_band_filter_name} data...')
            lfp_band_s_key = (ThetaDeltaRatioSelection().LFPBandV1() & key & {'filter_name': lfp_band_filter_name}).fetch1('KEY')
            lfp_band_s_key.pop('theta_delta_nwb_file_name')
            lfp_band_s_key.pop('theta_delta_lfp_merge_id')
            lfp_band_s_key.pop('interval_list_name')
            lfp_band_s_key.pop('smoothing_sigma')
            lfp_band_sampling_rate = (LFPBandV1() & lfp_band_s_key).fetch1('filter_sampling_rate')
            lfp_band_df = (LFPBandV1() & lfp_band_s_key).fetch1_dataframe()
            band_env = get_envelope(lfp_band_df.values)
            band_power = band_env**2
            band_power_smooth = gaussian_smooth(
                band_power,
                sigma=smoothing_sigma,
                sampling_frequency=lfp_band_sampling_rate,
            )
            mean_band_power = np.mean(band_power_smooth, axis=1)
            time = lfp_band_df.index.values
            if theta_delta_df is None:
                theta_delta_df = pd.DataFrame({'time': time, lfp_band_label: mean_band_power})
            else:
                theta_delta_df[lfp_band_label] = mean_band_power
        
        theta_delta_df['theta_delta_ratio'] = theta_delta_df['theta'].values / theta_delta_df['delta'].values

        # Insert into analysis nwb file
        nwb_analysis_file = AnalysisNwbfile()
        key["analysis_file_name"] = nwb_analysis_file.create(nwb_file_name)
        key["theta_delta_df_object_id"] = nwb_analysis_file.add_nwb_object(
            analysis_file_name=key['analysis_file_name'],
            nwb_object=theta_delta_df,
        )
        nwb_analysis_file.add(
            nwb_file_name=nwb_file_name,
            analysis_file_name=key['analysis_file_name'],
        )
        
        self.insert1(key)
    
    def fetch1_dataframe(self) -> pd.DataFrame:
        """Convenience function for returning the marks in a readable format"""
        _ = self.ensure_single_entry()
        return self.fetch_dataframe()[0]

    def fetch_dataframe(self) -> List[pd.DataFrame]:
        """Convenience function for returning all marks in a readable format"""
        return [data["theta_delta_df"] for data in self.fetch_nwb()]


# Other sleep structure - related functions:

def bools_to_intervals(bool_mask, time):
    # Find rising and falling edges
    diff = np.diff(bool_mask.astype(int))

    starts = np.where(diff == 1)[0] + 1
    ends   = np.where(diff == -1)[0] + 1

    # Handle edge cases
    if bool_mask[0]:
        starts = np.r_[0, starts]
    if bool_mask[-1]:
        ends = np.r_[ends, len(bool_mask)]

    start_times = time[starts]
    if ends[-1] == len(time):  # catch edge case where the last interval ends at the very last time
        ends[-1] = len(time) - 1
    end_times = time[ends]

    intervals = np.asarray(list(zip(start_times, end_times))) 
    
    return intervals


def get_nrem_intervals(nwb_file_name, epoch):

    interval_list_name = (
        sgc.TaskEpoch() & {"nwb_file_name": nwb_file_name, "epoch": epoch}
    ).fetch1("interval_list_name")
    pos_interval_list_name = (
        sgc.IntervalList()
        & {"nwb_file_name": nwb_file_name, "pipeline": "position"}
    ).fetch("interval_list_name")[epoch - 1]
    
    # set nrem interval list name
    nrem_interval_list_name = pos_interval_list_name.replace(
        "valid times", "nrem times"
    )

    # if it already exists, just return the intervals
    if (
        len(
            sgc.IntervalList()
            & {
                "nwb_file_name": nwb_file_name,
                "interval_list_name": nrem_interval_list_name,
            }
        )
        != 0
    ):
        nrem_intervals = (sgc.IntervalList() & {'nwb_file_name': nwb_file_name, 'interval_list_name': nrem_interval_list_name}).fetch1('valid_times')
        return nrem_interval_list_name, nrem_intervals

    # if not, calculate the intervals and inser them into sgc.IntervalList
    referenced = False

    lfp_electrode_group_name = 'good_single_elecs'
    lfp_filter_name = 'LFP 0-400 Hz'
    lfp_s_key = {
        'nwb_file_name': nwb_file_name,
        'lfp_electrode_group_name': lfp_electrode_group_name,
        'target_interval_list_name': interval_list_name,
        'filter_name': lfp_filter_name,
        'filter_sampling_rate': 30_000,  # sampling rate of the data (Hz)
        'target_sampling_rate': 1_000,  # sampling rate of the lfp output (Hz)
    }
    lfp_merge_id = (LFPOutput.LFPV1() & lfp_s_key).fetch1('merge_id')

    # select artifact detection parameters
    artifact_params_name = 'mad_7_0.66_thresh_200ms'
    lfp_filter_name = 'LFP 0-400 Hz'
    lfp_artifact_s_key = {
        'nwb_file_name': nwb_file_name,
        'lfp_electrode_group_name': lfp_electrode_group_name,
        'target_interval_list_name': interval_list_name,
        'filter_name': lfp_filter_name,
        'filter_sampling_rate': 30_000,  # I'm pretty sure this is the sampling rate for the original data but not sure
        'artifact_params_name': artifact_params_name,
    }
    artifact_removed_interval_list_name = (LFPArtifactRemovedIntervalList() & lfp_artifact_s_key).fetch1('artifact_removed_interval_list_name')

    smoothing_sigma = 0.004
    theta_delta_key = {
        'theta_delta_nwb_file_name': nwb_file_name,
        'theta_delta_lfp_merge_id': lfp_merge_id,
        'interval_list_name': artifact_removed_interval_list_name,
        'smoothing_sigma': smoothing_sigma,
        'referenced': int(referenced),
    }

    epoch_df = (ThetaDeltaRatio & theta_delta_key).fetch1_dataframe()

    # smooth the theta delta ratio
    smoothing_sigma=1
    theta_delta_ratio_smooth = gaussian_smooth(epoch_df['theta_delta_ratio'].values, smoothing_sigma, sampling_frequency=1000)
    epoch_df['theta_delta_ratio_smooth'] = theta_delta_ratio_smooth

    # import immobile time intervals and set immobile times
    trodes_pos_params_name = 'default'
    immobile_30s_interval_list_name = insert_immobile_times_interval(nwb_file_name, pos_interval_list_name, trodes_pos_params_name, speed_thresh=4, time_thresh=30, plot_pos=False)
    immobile_intervals = (sgc.IntervalList() & {'nwb_file_name': nwb_file_name, 'interval_list_name': immobile_30s_interval_list_name}).fetch1('valid_times')

    epoch_df['immobile'] = False
    for interval in immobile_intervals:
        interval_mask = (epoch_df['time'] >= interval[0]) & (epoch_df['time'] < interval[1])
        epoch_df.loc[interval_mask, 'immobile'] = True

    # find nrem times (based on immobility and nrem threshold)
    nrem_thresh = 5
    nrem_mask = (epoch_df['immobile']) & (epoch_df['theta_delta_ratio_smooth'] < nrem_thresh)

    nrem_intervals = bools_to_intervals(nrem_mask.values, epoch_df['time'].values)

    # cut out any intervals that are less than 1s long
    min_nrem_duration = 1  # seconds
    nrem_intervals = np.asarray([interval for interval in nrem_intervals if (interval[1] - interval[0]) >= min_nrem_duration])

    # insert nrem intervals into IntervalList
    sgc.IntervalList.insert1(
        {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": nrem_interval_list_name,
            "valid_times": np.asarray(nrem_intervals),
        },
        skip_duplicates=True,
    )
    print(f"Inserted new interval: {nrem_interval_list_name}")

    return nrem_interval_list_name, nrem_intervals
