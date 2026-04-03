import datajoint as dj
import numpy as np 
import pandas as pd
from scipy.stats import zscore
from typing import List

from spyglass.utils import SpyglassMixin, logger
import spyglass.common as sgc
from spyglass.lfp.analysis.v1 import LFPBandSelection, LFPBandV1
from spyglass.common.common_nwbfile import AnalysisNwbfile

from ripple_detection.core import gaussian_smooth, get_envelope

from gl_spyglass.custom_spyglass_tables.sleep_structure import bools_to_intervals, get_nrem_intervals

schema = dj.schema("gl_spindle")

@schema
class SpindleParameters(SpyglassMixin, dj.Lookup):
    """Parameters for spindle detection

    Attributes
    ----------
    spindle_param_name : str
        Name of the parameter set
    spindle_param_dict : dict
        Dictionary of parameters for the spindle detection, which
        may include...
        smoothing_sigma: float
            Smoothing sigma to use when smoothing the sigma power (sec)
        spindle_first_threshold : float
            First threshold for the processed sigma power
        minimum_spindle_duration : float
            Minimum duration for sigma power to exceed spindle_first_threshold 
            for spindle detection (sec)
        maximum_spindle_duration : float
            Maximum duration for sigma power to exceed spindle_first_threshold
            for spindle detection (sec)
        minimum_sigma_peak: float
            Peak threshold (second threshold) for the spindle interval to exceed
            at some point in the interval
        minimum_spindle_gap: float
            Minimum gap between neighboring spindles (sec)
            Any spindles separated by less than this value will get merged into one
    """

    definition = """
    spindle_param_name : varchar(80) # a name for this set of parameters
    ----
    spindle_param_dict : BLOB    # dictionary of parameters
    """

    def insert_default(self):
        """Insert the default parameter set"""
        default_dict = {
            "smoothing_sigma": 0.004,
            "spindle_first_threshold": 2.5,
            "minimum_spindle_duration": 0.25,
            "maximum_spindle_duration": 3,
            "minimum_sigma_peak": 10,
            "minimum_spindle_gap": 0.2,
        }
        self.insert1(
            {"spindle_param_name": "default", "spindle_param_dict": default_dict},
            skip_duplicates=True,
        )    


@schema
class SpindleLFPSelection(SpyglassMixin, dj.Manual):
    definition = """
     -> LFPBandV1
     group_name = 'cortical_spindle_elecs' : varchar(80)
     epoch: int
     """    

    class SpindleLFPElectrode(SpyglassMixin, dj.Part):
        definition = """
        -> SpindleLFPSelection
        -> LFPBandSelection.LFPBandElectrode
        """

    @staticmethod
    def validate_key(key):
        """Validates that the filter_name is a sigma filter"""
        filter_name = (LFPBandV1 & key).fetch1("filter_name")
        if "sigma" not in filter_name.lower():
            raise ValueError("Please use a sigma band filter")

    @staticmethod
    def set_lfp_electrodes(
        key,
        electrode_list=None,
        group_name="CA1",
        **kwargs,
    ):
        """Removes all electrodes for the specified nwb file and then
        adds back the electrodes in the list

        Parameters
        ----------
        key : dict
            dictionary corresponding to the LFPBand entry to use for
            ripple detection
        electrode_list : list
            list of electrodes from LFPBandSelection.LFPBandElectrode
            to be used as the ripple LFP during detection
        group_name : str, optional
            description of the electrode group, by default "CA1"
        """        
        if electrode_list is None:
            electrode_list = (
                (LFPBandSelection.LFPBandElectrode & key)
                .fetch("electrode_id")
                .tolist()
            )
        electrode_list.sort()
        try:
            electrode_keys = (
                pd.DataFrame(LFPBandSelection.LFPBandElectrode() & key)
                .set_index("electrode_id")
                .loc[np.asarray(electrode_list)]
                .reset_index()
                .loc[:, LFPBandSelection.LFPBandElectrode.primary_key]
            )
        except KeyError as err:
            logger.debug(err)
            raise KeyError(
                "Attempting to use electrode_ids that aren't in the associated"
                " LFPBand filtered dataset."
            ) from err
        electrode_keys["group_name"] = group_name
        electrode_keys["epoch"] = key['epoch']
        electrode_keys = electrode_keys.sort_values(by=["electrode_id"])
        SpindleLFPSelection.validate_key(key)
        SpindleLFPSelection().insert1(
            {**key, "group_name": group_name},
            skip_duplicates=True,
            **kwargs,
        )
        SpindleLFPSelection().SpindleLFPElectrode.insert(
            electrode_keys.to_dict(orient="records"),
            replace=True,
            **kwargs,
        )        


def merge_intervals(intervals, gap=0.4):
    """
    intervals: array-like of shape (N, 2), sorted by start time
    gap: maximum allowed gap between intervals to merge
    """
    intervals = np.asarray(intervals)
    if len(intervals) == 0:
        return intervals

    merged = [intervals[0].tolist()]

    for start, end in intervals[1:]:
        prev_start, prev_end = merged[-1]

        if start - prev_end < gap:
            # merge
            merged[-1][1] = max(prev_end, end)
        else:
            merged.append([start, end])

    return np.array(merged)


def get_mean_smooth_power(values, smoothing_sigma, sampling_frequency):
    env = get_envelope(values)
    power = env**2
    smooth_power = gaussian_smooth(
        power,
        smoothing_sigma,
        sampling_frequency,
    )
    mean_smooth_power = smooth_power.mean(axis=1)
    return mean_smooth_power


@schema
class SpindleTimes(SpyglassMixin, dj.Computed):
    definition = """
    -> SpindleLFPSelection
    -> SpindleParameters
    ---
    -> AnalysisNwbfile
    spindle_times_object_id : varchar(40)
     """

    def make(self, key):
        """Populate SpindleTimes table.

        Fetches...
            - Nwb file name from LFPBandV1
            - Parameters for spindle detection from SpindleParameters
            - Spindle LFPs from LFPBandV1
        Runs the spindle detection and inserts the results into the analysis nwb
        file, and inserts the key into the SpindleTimes table.

        """
        nwb_file_name = (LFPBandV1 & key).fetch1("nwb_file_name")
        epoch = key['epoch']

        logger.info(f"Computing spindle times for: {key}")

        # load in spindle parameters
        spindle_params = (
            SpindleParameters & {"spindle_param_name": key["spindle_param_name"]}
        ).fetch1("spindle_param_dict")

        smoothing_sigma = spindle_params['smoothing_sigma']
        spindle_thresh = spindle_params['spindle_first_threshold']
        min_spindle_duration = spindle_params['minimum_spindle_duration']
        max_spindle_duration = spindle_params['maximum_spindle_duration']
        min_sigma_peak = spindle_params['minimum_sigma_peak']
        min_spindle_gap = spindle_params['minimum_spindle_gap']
        sigma_band_sampling_rate = key['filter_sampling_rate']

        # load in sigma band data
        print('loading in sigma band...')
        sigma_band_df = (LFPBandV1 & key).fetch1_dataframe()

        # load in the nrem interval list name and intervals
        print('loading in nrem intervals...')
        _, nrem_intervals = get_nrem_intervals(nwb_file_name, epoch)

        # combine lfp band interval list with the nrem interval list to get their intersections
        sigma_band_interval_list_name = key['target_interval_list_name']
        sigma_band_interval = (sgc.IntervalList() & {'nwb_file_name': nwb_file_name, 'interval_list_name': sigma_band_interval_list_name}).fetch_interval()
        valid_nrem_intervals = sigma_band_interval.intersect(nrem_intervals)

        # isolate the sigma band df to just these valid nrem intervals
        valid_nrem_indices = valid_nrem_intervals.contains(timestamps=sigma_band_df.index.values, as_indices=True)
        nrem_sigma_band_df = sigma_band_df.iloc[valid_nrem_indices]

        print('processing sigma band into zscored, smoothed, average power...')
        # zscore across all intervals in this epoch
        zscore_nrem_sigma_band_df = nrem_sigma_band_df.apply(zscore)

        # do the power and smoothing calculations within each interval to avoid weird edge effects
        interval_dfs = []
        for interval in valid_nrem_intervals:
            interval_mask = (nrem_sigma_band_df.index >= interval[0]) & (nrem_sigma_band_df.index < interval[1])
            interval_time = zscore_nrem_sigma_band_df.loc[interval_mask].index.values
            interval_sigma_power = get_mean_smooth_power(
                zscore_nrem_sigma_band_df.loc[interval_mask].values,
                smoothing_sigma=smoothing_sigma,
                sampling_frequency=sigma_band_sampling_rate,
            )

            interval_dfs.append(pd.DataFrame({'time': interval_time, 'sigma_power': interval_sigma_power}))

        sigma_power_df = pd.concat(interval_dfs).reset_index(drop=True)

        print('detecting spindles...')
        # filter by initial spindle thresh
        spindle_mask = sigma_power_df['sigma_power'].values > spindle_thresh
        spindle_intervals = bools_to_intervals(spindle_mask, sigma_power_df['time'].values)

        # filter by minimum and maximum spindle duration
        spindle_durations = np.squeeze(np.diff(spindle_intervals))
        spindle_intervals = spindle_intervals[(spindle_durations > min_spindle_duration) & (spindle_durations < max_spindle_duration), :]  # exceeds 2.5 for more than 0.25 seconds and less than 3 seconds

        # filter by minimum sigma peak power
        max_sigma_powers = np.asarray([sigma_power_df.loc[(sigma_power_df['time'] >= start_time) & (sigma_power_df['time'] < end_time), 'sigma_power'].max() for start_time, end_time in spindle_intervals])
        spindle_intervals = spindle_intervals[max_sigma_powers > min_sigma_peak] # & peaks at >10

        # merge any spindles that are separated by less than the minimum spindle gap
        spindle_intervals = merge_intervals(spindle_intervals, gap=min_spindle_gap)

        # get final spindle properties and compile into a dataframe
        final_spindle_durations = np.squeeze(np.diff(spindle_intervals))
        final_spindle_median_amplitudes = [np.median(sigma_power_df.loc[(sigma_power_df['time'] >= interval[0]) & (sigma_power_df['time'] < interval[1]), 'sigma_power'].values) for interval in spindle_intervals]
        final_spindle_mean_amplitudes = [np.mean(sigma_power_df.loc[(sigma_power_df['time'] >= interval[0]) & (sigma_power_df['time'] < interval[1]), 'sigma_power'].values) for interval in spindle_intervals]
        final_spindle_max_amplitudes = [np.max(sigma_power_df.loc[(sigma_power_df['time'] >= interval[0]) & (sigma_power_df['time'] < interval[1]), 'sigma_power'].values) for interval in spindle_intervals]

        spindle_info_df = pd.DataFrame({
            'start_time': spindle_intervals[:, 0], 
            'end_time': spindle_intervals[:, 1], 
            'duration': final_spindle_durations, 
            'median_amplitude': final_spindle_median_amplitudes,
            'mean_amplitude': final_spindle_mean_amplitudes,
            'max_amplitude': final_spindle_max_amplitudes,
            })

        # Insert into analysis nwb file
        nwb_analysis_file = AnalysisNwbfile()
        key["analysis_file_name"] = nwb_analysis_file.create(nwb_file_name)
        key["spindle_times_object_id"] = nwb_analysis_file.add_nwb_object(
            analysis_file_name=key["analysis_file_name"],
            nwb_object=spindle_info_df,
        )
        nwb_analysis_file.add(
            nwb_file_name=nwb_file_name,
            analysis_file_name=key["analysis_file_name"],
        )

        self.insert1(key)
        print('spindle detection complete')

    def fetch1_dataframe(self) -> pd.DataFrame:
        """Convenience function for returning the marks in a readable format"""
        _ = self.ensure_single_entry()
        return self.fetch_dataframe()[0]

    def fetch_dataframe(self) -> List[pd.DataFrame]:
        """Convenience function for returning all marks in a readable format"""
        return [data["spindle_times"] for data in self.fetch_nwb()]


