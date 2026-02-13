from typing import List

import datajoint as dj
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sortingview.views as vv
from matplotlib.axes import Axes
from ripple_detection import Karlsson_ripple_detector, Kay_ripple_detector, Shvartsman_ripple_detector
from ripple_detection.core import gaussian_smooth, get_envelope
from scipy.stats import zscore, median_abs_deviation

from spyglass.common.common_interval import IntervalList
import spyglass.common as sgc
import spyglass.lfp as lfp
from spyglass.common.common_session import Session
from spyglass.common.common_nwbfile import AnalysisNwbfile
from spyglass.lfp.analysis.v1.lfp_band import LFPBandSelection, LFPBandV1
from spyglass.lfp.lfp_merge import LFPOutput
from spyglass.position import PositionOutput
from spyglass.utils import SpyglassMixin, logger
from spyglass.utils.nwb_helper_fn import get_electrode_indices

from gabby_analysis_gillespie.utils.interval_functions import *

# NOTE: these grouped ripple tables are relevant if you want to find a ripple threshold based on the standard deviations across multiple epochs within a day (rather than within each epoch)
# that's particularly relevant for sleep epochs where you don't want the standard deviation to be too high, otherwise you'll miss a lot of ripples that you would've normally counted in a run epoch

schema = dj.schema("gl_dev_ripple_detection")

RIPPLE_DETECTION_ALGORITHMS = {
    "Kay_ripple_detector": Kay_ripple_detector,
    "Karlsson_ripple_detector": Karlsson_ripple_detector,
    "Shvartsman_ripple_detector": Shvartsman_ripple_detector,
}

# Do we need this anymore given that LFPBand is no longer a merge table?
UPSTREAM_ACCEPTED_VERSIONS = ["LFPBandV1"]


@schema
class LFPBandGroup(SpyglassMixin, dj.Manual):
    definition = """
    band_group_name : varchar(80) # nwb file name
    band_group_filter_name=None : varchar(80) # filter name (i.e. 'Ripple 100-250 Hz')
    ---
    elec_baselines=NULL  : blob
    elec_deviations=NULL : blob
    """

    class LFPBandV1(SpyglassMixin, dj.Part):
        definition = """
        -> master
        -> LFPBandV1
        """

    def insert_from_date(self, nwb_file_name, lfp_band_filter_name):
        lfpband_s_key = (sgc.Session() & {'nwb_file_name': nwb_file_name}).fetch('KEY')[0]
        lfpband_s_key['filter_name'] = lfp_band_filter_name
        lfpband_keys = (LFPBandV1() & lfpband_s_key).fetch('KEY')

        baselines = None
        deviations = None
        if len(lfpband_keys) > 1:
            baselines, deviations = compute_mobile_median_mad_across_day(nwb_file_name)

        self.insert1(dict(band_group_name=nwb_file_name, band_group_filter_name=lfp_band_filter_name, elec_baselines=baselines, elec_deviations=deviations))
        self.LFPBandV1.insert(
            [{**k, "band_group_name": nwb_file_name, "band_group_filter_name": lfp_band_filter_name} for k in lfpband_keys]
        )


@schema
class RippleLFPSelection(SpyglassMixin, dj.Manual):
    definition = """
     -> LFPBandGroup
     group_name = 'CA1' : varchar(80)
     """
    
    class RippleLFPElectrode(SpyglassMixin, dj.Part):
        definition = """
        -> RippleLFPSelection
        -> LFPBandSelection.LFPBandElectrode
        """

    @staticmethod
    def validate_key(key):
        """Validates that the filter_name is a ripple filter"""
        filter_name = (LFPBandGroup & key).fetch1("band_group_filter_name")
        if "ripple" not in filter_name.lower():
            raise ValueError("Please use a ripple filter")

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
        lfp_band_indiv_keys = (LFPBandGroup.LFPBandV1() & key).fetch("KEY")

        for lfp_band_indiv_key in lfp_band_indiv_keys:
        
            if electrode_list is None:
                electrode_list = (
                    (LFPBandSelection.LFPBandElectrode & lfp_band_indiv_key)
                    .fetch("electrode_id")
                    .tolist()
                )
            electrode_list.sort()
            try:
                electrode_keys = (
                    pd.DataFrame(LFPBandSelection.LFPBandElectrode() & lfp_band_indiv_key)
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
            electrode_keys['band_group_name'] = key['band_group_name']
            electrode_keys['band_group_filter_name'] = key['band_group_filter_name']
            electrode_keys["group_name"] = group_name
            electrode_keys = electrode_keys.sort_values(by=["electrode_id"])
            RippleLFPSelection.validate_key(key)
            RippleLFPSelection().insert1(
                {**key, "group_name": group_name},
                skip_duplicates=True,
                **kwargs,
            )
            RippleLFPSelection().RippleLFPElectrode.insert(
                electrode_keys.to_dict(orient="records"),
                replace=True,
                **kwargs,
            )


@schema
class RippleParameters(SpyglassMixin, dj.Lookup):
    """Parameters for ripple detection

    Attributes
    ----------
    ripple_param_name : str
        Name of the parameter set
    ripple_param_dict : dict
        Dictionary of parameters for ripple detection, including...
        speed_name : str
            Name of the speed column in the PositionOutput table
        ripple_detection_algorithm : str
            Name of the ripple detection algorithm to use
        ignore_tasks : list, optional
            List of task_names to ignore during ripple detection (e.g., 'sleep')
        ripple_detection_params : dict
            Dictionary of parameters for the ripple detection algorithm, which
            may include...
            speed_threshold : float
                Speed threshold for ripple detection (cm/s)
            minimum_duration : float
                Minimum duration for ripple detection (sec)
            zscore_threshold : float
                Z-score threshold for ripple detection (std)
            smoothing_sigma : float
                Smoothing sigma for ripple detection (sec)
            close_ripple_threshold : float
                Close ripple threshold for ripple detection (sec)
    """

    definition = """
    ripple_param_name : varchar(80) # a name for this set of parameters
    ----
    ripple_param_dict : BLOB    # dictionary of parameters
    """

    def insert_default(self):
        """Insert the default parameter set"""
        default_dict = {
            "speed_name": "head_speed",
            "ripple_detection_algorithm": "Kay_ripple_detector",
            "ripple_detection_params": dict(
                speed_threshold=4.0,  # cm/s
                minimum_duration=0.015,  # sec
                zscore_threshold=2.0,  # std
                smoothing_sigma=0.004,  # sec
                close_ripple_threshold=0.0,  # sec
            ),
        }
        self.insert1(
            {"ripple_param_name": "default", "ripple_param_dict": default_dict},
            skip_duplicates=True,
        )
        default_dict_trodes = {
            "speed_name": "speed",
            "ripple_detection_algorithm": "Kay_ripple_detector",
            "ripple_detection_params": dict(
                speed_threshold=4.0,  # cm/s
                minimum_duration=0.015,  # sec
                zscore_threshold=2.0,  # std
                smoothing_sigma=0.004,  # sec
                close_ripple_threshold=0.0,  # sec
            ),
        }
        self.insert1(
            {
                "ripple_param_name": "default_trodes",
                "ripple_param_dict": default_dict_trodes,
            },
            skip_duplicates=True,
        )


@schema
class RippleTimesGroup(SpyglassMixin, dj.Computed):
    definition = """
    -> RippleLFPSelection
    -> RippleParameters
    """

    class RippleTimes(SpyglassMixin, dj.Part):
        definition = """
        -> master
        -> LFPBandV1
        -> PositionOutput.proj(pos_merge_id='merge_id')
        ---
        -> AnalysisNwbfile
        ripple_times_object_id : varchar(40)
        """
        
        def fetch1_dataframe(self) -> pd.DataFrame:
            """Convenience function for returning the marks in a readable format"""
            _ = self.ensure_single_entry()
            return self.fetch_dataframe()[0]

        def fetch_dataframe(self) -> List[pd.DataFrame]:
            """Convenience function for returning all marks in a readable format"""
            return [data["ripple_times"] for data in self.fetch_nwb()]

    def make(self, key):
        """Populate RippleTimesV1 table.

        Fetches...
            - Nwb file name from LFPBandV1
            - Parameters for ripple detection from RippleParameters
            - Ripple LFPs and position info from PositionOutput and LFPBandV1
        Runs the specified ripple detection algorithm (Karlsson or Kay from
        ripple_detection package), inserts the results into the analysis nwb
        file, and inserts the key into the RippleTimesV1 table.

        """
        # fetching nwb_file_name from the first LFPBandV1 part in case the band_group_name convention changes later
        # should somewhere have a check that the nwb file name for everything in the group is the same
        band_group_key = {'band_group_name': key['band_group_name'], 'band_group_filter_name': key['band_group_filter_name']}
        nwb_file_name = (LFPBandGroup().LFPBandV1() & band_group_key).fetch('nwb_file_name')[0]

        logger.info(f"Computing ripple times for: {key}")
        ripple_params = (
            RippleParameters & {"ripple_param_name": key["ripple_param_name"]}
        ).fetch1("ripple_param_dict")

        ripple_detection_algorithm = ripple_params["ripple_detection_algorithm"]
        ripple_detection_params = ripple_params["ripple_detection_params"]

        if ripple_detection_algorithm == "Shvartsman_ripple_detector":
            extra_params = (LFPBandGroup & band_group_key).fetch(
                "elec_baselines", "elec_deviations", as_dict=True, limit=1,
            )[0]
            ripple_detection_params.update(extra_params)

        # find the relevant pos_merge_ids
        lfp_merge_ids = (LFPBandGroup().LFPBandV1() & key).fetch('lfp_merge_id')

        interval_lists = [(LFPOutput().LFPV1() & {'merge_id': lfp_merge_id}).fetch1('target_interval_list_name') for lfp_merge_id in lfp_merge_ids]
        pos_interval_list_names = [(
            sgc.IntervalList()
            & {"nwb_file_name": nwb_file_name, "pipeline": "position"}
        ).fetch("interval_list_name")[int(interval_list[:2]) - 1] for interval_list in interval_lists]

        pos_merge_ids = []
        for pos_interval_list_name in pos_interval_list_names:
            trodes_pos_params_name = 'default'
            pos_s_key = {
                "nwb_file_name": nwb_file_name,
                "interval_list_name": pos_interval_list_name,
                "trodes_pos_params_name": trodes_pos_params_name,
            }
            pos_key = (sgp.v1.TrodesPosSelection() & pos_s_key).fetch1("KEY")
            pos_merge_key = (sgp.PositionOutput.merge_get_part(pos_key)).fetch1("KEY")
            pos_merge_id = pos_merge_key['merge_id']
            pos_merge_ids.append(pos_merge_id)

        # run the ripple detection for each LFPBandV1() part in the LFPBandGroup table
        lfp_band_indiv_keys = (LFPBandGroup().LFPBandV1() & key).fetch('KEY')

        for i, lfp_band_indiv_key in tqdm(enumerate(lfp_band_indiv_keys)):

            # add ripple_param_name and group_name so they're accessible
            lfp_band_indiv_key['ripple_param_name'] = key['ripple_param_name']
            lfp_band_indiv_key['group_name'] = key['group_name']

            # add corresponding pos_merge_id
            lfp_band_indiv_key['pos_merge_id'] = pos_merge_ids[i]

            (
                speed,
                interval_ripple_lfps,
                sampling_frequency,
            ) = self.get_ripple_lfps_and_position_info(lfp_band_indiv_key)
            ripple_times = RIPPLE_DETECTION_ALGORITHMS[ripple_detection_algorithm](
                time=np.asarray(interval_ripple_lfps.index),
                filtered_lfps=np.asarray(interval_ripple_lfps),
                speed=np.asarray(speed),
                sampling_frequency=sampling_frequency,
                **ripple_detection_params,
            )
            # Insert into analysis nwb file
            nwb_analysis_file = AnalysisNwbfile()
            lfp_band_indiv_key["analysis_file_name"] = nwb_analysis_file.create(nwb_file_name)
            lfp_band_indiv_key["ripple_times_object_id"] = nwb_analysis_file.add_nwb_object(
                analysis_file_name=lfp_band_indiv_key["analysis_file_name"],
                nwb_object=ripple_times,
            )
            nwb_analysis_file.add(
                nwb_file_name=nwb_file_name,
                analysis_file_name=lfp_band_indiv_key["analysis_file_name"],
            )

            self.insert1(key, ignore_extra_fields=True, skip_duplicates=True)
            self.RippleTimes().insert1(lfp_band_indiv_key)

    def fetch1_dataframe(self) -> pd.DataFrame:
        """Convenience function for returning the marks in a readable format"""
        _ = self.ensure_single_entry()
        return self.fetch_dataframe()[0]

    def fetch_dataframe(self) -> List[pd.DataFrame]:
        """Convenience function for returning all marks in a readable format"""
        return [data["ripple_times"] for data in self.fetch_nwb()]

    @staticmethod
    def get_ripple_lfps_and_position_info(key) -> tuple:
        """Return the ripple LFPs and position info for the specified key.

        Fetches...
            - Ripple parameters from RippleParameters
            - Electrode keys from RippleLFPSelection
            - LFP data from LFPBandV1
            - Position data from PositionOutput merge table
        Interpolates the position data to the LFP timestamps.
        """
        # TODO: Pass parameters from make func, instead of fetching again
        ripple_params = (
            RippleParameters & {"ripple_param_name": key["ripple_param_name"]}
        ).fetch1("ripple_param_dict")

        speed_name = ripple_params["speed_name"]

        electrode_keys = (RippleLFPSelection.RippleLFPElectrode() & key).fetch(
            "electrode_id"
        )

        # warn/validate that there is only one wire per electrode
        ripple_lfp_nwb = (LFPBandV1 & key).fetch_nwb()[0]
        ripple_lfp_electrodes = ripple_lfp_nwb["lfp_band"].electrodes.data[:]
        elec_mask = np.zeros_like(ripple_lfp_electrodes, dtype=bool)
        valid_elecs = [
            elec for elec in electrode_keys if elec in ripple_lfp_electrodes
        ]
        lfp_indexed_elec_ids = get_electrode_indices(
            ripple_lfp_nwb["lfp_band"], valid_elecs
        )
        elec_mask[lfp_indexed_elec_ids] = True
        ripple_lfp = pd.DataFrame(
            ripple_lfp_nwb["lfp_band"].data,
            index=pd.Index(ripple_lfp_nwb["lfp_band"].timestamps, name="time"),
        ).loc[:, elec_mask]
        sampling_frequency = ripple_lfp_nwb["lfp_band_sampling_rate"]

        position_valid_times = (
            IntervalList
            & {
                "nwb_file_name": key["nwb_file_name"],
                "interval_list_name": key["target_interval_list_name"],
            }
        ).fetch_interval()
        position_info = (
            PositionOutput() & {"merge_id": key["pos_merge_id"]}
        ).fetch1_dataframe()

        # restrict valid times to position time
        valid_times_interval = np.array(
            [position_info.index[0], position_info.index[-1]]
        )
        position_valid_times = position_valid_times.intersect(
            valid_times_interval
        ).times

        position_info = pd.concat(
            [
                position_info.loc[slice(valid_time[0], valid_time[1])]
                for valid_time in position_valid_times
            ],
            axis=0,
        )
        interval_ripple_lfps = pd.concat(
            [
                ripple_lfp.loc[slice(valid_time[0], valid_time[1])]
                for valid_time in position_valid_times
            ],
            axis=0,
        )

        position_info = interpolate_to_new_time(
            position_info, interval_ripple_lfps.index
        )

        return (
            position_info[speed_name],
            interval_ripple_lfps,
            sampling_frequency,
        )

    @staticmethod
    def get_Kay_ripple_consensus_trace(
        ripple_filtered_lfps, sampling_frequency, smoothing_sigma: float = 0.004
    ) -> pd.DataFrame:
        """Calculate the consensus trace for the ripple filtered LFPs"""
        ripple_consensus_trace = np.full_like(ripple_filtered_lfps, np.nan)
        not_null = np.all(pd.notnull(ripple_filtered_lfps), axis=1)

        ripple_consensus_trace[not_null] = get_envelope(
            np.asarray(ripple_filtered_lfps)[not_null]
        )
        ripple_consensus_trace = np.sum(ripple_consensus_trace**2, axis=1)
        ripple_consensus_trace[not_null] = gaussian_smooth(
            ripple_consensus_trace[not_null],
            smoothing_sigma,
            sampling_frequency,
        )
        return pd.DataFrame(
            np.sqrt(ripple_consensus_trace), index=ripple_filtered_lfps.index
        )

    @staticmethod
    def plot_ripple_consensus_trace(
        ripple_consensus_trace,
        ripple_times,
        ripple_label=1,
        offset=0.100,
        relative=True,
        ax=None,
    ):
        """Plot the consensus trace for a ripple event"""
        ripple_start = ripple_times.loc[ripple_label].start_time
        ripple_end = ripple_times.loc[ripple_label].end_time
        time_slice = slice(ripple_start - offset, ripple_end + offset)

        start_offset = ripple_start if relative else 0
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(12, 1))
        ax.plot(
            ripple_consensus_trace.loc[time_slice].index - start_offset,
            ripple_consensus_trace.loc[time_slice],
        )
        ax.axvspan(
            ripple_start - start_offset,
            ripple_end - start_offset,
            zorder=-1,
            alpha=0.5,
            color="lightgrey",
        )
        ax.set_xlabel("Time [s]")
        ax.set_xlim(
            (time_slice.start - start_offset, time_slice.stop - start_offset)
        )

    @staticmethod
    def plot_ripple(
        lfps,
        ripple_times,
        ripple_label: int = 1,
        offset: float = 0.100,
        relative: bool = True,
        ax: Axes = None,
    ):
        """Plot the LFPs for a ripple event"""
        lfp_labels = lfps.columns
        n_lfps = len(lfp_labels)
        ripple_start = ripple_times.loc[ripple_label].start_time
        ripple_end = ripple_times.loc[ripple_label].end_time
        time_slice = slice(ripple_start - offset, ripple_end + offset)
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(12, n_lfps * 0.20))

        start_offset = ripple_start if relative else 0

        for lfp_ind, lfp_label in enumerate(lfp_labels):
            lfp = lfps.loc[time_slice, lfp_label]
            ax.plot(
                lfp.index - start_offset,
                lfp_ind + (lfp - lfp.mean()) / (lfp.max() - lfp.min()),
                color="black",
            )

        ax.axvspan(
            ripple_start - start_offset,
            ripple_end - start_offset,
            zorder=-1,
            alpha=0.5,
            color="lightgrey",
        )
        ax.set_ylim((-1, n_lfps))
        ax.set_xlim(
            (time_slice.start - start_offset, time_slice.stop - start_offset)
        )
        ax.set_ylabel("LFPs")
        ax.set_xlabel("Time [s]")

    def create_figurl(
        self,
        zscore_ripple=False,
        ripple_times_color="red",
        consensus_color="black",
        speed_color="black",
        view_height=800,
        use_ripple_filtered_lfps=False,
        lfp_offset=1,
        lfp_channel_ind=None,
    ):
        """Generate a FigURL for the ripple detection"""
        ripple_times = self.fetch1_dataframe()

        def _add_ripple_times(
            view,
            ripple_times=ripple_times,
            ripple_times_color=ripple_times_color,
        ):
            return view.add_interval_series(
                name="Ripple Events",
                t_start=ripple_times.start_time.to_numpy(),
                t_end=ripple_times.end_time.to_numpy(),
                color=ripple_times_color,
            )

        key = self.fetch1("KEY")
        (
            speed,
            ripple_filtered_lfps,
            sampling_frequency,
        ) = self.get_ripple_lfps_and_position_info(key)
        ripple_consensus_trace = self.get_Kay_ripple_consensus_trace(
            ripple_filtered_lfps, sampling_frequency
        )

        if zscore_ripple:
            ripple_consensus_trace = zscore(ripple_consensus_trace)

        consensus_view = _add_ripple_times(vv.TimeseriesGraph())
        consensus_name = (
            "Z-Scored Consensus Trace" if zscore_ripple else "Consensus Trace"
        )
        consensus_view.add_line_series(
            name=consensus_name,
            t=np.asarray(ripple_consensus_trace.index).squeeze(),
            y=np.asarray(ripple_consensus_trace, dtype=np.float32).squeeze(),
            color=consensus_color,
            width=1,
        )
        if zscore_ripple:
            ripple_params = (
                RippleParameters
                & {"ripple_param_name": key["ripple_param_name"]}
            ).fetch1("ripple_param_dict")

            zscore_threshold = ripple_params["ripple_detection_params"].get(
                "zscore_threshold"
            )
            if zscore_threshold is not None:
                consensus_view.add_line_series(
                    name="Z-Score Threshold",
                    t=np.asarray(ripple_consensus_trace.index).squeeze(),
                    y=np.ones_like(
                        ripple_consensus_trace, dtype=np.float32
                    ).squeeze()
                    * zscore_threshold,
                    color=ripple_times_color,
                    width=1,
                )

        if use_ripple_filtered_lfps:
            interval_ripple_lfps = ripple_filtered_lfps
        else:
            lfp_merge_id = (LFPBandSelection & key).fetch1("lfp_merge_id")
            lfp_df = (LFPOutput & {"merge_id": lfp_merge_id}).fetch1_dataframe()
            interval_ripple_lfps = lfp_df.loc[speed.index[0] : speed.index[-1]]
        if lfp_channel_ind is not None:
            if lfp_channel_ind.max() >= interval_ripple_lfps.shape[1]:
                raise ValueError(
                    "lfp_channel_ind is out of range for the number of LFPs"
                )
            interval_ripple_lfps = interval_ripple_lfps.iloc[:, lfp_channel_ind]

        lfp_view = _add_ripple_times(vv.TimeseriesGraph())
        max_lfp_value = interval_ripple_lfps.to_numpy().max()
        lfp_offset *= max_lfp_value

        for i, lfp in enumerate(interval_ripple_lfps.to_numpy().T):
            lfp_view.add_line_series(
                name=f"LFP {i}",
                t=np.asarray(interval_ripple_lfps.index).squeeze(),
                y=np.asarray(lfp + lfp_offset * i, dtype=np.int16).squeeze(),
                color="black",
                width=1,
            )
        speed_view = _add_ripple_times(vv.TimeseriesGraph())
        speed_view.add_line_series(
            name="Speed [cm/s]",
            t=np.asarray(speed.index).squeeze(),
            y=np.asarray(speed, dtype=np.float32).squeeze(),
            color=speed_color,
            width=1,
        )
        vertical_panel_content = [
            vv.LayoutItem(consensus_view, stretch=2, title="Consensus"),
            vv.LayoutItem(lfp_view, stretch=8, title="LFPs"),
            vv.LayoutItem(speed_view, stretch=2, title="Speed"),
        ]

        view = vv.Box(
            direction="horizontal",
            show_titles=True,
            height=view_height,
            items=[
                vv.LayoutItem(
                    vv.Box(
                        direction="vertical",
                        show_titles=True,
                        items=vertical_panel_content,
                    )
                ),
            ],
        )

        return view.url(label="Ripple Detection")


def interpolate_to_new_time(
    df, new_time, upsampling_interpolation_method="linear"
):
    """Upsample a dataframe to a new time index"""
    old_time = df.index
    new_index = pd.Index(
        np.unique(np.concatenate((old_time, new_time))), name="time"
    )
    return (
        df.reindex(index=new_index)
        .interpolate(method=upsampling_interpolation_method)
        .reindex(index=new_time)
    )


def compute_mobile_median_mad_across_day(nwb_file_name):
    # automatically find all the relevant epochs
    all_epochs = (sgc.TaskEpoch() & {"nwb_file_name": nwb_file_name}).fetch(
        "epoch"
    )

    ripple_band_dfs = []
    for epoch in tqdm(all_epochs):
    # for epoch in [2]:
        # 1. load in relevant interval list names
        interval_list_name = (
            sgc.TaskEpoch() & {"nwb_file_name": nwb_file_name, "epoch": epoch}
        ).fetch1("interval_list_name")

        pos_interval_list_name = (sgc.IntervalList() & {'nwb_file_name': nwb_file_name, 'pipeline': 'position'}).fetch('interval_list_name')[epoch - 1]

        trodes_pos_params_name = 'default'
        mobile_interval_list_name = insert_mobile_times_interval(nwb_file_name, pos_interval_list_name, trodes_pos_params_name, speed_thresh=4, time_thresh=1)

        # 2. load in ripple band df
        lfp_electrode_group_name = 'good_single_elecs'

        lfp_sampling_rate = 1_000

        lfp_filter_name = 'LFP 0-400 Hz'
        lfp_s_key = {
            'nwb_file_name': nwb_file_name,
            'lfp_electrode_group_name': lfp_electrode_group_name,
            'target_interval_list_name': interval_list_name,
            'filter_name': lfp_filter_name,
            'filter_sampling_rate': 30_000,  # sampling rate of the data (Hz)
            'target_sampling_rate': lfp_sampling_rate,  # sampling rate of the lfp output (Hz)
        }

        lfp_merge_id = (lfp.LFPOutput.LFPV1() & lfp_s_key).fetch1('merge_id')
        lfp_key = {
            'merge_id': lfp_merge_id,
        }

        # select artifacts to include
        artifact_params_name = 'mad_7_0.66_thresh_200ms'

        # select ripples to include
        trodes_pos_params_name = 'default'
        pos_s_key = {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": pos_interval_list_name,
            "trodes_pos_params_name": trodes_pos_params_name,
        }
        pos_key = (sgp.v1.TrodesPosSelection() & pos_s_key).fetch1("KEY")
        pos_merge_key = (sgp.PositionOutput.merge_get_part(pos_key)).fetch1("KEY")
        pos_merge_id = pos_merge_key['merge_id']
        lfp_artifact_s_key = {
            'nwb_file_name': nwb_file_name,
            'lfp_electrode_group_name': lfp_electrode_group_name,
            'target_interval_list_name': interval_list_name,
            'filter_name': lfp_filter_name,
            'filter_sampling_rate': 30_000,  # I'm pretty sure this is the sampling rate for the original data but not sure
            'artifact_params_name': artifact_params_name,
        }
        artifact_removed_interval_list_name = (lfp.v1.LFPArtifactRemovedIntervalList() & lfp_artifact_s_key).fetch1('artifact_removed_interval_list_name')
        ripple_filter_name = 'Ripple 100-250 Hz'
        rip_interval_list_name = artifact_removed_interval_list_name
        lfp_band_s_key = {
            'lfp_merge_id': lfp_merge_id,
            'filter_name': ripple_filter_name,
            'filter_sampling_rate': lfp_sampling_rate,
            'target_interval_list_name': rip_interval_list_name,
            'lfp_band_sampling_rate': lfp_sampling_rate,
            'nwb_file_name': LFPOutput.merge_get_parent({"merge_id": lfp_merge_id}).fetch1("nwb_file_name"),
        }
        lfp_band_key = (LFPBandV1 & lfp_band_s_key).fetch1("KEY")
        # rip_s_key = (sgr.RippleLFPSelection & lfp_band_key).fetch1("KEY")
        # ripple_param_name = 'default_karlsson'
        # rip_key = {
        #     "ripple_param_name": ripple_param_name,
        #     **rip_s_key,
        #     "pos_merge_id": pos_merge_id,
        # }

        # load in ripple band df
        print('loading ripple band df...')
        ripple_band_df = (LFPBandV1() & lfp_band_key).fetch1_dataframe()

        # filter into mobile times only
        mobile_interval = (sgc.IntervalList() & {'nwb_file_name': nwb_file_name, 'interval_list_name': mobile_interval_list_name}).fetch_interval()
        mobile_inds = mobile_interval.contains(ripple_band_df.index.values, as_indices=True)
        mobile_ripple_band_df = ripple_band_df.iloc[mobile_inds]

        # remove any all-zero columns (reference electrodes that were included in here by accident)
        mobile_ripple_band_df = mobile_ripple_band_df.loc[:, (mobile_ripple_band_df != 0).any(axis=0)]
        mobile_ripple_band_df.columns = range(mobile_ripple_band_df.shape[1])

        ripple_band_dfs.append(mobile_ripple_band_df)

    # concatenate them together
    day_ripple_band_df = pd.concat(ripple_band_dfs).reset_index(drop=True)

    # process the ripple band into smoothed envelopes
    sampling_frequency = 1000
    smoothing_sigma = 0.004

    day_ripple_env = get_envelope(day_ripple_band_df.values)
    day_ripple_env = gaussian_smooth(
        day_ripple_env, sigma=smoothing_sigma, sampling_frequency=sampling_frequency
    )

    # find the median and the (scaled) MAD for each electrode
    elec_medians = np.nanmedian(day_ripple_env, axis=0)
    elec_mads = median_abs_deviation(day_ripple_env, axis=0, scale='normal', nan_policy='omit')

    return elec_medians, elec_mads
