import numpy as np
import pandas as pd
from scipy.stats import zscore
import warnings
from scipy.signal import hilbert
from scipy.fftpack import next_fast_len
import math
from tqdm.notebook import tqdm
import math
from matplotlib.collections import LineCollection
import math
from scipy.stats import zscore
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.decomposition import PCA


import spyglass.common as sgc
from spyglass.lfp.analysis.v1 import LFPBandV1, LFPBandSelection
from spyglass.lfp import LFPOutput
from spyglass.lfp.v1 import LFPArtifactRemovedIntervalList
import spyglass.lfp as lfp
import spyglass.position as sgp
from spyglass.common.common_interval import Interval
from spyglass.common.common_filter import FirFilterParameters
import spyglass.position as sgp
from spyglass.common.common_interval import Interval

from ripple_detection.core import gaussian_smooth

from gl_spyglass.utils.interval_functions import insert_mobile_times_interval
from gl_spyglass.custom_spyglass_tables.sleep_structure import bools_to_intervals
from gl_spyglass.utils.common_neural_functions import validate_references
from gl_spyglass.custom_spyglass_tables.grouped_ripple import RippleTimesGroup
from gl_spyglass.utils.interval_functions import insert_mobile_times_interval
from gl_spyglass.custom_spyglass_tables.sleep_structure import bools_to_intervals 


def get_next_prev_arm_coords(df):
    pos_track_segment_ids = df['pos_track_segment_id'].unique()
    arm_segments = pos_track_segment_ids[pos_track_segment_ids != 0]

    for arm_segment in tqdm(arm_segments):
        arm_segment_entrance = df.loc[df.loc[df['pos_track_segment_id'] == arm_segment, 'pos_linear_position'].idxmin(), ['pos_x', 'pos_y']].values
        
        # NOTE: this is isolated to the base area for now
        # set next (current) arm coordinates
        df.loc[(df['pos_track_segment_id'] == 0) & (df['trial_direction'] == 'outbound') & (df['curr_outer'] == arm_segment), 'next_arm_x'] = arm_segment_entrance[0]
        df.loc[(df['pos_track_segment_id'] == 0) & (df['trial_direction'] == 'outbound') & (df['curr_outer'] == arm_segment), 'next_arm_y'] = arm_segment_entrance[1]

        # set previous arm coordinates
        df.loc[(df['pos_track_segment_id'] == 0) & (df['trial_direction'] == 'outbound') & (df['prev_outer'] == arm_segment), 'prev_arm_x'] = arm_segment_entrance[0]
        df.loc[(df['pos_track_segment_id'] == 0) & (df['trial_direction'] == 'outbound') & (df['prev_outer'] == arm_segment), 'prev_arm_y'] = arm_segment_entrance[1]

    return df

def get_prev_goal_coords(df):
    pos_track_segment_ids = df['pos_track_segment_id'].unique()
    arm_segments = pos_track_segment_ids[pos_track_segment_ids != 0]

    for arm_segment in tqdm(arm_segments):
        arm_segment_entrance = df.loc[df.loc[df['pos_track_segment_id'] == arm_segment, 'pos_linear_position'].idxmin(), ['pos_x', 'pos_y']].values
        
        # NOTE: this is isolated to the base area for now
        # set next (current) arm coordinates
        df.loc[(df['pos_track_segment_id'] == 0) & (df['trial_direction'] == 'outbound') & (df['prev_goal'] == arm_segment), 'prev_goal_x'] = arm_segment_entrance[0]
        df.loc[(df['pos_track_segment_id'] == 0) & (df['trial_direction'] == 'outbound') & (df['prev_goal'] == arm_segment), 'prev_goal_y'] = arm_segment_entrance[1]

    return df

def find_angles_between_points(A, B):
    dx = B[:, 0] - A[:, 0]
    dy = B[:, 1] - A[:, 1]
    angles = np.arctan2(dy, dx)
    return angles


def find_vector(points, vector_type):

    if vector_type == 'start_end':
        # find the magnitude and direction of the endpoints-based vector
        v_start = points[0]
        v_end = points[-1]
        magn = np.linalg.norm(v_end - v_start)
        dire = math.atan2(v_end[1] - v_start[1], v_end[0] - v_start[0])

    if vector_type == 'best_fit':
        # project points into 1-dimensional space and then reconstruct into original coordinate space
        pca = PCA(n_components=1)
        X_recon = pca.inverse_transform(pca.fit_transform(points))

        # flip so vector points start -> end
        if np.dot(X_recon[-1] - X_recon[0], points[-1] - points[0]) < 0:
            X_recon = X_recon[::-1]

        # find the magnitude and direction of the reconstructed vector
        v_start = X_recon[0]
        v_end = X_recon[-1]
        magn = np.linalg.norm(v_end - v_start)
        dire = math.atan2(v_end[1] - v_start[1], v_end[0] - v_start[0])
    
    return magn, dire, v_start, v_end


def calc_move_dirs(df, win_len=3):
    # win_len is in number of time bins, where each time bin is currently 4ms (since the decoding sampling rate is 250 Hz)
    
    pos = df[['pos_x', 'pos_y']].values

    pad_len = win_len // 2
    pos_padded = np.pad(pos, ((pad_len, pad_len), (0, 0)), constant_values=np.nan)

    windows = sliding_window_view(pos_padded, window_shape=(win_len, 2))

    move_dirs = []

    # windows is a 2D array, each row is one window
    for win_pos_coords in tqdm(windows):
        
        if np.any(np.isnan(win_pos_coords)):
            # if there's nans, just skip and cut off these edge cases
            move_dirs.append(np.nan)
            continue

        _, move_dir, _, _ = find_vector(np.squeeze(win_pos_coords), vector_type='start_end')
        move_dirs.append(move_dir)
    
    return move_dirs

def add_move_dir(df):
    move_dirs = calc_move_dirs(df, win_len=3)
    df['move_dir'] = move_dirs 
    return df

def load_theta_power(nwb_file_name, interval_list_name):
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

    lfp_sampling_rate = LFPOutput.merge_get_parent(
        {"merge_id": lfp_merge_id}
    ).fetch1("lfp_sampling_rate")

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
    lfp_band_interval_list_name = artifact_removed_interval_list_name

    lfp_band_filter_name = 'Theta 5-11 Hz (unreferenced)'
    lfp_band_s_key = {
        'lfp_merge_id': lfp_merge_id,
        'filter_name': lfp_band_filter_name,
        'filter_sampling_rate': lfp_sampling_rate,
        'target_interval_list_name': lfp_band_interval_list_name,
        'lfp_band_sampling_rate': lfp_sampling_rate,
        'nwb_file_name': LFPOutput.merge_get_parent({"merge_id": lfp_merge_id}).fetch1("nwb_file_name"),
    }

    referenced=False

    # Set lfp band electrodes
    if len(LFPBandSelection() & lfp_band_s_key) == 0:

        # select electrode ids to run the lfp band processing on: all ca1 electrodes are included in ripple_electrode_list, can1ref and can2ref are included in ripple_ref_electrode_list
        electrodes_df, val_can_refs = validate_references(nwb_file_name, is_copy=True)
        electrodes_df = electrodes_df[electrodes_df['bad_channel'] == 'False']
        electrodes_df = pd.DataFrame(
            [
                electrodes_df[electrodes_df['electrode_group_name'] == i].iloc[0]
                for i in np.unique(electrodes_df['electrode_group_name'].values)
            ]
        )
        ca1_elecs_mask = (electrodes_df['region_name'] == 'ca1')
        ca1_electrode_list = electrodes_df.loc[ca1_elecs_mask, 'electrode_id'].values

        if referenced:
            ca1_ref_electrode_list = electrodes_df.loc[ca1_elecs_mask, 'val_ref'].values.astype('int')
        else:
            ca1_ref_electrode_list = -1

        # cross check with the electrodes used in the lfp_electrode_group_name used to process the lfp ('good_single_elecs')
        lfp_elecs = (lfp.lfp_electrode.LFPElectrodeGroup.LFPElectrode() & 
        {'nwb_file_name': nwb_file_name, 'lfp_electrode_group_name': 'good_single_elecs'}).fetch('electrode_id')
        if not np.all([elec in lfp_elecs for elec in ca1_electrode_list]):
            raise ValueError('not all of the electrodes selected for ripple detection have been processed in the lfp filtering step')

        LFPBandSelection().set_lfp_band_electrodes(
            nwb_file_name=lfp_band_s_key['nwb_file_name'],
            lfp_merge_id=lfp_merge_id,
            electrode_list=ca1_electrode_list,
            filter_name=lfp_band_filter_name,
            interval_list_name=lfp_band_interval_list_name,
            reference_electrode_list=ca1_ref_electrode_list,
            lfp_band_sampling_rate=lfp_sampling_rate,    
        )

    # Populate LFPBandV1
    if not (LFPBandV1 & lfp_band_s_key):
        print(f"Populating LFPBandV1 for {lfp_band_filter_name}...")
        LFPBandV1().populate(
            lfp_band_s_key,
        )

    ca1_electrodes = np.sort((LFPBandSelection.LFPBandElectrode() & lfp_band_s_key).fetch('electrode_id'))

    theta_power_df = (LFPBandV1() & lfp_band_s_key).compute_signal_power()

    theta_power_df.columns = ca1_electrodes

    return theta_power_df, lfp_sampling_rate

def load_ripple_times(nwb_file_name, interval_list_name, pos_interval_list_name):

    # select ripples to include
    ripple_filter_name = 'Ripple 100-250 Hz'
    ripple_param_name = 'shvartsman_sd4_part2'
    lfp_electrode_group_name = 'good_single_elecs'
    lfp_filter_name = 'LFP 0-400 Hz'

    lfp_s_key = {
        'nwb_file_name': nwb_file_name,
        'lfp_electrode_group_name': lfp_electrode_group_name,
        'target_interval_list_name': interval_list_name,
        'filter_name': lfp_filter_name,
        'filter_sampling_rate': 30_000,  # sampling rate of the data (Hz)  # TODO need to automate sampling rate detection since it's different for 8arm vs 6arm data
        'target_sampling_rate': 1_000,  # sampling rate of the lfp output (Hz)
    }
    lfp_merge_id = (lfp.LFPOutput.LFPV1() & lfp_s_key).fetch1('merge_id')

    trodes_pos_params_name = 'default'
    pos_s_key = {
        "nwb_file_name": nwb_file_name,
        "interval_list_name": pos_interval_list_name,
        "trodes_pos_params_name": trodes_pos_params_name,
    }
    pos_key = (sgp.v1.TrodesPosSelection() & pos_s_key).fetch1("KEY")
    pos_merge_key = (sgp.PositionOutput.merge_get_part(pos_key)).fetch1("KEY")
    pos_merge_id = pos_merge_key['merge_id']

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

    lfp_sampling_rate = LFPOutput.merge_get_parent(
        {"merge_id": lfp_merge_id}
    ).fetch1("lfp_sampling_rate")

    # select artifact detection parameters
    artifact_params_name = 'mad_7_0.66_thresh_200ms'
    lfp_filter_name = 'LFP 0-400 Hz'

    artifact_params_name = 'mad_7_0.66_thresh_200ms'
    lfp_artifact_s_key = {
        'nwb_file_name': nwb_file_name,
        'lfp_electrode_group_name': lfp_electrode_group_name,
        'target_interval_list_name': interval_list_name,
        'filter_name': lfp_filter_name,
        'filter_sampling_rate': 30_000,  # I'm pretty sure this is the sampling rate for the original data but not sure
        'artifact_params_name': artifact_params_name,
    }
    artifact_removed_interval_list_name = (lfp.v1.LFPArtifactRemovedIntervalList() & lfp_artifact_s_key).fetch1('artifact_removed_interval_list_name')
    rip_interval_list_name = artifact_removed_interval_list_name

    lfp_sampling_rate = LFPOutput.merge_get_parent(
        {"merge_id": lfp_merge_id}
    ).fetch1("lfp_sampling_rate")

    lfp_band_s_key = {
        'lfp_merge_id': lfp_merge_id,
        'filter_name': ripple_filter_name,
        'filter_sampling_rate': lfp_sampling_rate,
        'target_interval_list_name': rip_interval_list_name,
        'lfp_band_sampling_rate': lfp_sampling_rate,
        'nwb_file_name': lfp.LFPOutput.merge_get_parent({"merge_id": lfp_merge_id}).fetch1("nwb_file_name"),
    }
    lfp_band_key = (LFPBandV1 & lfp_band_s_key).fetch1("KEY")

    # Load in ripple data from the ripple group table
    lfp_band_key['ripple_param_name'] = ripple_param_name
    ripple_data = (RippleTimesGroup().RippleTimes() & lfp_band_key).fetch1_dataframe()
    ripple_times = ripple_data[['start_time', 'end_time']].values

    return ripple_times

def insert_high_theta_intervals(nwb_file_name, interval_list_name, ripple_buffer_time=None):
    print('inserting high theta intervals...')

    if "valid times" in interval_list_name:
        high_theta_interval_list_name = interval_list_name.replace(
            "valid times", f"high theta times{'' if ripple_buffer_time is None else f' rip_buffer_{ripple_buffer_time}'}"
        )
    else:
        high_theta_interval_list_name = f"{interval_list_name} high theta times{'' if ripple_buffer_time is None else f' rip_buffer_{ripple_buffer_time}'}"

    # check if this interval already exists
    if (
        len(
            sgc.IntervalList()
            & {
                "nwb_file_name": nwb_file_name,
                "interval_list_name": high_theta_interval_list_name,
            }
        )
        != 0
    ):
        print(f"{high_theta_interval_list_name} already exists, skipping")
        high_theta_intervals = (sgc.IntervalList() & {'nwb_file_name': nwb_file_name, 'interval_list_name': high_theta_interval_list_name}).fetch1('valid_times')
        return high_theta_interval_list_name, high_theta_intervals

    # load theta power
    print('loading theta power...')
    theta_power_df, lfp_sampling_rate = load_theta_power(nwb_file_name, interval_list_name)

    # process theta power
    print('processing theta power...')
    zscore_theta_power_df = theta_power_df.apply(zscore, axis=0)
    median_theta_power = zscore_theta_power_df.median(axis=1).values
    zscore_median_theta_power = zscore(median_theta_power)

    # smooth using a gaussian smoothing window
    lfp_sampling_rate = 1000
    smoothing_sigma=0.1
    zscore_theta_power_smooth = gaussian_smooth(
        zscore_median_theta_power,
        sigma=smoothing_sigma,
        sampling_frequency=lfp_sampling_rate,
    )

    # load in mobile intervals
    print('loading in mobile intervals...')
    pos_interval_list_name = (
        sgc.IntervalList()
        & {"nwb_file_name": nwb_file_name, "pipeline": "position"}
    ).fetch("interval_list_name")[int(interval_list_name[:2]) - 1]
    trodes_pos_params_name = 'default'  # NOTE: default_decoding upsamples (as opposed to just default which doesn't)
    mobile_interval_list_name = insert_mobile_times_interval(nwb_file_name, pos_interval_list_name, trodes_pos_params_name, speed_thresh=4, time_thresh=1)
    mobile_intervals = (sgc.IntervalList() & {'nwb_file_name': nwb_file_name, 'interval_list_name': mobile_interval_list_name}).fetch1('valid_times')

    # find initial high theta power interval (candidates before final pruning)
    print('finding high theta interval candidates...')
    zscore_theta_power_smooth_df = pd.DataFrame({'power': zscore_theta_power_smooth}, index=theta_power_df.index.values)

    # find the average theta power during mobile times
    interval_powers = []
    for interval in mobile_intervals:
        interval_power = zscore_theta_power_smooth_df.loc[(zscore_theta_power_smooth_df.index >= interval[0]) & (zscore_theta_power_smooth_df.index < interval[1]), 'power'].values
        interval_powers.append(interval_power)

    mobile_power = np.concatenate(interval_powers)

    # find high theta intervals based on this threshold
    mean_theta_power_mobile = np.mean(mobile_power)
    high_theta_threshold = mean_theta_power_mobile - 1
    bool_mask = zscore_theta_power_smooth > high_theta_threshold
    high_theta_intervals = bools_to_intervals(bool_mask, theta_power_df.index.values)

    # load in and remove ripple times
    print('loading in and subtracting ripple times')
    ripple_times = load_ripple_times(nwb_file_name, interval_list_name, pos_interval_list_name)
    if ripple_buffer_time is not None:
        # expand ripple times to include a narrow buffer window
        ripple_times[:, 0] = ripple_times[:, 0] - ripple_buffer_time 
        ripple_times[:, 1] = ripple_times[:, 0] + ripple_buffer_time
    high_theta_intervals = Interval(high_theta_intervals, no_overlap=True).subtract(ripple_times).times

    # isolate only to intervals that are > 1s
    print('final time duration pruning...')
    interval_durations = np.squeeze(np.diff(high_theta_intervals, axis=1))
    high_theta_intervals = high_theta_intervals[interval_durations > 1]

    # insert this interval list into IntervalList
    sgc.IntervalList.insert1(
        {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": high_theta_interval_list_name,
            "valid_times": np.asarray(high_theta_intervals),
        },
        skip_duplicates=True,
    )
    print(f"Inserted new interval: {high_theta_interval_list_name}")

    return high_theta_interval_list_name, high_theta_intervals


# Write functions that filter any data into a desired frequency band (matching exact calculations done in LFPBandV1.populate)
def filter_data(
    timestamps,
    data,
    filter_coeff,
    valid_times,
    electrodes,
    decimation,
):
    gsp = _import_ghostipy()

    n_dim = len(data.shape)
    n_samples = len(timestamps)
    time_axis = 0 if data.shape[0] == n_samples else 1
    electrode_axis = 1 - time_axis
    input_dim_restrictions = [None] * n_dim
    input_dim_restrictions[electrode_axis] = np.s_[electrodes]

    indices = []
    output_shape_list = [0] * n_dim
    output_shape_list[electrode_axis] = len(electrodes)
    output_offsets = [0]

    filter_delay = calc_filter_delay(filter_coeff)
    for a_start, a_stop in valid_times:
        frm, to = _time_bound_check(a_start, a_stop, timestamps, n_samples)
        if np.isclose(frm, to, rtol=0, atol=1e-8):
            continue
        indices.append((frm, to))

        shape, _ = gsp.filter_data_fir(
            data,
            filter_coeff,
            axis=time_axis,
            input_index_bounds=[frm, to],
            output_index_bounds=[filter_delay, filter_delay + to - frm],
            describe_dims=True,
            ds=decimation,
            input_dim_restrictions=input_dim_restrictions,
        )
        output_offsets.append(output_offsets[-1] + shape[time_axis])
        output_shape_list[time_axis] += shape[time_axis]

    filtered_data = np.empty(tuple(output_shape_list), dtype=data.dtype)
    new_timestamps = np.empty((output_shape_list[time_axis],), timestamps.dtype)

    indices = np.array(indices, ndmin=2)

    ts_offset = 0
    for ii, (start, stop) in enumerate(indices):
        extracted_ts = timestamps[start:stop:decimation]
        new_timestamps[ts_offset : ts_offset + len(extracted_ts)] = extracted_ts
        ts_offset += len(extracted_ts)

        gsp.filter_data_fir(
            data,
            filter_coeff,
            axis=time_axis,
            input_index_bounds=[start, stop],
            output_index_bounds=[filter_delay, filter_delay + stop - start],
            ds=decimation,
            input_dim_restrictions=input_dim_restrictions,
            outarray=filtered_data,
            output_offset=output_offsets[ii],
        )

    return filtered_data, new_timestamps

def _import_ghostipy():
    try:
        import ghostipy as gsp

        return gsp
    except (ImportError, ModuleNotFoundError) as e:
        raise ImportError(
            "You must install ghostipy to use filtering methods. Please note "
            "that to install ghostipy on an Mac M1, you must first install "
            "pyfftw from conda-forge."
        ) from e

def calc_filter_delay(filter_coeff):
    return (len(filter_coeff) - 1) // 2

def _time_bound_check(start, stop, all, nsamples):
    timestamp_warn = "Interval time warning: "
    if start < all[0]:
        warnings.warn(
            timestamp_warn
            + "start time smaller than first timestamp, "
            + f"substituting first: {start} < {all[0]}"
        )
        start = all[0]

    if stop > all[-1]:
        print(
            timestamp_warn
            + "stop time larger than last timestamp, "
            + f"substituting last: {stop} < {all[-1]}"
        )
        stop = all[-1]

    frm, to = np.searchsorted(all, (start, stop))
    to = min(to, nsamples)
    return frm, to

def load_ca1_lfp_theta(nwb_file_name, interval_list_name, return_lfp_band_s_key=False):
    
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

    lfp_sampling_rate = LFPOutput.merge_get_parent(
        {"merge_id": lfp_merge_id}
    ).fetch1("lfp_sampling_rate")

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
    lfp_band_interval_list_name = artifact_removed_interval_list_name

    lfp_band_filter_name = 'Theta 5-11 Hz (unreferenced)'
    lfp_band_s_key = {
        'lfp_merge_id': lfp_merge_id,
        'filter_name': lfp_band_filter_name,
        'filter_sampling_rate': lfp_sampling_rate,
        'target_interval_list_name': lfp_band_interval_list_name,
        'lfp_band_sampling_rate': lfp_sampling_rate,
        'nwb_file_name': LFPOutput.merge_get_parent({"merge_id": lfp_merge_id}).fetch1("nwb_file_name"),
    }

    referenced=False

    # Set lfp band electrodes
    if len(LFPBandSelection() & lfp_band_s_key) == 0:

        # select electrode ids to run the lfp band processing on: all ca1 electrodes are included in ripple_electrode_list, can1ref and can2ref are included in ripple_ref_electrode_list
        electrodes_df, val_can_refs = validate_references(nwb_file_name, is_copy=True)
        electrodes_df = electrodes_df[electrodes_df['bad_channel'] == 'False']
        electrodes_df = pd.DataFrame(
            [
                electrodes_df[electrodes_df['electrode_group_name'] == i].iloc[0]
                for i in np.unique(electrodes_df['electrode_group_name'].values)
            ]
        )
        ca1_elecs_mask = (electrodes_df['region_name'] == 'ca1')
        ca1_electrode_list = electrodes_df.loc[ca1_elecs_mask, 'electrode_id'].values

        if referenced:
            ca1_ref_electrode_list = electrodes_df.loc[ca1_elecs_mask, 'val_ref'].values.astype('int')
        else:
            ca1_ref_electrode_list = -1

        # cross check with the electrodes used in the lfp_electrode_group_name used to process the lfp ('good_single_elecs')
        lfp_elecs = (lfp.lfp_electrode.LFPElectrodeGroup.LFPElectrode() & 
        {'nwb_file_name': nwb_file_name, 'lfp_electrode_group_name': 'good_single_elecs'}).fetch('electrode_id')
        if not np.all([elec in lfp_elecs for elec in ca1_electrode_list]):
            raise ValueError('not all of the electrodes selected for ripple detection have been processed in the lfp filtering step')

        LFPBandSelection().set_lfp_band_electrodes(
            nwb_file_name=lfp_band_s_key['nwb_file_name'],
            lfp_merge_id=lfp_merge_id,
            electrode_list=ca1_electrode_list,
            filter_name=lfp_band_filter_name,
            interval_list_name=lfp_band_interval_list_name,
            reference_electrode_list=ca1_ref_electrode_list,
            lfp_band_sampling_rate=lfp_sampling_rate,    
        )

    # Populate LFPBandV1
    if not (LFPBandV1 & lfp_band_s_key):
        print(f"Populating LFPBandV1 for {lfp_band_filter_name}...")
        LFPBandV1().populate(
            lfp_band_s_key,
        )

    ca1_electrodes = np.sort((LFPBandSelection.LFPBandElectrode() & lfp_band_s_key).fetch('electrode_id'))

    theta_band_df = (LFPBandV1() & lfp_band_s_key).fetch1_dataframe()
    theta_phase_df = (LFPBandV1() & lfp_band_s_key).compute_signal_phase() - np.pi

    theta_band_df.columns = ca1_electrodes
    theta_phase_df.columns = ca1_electrodes

    if return_lfp_band_s_key:
        return theta_band_df, theta_phase_df, lfp_band_s_key
    else:
        return theta_band_df, theta_phase_df

def load_ca1_mua_theta(decode_info_df):
    mua = np.array([decode_info_df['multiunit_firing_rate'].values])
    timestamps = decode_info_df['time'].values
    filter_coeff = (FirFilterParameters() & {'filter_name': 'Theta 5-11 Hz', 'filter_sampling_rate': 250}).fetch1('filter_coeff')
    mua_theta_band_vals, new_timestamps = filter_data(timestamps, mua, filter_coeff, valid_times=np.asarray([[timestamps[0], timestamps[-1] + 1]]), electrodes=[0], decimation=1)

    n_samples = len(mua_theta_band_vals[0])
    n = next_fast_len(n_samples)
    analytic_signal = hilbert(mua_theta_band_vals[0], N=n)[:n_samples]
    mua_theta_phase = np.angle(analytic_signal)
    
    return pd.DataFrame({'time': new_timestamps, 'theta_band': mua_theta_band_vals[0], 'theta_phase': mua_theta_phase})

def get_sweep_info(df, theta_type):
    sweep_coords = df[['mean_hpd90.0_x', 'mean_hpd90.0_y']].values
    magn, dire, v_start, v_end = find_vector(sweep_coords, vector_type='start_end')
    total_dist = np.sum(np.linalg.norm(sweep_coords[1:] - sweep_coords[:-1], axis=1))
    phase_duration = df[f'{theta_type}_theta_phase_wrapped'].values[-1] - df[f'{theta_type}_theta_phase_wrapped'].values[0]
    time_duration = df['time'].values[-1] - df['time'].values[0]
    speed = total_dist / time_duration

    start_pos = df[['pos_x', 'pos_y']].values[0]
    start_sweep = sweep_coords[0]
    init_offset_magn = np.linalg.norm(start_sweep - start_pos)
    init_offset_dire = math.atan2(start_sweep[1] - start_pos[1], start_sweep[0] - start_pos[0])

    df[['sweep_v_magn', 'sweep_v_dire', 'sweep_total_dist', 'sweep_phase_duration',
        'sweep_time_duration', 'sweep_speed', 'init_offset_magn', 'init_offset_dire']] = (
        magn, dire, total_dist, phase_duration, time_duration, speed, init_offset_magn, init_offset_dire
    )

    return df

def get_theta_phase_df(
        nwb_file_name,
        interval_list_name,
        decode_info_df,
        phase_type='lfp_best_elec',        
):
    print(f'getting theta phase for {phase_type}...')
    
    lfp_best_tets = {
        'pippin20210421_.nwb': 56,
        'pippin20210423_.nwb': 48,
        'archibald20210714_.nwb': 4,
        'chip20220109_.nwb': 30,
        'herman20211113_.nwb': 32,
        'hugo20210812_.nwb': 43,
        'bobrick20231204_.nwb': 34,
        'reginald20241015_.nwb': 55,
        'teddy20250604_.nwb': 43,
        'teddy20250616_.nwb': 54,
        'teddy20250620_.nwb': 50,
        'timothy20241111_.nwb': 58,
        'tony20250430_.nwb': 10,
    }

    if phase_type == 'mua':
        theta_band_phase_df = load_ca1_mua_theta(decode_info_df)
    
    if phase_type == 'lfp_best_elec':
        # load in ca1 LFP theta
        theta_band, theta_phase, lfp_band_s_key = load_ca1_lfp_theta(nwb_file_name, interval_list_name, return_lfp_band_s_key=True)

        # select best tetrode
        best_tetrode = lfp_best_tets[nwb_file_name]
        ca1_electrodes_df = pd.DataFrame(LFPBandSelection.LFPBandElectrode() & lfp_band_s_key)
        best_elec_id = ca1_electrodes_df.loc[ca1_electrodes_df['electrode_group_name'] == str(best_tetrode - 1), 'electrode_id'].values[0]

        # put theta band and phase from this electrode together
        theta_band_phase_df = pd.DataFrame({'time': theta_band.index.values, 'theta_band': theta_band[best_elec_id].values, 'theta_phase': theta_phase[best_elec_id].values})

    if phase_type == 'lfp_elec_mean':
        # load in ca1 LFP theta
        theta_band, _ = load_ca1_lfp_theta(nwb_file_name, interval_list_name)

        # get the phase of the mean signal
        mean_theta_band_vals = theta_band.mean(axis=1).values
        n_samples = len(mean_theta_band_vals)
        n = next_fast_len(n_samples)
        analytic_signal = hilbert(mean_theta_band_vals, N=n)[:n_samples]
        mean_theta_phase = np.angle(analytic_signal)

        # put theta band and phase from the mean together
        theta_band_phase_df = pd.DataFrame({'time': theta_band.index.values, 'theta_band': mean_theta_band_vals, 'theta_phase': mean_theta_phase})

    return theta_band_phase_df

from sklearn.decomposition import PCA

def find_vector(points, vector_type):

    if vector_type == 'start_end':
        # find the magnitude and direction of the endpoints-based vector
        v_start = points[0]
        v_end = points[-1]
        magn = np.linalg.norm(v_end - v_start)
        dire = math.atan2(v_end[1] - v_start[1], v_end[0] - v_start[0])

    if vector_type == 'best_fit':
        # project points into 1-dimensional space and then reconstruct into original coordinate space
        pca = PCA(n_components=1)
        X_recon = pca.inverse_transform(pca.fit_transform(points))

        # flip so vector points start -> end
        if np.dot(X_recon[-1] - X_recon[0], points[-1] - points[0]) < 0:
            X_recon = X_recon[::-1]

        # find the magnitude and direction of the reconstructed vector
        v_start = X_recon[0]
        v_end = X_recon[-1]
        magn = np.linalg.norm(v_end - v_start)
        dire = math.atan2(v_end[1] - v_start[1], v_end[0] - v_start[0])
    
    return magn, dire, v_start, v_end

def get_sweep_info(df, theta_type):
    sweep_coords = df[['mean_hpd90.0_x', 'mean_hpd90.0_y']].values
    magn, dire, v_start, v_end = find_vector(sweep_coords, vector_type='start_end')
    total_dist = np.sum(np.linalg.norm(sweep_coords[1:] - sweep_coords[:-1], axis=1))
    phase_duration = df[f'{theta_type}_theta_phase_wrapped'].values[-1] - df[f'{theta_type}_theta_phase_wrapped'].values[0]
    time_duration = df['time'].values[-1] - df['time'].values[0]
    speed = total_dist / time_duration

    start_pos = df[['pos_x', 'pos_y']].values[0]
    start_sweep = sweep_coords[0]
    init_offset_magn = np.linalg.norm(start_sweep - start_pos)
    init_offset_dire = math.atan2(start_sweep[1] - start_pos[1], start_sweep[0] - start_pos[0])

    df[['sweep_v_magn', 'sweep_v_dire', 'sweep_total_dist', 'sweep_phase_duration',
        'sweep_time_duration', 'sweep_speed', 'init_offset_magn', 'init_offset_dire']] = (
        magn, dire, total_dist, phase_duration, time_duration, speed, init_offset_magn, init_offset_dire
    )

    return df

def get_next_prev_arm_coords(df):
    pos_track_segment_ids = df['pos_track_segment_id'].unique()
    arm_segments = pos_track_segment_ids[pos_track_segment_ids != 0]

    for arm_segment in tqdm(arm_segments):
        arm_segment_entrance = df.loc[df.loc[df['pos_track_segment_id'] == arm_segment, 'pos_linear_position'].idxmin(), ['pos_x', 'pos_y']].values
        
        # NOTE: this is isolated to the base area for now
        # set next (current) arm coordinates
        df.loc[(df['pos_track_segment_id'] == 0) & (df['trial_direction'] == 'outbound') & (df['curr_outer'] == arm_segment), 'next_arm_x'] = arm_segment_entrance[0]
        df.loc[(df['pos_track_segment_id'] == 0) & (df['trial_direction'] == 'outbound') & (df['curr_outer'] == arm_segment), 'next_arm_y'] = arm_segment_entrance[1]

        # set previous arm coordinates
        df.loc[(df['pos_track_segment_id'] == 0) & (df['trial_direction'] == 'outbound') & (df['prev_outer'] == arm_segment), 'prev_arm_x'] = arm_segment_entrance[0]
        df.loc[(df['pos_track_segment_id'] == 0) & (df['trial_direction'] == 'outbound') & (df['prev_outer'] == arm_segment), 'prev_arm_y'] = arm_segment_entrance[1]

    return df

def get_prev_goal_coords(df):
    pos_track_segment_ids = df['pos_track_segment_id'].unique()
    arm_segments = pos_track_segment_ids[pos_track_segment_ids != 0]

    for arm_segment in tqdm(arm_segments):
        arm_segment_entrance = df.loc[df.loc[df['pos_track_segment_id'] == arm_segment, 'pos_linear_position'].idxmin(), ['pos_x', 'pos_y']].values
        
        # NOTE: this is isolated to the base area for now
        # set next (current) arm coordinates
        df.loc[(df['pos_track_segment_id'] == 0) & (df['trial_direction'] == 'outbound') & (df['prev_goal'] == arm_segment), 'prev_goal_x'] = arm_segment_entrance[0]
        df.loc[(df['pos_track_segment_id'] == 0) & (df['trial_direction'] == 'outbound') & (df['prev_goal'] == arm_segment), 'prev_goal_y'] = arm_segment_entrance[1]

    return df

def find_angles_between_points(A, B):
    dx = B[:, 0] - A[:, 0]
    dy = B[:, 1] - A[:, 1]
    angles = np.arctan2(dy, dx)
    return angles

def calc_move_dirs(df, win_len=3):
    # win_len is in number of time bins, where each time bin is currently 4ms (since the decoding sampling rate is 250 Hz)
    
    pos = df[['pos_x', 'pos_y']].values

    pad_len = win_len // 2
    pos_padded = np.pad(pos, ((pad_len, pad_len), (0, 0)), constant_values=np.nan)

    windows = sliding_window_view(pos_padded, window_shape=(win_len, 2))

    move_dirs = []

    # windows is a 2D array, each row is one window
    for win_pos_coords in tqdm(windows):
        
        if np.any(np.isnan(win_pos_coords)):
            # if there's nans, just skip and cut off these edge cases
            move_dirs.append(np.nan)
            continue

        _, move_dir, _, _ = find_vector(np.squeeze(win_pos_coords), vector_type='start_end')
        move_dirs.append(move_dir)
    
    return move_dirs

def add_move_dir(df):
    move_dirs = calc_move_dirs(df, win_len=3)
    df['move_dir'] = move_dirs 
    return df

def get_sweep_dir_diff(df):
    df['sweep_dir_diff'] = np.concatenate([[np.nan], np.diff(df['sweep_v_dire'].values)])
    return df

def add_processed_theta_sweep_info(nwb_file_name, interval_list_name, decode_info_df, theta_type='mua'):
    # add euclidean mean decode error (distance)
    decode_info_df['euc_mean_decode_err'] = np.linalg.norm(decode_info_df[['mean_hpd90.0_x', 'mean_hpd90.0_y']].values - decode_info_df[['pos_x', 'pos_y']].values, axis=1)
    
    # add smoothed euclidean mean decode error
    smoothing_sigma=0.01
    decode_sampling_freq = int(1/np.mean(np.diff(decode_info_df['time'].values)))
    decode_info_df['euc_mean_decode_err_smooth'] = gaussian_smooth(decode_info_df['euc_mean_decode_err'].values, smoothing_sigma, sampling_frequency=decode_sampling_freq)

    # add smoothed mean hpd coordinates
    smoothing_sigma=0.01
    decode_sampling_freq = int(1/np.mean(np.diff(decode_info_df['time'].values)))
    decode_info_df['mean_hpd90.0_x_smooth'] = gaussian_smooth(decode_info_df['mean_hpd90.0_x'].values, smoothing_sigma, sampling_frequency=decode_sampling_freq)
    decode_info_df['mean_hpd90.0_y_smooth'] = gaussian_smooth(decode_info_df['mean_hpd90.0_y'].values, smoothing_sigma, sampling_frequency=decode_sampling_freq)

    # add in theta phase
    theta_phase_df = get_theta_phase_df(
                            nwb_file_name,
                            interval_list_name,
                            decode_info_df,
                            phase_type=theta_type,
                        )
    
    # rename columns so that they're all distinct
    theta_phase_df = theta_phase_df.rename(columns={'theta_band': f'{theta_type}_theta_band', 'theta_phase': f'{theta_type}_theta_phase'})

    decode_info_df = pd.merge_asof(
        decode_info_df,
        theta_phase_df,
        on='time',
        direction='nearest',
        tolerance=0.002   # max allowed mismatch (seconds)
    )

    # calculate and load in high theta intervals
    _, high_theta_intervals = insert_high_theta_intervals(nwb_file_name, interval_list_name)

    # label all high theta interval times
    time_vals = decode_info_df['time'].values
    decode_info_df['is_high_theta'] = False
    is_high_theta_col = decode_info_df.columns.get_loc('is_high_theta')
    for interval in high_theta_intervals:
        i_start = np.searchsorted(time_vals, interval[0], side='left')
        i_end = np.searchsorted(time_vals, interval[1], side='left')
        decode_info_df.iloc[i_start:i_end, is_high_theta_col] = True

    # split theta cycles by theta type
    print(f'splitting theta cycles by {theta_type}...')

    # find theta phase 0 crossings (these mark the start of the theta cycle since they're aligned to the peaks of the MUA activity)
    theta_phase = decode_info_df[[f'{theta_type}_theta_phase']]

    if theta_type == 'mua':  # look for transitions from high to low (zero crossings in the negative direction)
        phase_zeros_mask = (theta_phase.shift(1) > 0) & (theta_phase < 0)
    else:  # look for zero crossings (in the positive direction)
        phase_zeros_mask = (theta_phase.shift(1) < 0) & (theta_phase > 0)
    theta_cycle_start_times = decode_info_df.loc[phase_zeros_mask.values.ravel(), 'time'].values

    # assign theta cycles in the combined dataframe
    time = decode_info_df['time'].values
    cycle_indices = np.searchsorted(theta_cycle_start_times, time, side='right')
    decode_info_df[f'n_{theta_type}_theta_cycle'] = cycle_indices

    # also set the aligned theta phase (wrapped version so that it goes from 0 at the start of the theta cycle and 2*pi at the end)
    if theta_type == 'mua':
        decode_info_df[f'{theta_type}_theta_phase_wrapped'] = decode_info_df[f'{theta_type}_theta_phase'].values + np.pi
    else:
        decode_info_df[f'{theta_type}_theta_phase_wrapped'] = decode_info_df[f'{theta_type}_theta_phase'].values % (2 * np.pi)

    # split into theta cycle bins
    n_cycle_bins = 36
    cycle_bin_starts = np.linspace(0, 2*np.pi, n_cycle_bins)

    # assign theta cycles in the combined dataframe
    theta_phase_wrapped = decode_info_df[f'{theta_type}_theta_phase_wrapped'].values
    cycle_bins = np.searchsorted(cycle_bin_starts, theta_phase_wrapped, side='right')
    decode_info_df[f'n_{theta_type}_theta_cycle_bin'] = cycle_bins
    decode_info_df[f'{theta_type}_theta_cycle_bin_degr'] = cycle_bins*10
    
    # Use desired theta type to label sweeps within the predefined theta cycle boundaries
    print(f'labeling sweeps for {theta_type}')
    # (optimized version of previous code (optimized by claude))
    col_cycle = f'n_{theta_type}_theta_cycle'
    col_phase = f'{theta_type}_theta_phase_wrapped'
    not_first_and_not_last_mask = (decode_info_df[col_cycle] != 0) & (decode_info_df[col_cycle] != decode_info_df[col_cycle].unique()[-1])

    theta_cycles = decode_info_df.loc[not_first_and_not_last_mask, col_cycle].unique()

    phase_min_cutoff = np.pi / 3
    phase_max_cutoff = 5 * np.pi / 3

    # precompute once outside loop
    time_vals = decode_info_df['time'].values
    idx_vals  = decode_info_df.index.values
    phase_vals = decode_info_df[col_phase].values
    cycle_vals = decode_info_df[col_cycle].values
    decode_err_smooth_vals = decode_info_df['euc_mean_decode_err_smooth'].values
    decode_info_df['is_sweep'] = False  # initialize is_sweep column
    is_sweep_col = decode_info_df.columns.get_loc('is_sweep')

    # precompute phase mask once — same for every cycle
    phase_mask = (phase_vals >= phase_min_cutoff) & (phase_vals < phase_max_cutoff)

    for n_theta_cycle in tqdm(theta_cycles):

        cycle_mask = cycle_vals == n_theta_cycle
        combined_mask = cycle_mask & phase_mask

        if not combined_mask.any():
            continue

        # find min/max error times within combined mask
        combined_indices = np.where(combined_mask)[0]
        cycle_decode_err = decode_err_smooth_vals[combined_indices]
        sweep_min_time = time_vals[combined_indices[cycle_decode_err.argmin()]]
        sweep_max_time = time_vals[combined_indices[cycle_decode_err.argmax()]]

        if sweep_min_time > sweep_max_time:
            sweep_start_time, sweep_end_time = sweep_max_time, sweep_min_time
        else:
            sweep_start_time, sweep_end_time = sweep_min_time, sweep_max_time

        # use searchsorted for sweep boundary
        i_start = np.searchsorted(time_vals, sweep_start_time, side='left')
        i_end   = np.searchsorted(time_vals, sweep_end_time,   side='left')

        # only label rows that are also in cycle_mask
        sweep_indices = np.where(cycle_mask[i_start:i_end])[0] + i_start
        decode_info_df.iloc[sweep_indices, is_sweep_col] = True
    
    # exclude any sweeps that are < pi / 2 in duration
    sweep_phase_duration = (
        decode_info_df
        .loc[decode_info_df['is_sweep']]
        .groupby(col_cycle)[col_phase]
        .agg(lambda x: x.iloc[-1] - x.iloc[0])
    )
    decode_info_df['sweep_phase_duration'] = decode_info_df[col_cycle].map(sweep_phase_duration)

    mobile_whole_cycle_mask = decode_info_df.groupby(col_cycle)['speed'].transform('min') >= 4
    sweep_phase_duration_df = decode_info_df.loc[mobile_whole_cycle_mask].groupby([col_cycle]).first().reset_index()[[col_cycle, 'sweep_phase_duration']]
    n_less_than_quarter_cycle = len(sweep_phase_duration_df[sweep_phase_duration_df['sweep_phase_duration'].values <= np.pi/2])
    print(f'{n_less_than_quarter_cycle} cycles < 1/4 cycle out of {len(sweep_phase_duration_df)} mobile sweeps ({n_less_than_quarter_cycle / len(sweep_phase_duration_df)})')
            
    decode_info_df.loc[decode_info_df['sweep_phase_duration'] <= np.pi/2, 'is_sweep'] = False

    # set oubound / inbound trial times
    decode_info_df.loc[decode_info_df['trial_stage'].isin(['center_to_outer']), 'trial_direction'] = 'outbound'
    decode_info_df.loc[decode_info_df['trial_stage'].isin(['outer_to_home']), 'trial_direction'] = 'inbound'

    print('getting behaviorally-relevant coordinates...')
    # set the previous outer
    trial_values = decode_info_df.groupby(["n_trial"])["curr_outer"].first()
    trial_values_shifted = trial_values.shift(1)
    decode_info_df["prev_outer"] = decode_info_df['n_trial'].map(trial_values_shifted)

    if 'prev_goal' not in decode_info_df.columns:
        decode_info_df['prev_goal'] = float('nan')

    # get coordinates of the next arm, previous arm, and previous goal arm
    decode_info_df = get_next_prev_arm_coords(decode_info_df)
    decode_info_df = get_prev_goal_coords(decode_info_df)

    # find the angles between current position and each of these arms
    pos_x = decode_info_df['pos_x'].values
    pos_y = decode_info_df['pos_y'].values
    pos = np.vstack([pos_x, pos_y]).T

    print('getting angles between current position and behaviorally-relevant coordinates...')
    # angle between current position and next arm
    next_arm_x = decode_info_df['next_arm_x'].values
    next_arm_y = decode_info_df['next_arm_y'].values
    next_outers = np.vstack([next_arm_x, next_arm_y]).T
    decode_info_df['next_arm_dir'] = find_angles_between_points(pos, next_outers)

    # angle between current position and previous arm
    prev_arm_x = decode_info_df['prev_arm_x'].values
    prev_arm_y = decode_info_df['prev_arm_y'].values
    prev_outers = np.vstack([prev_arm_x, prev_arm_y]).T
    decode_info_df['prev_arm_dir'] = find_angles_between_points(pos, prev_outers)

    # angle between current position and previous goal
    prev_goal_x = decode_info_df['prev_goal_x'].values
    prev_goal_y = decode_info_df['prev_goal_y'].values
    prev_goals = np.vstack([prev_goal_x, prev_goal_y]).T
    decode_info_df['prev_goal_dir'] = find_angles_between_points(pos, prev_goals)

    # calculate and add movement direction at each time point
    decode_info_df = add_move_dir(decode_info_df)

    print('add in sweep-specific information...')
    # get all other sweep-specific info (only on user-defined sweeps)
    sweep_decode_info_df = decode_info_df.loc[decode_info_df['is_sweep']].drop(columns=['sweep_phase_duration']).groupby([f'n_{theta_type}_theta_cycle']).apply(get_sweep_info, theta_type=theta_type, include_groups=False).reset_index()
    
    # winnow down to per-sweep summary
    per_sweep_df = sweep_decode_info_df.groupby([f'n_{theta_type}_theta_cycle']).first().reset_index()

    return decode_info_df, sweep_decode_info_df, per_sweep_df