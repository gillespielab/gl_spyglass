import numpy as np
import xarray as xr

from spyglass.decoding.v0.visualization_2D_view import create_static_track_animation, get_ul_corners, process_decoded_data

import figpack.views as vv
from figpack_franklab.views import TrackAnimation


def create_track_animation_object_figpack(
    *, static_track_animation: any
) -> TrackAnimation:
    
    decoded_data = static_track_animation["decodedData"]

    timestamp_start = (
        static_track_animation["timestampStart"]
        if "timestampStart" in static_track_animation
        else None
    )
    head_direction = (
        static_track_animation["headDirection"]
        if "headDirection" in static_track_animation
        else None
    )
    return TrackAnimation(
        bin_width=decoded_data["binWidth"],
        bin_height=decoded_data["binHeight"],
        values=decoded_data["values"].astype(np.int16),
        locations=decoded_data["locations"],
        frame_bounds=decoded_data["frameBounds"].astype(np.int16),
        xmin=decoded_data["xmin"],
        xcount=decoded_data["xcount"],
        ymin=decoded_data["ymin"],
        ycount=decoded_data["ycount"],
        track_bin_width=static_track_animation["trackBinWidth"],
        track_bin_height=static_track_animation["trackBinHeight"],
        track_bin_corners=static_track_animation["trackBinULCorners"],
        total_recording_frame_length=static_track_animation[
            "totalRecordingFrameLength"
        ],
        timestamp_start=timestamp_start,
        timestamps=static_track_animation["timestamps"],
        positions=static_track_animation["positions"],
        # x_min=static_track_animation["xmin"],
        xmax=static_track_animation["xmax"],
        # y_min=static_track_animation["ymin"],
        ymax=static_track_animation["ymax"],
        sampling_frequency_hz=static_track_animation["samplingFrequencyHz"],
        head_direction=head_direction,
    )

def create_2D_decode_view_figpack(
    position_time: np.ndarray,
    position: np.ndarray,
    interior_place_bin_centers: np.ndarray,
    place_bin_size: np.ndarray,
    posterior: xr.DataArray,
    head_dir: np.ndarray = None,
) -> TrackAnimation:
    """Creates a 2D decoding movie view

    Parameters
    ----------
    position_time : np.ndarray, shape (n_time,)
    position : np.ndarray, shape (n_time, 2)
    interior_place_bin_centers: np.ndarray, shape (n_track_bins, 2)
    place_bin_size : np.ndarray, shape (2, 1)
    posterior : xr.DataArray, shape (n_time, n_position_bins)
    head_dir : np.ndarray, optional

    Returns
    -------
    view : vvf.TrackPositionAnimationV1

    """
    assert (
        position_time.shape[0] == position.shape[0]
    ), "position_time and position must have the same length"
    assert (
        posterior.shape[0] == position.shape[0]
    ), "posterior and position must have the same length"

    position_time = np.squeeze(np.asarray(position_time)).copy()
    position = np.asarray(position)
    if head_dir is not None:
        head_dir = np.squeeze(np.asarray(head_dir))

    track_bin_width, track_bin_height = place_bin_size
    upper_left_points = get_ul_corners(
        track_bin_width, track_bin_height, interior_place_bin_centers
    )

    data = create_static_track_animation(
        ul_corners=upper_left_points,
        track_rect_height=track_bin_height,
        track_rect_width=track_bin_width,
        timestamps=position_time,
        positions=position.T,
        head_dir=head_dir,
        compute_real_time_rate=True,
    )
    data["decodedData"] = process_decoded_data(posterior)

    return create_track_animation_object_figpack(static_track_animation=data)