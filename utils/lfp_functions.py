import numpy as np
import pandas as pd
from datajoint.errors import DuplicateError

import spyglass.lfp as lfp
import spyglass.position as sgp
from spyglass.lfp.lfp_merge import LFPOutput
from spyglass.lfp.analysis.v1 import LFPBandSelection, LFPBandV1
from spyglass.utils import logger

from gl_spyglass.utils.common_neural_functions import validate_references


def single_epoch_lfp_pipeline(nwb_file_name, interval_list_name, pos_interval_list_name, pop_ripple=True, rip_artifact_removed=True, pop_theta=False, ripple_param_name='default_karlsson'):
    '''
    This function walks through populating LFP, with an option to populate the ripple LFP band and detect ripples.
    Functionality for populating the theta band is not yet built out -- this function could be improved to populate an arbitrary lfp band.

    Currently the ripple detection step is commented out as the grouped_date_ripple_pipeline function below is used for the actual
    ripple detection step instead (since you need to utilize the grouped ripple tables in order to use the Shvartsman_ripple_detection method).

    Note additional parameters for including artifact detection (or not) in the ripple band filtering process.
    '''

    # 1) POPULATE LFP
    print('populating lfp...')
    # get validated references
    electrodes_df, val_can_refs = validate_references(nwb_file_name, is_copy=True)

    # create LFP electrode group with one good electrode from each tetrode (if any)
    good_elecs_df = electrodes_df[
        (electrodes_df['bad_channel'] == 'False')
    ]
    good_single_elecs_df = pd.DataFrame(
        [
            good_elecs_df[good_elecs_df["electrode_group_name"] == i].iloc[0]
            for i in np.unique(good_elecs_df["electrode_group_name"].values)
        ]
    )
    good_single_elecs = good_single_elecs_df['electrode_id'].values
    lfp_electrode_group_name = 'good_single_elecs'
    try:
        lfp.lfp_electrode.LFPElectrodeGroup.create_lfp_electrode_group(
            nwb_file_name=nwb_file_name,
            group_name=lfp_electrode_group_name,
            electrode_list=good_single_elecs,
        )
    except DuplicateError:
        print('DuplicateError: skipping electrode group creation because already exists')

    # populate LFP output
    # insert selection key into lfp selection table
    lfp_filter_name = 'LFP 0-400 Hz'
    lfp_s_key = {
        'nwb_file_name': nwb_file_name,
        'lfp_electrode_group_name': lfp_electrode_group_name,
        'target_interval_list_name': interval_list_name,
        'filter_name': lfp_filter_name,
        'filter_sampling_rate': 30_000,  # sampling rate of the data (Hz)
        'target_sampling_rate': 1_000,  # sampling rate of the lfp output (Hz)
    }
    lfp.v1.LFPSelection.insert1(lfp_s_key, skip_duplicates=True)

    # populate lfp table (downsample and low pass filter)
    lfp.v1.LFPV1().populate(lfp_s_key)

    # fetch lfp merge id
    lfp_merge_id = (lfp.LFPOutput.LFPV1() & lfp_s_key).fetch1('merge_id')


    # 2) LFP ARTIFACT DETECTION
    print('populating lfp artifact detection...')
    # insert custom artifact detection parameters
    params_name = 'mad_7_0.66_thresh_200ms'
    custom_mad_params = [
        params_name,
        {
            "artifact_detection_algorithm": "mad",
            "artifact_detection_algorithm_params": {
                # akin to z-score std dev if the distribution is normal
                "mad_thresh": 7.0,
                "proportion_above_thresh": 0.66,
                "removal_window_ms": 200.0,  # in milliseconds
            },
        },
    ]
    lfp.v1.LFPArtifactDetectionParameters().insert1(custom_mad_params, skip_duplicates=True)

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
    lfp.v1.LFPArtifactDetectionSelection().insert1(lfp_artifact_s_key, skip_duplicates=True)

    # populate artifact detection 
    lfp.v1.LFPArtifactDetection().populate(lfp_artifact_s_key)
    artifact_removed_interval_list_name = (lfp.v1.LFPArtifactRemovedIntervalList() & lfp_artifact_s_key).fetch1('artifact_removed_interval_list_name')

    # 3) POPULATE POSITION
    print('populating position...')
    # define trodes parameters to use
    trodes_pos_params_name = 'default'

    # pair nwb_file_name, interval_list, params into trodes pos selection
    pos_s_key = {
        "nwb_file_name": nwb_file_name,
        "interval_list_name": pos_interval_list_name,
        "trodes_pos_params_name": trodes_pos_params_name,
    }
    sgp.v1.TrodesPosSelection.insert1(pos_s_key, skip_duplicates=True)
    pos_key = (sgp.v1.TrodesPosSelection() & pos_s_key).fetch1("KEY")

    # populate trodes pos v1 table using trodes key
    sgp.v1.TrodesPosV1.populate(pos_key)

    # get merge id corresponding to our inserted trodes_key
    pos_merge_key = (sgp.PositionOutput.merge_get_part(pos_key)).fetch1("KEY")
    pos_merge_id = pos_merge_key['merge_id']

    if pop_ripple:
        # 4) RIPPLE DETECTION
        print('populating ripples...')
        # populate ripple LFP band using artifact removed interval list
        # select lfp band parameters
        lfp_sampling_rate = LFPOutput.merge_get_parent(
            {"merge_id": lfp_merge_id}
        ).fetch1("lfp_sampling_rate")

        ripple_filter_name = 'Ripple 100-250 Hz'

        if rip_artifact_removed:
            rip_interval_list_name = artifact_removed_interval_list_name
        else:
            rip_interval_list_name = interval_list_name

        lfp_band_s_key = {
            'lfp_merge_id': lfp_merge_id,
            'filter_name': ripple_filter_name,
            'filter_sampling_rate': lfp_sampling_rate,
            'target_interval_list_name': rip_interval_list_name,
            'lfp_band_sampling_rate': lfp_sampling_rate,
            'nwb_file_name': LFPOutput.merge_get_parent({"merge_id": lfp_merge_id}).fetch1("nwb_file_name"),
        }

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
            ripple_elecs_mask = (electrodes_df['region_name'] == 'ca1')
            ripple_electrode_list = electrodes_df.loc[ripple_elecs_mask, 'electrode_id'].values
            ripple_ref_electrode_list = electrodes_df.loc[ripple_elecs_mask, 'val_ref'].values.astype('int')

            # cross check with the electrodes used in the lfp_electrode_group_name used to process the lfp ('good_single_elecs')
            lfp_elecs = (lfp.lfp_electrode.LFPElectrodeGroup.LFPElectrode() & 
            {'nwb_file_name': nwb_file_name, 'lfp_electrode_group_name': 'good_single_elecs'}).fetch('electrode_id')
            if not np.all([elec in lfp_elecs for elec in ripple_electrode_list]):
                raise ValueError('not all of the electrodes selected for ripple detection have been processed in the lfp filtering step')

            LFPBandSelection().set_lfp_band_electrodes(
                nwb_file_name=lfp_band_s_key['nwb_file_name'],
                lfp_merge_id=lfp_merge_id,
                electrode_list=ripple_electrode_list,
                filter_name=ripple_filter_name,
                interval_list_name=rip_interval_list_name,
                reference_electrode_list=ripple_ref_electrode_list,
                lfp_band_sampling_rate=lfp_sampling_rate,    
            )

        # Populate LFPBandV1
        if not (LFPBandV1 & lfp_band_s_key):
            logger.info(f"Populating LFPBandV1 for {ripple_filter_name}...")
            LFPBandV1().populate(
                lfp_band_s_key,
            )
        else:
            logger.info(f"LFPBandV1 already populated for {ripple_filter_name}.")

        lfp_band_key = (LFPBandV1 & lfp_band_s_key).fetch1("KEY")

        # # NORMAL RIPPLE TIMES V1 PIPELINE
        # # select ripple detection parameters
        # electrodes_df, val_can_refs = validate_references(nwb_file_name, is_copy=True)
        # electrodes_df = electrodes_df[electrodes_df['bad_channel'] == 'False']
        # electrodes_df = pd.DataFrame(
        #     [
        #         electrodes_df[electrodes_df['electrode_group_name'] == i].iloc[0]
        #         for i in np.unique(electrodes_df['electrode_group_name'].values)
        #     ]
        # )
        # ca1_ripple_elecs = electrodes_df.loc[electrodes_df['region_name'] == 'ca1', 'electrode_id'].values

        # sgr.RippleLFPSelection.set_lfp_electrodes(
        #     lfp_band_key,
        #     electrode_list=ca1_ripple_elecs,
        #     group_name='ca1_ripple_elecs'
        # )
        # rip_s_key = (sgr.RippleLFPSelection & lfp_band_key).fetch1("KEY")

        # # populate ripple detection
        # ripple_param_name = ripple_param_name

        # rip_key = {
        #     "ripple_param_name": ripple_param_name,
        #     **rip_s_key,
        #     "pos_merge_id": pos_merge_id,
        # }
        # sgr.RippleTimesV1().populate(rip_key)

        # print('completed ripple detection pipeline')


def grouped_date_ripple_pipeline(nwb_file_name, ripple_param_name):
        # CUSTOM GROUPED RIPPLE TIMES PIPELINE

        from gl_spyglass.custom_spyglass_tables.grouped_ripple import LFPBandGroup, RippleLFPSelection, RippleParameters, RippleTimesGroup

        lfp_band_filter_name = 'Ripple 100-250 Hz'

        # insert lfp band group based on date --> will calculate and store the baselines and deviations based on all mobile times across epochs
        if len(LFPBandGroup() & {'band_group_name': nwb_file_name, 'band_group_filter_name': lfp_band_filter_name}) == 0:
            LFPBandGroup().insert_from_date(nwb_file_name, lfp_band_filter_name)

        # create group key
        lfp_band_group_key = {'band_group_name': nwb_file_name, 'band_group_filter_name': lfp_band_filter_name}

        # select ripple electrodes
        electrodes_df, val_can_refs = validate_references(nwb_file_name, is_copy=True)
        electrodes_df = electrodes_df[electrodes_df['bad_channel'] == 'False']
        electrodes_df = pd.DataFrame(
            [
                electrodes_df[electrodes_df['electrode_group_name'] == i].iloc[0]
                for i in np.unique(electrodes_df['electrode_group_name'].values)
            ]
        )
        ca1_ripple_elecs = electrodes_df.loc[electrodes_df['region_name'] == 'ca1', 'electrode_id'].values

        # set the ripple lfp electrodes group
        elec_group_name = 'ca1_ripple_elecs'
        RippleLFPSelection().set_lfp_electrodes(
            lfp_band_group_key,
            electrode_list=ca1_ripple_elecs,
            group_name=elec_group_name, 
        )

        # create rip key
        rip_key = {
            'ripple_param_name': ripple_param_name,
            'group_name': elec_group_name,
            **lfp_band_group_key,
        }

        # detect ripples across the lfp band group
        RippleTimesGroup().populate(rip_key)