import sys
from itertools import starmap

from spyglass.utils import logger
from spyglass.utils.dj_helper_fn import NonDaemonPool  # For parallel processing
import spyglass.spikesorting.v1 as sgs
from spyglass.position import PositionOutput
import spyglass.position as sgp

from gl_spyglass.utils.coincident_spike_functions import *
from gl_spyglass.utils.common_neural_functions import validate_references
from gl_spyglass.utils.parallel_spike_functions.clusterless import *
from gl_spyglass.utils.parallel_spike_functions.spikesorting import _process_single_sort_group
from gl_spyglass.utils.parallel_spike_functions.waveform_feature_extraction import *
from gl_spyglass.utils.interval_functions import interval_list_during_trials, insert_mobile_times_interval

def parallel_process(sort_group_ids, process_args_list, single_sort_group_fn, use_parallel, max_processes):

    if not use_parallel:
        effective_processes = 1
        logger.info(f"Running {single_sort_group_fn} pipeline sequentially...")
        results = list(starmap(single_sort_group_fn, process_args_list))
    else:
        effective_processes = min(max_processes, len(sort_group_ids))
        logger.info(
            f"Running {single_sort_group_fn} pipeline in parallel with up to {effective_processes} processes..."
        )
        try:
            with NonDaemonPool(processes=effective_processes, maxtasksperchild=1) as pool:
                results = list(
                    pool.starmap(single_sort_group_fn, process_args_list)
                )
        except Exception as e:
            logger.error(f"Parallel processing failed: {e}", exc_info=True)
            logger.warning("Attempting sequential processing as fallback...")
            try:
                results = list(
                    starmap(single_sort_group_fn, process_args_list)
                )
                effective_processes = 1
            except Exception as seq_e:
                logger.critical(
                    f"Sequential fallback failed after parallel error: {seq_e}",
                    exc_info=True,
                )
                results = [False] * len(sort_group_ids)
                effective_processes = 0

def single_interval_decoding_pipeline(
        nwb_file_name, 
        sort_interval_name, 
        team_name, 
        interval_type,
        preproc_param_name='default',
        artifact_param_name='amp_3000_0.5_prop',
        sorter_name='clusterless_thresholder',
        sorting_param_name='default_clusterless',
        waveform_param_name='default_whitened',
        metric_param_name='franklab_default',
        metric_curation_param_name='default',
        run_metric_curation=True,
        apply_curation_merges=True,
        description='Standard pipeline run',
        skip_duplicates=True,
        reserve_jobs=True,
        max_processes=20,
        kwargs=None,
        features_param_name='amplitude',
    ):
    '''
    interval_type: single or combined (useful for sleep decoding)
    '''

    if interval_type == 'combined':
        # Insert combined interval list into sgc.IntervalList if it doesn't already exist
        run_interval_list_name = sort_interval_name[:5]  # currently hard coded to be the first 5 characters
        sleep_interval_list_name = sort_interval_name[-5:]  # currently hard coded to be the last 5 characters
        
        run_valid_times = (sgc.IntervalList() & {'nwb_file_name': nwb_file_name, 'interval_list_name': run_interval_list_name}).fetch1('valid_times')
        sleep_valid_times = (sgc.IntervalList() & {'nwb_file_name': nwb_file_name, 'interval_list_name': sleep_interval_list_name}).fetch1('valid_times')  
        combined_valid_times = np.concatenate([run_valid_times, sleep_valid_times])

        sgc.IntervalList().insert1(
            {
                "nwb_file_name": nwb_file_name,
                "interval_list_name": sort_interval_name,
                "valid_times": combined_valid_times,
            },
            skip_duplicates=True,
        )
        print(f'inserted {sort_interval_name}')


    # 1) SET SORT GROUPS
    # get validated references
    electrodes_df, val_can_refs = validate_references(nwb_file_name, is_copy=True)
    val_ref_dict = dict(electrodes_df[['electrode_group_name', 'val_ref']].groupby('electrode_group_name').mean().reset_index().astype(int).astype({'electrode_group_name': str}).values)

    # populate SortGroup with validated references
    if not (sgs.SortGroup & {"nwb_file_name": nwb_file_name}):
        try:
            sgs.SortGroup.set_group_by_shank(nwb_file_name=nwb_file_name, references=val_ref_dict, omit_unitrode=False)
            if not (sgs.SortGroup & {"nwb_file_name": nwb_file_name}):
                raise ValueError(
                    f"Failed to create/find SortGroups for {nwb_file_name} after set_group_by_shank."
                )
            logger.info(f"Successfully created SortGroups for {nwb_file_name}.")
        except Exception as e:
            raise ValueError(
                f"Failed to create SortGroups for {nwb_file_name}: {e}"
            ) from e
    else:
        logger.info(f"Sort groups already exist for {nwb_file_name}.")

    # only process the ca1 electrodes to save time
    ca1_elecs = electrodes_df.loc[(electrodes_df['bad_channel'] == 'False') & (electrodes_df['region_name'] == 'ca1'), 'electrode_id'].values
    ca1_sort_group_ids = [(sgs.SortGroup.SortGroupElectrode() & {'nwb_file_name': nwb_file_name, 'electrode_id': ca1_elec}).fetch1('sort_group_id') for ca1_elec in ca1_elecs]

    sort_group_ids = np.unique(ca1_sort_group_ids)

    # Final check
    if len(sort_group_ids) == 0:
        raise ValueError(
            f"No processable sort groups identified for nwb_file_name '{nwb_file_name}' "
        )

    logger.info(
        f"Identified {len(sort_group_ids)} sort group(s) to process: {sort_group_ids.tolist()}"
    )

    # 2) POPULATE SpikeSortingRecording
    # --- Prepare arguments for each sort group ---
    process_args_list: List[tuple] = []
    for sort_group_id in sort_group_ids:
        process_args_list.append(
            (
                nwb_file_name,
                sort_interval_name,
                int(sort_group_id),
                team_name,
                preproc_param_name,
                artifact_param_name,
                sorter_name,
                sorting_param_name,
                waveform_param_name,
                metric_param_name,
                metric_curation_param_name,
                run_metric_curation,
                apply_curation_merges,
                description,
                skip_duplicates,
                reserve_jobs,
                kwargs,
            )
        )
    use_parallel = (
        max_processes is not None
        and max_processes > 0
        and len(sort_group_ids) > 1
    )
    # TODO: check that this works
    parallel_process(sort_group_ids, process_args_list, _process_single_sort_group, use_parallel, max_processes)

    # 3) POPULATE UnitWaveformFeatures
    recording_ids = (sgs.SpikeSortingRecordingSelection() & {
                    'nwb_file_name': nwb_file_name, 
                    'interval_list_name': sort_interval_name,
                    'preproc_param_name': 'default',
                }).fetch('recording_id')
    
    spikesorting_merge_ids = [((SpikeSortingOutput.CurationV1 * sgs.SpikeSortingSelection) & {
        'nwb_file_name': nwb_file_name,
        'recording_id': recording_id,
        'sorter': sorter_name,
        'sorter_param_name': sorting_param_name,
    }).fetch1('merge_id') for recording_id in recording_ids]

    process_args_list: List[tuple] = []
    for spikesorting_merge_id in spikesorting_merge_ids:
        process_args_list.append(
            (
                spikesorting_merge_id,
                features_param_name,
                sort_interval_name,
            )
        )
    use_parallel = (
        max_processes is not None
        and max_processes > 0
        and len(spikesorting_merge_ids) > 1
    )
    parallel_process(spikesorting_merge_ids, process_args_list, populate_spyglass_waveform_features_v1, use_parallel, max_processes)

    # 4) POPULATE UnitWaveformFeaturesGroup
    waveform_s_keys = [
        {
            'spikesorting_merge_id': spikesorting_merge_id,
            'features_param_name': features_param_name,
        }
        for spikesorting_merge_id in spikesorting_merge_ids
    ]

    wf_group_name = f'ca1_waveforms {sort_interval_name}'

    UnitWaveformFeaturesGroup().create_group(
        nwb_file_name=nwb_file_name,
        group_name=wf_group_name,
        keys=waveform_s_keys,
    )

    if interval_type == 'single':
        # 5) POPULATE PositionOutput
        interval_list_name = sort_interval_name
        epoch = int(interval_list_name[:2])
        pos_interval_list_name = (sgc.IntervalList() & {'nwb_file_name': nwb_file_name, 'pipeline': 'position'}).fetch('interval_list_name')[epoch - 1]

        # define trodes parameters to use
        trodes_pos_params_name = 'default_decoding'  # NOTE: default_decoding upsamples (as opposed to just default which doesn't)

        # pair nwb_file_name, interval_list, params into trodes pos selection
        trodes_s_key = {
            'nwb_file_name': nwb_file_name,
            'interval_list_name': pos_interval_list_name,
            'trodes_pos_params_name': trodes_pos_params_name,
        }
        sgp.v1.TrodesPosSelection.insert1(trodes_s_key, skip_duplicates=True)
        trodes_key = (sgp.v1.TrodesPosSelection() & trodes_s_key).fetch1("KEY")

        # populate trodes pos v1 table using trodes key
        sgp.v1.TrodesPosV1.populate(trodes_key)

        # get merge id corresponding to our inserted trodes_key
        merge_key = (sgp.PositionOutput.merge_get_part(trodes_key)).fetch1("KEY")

        # 6) POPULATE PositionGroup
        position_merge_ids = (
            PositionOutput.TrodesPosV1
            & {
                "nwb_file_name": nwb_file_name,
                "interval_list_name": pos_interval_list_name,
                "trodes_pos_params_name": trodes_pos_params_name,
            }
        ).fetch("merge_id")

        pos_group_name = f'{interval_list_name} decoding'

        PositionGroup().create_group(
            nwb_file_name=nwb_file_name,
            group_name=pos_group_name,
            keys=[{"pos_merge_id": merge_id} for merge_id in position_merge_ids],
        )

        # 7) Find the correct encoding interval list (mobile, without coincident spikes, during trials)
        #    and correct decoding interval list (without coincident spikes, during trials)
        # mobile        
        mobile_interval_list_name = insert_mobile_times_interval(nwb_file_name, pos_interval_list_name, trodes_pos_params_name, speed_thresh=4, time_thresh=1)

        waveform_s_keys = [
            {
                'spikesorting_merge_id': spikesorting_merge_id,
                'features_param_name': features_param_name,
            }
            for spikesorting_merge_id in spikesorting_merge_ids
        ]

        # coincident spike times
        spike_closeness_threshold = 0.00005
        max_coincident_fraction = 0.5
        removal_window_s = 0.001

        filt_mobile_interval_list_name = f'{mobile_interval_list_name} coincident_spikes_removed_times rem_{removal_window_s} close_{spike_closeness_threshold} frac_{max_coincident_fraction}'
        filt_pos_interval_list_name = f'{pos_interval_list_name} coincident_spikes_removed_times rem_{removal_window_s} close_{spike_closeness_threshold} frac_{max_coincident_fraction}'
        filt_mobile_complete = len(sgc.IntervalList() & {'nwb_file_name': nwb_file_name, 'interval_list_name': filt_mobile_interval_list_name}) != 0
        filt_pos_complete = len(sgc.IntervalList() & {'nwb_file_name': nwb_file_name, 'interval_list_name': filt_pos_interval_list_name}) != 0
        if filt_mobile_complete & filt_pos_complete:
            print(f'{filt_mobile_interval_list_name} & {filt_pos_interval_list_name} already in sgc.IntervalList() for this session')
        else:
            print('loading in spike times...')
            spike_times, spike_waveform_features = (
                UnitWaveformFeatures & waveform_s_keys
            ).fetch_data()

            filtered_spike_times, filtered_time_bin_ind, med_coinc_spike_times = detect_coincident_spikes(spike_times, spike_closeness_threshold, max_coincident_fraction)

            filt_mobile_interval_list_name = insert_coincident_spike_interval_list(
                med_coinc_spike_times, 
                removal_window_s, 
                nwb_file_name,
                mobile_interval_list_name,
                spike_closeness_threshold,
                max_coincident_fraction,
            )

            filt_pos_interval_list_name = insert_coincident_spike_interval_list(
                med_coinc_spike_times, 
                removal_window_s, 
                nwb_file_name,
                pos_interval_list_name,
                spike_closeness_threshold,
                max_coincident_fraction,
            )

        # during trials
        epoch = int(interval_list_name[:2])

        during_trials_filt_mobile_interval = interval_list_during_trials(nwb_file_name, filt_mobile_interval_list_name, epoch)
        during_trials_filt_pos_interval = interval_list_during_trials(nwb_file_name, filt_pos_interval_list_name, epoch)

        if (nwb_file_name == 'teddy20250620_.nwb') | (nwb_file_name == 'teddy20250626_.nwb'):
            wf_group_name = f'ca1_waveforms without outliers {sort_interval_name}'
        else:
            wf_group_name = f'ca1_waveforms {sort_interval_name}'
        pos_group_name = f'{interval_list_name} decoding'
        decoding_param_name = 'contfrag_clusterless_placebin3_100chunks_blocksize100__nocache'
        encoding_interval = during_trials_filt_mobile_interval
        decoding_interval = during_trials_filt_pos_interval 
        estimate_decoding_params = True
    
    # TODO: check if the combined version works
    if interval_type == 'combined':

        # 5) POPULATE PositionOutput
        run_interval_list_name = sort_interval_name[:5]  # currently hard coded to be the first 5 characters
        sleep_interval_list_name = sort_interval_name[-5:]  # currently hard coded to be the last 5 characters
        run_epoch = int(run_interval_list_name[:2])
        sleep_epoch = int(sleep_interval_list_name[:2])
        run_pos_interval_list_name = (sgc.IntervalList() & {'nwb_file_name': nwb_file_name, 'pipeline': 'position'}).fetch('interval_list_name')[run_epoch - 1]
        sleep_pos_interval_list_name = (sgc.IntervalList() & {'nwb_file_name': nwb_file_name, 'pipeline': 'position'}).fetch('interval_list_name')[sleep_epoch - 1]

        # get the merge ids
        trodes_pos_params_name = 'default_decoding'  # NOTE: default_decoding upsamples (as opposed to just default which doesn't)
        
        # get run merge id
        # pair nwb_file_name, interval_list, params into trodes pos selection
        run_trodes_s_key = {
            'nwb_file_name': nwb_file_name,
            'interval_list_name': run_pos_interval_list_name,
            'trodes_pos_params_name': trodes_pos_params_name,
        }
        sgp.v1.TrodesPosSelection.insert1(run_trodes_s_key, skip_duplicates=True)
        run_trodes_key = (sgp.v1.TrodesPosSelection() & run_trodes_s_key).fetch1("KEY")

        # populate trodes pos v1 table using trodes key
        sgp.v1.TrodesPosV1.populate(run_trodes_key)

        # get merge id corresponding to our inserted trodes_key
        run_pos_merge_id = (sgp.PositionOutput.merge_get_part(run_trodes_key)).fetch1("merge_id")

        # get sleep merge id
        # pair nwb_file_name, interval_list, params into trodes pos selection
        sleep_trodes_s_key = {
            'nwb_file_name': nwb_file_name,
            'interval_list_name': sleep_pos_interval_list_name,
            'trodes_pos_params_name': trodes_pos_params_name,
        }
        sgp.v1.TrodesPosSelection.insert1(sleep_trodes_s_key, skip_duplicates=True)
        sleep_trodes_key = (sgp.v1.TrodesPosSelection() & sleep_trodes_s_key).fetch1("KEY")

        # populate trodes pos v1 table using trodes key
        sgp.v1.TrodesPosV1.populate(sleep_trodes_key)

        # get merge id corresponding to our inserted trodes_key
        sleep_pos_merge_id = (sgp.PositionOutput.merge_get_part(sleep_trodes_key)).fetch1("merge_id")

        # 6) POPULATE PositionGroup
        position_merge_ids = [run_pos_merge_id, sleep_pos_merge_id]

        pos_group_name = f'{run_interval_list_name} encoding {sleep_interval_list_name} decoding'

        PositionGroup().create_group(
            nwb_file_name=nwb_file_name,
            group_name=pos_group_name,
            keys=[{"pos_merge_id": merge_id} for merge_id in position_merge_ids],
        )

        # 7) Find the correct encoding interval list (mobile, without coincident spikes, during trials)
        #    and correct decoding interval list (without coincident spikes, during trials)
        spike_closeness_threshold = 0.00005
        max_coincident_fraction = 0.5
        removal_window_s = 0.001

        # run intervals
        run_mobile_interval_list_name = insert_mobile_times_interval(nwb_file_name, run_pos_interval_list_name, trodes_pos_params_name, speed_thresh=4, time_thresh=1)

        # check that the coincident spike - filtered run intervals exist and insert them if they don't
        filt_run_mobile_interval_list_name = f'{run_mobile_interval_list_name} coincident_spikes_removed_times rem_{removal_window_s} close_{spike_closeness_threshold} frac_{max_coincident_fraction}'
        filt_run_pos_interval_list_name = f'{run_pos_interval_list_name} coincident_spikes_removed_times rem_{removal_window_s} close_{spike_closeness_threshold} frac_{max_coincident_fraction}'
        filt_run_mobile_complete = len(sgc.IntervalList() & {'nwb_file_name': nwb_file_name, 'interval_list_name': filt_run_mobile_interval_list_name}) != 0
        filt_run_pos_complete = len(sgc.IntervalList() & {'nwb_file_name': nwb_file_name, 'interval_list_name': filt_run_pos_interval_list_name}) != 0

        # sleep intervals
        sleep_mobile_interval_list_name = insert_mobile_times_interval(nwb_file_name, sleep_pos_interval_list_name, trodes_pos_params_name, speed_thresh=4, time_thresh=1)

        # check that the coincident spike - filtered sleep intervals exist and insert them if they don't
        filt_sleep_mobile_interval_list_name = f'{sleep_mobile_interval_list_name} coincident_spikes_removed_times rem_{removal_window_s} close_{spike_closeness_threshold} frac_{max_coincident_fraction}'
        filt_sleep_pos_interval_list_name = f'{sleep_pos_interval_list_name} coincident_spikes_removed_times rem_{removal_window_s} close_{spike_closeness_threshold} frac_{max_coincident_fraction}'
        filt_sleep_mobile_complete = len(sgc.IntervalList() & {'nwb_file_name': nwb_file_name, 'interval_list_name': filt_sleep_mobile_interval_list_name}) != 0
        filt_sleep_pos_complete = len(sgc.IntervalList() & {'nwb_file_name': nwb_file_name, 'interval_list_name': filt_sleep_pos_interval_list_name}) != 0

        # check that filtered and during trials run intervals exist and insert them if they don't
        if not filt_sleep_pos_complete:
            print('loading in spike times...')
            
            recording_ids = (sgs.SpikeSortingRecordingSelection() & {
                'nwb_file_name': nwb_file_name, 
                'interval_list_name': sleep_pos_interval_list_name,
                'preproc_param_name': 'default',
            }).fetch('recording_id')
    
            spikesorting_merge_ids = [((SpikeSortingOutput.CurationV1 * sgs.SpikeSortingSelection) & {
                'nwb_file_name': nwb_file_name,
                'recording_id': recording_id,
                'sorter': sorter_name,
                'sorter_param_name': sorting_param_name,
            }).fetch1('merge_id') for recording_id in recording_ids]

            waveform_s_keys = [
                {
                    'spikesorting_merge_id': spikesorting_merge_id,
                    'features_param_name': features_param_name,
                }
                for spikesorting_merge_id in spikesorting_merge_ids
            ]
            
            spike_times, _ = (
                UnitWaveformFeatures & waveform_s_keys
            ).fetch_data()

            _, _, med_coinc_spike_times = detect_coincident_spikes(spike_times, spike_closeness_threshold, max_coincident_fraction)

            filt_sleep_pos_interval_list_name = insert_coincident_spike_interval_list(
                med_coinc_spike_times, 
                removal_window_s, 
                nwb_file_name,
                sleep_pos_interval_list_name,
                spike_closeness_threshold,
                max_coincident_fraction,
            )

        # during trials
        during_trials_filt_run_mobile_interval_list_name = interval_list_during_trials(nwb_file_name, filt_run_mobile_interval_list_name, run_epoch)

        # IF USING COMBINED ENCODING INTERVAL: insert combined encoding interval
        run_mobile_valid_times = (sgc.IntervalList() & {'nwb_file_name': nwb_file_name, 'interval_list_name': during_trials_filt_run_mobile_interval_list_name}).fetch1('valid_times')
        sleep_mobile_valid_times = (sgc.IntervalList() & {'nwb_file_name': nwb_file_name, 'interval_list_name': filt_sleep_mobile_interval_list_name}).fetch1('valid_times')  
        combined_encoding_times = np.concatenate([run_mobile_valid_times, sleep_mobile_valid_times])
        combined_encoding_interval_name = f'pos 3 and pos 4 mobile times rem_{removal_window_s} close_{spike_closeness_threshold} frac_{max_coincident_fraction}'

        sgc.IntervalList().insert1(
            {
                "nwb_file_name": nwb_file_name,
                "interval_list_name": combined_encoding_interval_name,
                "valid_times": combined_encoding_times,
            },
            skip_duplicates=True,
        )
        print(f'inserted {combined_encoding_interval_name}')

        pos_group_name = f'04_r2 encoding 05_s3 decoding'
        decoding_param_name = 'contfrag_clusterless_placebin3_100chunks_blocksize100__nocache'
        wf_group_name = f'ca1_waveforms {sort_interval_name}'
        encoding_interval = combined_encoding_interval_name
        decoding_interval = filt_sleep_pos_interval_list_name
        estimate_decoding_params = False 
    
    # 8) POPULATE ClusterlessDecodingSelection
    # insert a ClusterlessDecodingSelection entry into the database
    selection_key = {
        "waveform_features_group_name": wf_group_name,
        "position_group_name": pos_group_name,
        "decoding_param_name": decoding_param_name,
        "nwb_file_name": nwb_file_name,
        "encoding_interval": encoding_interval,
        "decoding_interval": decoding_interval,
        "estimate_decoding_params": estimate_decoding_params,
    }

    ClusterlessDecodingSelection.insert1(
        selection_key,
        skip_duplicates=True,
    )

    # run with gpu
    ClusterlessDecodingV1.populate(selection_key)
