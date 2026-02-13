import time
from itertools import starmap
from typing import Any, Dict, List, Optional, Union

import datajoint as dj
import numpy as np

from spyglass.common import (
    ElectrodeGroup,
    IntervalList,
    LabMember,
    LabTeam,
    Nwbfile,
    Probe,
)
from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
from spyglass.spikesorting.v1 import (
    ArtifactDetection,
    ArtifactDetectionParameters,
    ArtifactDetectionSelection,
    CurationV1,
    MetricCuration,
    MetricCurationParameters,
    MetricCurationSelection,
    MetricParameters,
    SortGroup,
    SpikeSorterParameters,
    SpikeSorting,
    SpikeSortingPreprocessingParameters,
    SpikeSortingRecording,
    SpikeSortingRecordingSelection,
    SpikeSortingSelection,
    WaveformParameters,
)
from spyglass.utils import logger
from spyglass.utils.dj_helper_fn import NonDaemonPool  # For parallel processing
from spyglass.utils.nwb_helper_fn import close_nwb_files

# --- Constants ---
INITIAL_CURATION_ID = 0
PARENT_CURATION_ID = -1

# --- Helper Function for DataJoint Population Pattern ---
def _ensure_selection_and_populate(
    selection_table: dj.Table,
    computed_table: dj.Table,
    selection_key: Dict[str, Any],
    description: str,
    reserve_jobs: bool = True,
    populate_kwargs: Optional[Dict] = None,
) -> Optional[Dict[str, Any]]:
    """
    Ensures a selection entry exists (by inserting or fetching) and populates
    the corresponding computed table if the entry was newly inserted.

    Handles the return signature of insert_selection (list for existing,
    dict for new) based on the user-provided implementation.

    Parameters
    ----------
    selection_table : dj.Table
        The DataJoint Selection table class (e.g., SpikeSortingRecordingSelection).
    computed_table : dj.Table
        The DataJoint Computed table class linked to the selection table.
    selection_key : Dict[str, Any]
        The key defining the selection (used as input for insert_selection).
    description : str
        A description of the step for logging.
    reserve_jobs : bool, optional
        Passed to `populate`. Defaults to True.
    populate_kwargs : Optional[Dict], optional
        Additional keyword arguments for the `populate` call. Defaults to None.

    Returns
    -------
    Optional[Dict[str, Any]]
        The primary key dictionary of the selection entry if successful,
        otherwise None.
    """
    if populate_kwargs is None:
        populate_kwargs = {}

    logger.debug(
        f"Ensuring selection for {description} with key: {selection_key}"
    )
    final_key: Optional[Dict[str, Any]] = None

    try:
        inserted_or_fetched_key: Union[Dict, List[Dict]] = (
            selection_table.insert_selection(selection_key)
        )

        if isinstance(inserted_or_fetched_key, list):
            if not inserted_or_fetched_key:
                logger.error(
                    f"insert_selection found existing entries but returned an empty list for {description}."
                )
                return None
            if len(inserted_or_fetched_key) > 1:
                raise ValueError(
                    f"Multiple entries found for {description}: {inserted_or_fetched_key}"
                )
            final_key = inserted_or_fetched_key[0]
            logger.info(
                f"Using existing selection entry for {description}: {final_key}"
            )

        elif isinstance(inserted_or_fetched_key, dict):
            final_key = inserted_or_fetched_key
            logger.info(
                f"New selection entry inserted for {description}: {final_key}"
            )
        else:
            logger.error(
                f"Unexpected return type from insert_selection for {description}: {type(inserted_or_fetched_key)}"
            )
            return None

        if final_key:
            try:
                computed_table.populate(
                    final_key, reserve_jobs=reserve_jobs, **populate_kwargs
                )
            except dj.errors.DataJointError as e:
                logger.warning(
                    f"DataJointError checking computed table {computed_table.__name__} for {description}: {e}. Assuming population needed."
                )

            return final_key
        else:
            return None

    except dj.errors.DataJointError as e:
        logger.error(
            f"DataJointError during selection/population for {description}: {e}",
            exc_info=True,
        )
        return None
    except Exception as e:
        logger.error(
            f"Unexpected error during selection/population for {description}: {e}",
            exc_info=True,
        )
        return None


# --- Worker Function for Parallel Processing ---
def _process_single_sort_group(
    nwb_file_name: str,
    sort_interval_name: str,
    sort_group_id: int,
    team_name: str,
    preproc_param_name: str,
    artifact_param_name: str,
    sorter_name: str,
    sorting_param_name: str,
    waveform_param_name: str,
    metric_param_name: str,
    metric_curation_param_name: str,
    run_metric_curation: bool,
    apply_curation_merges: bool,
    base_curation_description: str,
    skip_duplicates: bool,
    reserve_jobs: bool,
    populate_kwargs: Dict,
) -> bool:
    """Processes a single sort group for the v1 pipeline (worker function)."""
    sg_description = (
        f"{nwb_file_name} | SG {sort_group_id} | Intvl {sort_interval_name}"
    )
    final_curation_key: Optional[Dict[str, Any]] = None

    try:
        # --- 1. Recording Selection and Population ---
        logger.info(f"---- Step 1: Recording | {sg_description} ----")
        recording_selection_key = {
            "nwb_file_name": nwb_file_name,
            "sort_group_id": sort_group_id,
            "interval_list_name": sort_interval_name,
            "preproc_param_name": preproc_param_name,
            "team_name": team_name,
        }
        recording_id_dict = _ensure_selection_and_populate(
            SpikeSortingRecordingSelection,
            SpikeSortingRecording,
            recording_selection_key,
            f"Recording | {sg_description}",
            reserve_jobs,
            populate_kwargs,
        )
        if not recording_id_dict:
            logger.error(f"Recording step failed for {sg_description}.")
            return False

        # # --- 2. Artifact Detection Selection and Population (optional) ---
        # logger.info(f"---- Step 2: Artifact Detection | {sg_description} ----")
        # artifact_selection_key = {
        #     "recording_id": recording_id_dict["recording_id"],
        #     "artifact_param_name": artifact_param_name,
        # }
        # artifact_id_dict = _ensure_selection_and_populate(
        #     ArtifactDetectionSelection,
        #     ArtifactDetection,
        #     artifact_selection_key,
        #     f"Artifact Detection | {sg_description}",
        #     reserve_jobs,
        #     populate_kwargs,
        # )
        # if not artifact_id_dict:
        #     logger.error(
        #         f"Artifact Detection step failed for {sg_description}."
        #     )
        #     return False

        # --- 3. Spike Sorting Selection and Population ---
        logger.info(f"---- Step 3: Spike Sorting | {sg_description} ----")
        sorting_selection_key = {
            "recording_id": recording_id_dict["recording_id"],
            "sorter": sorter_name,
            "sorter_param_name": sorting_param_name,
            "nwb_file_name": nwb_file_name,
            # "interval_list_name": str(artifact_id_dict["artifact_id"]),
            'interval_list_name': sort_interval_name,
        }
        sorting_id_dict = _ensure_selection_and_populate(
            SpikeSortingSelection,
            SpikeSorting,
            sorting_selection_key,
            f"Spike Sorting | {sg_description}",
            reserve_jobs,
            populate_kwargs,
        )
        if not sorting_id_dict:
            logger.error(f"Spike Sorting step failed for {sg_description}.")
            return False

        # --- 4. Initial Automatic Curation ---
        logger.info(
            f"---- Step 4: Initial Automatic Metric Curation | {sg_description} ----"
        )
        initial_curation_key_base = {
            "sorting_id": sorting_id_dict["sorting_id"],
            "curation_id": INITIAL_CURATION_ID,
        }
        initial_curation_key = None
        if CurationV1 & initial_curation_key_base:
            logger.warning(
                f"Initial curation already exists for {sg_description}, fetching key."
            )
            initial_curation_key = (
                CurationV1 & initial_curation_key_base
            ).fetch1("KEY")
        else:
            try:
                inserted: Union[Dict, List[Dict]] = CurationV1.insert_curation(
                    sorting_id=sorting_id_dict["sorting_id"],
                    parent_curation_id=PARENT_CURATION_ID,
                    description=f"Initial: {base_curation_description} (SG {sort_group_id})",
                )
                if not inserted:
                    logger.error(
                        f"CurationV1.insert_curation returned None/empty for initial curation for {sg_description}"
                    )
                    if CurationV1 & initial_curation_key_base:
                        initial_curation_key = (
                            CurationV1 & initial_curation_key_base
                        ).fetch1("KEY")
                    else:
                        return False
                elif isinstance(inserted, list):
                    initial_curation_key = inserted[0]
                elif isinstance(inserted, dict):
                    initial_curation_key = inserted

                if not (
                    initial_curation_key
                    and initial_curation_key["curation_id"]
                    == INITIAL_CURATION_ID
                ):
                    logger.error(
                        f"Initial curation key mismatch or not found after insertion attempt for {sg_description}"
                    )
                    return False
            except Exception as e:
                logger.error(
                    f"Failed to insert initial curation for {sg_description}: {e}",
                    exc_info=True,
                )
                return False
        final_curation_key = initial_curation_key

        # # --- 5. Metric-Based Curation (Optional) ---
        # if run_metric_curation:
        #     logger.info(f"---- Step 5: Metric Curation | {sg_description} ----")
        #     metric_selection_key = {
        #         **initial_curation_key,
        #         "waveform_param_name": waveform_param_name,
        #         "metric_param_name": metric_param_name,
        #         "metric_curation_param_name": metric_curation_param_name,
        #     }
        #     metric_curation_id_dict = _ensure_selection_and_populate(
        #         MetricCurationSelection,
        #         MetricCuration,
        #         metric_selection_key,
        #         f"Metric Curation Selection | {sg_description}",
        #         reserve_jobs,
        #         populate_kwargs,
        #     )
        #     if not metric_curation_id_dict:
        #         logger.error(
        #             f"Metric Curation Selection/Population step failed for {sg_description}."
        #         )
        #         return False

        #     if not (MetricCuration & metric_curation_id_dict):
        #         logger.error(
        #             f"Metric Curation table check failed after populate call for {sg_description} | Key: {metric_curation_id_dict}"
        #         )
        #         return False

        #     logger.info(
        #         f"---- Inserting Metric Curation Result into CurationV1 | {sg_description} ----"
        #     )
        #     metric_result_description = f"metric_curation_id: {metric_curation_id_dict['metric_curation_id']}"
        #     metric_curation_result_check_key = {
        #         "sorting_id": sorting_id_dict["sorting_id"],
        #         "parent_curation_id": initial_curation_key["curation_id"],
        #         "description": metric_result_description,
        #     }
        #     final_metric_curation_key = None
        #     if CurationV1 & metric_curation_result_check_key:
        #         logger.warning(
        #             f"Metric curation result already in CurationV1 for {sg_description}, fetching key."
        #         )
        #         final_metric_curation_key = (
        #             CurationV1 & metric_curation_result_check_key
        #         ).fetch1("KEY")
        #     else:
        #         try:
        #             inserted = CurationV1.insert_metric_curation(
        #                 metric_curation_id_dict,
        #                 apply_merge=apply_curation_merges,
        #             )
        #             if not inserted:
        #                 logger.error(
        #                     f"CurationV1.insert_metric_curation returned None/empty for {sg_description}"
        #                 )
        #                 if CurationV1 & metric_curation_result_check_key:
        #                     final_metric_curation_key = (
        #                         CurationV1 & metric_curation_result_check_key
        #                     ).fetch1("KEY")
        #                 else:
        #                     return False
        #             elif isinstance(inserted, list):
        #                 final_metric_curation_key = inserted[0]
        #             elif isinstance(inserted, dict):
        #                 final_metric_curation_key = inserted

        #             if not final_metric_curation_key:
        #                 logger.error(
        #                     f"Metric curation result key not obtained after insertion attempt for {sg_description}"
        #                 )
        #                 return False
        #         except Exception as e:
        #             logger.error(
        #                 f"Failed to insert metric curation result for {sg_description}: {e}",
        #                 exc_info=True,
        #             )
        #             return False
        #     final_curation_key = final_metric_curation_key

        # --- 6. Insert into Merge Table ---
        if final_curation_key is None:
            logger.error(
                f"Final curation key is None before Merge Table Insert for {sg_description}. Aborting."
            )
            return False

        logger.info(f"---- Step 6: Merge Table Insert | {sg_description} ----")
        logger.debug(f"Merge table insert key: {final_curation_key}")

        merge_part_table = SpikeSortingOutput.CurationV1()
        if not (merge_part_table & final_curation_key):
            try:
                SpikeSortingOutput.insert(
                    [final_curation_key],
                    part_name="CurationV1",
                    skip_duplicates=skip_duplicates,
                )
                logger.info(
                    f"Successfully inserted final curation into merge table for {sg_description}."
                )
            except Exception as e:
                logger.error(
                    f"Failed to insert into merge table for {sg_description}: {e}",
                    exc_info=True,
                )
                return False
        else:
            logger.warning(
                f"Final curation {final_curation_key} already in merge table part "
                f"{merge_part_table.table_name} for {sg_description}. Skipping merge insert."
            )

        logger.info(
            f"==== Successfully Completed Sort Group ID: {sort_group_id} ===="
        )

        # close all remaining open nwb files to save memory
        close_nwb_files()
        
        return True

    except dj.errors.DataJointError as e:
        logger.error(
            f"DataJoint Error processing Sort Group ID {sort_group_id}: {e}",
            exc_info=True,
        )
        return False
    except Exception as e:
        logger.error(
            f"General Error processing Sort Group ID {sort_group_id}: {e}",
            exc_info=True,
        )
        return False
    
