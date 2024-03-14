import datajoint as dj
from spyglass.utils import logger
from spyglass.common.common_task import TaskEpoch
from spyglass.common.common_behav import StateScriptFile
from spyglass.common.common_interval import IntervalList
from spyglass.utils.nwb_helper_fn import get_nwb_file
from spyglass.common.common_nwbfile import Nwbfile
from spyglass.common.common_nwbfile import AnalysisNwbfile

from utils.parse_trials_helper import V8TrialParser
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

schema = dj.schema("TrialsInfo")


@schema
class TrialInfo(dj.Computed):
    definition = """
    -> StateScriptFile
    ---
    -> AnalysisNwbfile                  # analysis file containing trial-by-trial analysis
    trial_info_object_id : varchar(40)  # the NWB object ID for loading this object from the file
    parser: varchar(100)                # type of parser used to interpret statescript log
    descriptors = null : blob           # global descriptors for task
    """
    

    def make(self, key):
        '''
        Parses the given StateScriptFile into landmark behavioral events
        and saves them as an NWB analysis file.
        '''

        nwb_file_name = key["nwb_file_name"]
        nwb_file_abspath = Nwbfile().get_abs_path(nwb_file_name)
        nwbf = get_nwb_file(nwb_file_abspath)

        # get homedio start timestamp to calculate offset
        behav_events = nwbf.processing.get("behavior").data_interfaces['behavioral_events']
        diomap = {} # map of event name to channel
        for (name,series) in behav_events.time_series.items():
            diomap[name] = series.description 
        # get timestamps of all homedio events in the session
        homediotimesall = np.asarray(behav_events.time_series['homebeam'].timestamps)

        # extract time range of TaskEpoch (use dataframe for filtering)
        epoch_valid_times = (pd.DataFrame(
            IntervalList & {"nwb_file_name": nwb_file_name})
            .set_index("interval_list_name")
            .filter(regex=r"^[0-9]", axis=0)
            .valid_times
        )
        # get timestamp of 1st homewell trigger the given epoch
        epoch_name = (TaskEpoch & {"nwb_file_name": key["nwb_file_name"], "epoch": key["epoch"]}).fetch1("interval_list_name")
        start_time, end_time = epoch_valid_times[epoch_name].squeeze()
        home_times = homediotimesall[np.where((homediotimesall >= start_time) & (homediotimesall < end_time))]
        if home_times.size > 0:
            first_home_time = home_times[0]
        else:
            logger.info(f"No home dio events detected for epoch {epoch_name}")
            return

        # get and parse trials from each statescript

        associated_files = nwbf.processing.get("associated_files")
        if associated_files is None:
            logger.info(f"No associated files found for {epoch_name}")
            return
        
        # get and parse trials from each statescript
        file_id = (StateScriptFile & key).fetch1("file_object_id")
        sc = nwbf.objects[file_id]

        # parse statescript according to the task type (currently there's only the V8, but future variants will be added here)
        task_name = (TaskEpoch & {"nwb_file_name": key["nwb_file_name"], "epoch": key["epoch"]}).fetch1("task_name")
        if task_name == 'Eight arm flexible spatial task':
            key['parser'] = 'V8_delay'
            parser = V8TrialParser(sc.content, diomap, first_home_time, key)
        elif task_name == 'Sleep':
            logger.info(f"Skipping sleep epoch: {epoch_name}")
            return 
        else:
            logger.info(f"Skipping unsupported task type: {task_name}")
            return

        trials_df = parser.parse_trials()

        # set trial descriptors using statescript

        # Insert into analysis nwb file
        nwb_analysis_file = AnalysisNwbfile()
        key["analysis_file_name"] = nwb_analysis_file.create(key['nwb_file_name'])
        key["trial_info_object_id"] = nwb_analysis_file.add_nwb_object(
            analysis_file_name=key["analysis_file_name"],
            nwb_object=trials_df,
        )
        nwb_analysis_file.add(
            nwb_file_name=nwb_file_name,
            analysis_file_name=key["analysis_file_name"],
        )
        self.insert1(key)
