import datajoint as dj
from spyglass.utils import logger
import spyglass.common as sgc
from spyglass.common.common_behav import StateScriptFile
from spyglass.common.common_dio import DIOEvents
from spyglass.common.common_nwbfile import AnalysisNwbfile, Nwbfile
from spyglass.utils.nwb_helper_fn import get_nwb_file
from spyglass.utils.dj_mixin import SpyglassMixin
from gl_spyglass_tables.utils.parse_trials_helper import V8TrialParser
import numpy as np
import re

# TODO consider adding TrialInfoSelection table?
schema = dj.schema("trial_info")

@schema
class TrialInfo(SpyglassMixin, dj.Computed):
    definition = """
    -> StateScriptFile                  # log containing records of behavior during task
    -> DIOEvents                        # home poke dio event marking start of trials
    ---
    -> AnalysisNwbfile                  # analysis file containing trial-by-trial analysis
    trial_info_object_id : varchar(40)  # the NWB object ID for loading this object from the file
    parser: varchar(100)                # type of parser used to interpret statescript log
    descriptors = null : blob           # global descriptors for task
    """

    def make(self, key):
        """
        Parses the given StateScriptFile into landmark behavioral events
        and saves them as an NWB analysis file.
        """
        # ignore all DIO events that aren't homebeam to avoid repeats
        if key["dio_event_name"] != "homebeam":
            return

        # retrieve nwb file object
        nwb_file_abspath = Nwbfile().get_abs_path(key["nwb_file_name"])
        nwbf = get_nwb_file(nwb_file_abspath)
        
        # get first home poke time of the epoch for calculating trodes to ptp time offset
        nwb_file_name = key["nwb_file_name"]
        epoch_num = key["epoch"]
        home_times = get_beam_times(key, nwbf, "homebeam")
        if home_times.size == 0:
            logger.info(f"Skipping epoch: No home dio events detected for {nwb_file_name}, epoch {epoch_num}")
            return
        
        # get dio mapping for arms
        dio_map = get_dio_mapping(key, nwbf)
        
        # get statescriptlog contents and get descriptors
        file_id = (StateScriptFile & key).fetch1("file_object_id")
        sc = nwbf.objects[file_id]
        key["descriptors"] = get_sc_descriptors(sc.content)

        # parse statescript log according to the task type (currently there's only the V8, but future variants will be added here)
        task_name = (sgc.TaskEpoch & {"nwb_file_name": key["nwb_file_name"], "epoch": key["epoch"]}).fetch1("task_name")
        if task_name == "Eight arm flexible spatial task":
            key["parser"] = "V8_delay"
            parser = V8TrialParser(sc.content, dio_map, home_times, key)
        elif task_name == "Sleep": # just in case
            logger.info(f"Skipping sleep epoch: {nwb_file_name}, epoch {epoch_num}")
            return 
        else:
            logger.info(f"Skipping unsupported task type: {task_name}")
            return

        trials_df = parser.parse_trials()

        # insert parsed trial info into analysis nwb file
        session = (sgc.Session & {"nwb_file_name": key["nwb_file_name"], "epoch": key["epoch"]}).fetch1("session_id")
        nwb_analysis_file = AnalysisNwbfile()
        key["analysis_file_name"] = nwb_analysis_file.create(key["nwb_file_name"])
        key["trial_info_object_id"] = nwb_analysis_file.add_nwb_object(
            analysis_file_name=key["analysis_file_name"],
            nwb_object=trials_df,
            table_name=f"Trials dataframe for {session}, epoch {epoch_num}"
        )
        nwb_analysis_file.add(
            nwb_file_name=key["nwb_file_name"],
            analysis_file_name=key["analysis_file_name"],
        )
        self.insert1(key)
    
    def fetch1_dataframe(self):
        """
        Fetch the trial-by-trial analysis dataframe for a given epoch on a given day. 
        Only valid when a single epoch is selected.

        Example:
        restr = {"nwb_file_name": "bobrick20231114_.nwb", "epoch": 4}
        (TrialInfo & restr).fetch1_dataframe()
        """
        num_tuples = self.fetch().size
        if num_tuples != 1:
            logger.info(f"Can only fetch dataframe for exactly 1 epoch, but {num_tuples} were given")
            return

        nwbf = self.fetch_nwb()[0]
        trials_df = nwbf["trial_info"]
        return trials_df
    
    def plot_trials(self, start=0, end=None):
        """
        Visualize trial-by-trial behavioral data for a given epoch on a given day. 
        Only valid when a single epoch is selected.
        Top plot: timestamps when certain landmarks were triggered
        Bottom plot: history of outerwells visited.        

        Example:
        restr = {"nwb_file_name": "bobrick20231114_.nwb", "epoch": 4}
        (TrialInfo & restr).plot_trials()
        """

        num_tuples = self.fetch().size
        if num_tuples != 1:
            logger.info(f"Can only plot exactly 1 epoch, but {num_tuples} were given")
            return

        trials_df = self.fetch1_dataframe()
        if end is None:
            end = len(trials_df)
        session = (sgc.Session & self).fetch1("session_id")
        epoch = self.fetch1("epoch")
        task_name = (sgc.TaskEpoch & self).fetch1("task_name")
        if task_name == "Eight arm flexible spatial task":
            V8TrialParser.plot_trials(trials_df, session, epoch, start, end)
        else:
            print(f"No parsing logic implemented for task: {task_name}")

def get_beam_times(key, nwbf, beam):
    # get homebeam dio event timestamps
    dio_obj_id = (
        sgc.DIOEvents & {"nwb_file_name": key["nwb_file_name"], "dio_event_name": beam}
    ).fetch1("dio_object_id")
    dios = nwbf.objects[dio_obj_id]
    diotimesall = np.asarray(dios.timestamps)

    # get timestamp of 1st homewell trigger the given epoch to calculate time offset
    epoch_name = (sgc.TaskEpoch & {"nwb_file_name": key["nwb_file_name"], "epoch": key["epoch"]}).fetch1("interval_list_name")
    epoch_valid_times = ( # gets time bounds of epoch
        sgc.IntervalList & {"nwb_file_name" : key["nwb_file_name"], "interval_list_name": epoch_name}
    ).fetch1("valid_times")
    start_time, end_time = epoch_valid_times.squeeze()
    return diotimesall[(diotimesall >= start_time) & (diotimesall < end_time)]

def get_dio_mapping(key, nwbf):
    # get dio mapping for arms
    diomap = {}
    dio_obj_ids = (DIOEvents & {"nwb_file_name": key["nwb_file_name"]}).fetch("dio_object_id")
    for id in dio_obj_ids:
        dio_obj = nwbf.objects[id]
        desc = dio_obj.description
        chan_num = re.search(r'\d+', desc).group()
        diomap[dio_obj.name] = chan_num
    return diomap

def get_sc_descriptors(sc_text):
    """
    Helper method to retrieve key descriptors from the statescript log. 

    sc_text: str    # text contents of statescript file
    """
    descriptors = {}

    sc_lines = sc_text.split("\n")

    lines = [line[1:] for line in sc_lines if len(line) > 0 and line[0]=="#"]

    for line in lines:
        if "%" in line:
            line = line[:line.index("%")].strip() # strip comments

        if re.match(r"<.*_uw\.sc>$", line): # get statescript file name
            descriptors["statescript"] = line[1:-1]
        elif re.match(r"<.*\.py>$", line): # get python script file name
            descriptors["python_script"] = line[1:-1]
        elif re.match(r"^(int lockoutPeriod\s*=?).*", line): # get lockout period length (in seconds)
            descriptors["lockout_period"] = int(line[line.index("=") + 1 : ].strip()) / 1000
        elif re.match(r"^(outerReps\s*=?).*", line):
            if "np.random.randint" in line:
                rangestr = line[line.index("(") + 1 : line.index(")")]
                range = rangestr.split(",")
                descriptors["outer_reps"] = [int(range[0]), int(range[1])]
            else:
                descriptors["outer_reps"] = int(line[line.index("=") + 1:].strip())
        elif re.match(r"^(numgoals\s*=?).*", line):
            descriptors["num_goals"] = int(line[line.index("=") + 1:].strip())
        elif re.match(r"^(forageNum\s*=?).*", line):
            descriptors["forage_num"] = int(line[line.index("=") + 1:].strip())
        
    # if not "lockout_period" in descriptors.keys():
    #     raise Exception("No lockout period found in StateScriptLog")

    return descriptors