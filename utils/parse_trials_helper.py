import numpy as np
import pandas as pd
import itertools
import sys
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

MILLISECONDS_PER_SECOND = 1000

class TrialParser(ABC):
    @abstractmethod
    def parse_trials(self):
        pass


class V8TrialParser(TrialParser):
    def __init__(self, script, diomap, home_dio_times, key):
        """
        script (str): raw script from statescript log
        diomap (dict): map event names to their dio channel (e.g. {"homebeam": "Din1", ...})
        firsthometime (float): timestamp of 1st time homebeam was triggered. used for calculating timestamp offset.
        desc (dict): key detailing session name, epoch number, and epoch descriptors
        """
        self.script = script
        self.diomap = diomap
        self.home_dio_times = home_dio_times
        self.key = key
        self.trials_df = None

    def parse_trials(self):
        """Return a dataframe containing the following trial metrics
        ---
        start_time: double       # start time of trial
        end_time: double         # end time of trial
        leave_home: double       # last time when home well was triggered before going to RW
        leave_rw: double         # last time when either R or W is triggered before going to an outer arm
        rw_start: double         # time when the center port is hit
        rw_end: double           # time when reward has been delivered
        rw_success: int          # whether the animal got reward at center wells (yes=1, no=0)
        trial_type: int          # rip = type 1, wait = type 2
        outer_well: int          # which outerwell was visited
        outer_time: double       # time when outerwell was visited
        leave_outer: double      # last time outerwell was triggered
        outer_success: int       # whether the animal got reward at outerwell (yes=1, no=0)
        goal_well: int           # which arm was the goal arm
        lockout_starts: blob     # start times of lockouts
        lockout_ends: blob       # end times of lockouts
        lockout_type: int        # type of error that caused lockout
        during_lockout: blob     # wells visited during lockout
        """
        
        # script: contents of the epoch's statescript file
        parsed_events = self.__parse_statescript()
        events_df = self.__filter_events(*parsed_events)
        self.trials_df = events_df
        return events_df

    @staticmethod
    def plot_trials(df, session, epoch_num, start, end, return_fig=False):
        """Plots trial-by-trial behavioral data

        Params:
        - start, end: starting and end indices for trials to be included in the plot.
        """
        if start < 0 or end > len(df):
            print(f"Invalid interval ({start}, {end}) for dataframe of length {len(df)}")
            return
        if end is None:
            end = len(df)
        trial_num = df["trial_num"].to_numpy()[start:end]
        trialtype = df["trial_type"].to_numpy()[start:end]
        RWstart = df["rw_start"].to_numpy()[start:end]
        RWend = df["rw_end"].to_numpy()[start:end]
        outertime = df["outer_time"].to_numpy()[start:end]
        outerwell = df["outer_well"].to_numpy()[start:end]
        goalwell = df["goal_well"].to_numpy()[start:end]

        lockstarts = list(itertools.chain(*list(df["lockout_starts"][start:end])))
        lockends = list(itertools.chain(*list(df["lockout_ends"][start:end])))
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,10))
        
        ax1.set_title("Landmark times")
        ax1.set_xlim(df["start_time"].iat[start], df["end_time"].iat[end-1])
        ax1.set_xlabel("Time (ms)")
        ax1.set_ylim(-1, 3)
        ax1.set_yticks([])
        ax1.plot(df["start_time"][start:end], np.zeros(len(df["start_time"][start:end])), "b|")
        ax1.plot(df["leave_home"][start:end], np.zeros(len(df["start_time"][start:end])), "r|")
        ax1.plot(RWstart[np.where(trialtype==1)[0]], np.ones(sum(trialtype==1)), "g.")
        ax1.plot(RWstart[np.where(trialtype==2)[0]], np.ones(sum(trialtype==2)), "b.")
        ax1.plot(RWend[np.where(trialtype==1)[0]], np.ones(sum(trialtype==1)), "g|")
        ax1.plot(RWend[np.where(trialtype==2)[0]], np.ones(sum(trialtype==2)), "b|")
        ax1.plot(outertime[np.equal(outerwell, goalwell).nonzero()[0]], np.ones(sum(outerwell == goalwell)), "c.")
        ax1.plot(outertime[np.not_equal(outerwell, goalwell).nonzero()[0]],np.ones(sum(outerwell != goalwell)), "m.")
        ax1.plot(lockstarts, np.ones(len(lockends)), "rx")
        ax1.plot(lockends, np.ones(len(lockstarts)), "bx")
        ax1.legend(
            [
                "starttime", 
                "leavehome", 
                "RWstart (rip trial)", 
                "RWstart (wait trial)", 
                "RWend (rip)", 
                "RWend (wait)", 
                "goal well times", 
                "non-goal well times", 
                "lockstarts", 
                "lockends"
            ], 
            loc="upper right", 
            ncol=2
        )
        
        ax2.set_title("Goal well visits")
        ax2.set_xlim(start, end)
        ax2.set_xticks(np.arange(start // 10 * 10, end, 10))
        ax2.set_xticks(trial_num, minor=True)
        ax2.set_xlabel("Trial")
        ax2.set_ylabel("Well number")
        ax2.plot(trial_num[np.equal(outerwell, goalwell).nonzero()[0]], outerwell[np.equal(outerwell, goalwell).nonzero()[0]], "co")
        ax2.plot(trial_num[np.not_equal(outerwell, goalwell).nonzero()[0]], outerwell[np.not_equal(outerwell, goalwell).nonzero()[0]], "mo")
        #ax2.plot(df["leave_outer"][start:end], df["outer_well"][start:end], "m|")
        ax2.legend(["goal well arms", "non-goal well arms"])
        fig.suptitle(f"{session}, epoch {epoch_num}", fontsize=16)

        ax2.grid(which="both")

        if return_fig:
            return plt.gcf()

    def __parse_statescript(self):
        """ Helper function to parse statescript log into timestamp arrays or 
        records of various behavioral landmarks (home well / center wells / outer well visits,
        start & end times of rip/wait trials, start & end times of lockouts, record of goals)
        """
        home_label = int(self.diomap["homebeam"])
        rip_label = int(self.diomap["Rbeam"])
        wait_label = int(self.diomap["Wbeam"])
        arm1_label = int(self.diomap["arm1beam"])

        # process statescript log
        lines = self.script.split("\n")
        data = [line.split(" ") for line in lines if len(line) > 0 and line[0] != "#"]
        dataArray = np.array([d+[""]*(6-len(d)) for d in data])
        descriptors = self.key["descriptors"]

        # initialize uptimesall, downtimesall, lockends, lockstarts, goalcount, goalcounttimes, waitends, ripends
        up_mask = dataArray[:,1]=="UP"
        uptimesall = dataArray[up_mask,0].astype(int) / MILLISECONDS_PER_SECOND
        upwellsall = dataArray[up_mask,2].astype(int)
        sc_home_times = uptimesall[upwellsall == home_label]

        # TODO: choose the first valid homedio times to align to
        offset = self.__get_time_offset(sc_home_times)
        uptimesall = uptimesall + offset

        down_mask = dataArray[:,1]=="DOWN"
        downtimesall = dataArray[down_mask,0].astype(int) / MILLISECONDS_PER_SECOND + offset
        downwellsall = dataArray[down_mask,2].astype(int)
        
        lockend_mask = dataArray[:,1]=="LOCKEND"
        lockends = dataArray[lockend_mask,0].astype(int) / MILLISECONDS_PER_SECOND + offset
        lockstarts = lockends - descriptors["lockout_period"]  # e.g lockout_period= 30.0
        
        goalcount_mask = dataArray[:,1]=="goalTotal"
        goalcount = dataArray[goalcount_mask,3].astype(int)
        goalcounttimes = dataArray[goalcount_mask,0].astype(int) / MILLISECONDS_PER_SECOND + offset

        # CHANGE BASED ON RAT
        #TODO: consider optional param per subject
        waitends = dataArray[dataArray[:,1]=="CLICK1",0].astype(int) / MILLISECONDS_PER_SECOND + offset
        ripends = dataArray[dataArray[:,1]=="BEEP1",0].astype(int) / MILLISECONDS_PER_SECOND + offset

        # filter to timestamps that weren't repeat pokes        
        nonrepinds = np.where(np.diff(upwellsall, prepend=0) != 0)[0] #add prepend?

        # include home pokes that followed a lockout
        homeindsall = np.where(upwellsall == home_label)[0]
        afterlockinds = np.intersect1d(lookup(lockends, uptimesall), homeindsall)
        valid_poke_mask = np.concatenate((nonrepinds, afterlockinds))
        uptimes = uptimesall[valid_poke_mask]
        upwells = upwellsall[valid_poke_mask]

        # get timestamps where home, rip, wait, and outer wells were visited
        home = uptimes[upwells == home_label]
        rip = uptimes[upwells == rip_label]
        wait = uptimes[upwells == wait_label]
        outer_mask = (upwells >= arm1_label)
        if home_label > arm1_label: # TODO: eliminate assumption that outer arm dio channels are consecutive
            outer_mask = (upwells >= arm1_label) & (upwells < home_label)

        outer = np.array([uptimes[outer_mask], upwells[outer_mask] - arm1_label + 1]) # convert dio channel to arm number
        # filter goal records to only count times where goalcount increased
        goalrec = goalcounttimes[np.where(np.diff(goalcount, prepend=0) > 0)[0]]

        return home, rip, wait, outer, uptimes, upwells, downtimesall, downwellsall, lockstarts, lockends, waitends, ripends, goalrec

    def __filter_events(self, home, rip, wait, outer, uptimes, upwells, downtimesall, downwellsall, lockstarts, lockends, waitends, ripends, goalrec):
        """
        Filters parsed behavioral events for epoch based on task rules and stores results in a dataframe
        """
        
        # only use start times that are NOT within 0.3 s of a lockstart
        goodhome = home[goodhome_filter(home, lockstarts)]

        # initialize dataframe to be populated
        trial_data = []
        start_times = goodhome[:-1]
        end_times = goodhome[1:]

        for t in range(len(goodhome) - 1):
            trial = dict(
                trial_num=t+1,
                start_time=start_times[t], 
                end_time=end_times[t],
                leave_home=0.0,
                trial_type=0,
                rw_start=0.0,
                rw_end=0.0,
                leave_rw=0.0,
                rw_success=0,
                outer_well=0,
                goal_well=0,
                outer_time=0.0,
                leave_outer=0.0,
                outer_success=0,
                lockout_starts=[],
                lockout_ends = [],
                during_lockout=[],
                lockout_type=0,
            )
            
            try:
                start_time = trial["start_time"]
                end_time = trial["end_time"]
                # error trials
                if len(valid_indices(lockstarts, [start_time, end_time])) > 0:
                    trial["lockout_starts"] = lockstarts[valid_indices(lockstarts, [start_time, end_time])].tolist()
                    trial["lockout_ends"] = lockends[valid_indices(lockends, [start_time, end_time])].tolist()
                    trial["during_lockout"] = upwells[valid_indices(uptimes, [trial["lockout_starts"][0], end_time])].tolist() # TODO: add uptimes in addition to wells
                    # completed rip or wait well succesfully
                    if len(valid_indices(ripends, [start_time, end_time])) > 0 or len(valid_indices(waitends, [start_time, end_time])) > 0:
                        trial["lockout_type"] = 1
                        trial["rw_success"] = 1
                        if len(valid_indices(rip, [start_time, trial["lockout_starts"][0]-0.1])):
                            rw_lockout("rip", trial, start_time, end_time, rip, ripends, downtimesall)
                        elif len(valid_indices(wait, [start_time, trial["lockout_starts"][0]-.1])):
                            rw_lockout("wait", trial, start_time, end_time, wait, waitends, downtimesall)
                        
                        # also completed outer successfully (lockedout on way home, ie by going to r/w), still considered locktype1, order error
                        if len(valid_indices(outer[:, 0], [start_time, trial["lockout_starts"][0]-.1])) > 0:
                            trial["outer_time"] = outer[0, valid_indices(outer[0], [start_time, trial["lockout_starts"][0]-.1])[0]]
                            trial["outer_well"] = outer[1, valid_indices(outer[0], [start_time, trial["lockout_starts"][0]-.1])[0]]
                            trial["leave_outer"] = downtimesall[(downtimesall >= trial["outer_time"]) & (downtimesall < trial["lockout_starts"][0]) & (downwellsall == trial["outer_well"])][0]
                            if len(valid_indices(goalrec, [trial["start_time"],trial["lockout_starts"][0]])) > 0: # received outer reward
                                trial["goal_well"] = trial["outer_well"]
                                trial["outer_success"] = 1
                    
                    # did not complete rip/wait successfully
                    else:
                        trial["rw_success"] = 0
                        #if he locks out by going straight out (locktype1 order error)
                        if len(valid_indices(outer[0], [start_time, trial["lockout_starts"][0]])):
                            trial["leave_home"] = downtimesall[(downtimesall == 1) & (downtimesall >= start_time) & (downtimesall < trial["lockout_starts"])][-1]
                            trial["lockout_type"] = 1
                            trial["trial_type"] = 0 # type=error, cannot define r or w
                        else:
                            # lockout bc visited rip on a wait trial
                            if len(valid_indices(rip,[trial["lockout_starts"][0]-.01, trial["lockout_starts"][0]])) > 0: # rip visit was the lock cause
                                rw_mismatch_lockout("wait", trial, downtimesall, downwellsall) 
                            # lockout bc visited wait on a rip trial
                            elif len(valid_indices(wait,[trial["lockout_starts"][0]-.01, trial["lockout_starts"][0]])) > 0: # wait visit was the lock cause
                                rw_mismatch_lockout("rip", trial, downtimesall, downwellsall)
                            # correctly visited rip but was impatient
                            elif len(valid_indices(rip,[start_time, trial["lockout_starts"][0]])) > 0:
                                rw_impatience_lockout("rip", trial, rip, downtimesall, start_time)
                            # correctly visited wait but was impatient
                            elif len(valid_indices(wait,[start_time, trial["lockout_starts"][0]])) > 0:
                                rw_impatience_lockout("wait", trial, wait, downtimesall, start_time)
                # COMPLETE TRIAL no lockouts
                else:
                    trial["rw_success"] = 1
                    trial["outer_time"] = outer[0, valid_indices(outer[0], [start_time, end_time])][0]
                    trial["outer_well"] = outer[1, valid_indices(outer[0], [start_time, end_time])][0]
                    trial["leave_outer"] = downtimesall[valid_indices(downtimesall, [trial["outer_time"], end_time])][-1]
                    if len(valid_indices(goalrec, [start_time, end_time])): # received outer reward
                        trial["goal_well"] = trial["outer_well"]
                        trial["outer_success"] = 1
                    
                    if len(valid_indices(rip, [start_time, end_time-.001])): # rip trial -.001 to catch trodes freeze trials
                        rw_normal("rip", trial, start_time, end_time, rip, ripends, downtimesall)
                    elif len(valid_indices(wait, [start_time, end_time-.001])): # wait trial
                        rw_normal("wait", trial, start_time, end_time, wait, waitends, downtimesall)
                
                # sanity checks:
                assert(trial["start_time"] < trial["leave_home"])
                assert(trial["rw_start"] <= trial["rw_end"])
                assert(trial["rw_end"] <= trial["leave_rw"])
                assert(trial["outer_time"] <= trial["leave_outer"])

            except Exception as e:
                _, _, e_traceback = sys.exc_info()
                e_line = e_traceback.tb_lineno

                print("bug trial #%d, epoch %d! line %d: %s" % (t, self.key["epoch"], e_line, str(e)))
                #zero out all measures for bug trials!
                trial["lockout_starts"] = []
                trial["lockout_ends"] = []
                trial["during_lockout"] = []
                trial["lockout_type"] = 0
                trial["rw_start"] = 0.0
                trial["rw_end"] = 0.0
                trial["leave_home"] = 0.0
                trial["leave_rw"] = 0.0
                trial["trial_type"] = 0
                trial["outer_well"] = 0
                trial["outer_time"] = 0.0
                trial["leave_outer"] = 0.0
                trial["goal_well"] = 0
                trial["rw_success"] = 0

            trial_data.append(trial)
        
        trial_df = pd.DataFrame(trial_data)
        # work backwards to fill in goal info based on rewarded locations (only know once he gets goal for the first time)
        # this also works when the end of the ep ends in zeros ( will just overwrite 0 with 0), just can"t know goal for those trials
        for t in range(len(trial_df["goal_well"])-1, 0, -1):
            if trial_df["goal_well"].iat[t-1] == 0:
                trial_df["goal_well"].iat[t-1] = trial_df["goal_well"].iat[t]

        return trial_df
    
    def __get_time_offset(self, sc_home_times):
        # finds the offset that best aligns the statescript homebeam times to dio home times
        # DIO time = unix time (s), SC times = time since Trodes booted up (ms)
        event_idx = 0
        while event_idx < self.home_dio_times.size:
            offset = self.home_dio_times[event_idx] - sc_home_times[0]
            # mismatch score: sum of differences between the first 4 sc timestamps and their closest dio timestamp
            mismatch = 0
            # using the first 4 sc homebeam times and calculating their mismatch to dio homebeams times
            for i in range(4): 
                target = sc_home_times[i] + offset
                lim = sc_home_times[4] + offset
                diffs = np.absolute(self.home_dio_times[self.home_dio_times < lim] - target)
                mismatch += np.min(diffs) # dio home time that was closest to the target sc home time
            if mismatch < 0.1: # if below threshold, return the offset for aligning timestamps
                return offset
            event_idx += 1
            

# HELPER FUNCTIONS

def rw_normal(rw_type, trial, start, end, events, event_ends, downtimesall):
    """
    rw_type: str, "rip" or "wait"
    trial: df, the current trial dictionary being populated
    events: rip or wait
    event_ends: ripends or waitends
    """

    trial_type = 1 if rw_type == "rip" else 2
    trial["trial_type"] = trial_type
    trial["rw_start"] = events[valid_indices(events, [start, end])][0]
    trial["rw_end"] = event_ends[valid_indices(event_ends, [start, end])][0] 
    trial["leave_rw"] = downtimesall[valid_indices(downtimesall, [trial["rw_start"], trial["outer_time"]])][-1]
    trial["leave_home"] = downtimesall[valid_indices(downtimesall, [start, trial["rw_start"]])][-1]
    #those trials when he gets click/beep just as he leaves
    if (trial["rw_end"] - trial["leave_rw"] < .3) and (trial["rw_end"] - trial["leave_rw"] > 0):
        trial["leave_rw"] = trial["rw_end"]

def rw_lockout(rw_type, trial, start, end, events, event_ends, downtimesall):
    trial_type = 1 if rw_type == "rip" else 2
    trial["trial_type"] = trial_type
    trial["rw_start"] = events[valid_indices(events, [start, trial["lockout_starts"][0]-.1])][0]
    trial["leave_home"] = downtimesall[valid_indices(downtimesall, [start, trial["rw_start"]])][-1]
    trial["rw_end"] = event_ends[valid_indices(event_ends, [start, end])][0]
    trial["leave_rw"] = downtimesall[valid_indices(downtimesall, [trial["rw_start"], trial["lockout_starts"][0]])][-1]
    # those trials when he gets click/beep just as he leaves
    if (trial["rw_end"] - trial["leave_rw"]) < .3 and (trial["rw_end"] - trial["leave_rw"]) > 0:
        trial["leave_rw"] = trial["rw_end"]
    
def outerwell_lockout(trial, outer, start, downtimesall, downwellsall, goalrec):
    trial["outer_time"] = outer[0, valid_indices(outer[0], [start, trial["lockout_starts"][0]-.1])][0]
    trial["outer_well"] = outer[1, valid_indices(outer[0], [start, trial["lockout_starts"][0]-.1])][0]
    trial["leave_outer"] = downtimesall[(downtimesall >= trial["outer_time"]) & (downtimesall < trial["lockout_starts"][0]) & (downwellsall == trial["outer_well"])][0]
    if len(valid_indices(goalrec, [trial["start_time"],trial["lockout_starts"][0]])) > 0: # received outer reward
        trial["goal_well"] = trial["outer_well"]
        trial["outer_success"] = 1

def rw_mismatch_lockout(rw_type, trial, downtimesall, downwellsall, start):
    trial_type = 1 if rw_type == "rip" else 2
    trial["trial_type"] = trial_type
    trial["lockout_type"] = 2
    trial["leave_home"] = downtimesall[(downwellsall == 1) & (downtimesall >= start) & (downtimesall < trial["lockout_starts"][0])][-1]

def rw_impatience_lockout(rw_type, trial, events, downtimesall, start):
    trial_type = 1 if rw_type == "rip" else 2
    trial["trial_type"] = trial_type
    trial["lockout_type"] = 3
    trial["rw_start"] = events[valid_indices(events, [start, trial["lockout_starts"][0]])][0]
    trial["leave_rw"] = trial["lockout_starts"][0]
    trial["rw_end"] = trial["lockout_starts"][0]
    trial["leave_home"] = downtimesall[valid_indices(downtimesall, [start, trial["rw_start"]])][-1]

def valid_indices(values, bounds):
    """
    values (array/list of ints)
    bounds (tuple): [lowerbound, upperbound]
    
    Returns new array containing indices i such that
    where lowerbound <= values[i] <= upperbound
    """
    if bounds[0] > bounds[1]:
        raise Exception("Invalid bounds provided to valid_indices: lowerbound cannot be higher than upperbound")
    return np.nonzero((values >= bounds[0]) & (values < bounds[1]))[0]

def lookup(reference, target):
    """
    For each timestamp t in reference, returns the list of indices i where target[i] is
    the lowest timestamp greater than reference[t].

    Assumes reference and target are 1D ndarrays containing monotonically increasing values (timestamps).
    """
    indices = []
    left = 0
    for ref in reference:
        while (left < target.size and target[left] <= ref):
            left += 1
        if left < target.size:
            indices.append(left)
            left += 1
    return np.array(indices)

def goodhome_filter(home, lockstarts):
    """
    Returns a list of indices i such that home[i] < lockstart-.3 or home[i] > lockstart FOR ALL times in lockstarts

    Assumes home and lockstarts are 1D ndarrays containing monotonically increasing values (timestamps).
    """
    if lockstarts.size == 0:
        return np.arange(home.size)
    indices = []
    for i in range(home.size):
        if np.max(lockstarts > home[i]) == 0: # case where no lockstarts occurred after hometime
            return np.array(indices)
        j = np.argmax(lockstarts > home[i]) # first index where lockstart > hometime
        if lockstarts[j]-0.3 > home[i]: # include i if hometime[i] < lockstart-3
            indices.append(i)
    return np.array(indices)
