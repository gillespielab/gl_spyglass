import numpy as np
import pandas as pd
import itertools
import sys
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import math
import ast
from numpy import array_equal

MILLISECONDS_PER_SECOND = 1000

class TrialParser(ABC):
    @abstractmethod
    def parse_trials(self):
        pass


class V8TrialParser(TrialParser):
    def __init__(self, script, diomap, home_dio_times, key, new):
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
        self.new = new

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

        IF 'NEW' (specific to 8Arm trial):
        search_trial             # true if trial is search trial, false if repeat 

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
        ax2.legend(["goal well arms", "non-goal well arms"])
        fig.suptitle(f"{session}, epoch {epoch_num}", fontsize=16)

        ax2.grid(which="both")

        if return_fig:
            return plt.gcf()

    @staticmethod
    def plot_com_trials(df, session, epoch_num, start, end, com_trial_nums, return_fig=False):
        """Plots trial-by-trial behavioral data

        Params:
        - start, end: starting and end indices for trials to be included in the plot.
        """
        if start < 0 or end > len(df):
            print(f"Invalid interval ({start}, {end}) for dataframe of length {len(df)}")
            return
        if end is None:
            end = len(df)
        
        com_trial_nums = np.array(com_trial_nums)
        trial_num = df["trial_num"].to_numpy()[start:end]
        trialtype = df["trial_type"].to_numpy()[start:end]
        RWstart = df["rw_start"].to_numpy()[start:end]
        RWend = df["rw_end"].to_numpy()[start:end]
        outertime = df["outer_time"].to_numpy()[start:end]
        outerwell = df["outer_well"].to_numpy()[start:end]
        goalwell = df["goal_well"].to_numpy()[start:end]

        lockstarts = list(itertools.chain(*list(df["lockout_starts"][start:end])))
        lockends = list(itertools.chain(*list(df["lockout_ends"][start:end])))
        
        fig, ax = plt.subplots(figsize=(math.ceil(len(trial_num)/8),5))
        
        ax.set_title("Goal well visits")
        ax.set_xlim(start, end)
        ax.set_xticks(np.arange(start // 10 * 10, end, 10))
        ax.set_xticks(trial_num, minor=True)
        ax.set_xlabel("Trial")
        ax.set_ylabel("Well number")
        if len(com_trial_nums) != 0:
            ax.plot(trial_num[np.equal(outerwell, goalwell).nonzero()[0]], outerwell[np.equal(outerwell, goalwell).nonzero()[0]], "co", alpha = .5)
            ax.plot(trial_num[np.not_equal(outerwell, goalwell).nonzero()[0]], outerwell[np.not_equal(outerwell, goalwell).nonzero()[0]], "mo", alpha = .5)
            colors = ["c" if tn in trial_num[np.equal(outerwell, goalwell).nonzero()[0]] else "m" for tn in com_trial_nums]
            ax.scatter(com_trial_nums, outerwell[com_trial_nums-1], facecolors=colors, zorder = 10, edgecolors='black')
            ax.legend(["goal well arms", "non-goal well arms", "change of mind trials"])
        else:
            ax.plot(trial_num[np.equal(outerwell, goalwell).nonzero()[0]], outerwell[np.equal(outerwell, goalwell).nonzero()[0]], "co")
            ax.plot(trial_num[np.not_equal(outerwell, goalwell).nonzero()[0]], outerwell[np.not_equal(outerwell, goalwell).nonzero()[0]], "mo")    
        #ax2.plot(df["leave_outer"][start:end], df["outer_well"][start:end], "m|")
            ax.legend(["goal well arms", "non-goal well arms"])
        fig.suptitle(f"{session}, epoch {epoch_num}", fontsize=16)

        ax.grid(which="both")

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

        # Processing the statescript log: each row in the file contains a series of space-separated
        # values that represent a behavioral event (see any of the .stateScriptLog files as an example).
        # The categories below describe the format of rows we will use for parsing:
        # ----------------------------------------------------------------------------------------
        # - 2 numbers, "<timestamp> <event>"
        #       <event> can be either LOCKEND, CLICK1, or BEEP1. These correspond to the end of lockouts,
        #       and rip/wait events.
        # - 3 numbers, "<timestamp> UP/DOWN <DIO>":
        #       <DIO> is a number that maps to an arm on the maze (saved in self.diomap)
        #       These rows indicate that the animal hit <DIO> UP/DOWN at <timestamp>.
        # - <timestamp> <score> = <value>:
        #       e.g. goalTotal = 15

        lines = self.script.split("\n")
        data = [line.split(" ") for line in lines if len(line) > 0 and line[0] != "#"]
        # bug fix, 5/8/25 -- changed from extending to length 6 to extending to length 8
        #   When there are multiple goals, the lines can be longer than 6 without any extension, meaning that the array is 
        #   inhomogenous and numpy doesn't like that 
        dataArray = np.array([d+[""]*(8-len(d)) for d in data], dtype=list)
        descriptors = self.key["descriptors"]

        # If lockout period wasn't set in the StateScript, we have to manually calculate it 
        if "lockout_period" not in descriptors.keys():
            self.key["descriptors"]["lockout_period"] = calculate_lockout_period(dataArray)

        # initialize uptimesall, downtimesall, lockends, lockstarts, goalcount, goalcounttimes, waitends, ripends
        up_mask = dataArray[:,1]=="UP"
        uptimesall = dataArray[up_mask,0].astype(int) / MILLISECONDS_PER_SECOND
        upwellsall = dataArray[up_mask,2].astype(int)
        sc_home_times = uptimesall[upwellsall == home_label]

        # given the homebeam times in the statescript log vs. home DIO times in the nwb,
        # we can calculate the offset between trodes vs. unix time.
        offset = self.__get_time_offset(sc_home_times)
        uptimesall = uptimesall + offset

        down_mask = dataArray[:,1]=="DOWN"
        downtimesall = dataArray[down_mask,0].astype(int) / MILLISECONDS_PER_SECOND + offset
        downwellsall = dataArray[down_mask,2].astype(int)
        
        lockend_mask = dataArray[:,1]=="LOCKEND"
        # These have the correct number of lockouts 
        lockends = dataArray[lockend_mask,0].astype(int) / MILLISECONDS_PER_SECOND + offset
        lockstarts = lockends - descriptors["lockout_period"]  # e.g lockout_period= 30.0
        
        
        goalcount_mask = dataArray[:,1]=="goalTotal"
        goalcount = dataArray[goalcount_mask,3].astype(int)
        goalcounttimes = dataArray[goalcount_mask,0].astype(int) / MILLISECONDS_PER_SECOND + offset

        #TODO: be aware that the CLICK1=wait and BEEP1=rip assignment may be reversed for some subjects.
        waitends = dataArray[dataArray[:,1]=="CLICK1",0].astype(int) / MILLISECONDS_PER_SECOND + offset
        ripends = dataArray[dataArray[:,1]=="BEEP1",0].astype(int) / MILLISECONDS_PER_SECOND + offset

        # filter to timestamps that weren't repeat pokes        
        nonrepinds = np.where(np.diff(upwellsall, prepend=0) != 0)[0]

        # include home pokes that followed a lockout
        homeindsall = np.where(upwellsall == home_label)[0]
        afterlockinds = np.intersect1d(lookup(lockends, uptimesall), homeindsall)
        # Bug fix 5/1/25 -- nonrepinds and afterlockinds may have overlap, results in non-sequential trials and double counting
        valid_poke_mask = np.unique(np.concatenate((nonrepinds, afterlockinds)).astype(int))
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

        # get currentgoal information from 'CURRENTGOAL' printouts
        currentgoal_mask = dataArray[:, 1] =='CURRENTGOAL'
        currentgoals = [ast.literal_eval(','.join([g for g in goal if g])) for goal in dataArray[currentgoal_mask, 3:]]
        currentgoal_times = dataArray[currentgoal_mask, 0].astype(int) / MILLISECONDS_PER_SECOND + offset
        
        goals, goal_switch_times, num_goals, forage_num = detect_goal_info(currentgoals, currentgoal_times, descriptors, home)

        if "num_goals" not in descriptors.keys():
            self.key['descriptors']['num_goals'] = num_goals
        if "forage_num" not in descriptors.keys():
            self.key['descriptors']['forage_num'] = forage_num

        # get more detailed outerreps information from 'outerreps' printouts
        outerreps_mask = dataArray[:, 1] == 'outerreps'
        outerreps = dataArray[outerreps_mask, 3].astype(int)
        outerreps_times = dataArray[outerreps_mask, 0].astype(int) / MILLISECONDS_PER_SECOND + offset

        if len(np.unique(outerreps)) == 1:
            outerreps = outerreps[0]
            outerreps_times = []
        else:
            outerreps_times = [home[home >= outerreps_time][0] for outerreps_time in outerreps_times]  # set the switch to be the next trial start (outerreps get printed out as soon as they finish a goal block, before they initiate the next trial)
            # might not line up exactly with the goal switch times because there might be multiple pokes at the home well in between when the two get printed out, but should correspond to the same trial! hopefully!
            # if not array_equal(goal_switch_times, outerreps_times):
            #     raise RuntimeError(f'goal switch times {goal_switch_times} are not equal to the outerreps times {outerreps_times}')

        if "outer_reps" not in descriptors.keys():
            self.key['descriptors']['outer_reps'] = outerreps

        return home, rip, wait, outer, uptimes, upwells, downtimesall, downwellsall, lockstarts, lockends, waitends, ripends, goalrec, goals, goal_switch_times, outerreps, outerreps_times

    def __get_search_repeat(self, trial_df):
        # classifies trials as search or repeat
            # trials up to and including the first trial where reward is found are considered search
            # then trials up to and including the first trial where goal arm is unrewarded are considered repeat
        t = 0
        while t < len(trial_df["search_trial"]):
            i = t
            while(i < len(trial_df["search_trial"]) and trial_df["outer_well"].iloc[i] != trial_df["goal_well"].iloc[i]):
                trial_df["search_trial"].iat[i] = True
                i += 1
            if i < len(trial_df["search_trial"]):
                trial_df["search_trial"].iat[i] = True
                i += 1
            while(i < len(trial_df["search_trial"]) and  trial_df["goal_well"].iloc[i] == trial_df["goal_well"].iloc[i-1]):
                trial_df["search_trial"].iat[i] = False
                i += 1
            if i < len(trial_df["search_trial"]):
                trial_df["search_trial"].iat[i] = False
                i += 1
            t = i 

        return trial_df

    def __filter_events(self, home, rip, wait, outer, uptimes, upwells, downtimesall, downwellsall, lockstarts, lockends, waitends, ripends, goalrec, goals, goal_switch_times, outerreps, outerreps_times):
        """
        Filters parsed behavioral events for epoch based on task rules and stores results in a dataframe
        """
        # print(len(home))
        
        # only use start times that are NOT within 0.3 s of a lockstart
        # print(goodhome_filter(home, lockstarts))
        goodhome = home[goodhome_filter(home, lockstarts, lockends)]
        home_label = int(self.diomap["homebeam"])
        # print(len(goodhome))

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
                goal_well=None,
                outer_time=0.0,
                leave_outer=0.0,
                outer_success=0,
                lockout_starts=[],
                lockout_ends = [],
                during_lockout=[],
                lockout_type=0,
            )
            if self.new:
                trial["search_trial"] = False
            
            
            try:
                start_time = trial["start_time"]
                end_time = trial["end_time"]
                # error trials -- if there is a lockstart recorded within this trial 
                if len(valid_indices(lockstarts, [start_time, end_time])) > 0:
                    trial["lockout_starts"] = lockstarts[valid_indices(lockstarts, [start_time, end_time])].tolist()
                    trial["lockout_ends"] = lockends[valid_indices(lockends, [start_time, end_time])].tolist()
                    # Any time there is a lockout, every upwell after is during lockout ?
                    trial["during_lockout"] = upwells[valid_indices(uptimes, [trial["lockout_starts"][0], end_time])].tolist() # TODO: need to update this to be in regular goal notation, not in whatever it is now
                    # completed rip or wait well succesfully 
                    if len(valid_indices(ripends, [start_time, trial["lockout_starts"][0]-.1])) > 0 or len(valid_indices(waitends, [start_time, trial["lockout_starts"][0]-.1])) > 0:
                        trial["lockout_type"] = 2
                        trial["rw_success"] = 1

                        if len(valid_indices(rip, [start_time, trial["lockout_starts"][0]-0.1])):
                            rw_lockout("rip", trial, start_time, end_time, rip, ripends, downtimesall)
                        elif len(valid_indices(wait, [start_time, trial["lockout_starts"][0]-.1])):
                            rw_lockout("wait", trial, start_time, end_time, wait, waitends, downtimesall)
                        
                        # also completed outer successfully (lockedout on way home, ie by going to r/w), still considered locktype1, order error
                        # bug fix! 7/2/25 was feeding in outer[:, 0] instead of outer[0, :] before!
                        if len(valid_indices(outer[0, :], [start_time, trial["lockout_starts"][0]-.1])) > 0:
                            trial["outer_time"] = outer[0, valid_indices(outer[0], [start_time, trial["lockout_starts"][0]-.1])[0]]
                            trial["outer_well"] = outer[1, valid_indices(outer[0], [start_time, trial["lockout_starts"][0]-.1])[0]]
                            trial["leave_outer"] = downtimesall[(downtimesall >= trial["outer_time"]) & (downtimesall < trial["lockout_starts"][0]) & (downwellsall == trial["outer_well"])][0]
                            if len(valid_indices(goalrec, [trial["start_time"],trial["lockout_starts"][0]])) > 0: # received outer reward
                                # trial["goal_well"] = trial["outer_well"]
                                trial["outer_success"] = 1

                    # did not complete rip/wait successfully
                    else:
                        trial["rw_success"] = 0
                        # if he locks out by going straight to an outer arm, skipping rip/wait
                        if len(valid_indices(outer[0], [start_time, trial["lockout_starts"][0]])):
                            # Bug fix 5/14/25 --
                            # Changed from 1 to 'home_label' - this was the line that is causing a lot of bug trials that should just be lockouts 
                            trial["leave_home"] = downtimesall[(downwellsall == home_label) & (downtimesall >= start_time) & (downtimesall < trial["lockout_starts"])][-1]
                            trial["lockout_type"] = 2
                            trial["trial_type"] = 0 # type=error, cannot define r or w
                        else:
                            # if a lockout occurred immediately after rip/wait was visited, it was because either
                            # the rip well was visited on a wait trial or vice versa
                            if len(valid_indices(rip,[trial["lockout_starts"][0]-.01, trial["lockout_starts"][0]])) > 0: # rip visit was the lock cause
                                rw_mismatch_lockout("wait", trial, downtimesall, downwellsall) 
                            elif len(valid_indices(wait,[trial["lockout_starts"][0]-.01, trial["lockout_starts"][0]])) > 0: # wait visit was the lock cause
                                rw_mismatch_lockout("rip", trial, downtimesall, downwellsall)

                            # if a lockout occurred sometime after rip/wait, it was due to an impatience error
                            elif len(valid_indices(rip,[start_time, trial["lockout_starts"][0]])) > 0:
                                rw_impatience_lockout("rip", trial, rip, downtimesall, start_time)
                            elif len(valid_indices(wait,[start_time, trial["lockout_starts"][0]])) > 0:
                                rw_impatience_lockout("wait", trial, wait, downtimesall, start_time)
                # COMPLETE TRIAL no lockouts
                else:
                    trial["rw_success"] = 1
                    # The bug trials on this line means that we didn't record a lockout, but we also don't have any outer well visits
                    # This is likely an actual bug in the recording of trials, and will be saved as such 
                    trial["outer_time"] = outer[0, valid_indices(outer[0], [start_time, end_time])][0]
                    trial["outer_well"] = outer[1, valid_indices(outer[0], [start_time, end_time])][0]
                    trial["leave_outer"] = downtimesall[valid_indices(downtimesall, [trial["outer_time"], end_time])][-1]
                    if len(valid_indices(goalrec, [start_time, end_time])): # received outer reward
                        # trial["goal_well"] = trial["outer_well"]
                        trial["outer_success"] = 1
                    
                    if len(valid_indices(rip, [start_time, end_time-.001])): # rip trial -.001 to catch trodes freeze trials
                        rw_normal("rip", trial, start_time, end_time, rip, ripends, downtimesall)
                    elif len(valid_indices(wait, [start_time, end_time-.001])): # wait trial
                        rw_normal("wait", trial, start_time, end_time, wait, waitends, downtimesall)

                # sanity checks:
                assert (trial["start_time"] < trial["leave_home"]), "leave home <= start time"
                assert(trial["rw_start"] <= trial["rw_end"]), "rw end < rw start"
                assert(trial["rw_end"] <= trial["leave_rw"]), f"leave rw ({trial['leave_rw']}) < rw end ({trial['rw_end']})"
                assert(trial["outer_time"] <= trial["leave_outer"]), "leave outer < outer time"

            except Exception as e:
                _, _, e_traceback = sys.exc_info()
                e_line = e_traceback.tb_lineno

                print("bug trial #%d, epoch %d! line %d: %s" % (t + 1, self.key["epoch"], e_line, str(e)))
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
                trial["goal_well"] = None
                trial["rw_success"] = 0

                if self.new:
                    trial["search_trial"] = False

            # Assigning a type to empty lists to suprress hdmf parsing errors
            trial["lockout_starts"] = np.array(trial["lockout_starts"], dtype=np.float64)
            trial["lockout_ends"] = np.array(trial["lockout_ends"], dtype=np.float64)
            trial["during_lockout"] = np.array(trial["during_lockout"], dtype=np.float64)
            trial_data.append(trial)
        
        trial_df = pd.DataFrame(trial_data)
        # # work backwards to fill in goal info based on rewarded locations (only know once he gets goal for the first time)
        # # this also works when the end of the ep ends in zeros ( will just overwrite 0 with 0), just can"t know goal for those trials
        # for t in range(len(trial_df["goal_well"])-1, 0, -1):
        #     if trial_df["goal_well"].iat[t-1] == 0:
        #         trial_df["goal_well"].iat[t-1] = trial_df["goal_well"].iat[t]

        # use the currentgoal printouts from statescriptlog to fill in the goal_well information

        if self.new:
            trial_df = self.__get_search_repeat(trial_df)

        # print(len(trial_df[trial_df["lockout_type"] != 0]))

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
            # NOTE: pippin20210407_.nwb seems to have some sort of issue with dio recording, therefore we do not find 
            #       a small enough offset. This epoch should be discarded. 
            if mismatch < 0.1: # if below threshold, return the offset for aligning timestamps
                return offset
            event_idx += 1
            

    


            

# HELPER FUNCTIONS

def rw_normal(rw_type, trial, start, end, events, event_ends, downtimesall):
    """
    Records a normal visit to a rip/wait well in the trial record

    Parameters:
    rw_type (str): str, "rip" or "wait"
    trial (pd.DataFrame): the current trial dictionary being populated
    start (float): trial start time
    end (float): trial end time
    events (np.ndarray): start times of visits to the rip/wait well
    event_ends (np.ndarray): times when the subject left the rip/wait well
    """
    trial_type = 1 if rw_type == "rip" else 2
    trial["trial_type"] = trial_type
    trial["rw_start"] = events[valid_indices(events, [start, end])][0]
    trial["rw_end"] = event_ends[valid_indices(event_ends, [start, end])][0] 
    # use the time of next outer arm visit as the upper bound to end of the rip/wait visit
    trial["leave_rw"] = downtimesall[valid_indices(downtimesall, [trial["rw_start"], trial["outer_time"]])][-1]
    trial["leave_home"] = downtimesall[valid_indices(downtimesall, [start, trial["rw_start"]])][-1]
    #those trials when he gets click/beep just as he leaves
    if (trial["rw_end"] - trial["leave_rw"] < .3) and (trial["rw_end"] - trial["leave_rw"] > 0):
        trial["leave_rw"] = trial["rw_end"]

def rw_lockout(rw_type, trial, start, end, events, event_ends, downtimesall):
    """
    Records a visit to the rip/wait well that included a lockout

    Parameters:
    rw_type (str): str, "rip" or "wait"
    trial (pd.DataFrame): the current trial dictionary being populated
    start (float): trial start time
    end (float): trial end time
    events (np.ndarray): start times of visits to the rip/wait well
    event_ends (np.ndarray): times when the subject left the rip/wait well
    """
    trial_type = 1 if rw_type == "rip" else 2
    trial["trial_type"] = trial_type
    trial["rw_start"] = events[valid_indices(events, [start, trial["lockout_starts"][0]-.1])][0]
    trial["leave_home"] = downtimesall[valid_indices(downtimesall, [start, trial["rw_start"]])][-1]
    trial["rw_end"] = event_ends[valid_indices(event_ends, [start, end])][0]
    # use the time of the lockout as the upper bound to end of the rip/wait visit
    trial["leave_rw"] = downtimesall[valid_indices(downtimesall, [trial["rw_start"], trial["lockout_starts"][0]])][-1]
    # those trials when he gets click/beep just as he leaves
    # if (trial["rw_end"] - trial["leave_rw"]) < .3 and (trial["rw_end"] - trial["leave_rw"]) > 0:
    #     trial["leave_rw"] = trial["rw_end"]
    if trial["leave_rw"] < trial["rw_end"]:
        trial["leave_rw"] = trial["rw_end"]
    
def outerwell_lockout(trial, outer, start, downtimesall, downwellsall, goalrec):
    """
    Records a lockout visit to an outer well in the trial record

    Parameters:
    trial (pd.DataFrame): the current trial dictionary being populated
    outer (Tuple[np.ndarray]): outer[0] = start times of visits to the outer wells, outer[1] = wells visited
    start (float): trial start time
    downtimesall (np.ndarray): times when the subject poked down into a well
    downarmsall (np.ndarray): the well associated with the times above
    goalrec (np.ndarray): goal times
    """
    trial["outer_time"] = outer[0, valid_indices(outer[0], [start, trial["lockout_starts"][0]-.1])][0]
    trial["outer_well"] = outer[1, valid_indices(outer[0], [start, trial["lockout_starts"][0]-.1])][0]
    trial["leave_outer"] = downtimesall[(downtimesall >= trial["outer_time"]) & (downtimesall < trial["lockout_starts"][0]) & (downwellsall == trial["outer_well"])][0]
    if len(valid_indices(goalrec, [trial["start_time"],trial["lockout_starts"][0]])) > 0: # received outer reward
        # trial["goal_well"] = trial["outer_well"]
        trial["outer_success"] = 1

def rw_mismatch_lockout(rw_type, trial, downtimesall, downwellsall, start):
    """
    Handles trial where a lockout was triggered the the subject visiting rip/wait
    on the wrong trial (e.g. visiting the rip well on a wait trial)

    Parameters:
    rw_type (str): str, "rip" or "wait"
    trial (pd.DataFrame): the current trial dictionary being populated
    downtimesall (np.ndarray): times when the subject poked down into a well
    downarmsall (np.ndarray): the well associated with the times above
    start (float): trial start time
    """
    trial_type = 1 if rw_type == "rip" else 2
    trial["trial_type"] = trial_type
    trial["lockout_type"] = 1
    trial["leave_home"] = downtimesall[(downwellsall == 1) & (downtimesall >= start) & (downtimesall < trial["lockout_starts"][0])][-1]

def rw_impatience_lockout(rw_type, trial, events, downtimesall, start):
    """
    Handles trial where a lockout was triggered by the subject.

    Parameters:
    rw_type (str): str, "rip" or "wait"
    trial (pd.DataFrame): the current trial dictionary being populated
    downtimesall (np.ndarray): times when the subject poked down into a well
    downarmsall (np.ndarray): the well associated with the times above
    start (float): trial start time
    """
    trial_type = 1 if rw_type == "rip" else 2
    trial["trial_type"] = trial_type
    trial["lockout_type"] = 3
    trial["rw_start"] = events[valid_indices(events, [start, trial["lockout_starts"][0]])][0]
    trial["leave_rw"] = trial["lockout_starts"][0]
    trial["rw_end"] = trial["lockout_starts"][0]
    trial["leave_home"] = downtimesall[valid_indices(downtimesall, [start, trial["rw_start"]])][-1]

def valid_indices(values, bounds):
    """
    values (np.ndarray/list of ints)
    bounds (tuple): [lowerbound, upperbound]

    Just a shorthand for np.nonzero to avoid clogging up the main program.
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

# bug fix 5/9/25 -- didn't include the home pokes that occurred after all lockstarts, included homepokes that occurred during lockout 
def goodhome_filter(home, lockstarts, lockends):
    """
    Returns a list of indices i such that home[i] < lockstart-.3 or home[i] > lockstart FOR ALL times in lockstarts

    Assumes home and lockstarts are 1D ndarrays containing monotonically increasing values (timestamps).
    """
    if lockstarts.size == 0:
        return np.arange(home.size)
    indices = []
    for i in range(home.size):
        if np.max(lockstarts > home[i]) == 0: # case where no lockstarts occurred after hometime
            indices = np.concatenate((np.array(indices), np.arange(i, home.size))).astype(int)
            break
        j = np.argmax(lockstarts > home[i]) # first index where lockstart > hometime
        if lockstarts[j]-0.3 > home[i]: # include i if hometime[i] < lockstart-3
            indices.append(i)
    j = 0
    k = 0  
    uncovered = []
    for j in indices:
        while k < len(lockstarts) and lockends[k] < home[j]:
            k += 1
        if k < len(lockstarts) and lockstarts[k] <= home[j] <= lockends[k]:
            j += 1  # it's covered, move to next home
        else:
            uncovered.append(j)
            j += 1  
    indices = uncovered

    return np.array(indices)


# bug fix, 5/8/2025 -- 
#   some StateScript files don't save the lockout period, so that is empty in the descriptors

#   Manually calculates the average difference between white noise delivery and registering of lockend
#   Returns this value to be set in descriptors["lockout_period"]
def calculate_lockout_period(dataArray):
    lockend_mask = dataArray[:,1]=="LOCKEND"
    white_noise_mask = dataArray[:,1] == "WHITENOISE"
    lockend_times = dataArray[lockend_mask, 0].astype(int)
    white_noise_times = dataArray[white_noise_mask, 0].astype(int)
    # if they're not the same length, it should really only be a obob error
    #   either whitenoise happened before we started recording so there's an extra lockend, or we stopped recording
    #   after whitenoise started but before registering a lockend
    if len(lockend_times) != len(white_noise_times):
        if len(lockend_times) == 0 or len(white_noise_times) == 0:
            return 0
        if white_noise_times[-1] > lockend_times[-1]:
            white_noise_times = white_noise_times[:-1]
        if lockend_times[0] < white_noise_times[0]:
            lockend_times = lockend_times[1:]
    assert (len(lockend_times) == len(white_noise_times)), "Mismatch between lock starts and lock ends "
    differences = np.array(lockend_times - white_noise_times) / MILLISECONDS_PER_SECOND
    return float(round(np.mean(differences)))


def detect_goal_info(currentgoals, currentgoal_times, descriptors, home):
    # Goal: get current goals and current goal times via StateScriptLog 'CURRENTGOAL' printouts
    # Side goal: in the process, find descriptors forage_num, num_goal if they don't exist in the descriptors

    forage_num = None
    num_goals = None
    # If forage_num was set in the StateScript, retrieve:
    if "forage_num" in descriptors.keys():
        forage_num = descriptors['forage_num']
    # If num_goal was set in the StateScript, retrieve:
    if "num_goals" in descriptors.keys():
        num_goals = descriptors['num_goals']

    # initialize detected descriptors
    detect_forage_num = None
    detect_num_goals = None
    forage_trial_start_time = None

    # iterate through values instead of np.diff because sometimes there multiple goals (list of len >1)
    goals = []
    goal_switch_times = []

    for g, (goal, goal_time) in enumerate(zip(currentgoals, currentgoal_times)):
        if g == 0:
            if len(goal) == 1:
                goals.append(goal[0])
            else:
                goals.append(goal)
            continue

        prev_goal = currentgoals[g - 1]
        curr_goal = goal

        prev_goal_time = currentgoal_times[g - 1]
        curr_goal_time = goal_time

        # Compare the goal to the previous one to check if it was a goal switch

        # if there's only one goal now and there was only one goal before, compare
        # this is the most common case for the final stage of the task
        if (len(curr_goal) == 1) & (len(prev_goal) == 1):
            detect_num_goals = 1
            detect_forage_num = 1
            if curr_goal[0] != prev_goal[0]:
                goals.append(curr_goal[0])
                trial_start_time = home[home <= curr_goal_time][-1]  # most recent trial start
                goal_switch_times.append(trial_start_time)

        # if there's only one goal now and there was > 1 before, this was a forageassist trial where they found a goal:
        #   - set detect_forageassist = len(prev_goal)
        #   - check that forage_num = detect_forageassist
        #   - check that curr_goal is within prev_goal
        #   - assign the goal to be the curr goal (and get rid of the previous goal entry that was based on the forageassist start)
        #   - if this is not the first goal block (forage_trial_start_time is not None), add that time to the goal switch times
        if (len(curr_goal) == 1) & (len(prev_goal) > 1):
            detect_forage_num = len(prev_goal)
            if forage_num is not None:
                if forage_num != detect_forage_num:
                    raise RuntimeError(f'detected forage_num ({detect_forage_num}) and StateScript-extracted forage_num ({forage_num}) are not equal!')
            
            if curr_goal[0] not in prev_goal:
                raise RuntimeError(f'current goal ({curr_goal[0]}) is not in previous goal options ({prev_goal})!')

            if array_equal(np.asarray(goals[-1]), np.asarray(prev_goal)):
                goals[-1] = curr_goal[0]
            
            if (forage_trial_start_time is not None):
                goal_switch_times.append(forage_trial_start_time)
                forage_trial_start_time = None

        # if there's > 1 goal now and there was only one goal before:
        #   - set detect_forageassist = len(curr_goal)
        #   - check that forage_num = detect_forageassist
        #   - save the start of this trial in case this is a forageassist trial and it'll switch to one goal later
        if (len(curr_goal) > 1) & (len(prev_goal) == 1):
            detect_forage_num = len(curr_goal)
        
            if forage_num is not None:
                if forage_num != detect_forage_num:
                    raise RuntimeError(f'detected forage_num ({detect_forage_num}) and StateScript-extracted forage_num ({forage_num}) are not equal!')

            goals.append(curr_goal)    
            forage_trial_start_time = home[home <= curr_goal_time][-1]

        # if there's > 1 goal now and there was > 1 goal before, compare
        #   - check that num_goals = len(prev_goal) & num_goals = len(curr_goal)
        #   - if they're different, set detect_num_goal = len(curr_goal)
        if (len(curr_goal) > 1) & (len(prev_goal) > 1):
            # if num_goals is not None:
            #     if (num_goals != len(prev_goal)) | (num_goals != len(curr_goal)):
            #         raise RuntimeError(f'num_goals was set in state script as {num_goals} while the current goal has {len(curr_goal)} goals and the previous goal has {len(prev_goal)} goals!')
            if array_equal(np.asarray(curr_goal), np.asarray(prev_goal)):
                if forage_trial_start_time == None:
                    # use the first trial to save forage trial start time just in case it's forageassist instead of multiple goals
                    prev_trial_start_time = home[home <= prev_goal_time][-1]  # most recent trial start
                    forage_trial_start_time = prev_trial_start_time
            if not array_equal(np.asarray(curr_goal), np.asarray(prev_goal)):
                goals.append(curr_goal)
                trial_start_time = home[home <= curr_goal_time][-1]  # most recent trial start
                goal_switch_times.append(trial_start_time)
                detect_num_goals = len(curr_goal)

        if g == len(currentgoals) - 1:
            # if the last goal has multiple goals and detectforageassist is not None:
            #   - assign last goal to be 0
            if (len(curr_goal) > 1) & (detect_forage_num is not None):
                if len(curr_goal) != detect_forage_num:
                    raise RuntimeError(f'length of curr goal is {len(curr_goal)} does not match detected forage num {detect_forage_num}!')              
                goals[-1] = 0

            # if the last goal has multiple goals and forageassist is None:
            #   - check that num_goals = len(prev_goal)
            #   - check that forage_num = 0
            if (len(curr_goal) > 1) & (detect_forage_num is None):
                if num_goals is not None:
                    if num_goals != len(curr_goal):
                        raise RuntimeError(f'statescript-derived num_goals ({num_goals}) does not match length of the current goal ({len(curr_goal)})!')

    if detect_forage_num is not None:
        detect_num_goals = 1

    if detect_num_goals > 1:
        detect_forage_num = 0
    
    return goals, goal_switch_times, detect_num_goals, detect_forage_num