import numpy as np
import pandas as pd
import itertools
import sys
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import math
import ast
import warnings
from numpy import array_equal
import spyglass.common as sgc

MILLISECONDS_PER_SECOND = 1000

class TrialParser(ABC):
    @abstractmethod
    def parse_trials(self):
        pass


class V8TrialParser(TrialParser):
    def __init__(self, script, dio_map, home_dio_times, key, nwbf):
        """
        script (str): raw script from statescript log
        diomap (dict): map event names to their dio channel (e.g. {"homebeam": "Din1", ...})
        firsthometime (float): timestamp of 1st time homebeam was triggered. used for calculating timestamp offset.
        desc (dict): key detailing session name, epoch number, and epoch descriptors
        """
        self.script = script
        self.dio_map = dio_map
        self.home_dio_times = home_dio_times
        self.key = key
        self.trials_df = None
        self.nwbf = nwbf

    def parse_trials(self):
        """Return a dataframe containing the following trial metrics
        ---
        n_goal_block: float      # goal block number (1-indexed)
        trial_num: int           # trial number (1-indexed)
        start_time: float        # start time of trial
        end_time: float          # end time of trial
        leave_home: float        # last time when home well was triggered before going to center
        lockout_type: int        # type of error that caused lockout (0 = no lockout, 1 = center mismatch, 2 = some other order error, 3 = impatience error)
        lockout_desc: str        # description of what happened during this lockout, if applicable
        bug_trial: float         # 1 if this was a bug trial, 0 otherwise
        trial_type: int          # rip = type 1, wait = type 2
        center_start: float      # center well start time (first poke)
        center_end: float        # center well end time (well turns off and reward delivered)
        leave_center: float      # last time when the appropriate center well was triggered before going to an outer arm
        center_success: float    # whether the animal got reward at center wells (yes=1, no=0)
        lockout_start: float     # lockout start time, if applicable
        lockout_end: float       # lockout end time, if applicable
        during_lockout: str      # string of any arms that were visited during lockouts (i.e. '253' or 'H2W' where 'H', 'W', and 'R' represent home and center wells respectively)
        goal_block_phase: str    # 'search' or 'repeat' (starting with the first trial after the goal is found and ending after the trial where the previous goal is no longer rewarded)
        complete_goal_block: bool  # True if both the search and repeat phase were completed, False if not
        outer_reps: float        # number of outer reps required to complete this goal block

        """
        
        # script: contents of the epoch's statescript file
        parsed_events = self.__parse_statescript()
        home, rip, wait, outer, uptimes, upwells, downtimesall, downwellsall, lockstarts, lockends, waitends, ripends, goalrec, goals, goal_switch_times, outerreps, outerreps_times, lockout3_starts = parsed_events
        trials_df = self.__get_trials_df(*parsed_events)
        trials_df = self.__add_goal_block_info(trials_df, goals, goal_switch_times, outerreps)

        # reformatting lists of lists to avoid hdmf datatype issues
        trials_goals = trials_df['goal_well'].values
        str_goal_wells = [''.join(str(item) for item in trial_goal) for trial_goal in trials_goals]
        trials_df['goal_well'] = str_goal_wells

        trials_lockout_vals = trials_df['during_lockout'].values
        str_lockout_wells = [''.join(str(item) for item in trial_lockout) for trial_lockout in trials_lockout_vals]
        trials_df['during_lockout'] = str_lockout_wells

        self.trials_df = trials_df
        return trials_df

    @staticmethod
    def plot_trials(df, session, epoch_num, start, end, return_fig=False):
        """Plots trial-by-trial behavioral data

        Params:
        - start, end: starting and end indices for trials to be included in the plot.
        """
        if start < 0 or end > len(df):
            print(f"Invalid interval ({start}, {end}) for dataframe of length {len(df)}")
        if end is None:
            end = len(df)
        trial_num = df["trial_num"].to_numpy()[start:end]
        trial_type = df["trial_type"].to_numpy()[start:end]
        center_start = df["center_start"].to_numpy()[start:end]
        center_end = df["center_end"].to_numpy()[start:end]
        outer_start = df["outer_start"].to_numpy()[start:end]
        outer_well = df["outer_well"].to_numpy()[start:end]
        goal_well = df["goal_well"].to_numpy()[start:end].astype('int')
        outer_success = df['outer_success'].to_numpy()[start:end].astype('bool')
        lock_start = df["lockout_start"].to_numpy()[start:end]
        lock_end = df["lockout_end"].to_numpy()[start:end]
        goal_block_phase = df['goal_block_phase'].to_numpy()[start:end]
        n_goal_block = df['n_goal_block'].to_numpy()[start:end]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,10))

        ax1.set_title("Landmark times")
        ax1.set_xlim(df["start_time"].iat[start], df["end_time"].iat[end-1])
        ax1.set_xlabel("Time (ms)")
        ax1.set_ylim(-1, 3)
        ax1.set_yticks([])
        ax1.plot(df["start_time"][start:end], np.zeros(len(df["start_time"][start:end])), "b|")
        ax1.plot(df["leave_home"][start:end], np.zeros(len(df["start_time"][start:end])), "r|")
        ax1.plot(center_start[np.where(trial_type==1)[0]], np.ones(sum(trial_type==1)), "g.")
        ax1.plot(center_start[np.where(trial_type==2)[0]], np.ones(sum(trial_type==2)), "b.")
        ax1.plot(center_end[np.where(trial_type==1)[0]], np.ones(sum(trial_type==1)), "g|")
        ax1.plot(center_end[np.where(trial_type==2)[0]], np.ones(sum(trial_type==2)), "b|")
        ax2.plot(outer_start[outer_success], np.ones(len(outer_success[outer_success])), "c.")
        ax2.plot(outer_start[~outer_success], np.ones(len(outer_success[~outer_success])), "m.") 
        ax1.plot(lock_start, np.ones(len(lock_end)), "rx")
        ax1.plot(lock_end, np.ones(len(lock_start)), "bx")
        ax1.legend(
            [
                "start time", 
                "leave home", 
                "center start (rip trial)", 
                "center start (wait trial)", 
                "center end (rip)", 
                "center end (wait)", 
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
        ax2.plot(trial_num[outer_success], outer_well[outer_success], "co")
        ax2.plot(trial_num[~outer_success], outer_well[~outer_success], "mo")  

        search_phases = goal_block_phase == 'search'
        repeat_phases = ~search_phases
        search_bounds = find_phase_bounds(search_phases)
        repeat_bounds = find_phase_bounds(repeat_phases)
        for start_bound, end_bound in search_bounds:
            ax2.axvspan(start_bound, end_bound, color='red', alpha=0.1)
        for start_bound, end_bound in repeat_bounds:
            ax2.axvspan(start_bound, end_bound, color='green', alpha=0.1)

        goal_block_starts = np.where(np.diff(n_goal_block) != 0)[0] + 1
        goal_block_starts = np.insert(goal_block_starts, 0, 0)
        for block_start in goal_block_starts:
            ax2.axvline(block_start, color='black', linestyle='--')

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

        # NOTE: currently out of date with gabby's 07/2025 updates to the TrialInfo8Arm table
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
        home_label = int(self.dio_map["homebeam"])
        rip_label = int(self.dio_map["Rbeam"])
        wait_label = int(self.dio_map["Wbeam"])
        arm1_label = int(self.dio_map["arm1beam"])

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
        
        # These have the correct number of lockouts 
        lockend_mask = dataArray[:,1]=="LOCKEND"
        lockends = dataArray[lockend_mask,0].astype(int) / MILLISECONDS_PER_SECOND + offset
        lockstarts = lockends - descriptors["lockout_period"]  # e.g lockout_period= 30.0

        # lockout 3s are explicitly triggered by statescript (not python) so there's the right number!
        # good confirmation for impatience errors
        lockout3_mask = (dataArray[:,1]=='LOCKOUT') & (dataArray[:,2] == '3')
        lockout3_starts = dataArray[lockout3_mask, 0].astype(int) / MILLISECONDS_PER_SECOND + offset
                
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
        downtimes = downtimesall[valid_poke_mask]
        downwells = downwellsall[valid_poke_mask]

        # get timestamps where home, rip, wait, and outer wells were visited
        home = uptimes[upwells == home_label]
        home_ends = downtimes[downwells == home_label]
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
        currentgoals = [ast.literal_eval(','.join([g for g in goal_format(goal) if g])) for goal in dataArray[currentgoal_mask, 3:]]
        currentgoal_times = dataArray[currentgoal_mask, 0].astype(int) / MILLISECONDS_PER_SECOND + offset
        
        # set exception for reggie, weird bug where the forageNum doesn't match what it actually was
        if (self.key['nwb_file_name'] == 'reginald20241021_.nwb') & (self.key['epoch'] == 4):
            self.key['descriptors']['forage_num'] = 2
        goals, goal_switch_times, num_goals, forage_num = detect_goal_info(currentgoals, currentgoal_times, descriptors, home, self.dio_map)

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
            for t, outerreps_time in enumerate(outerreps_times):
                if len(home[home >= outerreps_time]) == 0:  # situation where this was the last trial
                    outerreps = outerreps[:-1]
                    outerreps_times = outerreps_times[:-1]
                else:  # set the switch to be the next trial start (outerreps get printed out as soon as they finish a goal block, before they initiate the next trial)
                    outerreps_times[t] = home[home >= outerreps_time][0]
            # outerreps_times = [home[home >= outerreps_time][0] for outerreps_time in outerreps_times]  

        if "outer_reps" not in descriptors.keys():
            self.key['descriptors']['outer_reps'] = outerreps

        return home, rip, wait, outer, uptimes, upwells, downtimesall, downwellsall, lockstarts, lockends, waitends, ripends, goalrec, goals, goal_switch_times, outerreps, outerreps_times, lockout3_starts

    def __get_trials_df(self, home, rip, wait, outer, uptimes, upwells, downtimesall, downwellsall, lockstarts, lockends, waitends, ripends, goalrec, goals, goal_switch_times, outerreps, outerreps_times, lockout3_starts):
        """
        Filters parsed behavioral events for epoch based on task rules and stores results in a dataframe
        """
        # print(len(home))
        
        # only use start times that are NOT within 0.3 s of a lockstart
        goodhome = home[goodhome_filter(home, lockstarts, lockends)]
        home_label = int(self.dio_map["homebeam"])

        rip_light_on_times, rip_light_off_times, _ = get_dio_event_times(self.key, self.nwbf, 'Rlight')
        wait_light_on_times, wait_light_off_times, _ = get_dio_event_times(self.key, self.nwbf, 'Wlight')

        trial_data = []
        start_times = goodhome[:-1]
        end_times = goodhome[1:]

        n_home = 0
        n_wait = 0
        n_rip = 0
        n_lock1s = 0
        n_lock2s = 0
        n_lock3s = 0
        n_locks = 0
        bug_trials = 0
        untracked_lockouts = 0
        center_well = None 

        trial_data = []

        for t in range(len(goodhome) - 1):
        # for t in range(8):
            n_home += 1

            bug_trial = 0.0

            start_time = start_times[t]
            end_time = end_times[t]
            if start_time > end_time:
                print('uh oh!')
            lockout_trial = len(valid_indices(lockstarts, [start_time, end_time])) > 0

            # identify if the lights were turned on / off during the trial
            rip_light_on = len(valid_indices(rip_light_on_times, [start_time, end_time])) > 0
            wait_light_on = len(valid_indices(wait_light_on_times, [start_time, end_time])) > 0

            rip_light_off = len(valid_indices(rip_light_off_times, [start_time, end_time])) > 0
            wait_light_off = len(valid_indices(wait_light_off_times, [start_time, end_time])) > 0
            
            if rip_light_on & wait_light_on:
                print(f'known bug trial! trial {t + 1}, both rip and wait light turned on during this trial')
                bug_trial = 1.0
                bug_trials += 1
            
            if rip_light_off & wait_light_off:
                print(f'known bug trial! trial {t + 1}, both rip and wait light turned off during this trial')
                bug_trial = 1.0
                bug_trials += 1
            
            # lockout trial
            if lockout_trial:   
                n_locks += 1   
                lockout_start_time = lockstarts[valid_indices(lockstarts, [start_time, end_time])].tolist()[0]
                lockout_end_time = lockends[valid_indices(lockends, [start_time, end_time])].tolist()[0]

                # collect any wells that were visited during the lockout period, translating them to [1, 8] for outer arms and ['H', 'R', 'W']
                during_lockout = np.asarray(upwells[valid_indices(uptimes, [lockout_start_time, end_time])].tolist(), dtype='str')
                for beam_name in ['homebeam', 'Rbeam', 'Wbeam']:
                    n_beam = str(self.dio_map[beam_name])
                    if beam_name == 'homebeam':
                        during_lockout[during_lockout == n_beam] = 'H'
                    else:
                        during_lockout[during_lockout == n_beam] = beam_name.replace('beam', '')
                beam_to_arm = get_beam_to_arm(self.dio_map)
                for b, beam in enumerate(during_lockout):
                    if beam not in ['H', 'W', 'R']:
                        during_lockout[b] = beam_to_arm[int(beam)]

                rip_complete = len(valid_indices(ripends, [start_time, lockout_start_time -.1])) > 0   
                wait_complete = len(valid_indices(waitends, [start_time, lockout_start_time -.1])) > 0
                center_complete = rip_complete or wait_complete

                if rip_complete:
                    center_well = 'rip'
                    center_starts = rip
                    center_ends = ripends
                    n_rip += 1
                if wait_complete:
                    center_well = 'wait'
                    center_starts = wait
                    center_ends = waitends
                    n_wait += 1

                # center completed
                if center_complete:

                    # catch exception
                    center_start = center_starts[valid_indices(center_starts, [start_time, end_time])][0]
                    leave_center_registered = len(downtimesall[valid_indices(downtimesall, [center_start, lockout_start_time])]) > 0
                    if not leave_center_registered:
                        print(f'unexpected bug trial! trial {t + 1}, leaving center was never registered')
                        bug_trial = 1.0
                        bug_trials += 1
                        untracked_lockouts += 1
                        trial_df = lockout_untracked(t, start_time, end_time, lockout_start_time, lockout_end_time,
                                                    home_label, center_well, downtimesall, goalrec, bug_trial, during_lockout)
                        trial_data.append(trial_df)
                        continue

                    outer_complete = len(valid_indices(outer[0, :], [start_time, lockout_start_time-.1])) > 0
                    # outer completed
                    if outer_complete:
                        # print('center completed, outer completed, lockout on the way home')
                        n_lock2s += 1
                        trial_df = lockout_center_outer_complete(t, start_time, end_time, lockout_start_time, lockout_end_time,
                                                                home_label, center_well, center_starts, center_ends, downtimesall, downwellsall,
                                                                outer, goalrec, bug_trial, during_lockout, self.dio_map)
                        trial_data.append(trial_df)
                        continue

                    # outer not completed
                    else:
                        center_then_home = False
                        center_then_opp = False
                        # if rip trial, check if there's any home pokes between ripend and lockstart
                        if rip_complete:
                            rip_end = ripends[valid_indices(ripends, [start_time, lockout_start_time -.1])[0]]
                            center_then_home = len(valid_indices(home, [rip_end, lockout_start_time])) > 0
                            center_then_opp = len(valid_indices(wait, [rip_end, lockout_start_time])) > 0
                        
                        # if wait trial, check if there's any home pokes between waitend and lockstart
                        if wait_complete:
                            wait_end = waitends[valid_indices(waitends, [start_time, lockout_start_time -.1])[0]]
                            center_then_home = len(valid_indices(home, [wait_end, lockout_start_time])) > 0
                            center_then_opp = len(valid_indices(rip, [wait_end, lockout_start_time])) > 0
                        
                        if center_then_home:
                            # print('center completed, but then back to home')
                            n_lock2s += 1
                            trial_df = lockout_center_then_home(t, start_time, end_time, lockout_start_time, lockout_end_time,
                                                                home_label, center_well, center_starts, center_ends, downtimesall, downwellsall,
                                                                goalrec, bug_trial, during_lockout)
                            trial_data.append(trial_df)
                            continue
                        
                        elif center_then_opp:
                            # print('center completed, but then opposite center')
                            n_lock2s += 1
                            trial_df = lockout_center_then_opp(t, start_time, end_time, lockout_start_time, lockout_end_time,
                                                            home_label, center_well, center_starts, center_ends, downtimesall, downwellsall,
                                                            goalrec, bug_trial, during_lockout)
                            trial_data.append(trial_df)
                            continue

                # center not completed
                else:
                    
                    center_well = None

                    # center mismatch if the rip well was lit up before lockstart but the wait well was visited or vice versa
                    center_mismatch = False

                    rip_visited = len(valid_indices(rip, [start_time, lockout_start_time - .01])) > 0
                    wait_visited = len(valid_indices(wait, [start_time, lockout_start_time - .01])) > 0

                    if rip_light_on & wait_visited:
                        center_mismatch = True
                    if wait_light_on & rip_visited:
                        center_mismatch = True 

                    # outer first if center was not completed and an outer arm was visited
                    outer_first = len(valid_indices(outer[0, :], [start_time, lockout_start_time-.01])) > 0

                    # check if the leave home poke was registered (some weird bug trials that lack this)
                    leave_home_registered = len(downtimesall[(downwellsall == home_label) & (downtimesall >= start_time) & (downtimesall < lockout_start_time)]) > 0

                    # impatience error trial (lockout 3 gets printed out in statescript)
                    lockout3_trial = len(valid_indices(lockout3_starts, [start_time, end_time])) > 0

                    if not leave_home_registered:
                        print(f'unexpected bug trial! trial {t + 1}, leaving home was never registered')
                        bug_trial = 1.0
                        bug_trials += 1
                        untracked_lockouts += 1
                        trial_df = lockout_untracked(t, start_time, end_time, lockout_start_time, lockout_end_time,
                                                    home_label, center_well, downtimesall, goalrec, bug_trial, during_lockout)
                        trial_data.append(trial_df)
                        continue

                    # center mismatch
                    elif center_mismatch:
                        # print('center not completed, mismatch center well')
                        n_lock1s += 1
                        trial_df = lockout_center_mismatch(t, start_time, end_time, lockout_start_time, lockout_end_time,
                                                        home_label, center_well, downtimesall, downwellsall, goalrec, bug_trial, during_lockout)
                        trial_data.append(trial_df)
                        continue

                    elif outer_first:  # dual conditions because of some exceptions that were flagged
                        # print('center not completed, went to outer first')
                        n_lock2s += 1
                        trial_df = lockout_outer_first(t, start_time, end_time, lockout_start_time, lockout_end_time,
                                                    home_label, center_well, downtimesall, downwellsall, goalrec, bug_trial, during_lockout)
                        trial_data.append(trial_df)
                        continue

                    elif lockout3_trial:    
                        # print('center not completed, impatience error')
                        n_lock3s += 1
                        trial_df = lockout_center_impatience_error(t, start_time, end_time, lockout_start_time, lockout_end_time,
                                                                    home_label, center_well, downtimesall, downwellsall, goalrec, bug_trial, during_lockout)
                        trial_data.append(trial_df)
                        continue
            
                    elif (not rip_light_on) & (not wait_light_on):
                        print(f'known bug trial! trial {t + 1}, neither rip or wait lights turned on during this trial. this gets classified as a lock1')
                        bug_trial = 1.0
                        bug_trials += 1
                        n_lock1s += 1
                        trial_df = lockout_no_center_lights_bug(t, start_time, end_time, lockout_start_time, lockout_end_time,
                                                                    home_label, center_well, downtimesall, goalrec, bug_trial, during_lockout)
                        trial_data.append(trial_df)
                        continue
                    
                    else:
                        print(f'unexpected bug trial! trial {t + 1}, cause unknown')
                        bug_trial = 1.0
                        bug_trials += 1
                        untracked_lockouts += 1
                        trial_df = lockout_untracked(t, start_time, end_time, lockout_start_time, lockout_end_time,
                                                    home_label, center_well, downtimesall, goalrec, bug_trial, during_lockout)
                        trial_data.append(trial_df)
                        continue

            # valid trial  
            else:
                # print('center completed, outer completed')

                rip_complete = len(valid_indices(ripends, [start_time, end_time])) > 0   
                wait_complete = len(valid_indices(waitends, [start_time, end_time])) > 0

                if rip_complete:
                    center_well = 'rip'
                    center_starts = rip
                    center_ends = ripends
                    n_rip += 1
                if wait_complete:
                    center_well = 'wait'
                    center_starts = wait
                    center_ends = waitends
                    n_wait += 1

                # set values for the bug trials
                lockout_start_time = float('nan')
                lockout_end_time = float('nan')

                during_lockout = np.asarray(upwells[valid_indices(uptimes, [start_time, end_time])].tolist(), dtype='str')
                for beam_name in ['homebeam', 'Rbeam', 'Wbeam']:
                    n_beam = str(self.dio_map[beam_name])
                    if beam_name == 'homebeam':
                        during_lockout[during_lockout == n_beam] = 'H'
                    else:
                        during_lockout[during_lockout == n_beam] = beam_name.replace('beam', '')
                beam_to_arm = get_beam_to_arm(self.dio_map)
                for b, beam in enumerate(during_lockout):
                    if beam not in ['H', 'W', 'R']:
                        during_lockout[b] = beam_to_arm[int(beam)]

                # check that at least one of the center wells was completed
                center_complete = rip_complete | wait_complete
                if not center_complete:
                    print(f'unexpected bug trial! trial {t + 1}, center completion wasnt registered, a weird one...')
                    bug_trial = 1.0
                    bug_trials += 1
                    trial_df = lockout_untracked(t, start_time, end_time, lockout_start_time, lockout_end_time,
                                    home_label, 'NaN', downtimesall, goalrec, bug_trial, during_lockout, lockout=False)
                    trial_data.append(trial_df)
                    continue

                center_start_registered = len(center_starts[valid_indices(center_starts, [start_time, end_time])]) > 0
                if not center_start_registered:
                    print(f'unexpected bug trial! trial {t + 1}, center start was never registered')
                    bug_trial = 1.0
                    bug_trials += 1
                    trial_df = lockout_untracked(t, start_time, end_time, float('nan'), float('nan'),
                                                home_label, 'NaN', downtimesall, goalrec, bug_trial, during_lockout, lockout=False)
                    trial_data.append(trial_df)
                    continue

                center_start = center_starts[valid_indices(center_starts, [start_time, end_time])][0]

                # check if the leave home poke was registered (some weird bug trials that lack this)
                leave_home_registered = len(downtimesall[(downwellsall == home_label) & (downtimesall >= start_time) & (downtimesall < center_start)]) > 0

                if not leave_home_registered:
                    print(f'unexpected bug trial! trial {t + 1}, leaving home was never registered')
                    bug_trial = 1.0
                    bug_trials += 1
                    trial_df = lockout_untracked(t, start_time, end_time, float('nan'), float('nan'),
                                                home_label, 'NaN', downtimesall, goalrec, bug_trial, during_lockout, lockout=False)
                    trial_data.append(trial_df)
                    continue

                # check if the leave center poke was registered (some weird bug trials that lack this)
                outer_start_registered = len(outer[0, valid_indices(outer[0, :], [start_time, end_time])]) > 0
                if not outer_start_registered:
                    print(f'unexpected bug trial! trial {t + 1}, outer start was never registered')
                    bug_trial = 1.0
                    bug_trials += 1
                    trial_df = lockout_untracked(t, start_time, end_time, float('nan'), float('nan'),
                                                home_label, 'NaN', downtimesall, goalrec, bug_trial, during_lockout, lockout=False)
                    trial_data.append(trial_df)
                    continue

                outer_start = outer[0, valid_indices(outer[0, :], [start_time, end_time])[0]]
                leave_center_registered = len(downtimesall[valid_indices(downtimesall, [center_start, outer_start])]) > 0

                if not leave_center_registered:
                    print(f'unexpected bug trial! trial {t + 1}, leaving center was never registered')
                    bug_trial = 1.0
                    bug_trials += 1
                    trial_df = lockout_untracked(t, start_time, end_time, float('nan'), float('nan'),
                                                home_label, 'NaN', downtimesall, goalrec, bug_trial, during_lockout, lockout=False)
                    trial_data.append(trial_df)
                    continue

                # check that whichever trial this was, the poke was also initiated during this trial
                center_init = len(valid_indices(center_starts, [start_time, end_time])) > 0
                if not center_init:
                    print(f'unexpected bug trial! trial {t + 1}, center initiation wasnt registered, a weird one...')
                    bug_trial = 1.0
                    trial_df = lockout_untracked(t, start_time, end_time, lockout_start_time, lockout_end_time,
                                    home_label, center_well, downtimesall, goalrec, bug_trial, during_lockout, lockout=False)
                    trial_data.append(trial_df)
                    continue
                    
                # check that outer well was poked during this trial
                outer_init = len(valid_indices(outer[0, :], [start_time, end_time])) > 0
                if not outer_init:
                    print(f'unexpected bug trial! trial {t + 1}, outer poke wasnt registered, a weird one...')
                    bug_trial = 1.0
                    trial_df = lockout_untracked(t, start_time, end_time, lockout_start_time, lockout_end_time,
                                    home_label, center_well, downtimesall, goalrec, bug_trial, during_lockout, lockout=False)
                    trial_data.append(trial_df)
                    continue 

                trial_df = valid_trial(t, start_time, end_time, home_label, center_well, center_starts, center_ends,
                                    downtimesall, downwellsall, outer, goalrec, bug_trial)
                trial_data.append(trial_df)
                continue

        print(f'home: {n_home}, wait: {n_wait}, rip: {n_rip}, lock1s: {n_lock1s}, lock2s: {n_lock2s}, lock3s: {n_lock3s}, total locks: {n_locks}')

        trials_df = pd.concat(trial_data).reset_index(drop=True)

        return trials_df
    
    def __add_goal_block_info(self, trials_df, goals, goal_switch_times, outerreps):
        # helper function for addding all the info for each goal block
        trials_df = self.__assign_goals(trials_df, goals, goal_switch_times)
        trials_df = self.__assign_goal_blocks(trials_df)
        trials_df = self.__assign_outerreps(trials_df, outerreps)
        trials_df = trials_df.groupby('n_goal_block').apply(assign_search_repeat, include_groups=False).reset_index()

        return trials_df
    
    def __assign_goals(self, trials_df, goals, goal_switch_times):
        # assign the goal for each trial based on the goals and goal switch times derived from the sc current goals
        if len(goals) == 1:
            goal = goals[0]

            goal_start_time = trials_df['start_time'].values[0]
            goal_end_time = trials_df['end_time'].values[-1]
            goal_mask =  (trials_df['end_time'].values > goal_start_time) & (trials_df['end_time'] <= goal_end_time)

            if self.key['descriptors']['num_goals'] > 1:
                goal_indices = np.where(goal_mask)[0]
                for i in goal_indices:
                    trials_df.at[i, 'goal_well'] = goal
            else:
                trials_df.loc[goal_mask, 'goal_well'] = goal

        else:
            for g, goal in enumerate(goals):
                if g == 0:
                    goal_start_time = trials_df['start_time'].values[0]
                    goal_end_time = goal_switch_times[g]
                elif g == (len(goals) - 1):
                    goal_start_time = goal_switch_times[g - 1]
                    goal_end_time = trials_df['end_time'].values[-1]
                else:
                    goal_start_time = goal_switch_times[g - 1]
                    goal_end_time = goal_switch_times[g]

                goal_mask =  (trials_df['end_time'].values > goal_start_time) & (trials_df['end_time'] <= goal_end_time)

                if self.key['descriptors']['num_goals'] > 1:
                    goal_indices = np.where(goal_mask)[0]
                    for i in goal_indices:
                        trials_df.at[i, 'goal_well'] = goal
                else:
                    trials_df.loc[goal_mask, 'goal_well'] = goal
        
        # handle exceptions where currentgoal didn't get printed out during a lockout trial that happened right after a goal switch
        goal_switch_trials = get_goal_switch_trials(trials_df, self.key['descriptors'])
        for goal_switch_trial in goal_switch_trials:
            curr_goal = trials_df.loc[trials_df['trial_num'] == goal_switch_trial, 'goal_well'].values[0]
            last_trial = goal_switch_trial - 1 
            is_last_trial_lockout = trials_df.loc[trials_df['trial_num'] == last_trial, 'lockout_type'].values[0] != 0
            while is_last_trial_lockout:
                # print(f'updating goal_well for lockout trial preceding switch (trial {last_trial})')
                last_trial_index = np.where(trials_df['trial_num'] == last_trial)[0][0]
                trials_df.at[last_trial_index, 'goal_well'] = curr_goal 
                last_trial -= 1
                if last_trial == 0:
                    is_last_trial_lockout = False  # get out of the while loop if last trial == 0 (because trials are 1-indexed, so this doesn't exist)
                else:
                    is_last_trial_lockout = trials_df.loc[trials_df['trial_num'] == last_trial, 'lockout_type'].values[0] != 0

        return trials_df
    
    def __assign_goal_blocks(self, trials_df):
        # set goal block number here: use the goal switches to separate the different goal blocks
        goal_switch_trials = get_goal_switch_trials(trials_df, self.key['descriptors'])
        if len(goal_switch_trials) == 0:
            trials_df['n_goal_block'] = 1
            trials_df['complete_goal_block'] = False
        else:
            for t, goal_switch_trial in enumerate(goal_switch_trials):
                if t == 0:
                    block_start = 0
                else:
                    block_start = goal_switch_trials[t - 1]
                
                # default values
                block_end = goal_switch_trial
                complete_goal_block = True
                n_goal_block = t + 1

                # find and assign all relevant trials
                trials_df.loc[(trials_df['trial_num'] > block_start) & (trials_df['trial_num'] <= block_end), 'n_goal_block'] = n_goal_block
                trials_df.loc[trials_df['n_goal_block'] == n_goal_block, 'complete_goal_block'] = complete_goal_block

                # add the last, incomplete goal block! (special start and end because it isn't concluded by a goal switch)
                if t == len(goal_switch_trials) - 1:
                    block_start = goal_switch_trials[t]
                    block_end = len(trials_df) + 1
                    complete_goal_block = False

                    n_goal_block = t + 2

                    trials_df.loc[(trials_df['trial_num'] > block_start) & (trials_df['trial_num'] <= block_end), 'n_goal_block'] = n_goal_block
                    trials_df.loc[trials_df['n_goal_block'] == n_goal_block, 'complete_goal_block'] = complete_goal_block

        return trials_df
    
    def __assign_outerreps(self, trials_df, outerreps):
        # use outerreps readouts to fill in outer reps for each block (this is especially important for variable outer reps!)
        
        n_goal_blocks = len(trials_df['n_goal_block'].unique())

        if (not isinstance(outerreps, list)) & (not isinstance(outerreps, np.ndarray)):
            trials_df['outer_reps'] = outerreps

        elif len(outerreps) == 0:
            # wasn't detected in the sc printouts (this often happens with multi-goal data since it doesn't get printed out there)
            outerreps = self.key['descriptors']['outer_reps']
            if isinstance(outerreps, list) | isinstance(outerreps, np.ndarray):  # if the descriptor outerreps is > 1, but the detected ones are empty, it means that it's variable outerreps but the first goal block wasn't completed
                if len(outerreps) == 0:  # no outerreps found in the descriptors or in the statescript printouts (1 goal block and the python script wasn't attached)
                    trials_df['outer_reps'] = float('nan')
                elif n_goal_blocks == 1:
                    # NOTE: we don't have any sc printouts for outerreps for the first goal block
                    trials_df['outer_reps'] = float('nan')
            else:
                trials_df['outer_reps'] = outerreps

        else:
            for n_goal_block in np.arange(1, n_goal_blocks + 1):
                if n_goal_block == 1:
                    # NOTE: we don't have any sc printouts for outerreps for the first goal block
                    continue
                block_indices = trials_df['n_goal_block'] == n_goal_block

                trials_df.loc[block_indices, 'outer_reps'] = outerreps[n_goal_block - 2]

        # back-calculate the outerreps in the first goal block, if possible
        # (if it's variable outerreps and the goal block was not completed, then this is not possible and it will remain as 'nan')
        first_complete = trials_df.loc[trials_df['n_goal_block'] == 1, 'complete_goal_block'].values[0]
        if first_complete:
            first_block_indices = trials_df['n_goal_block'] == 1
            n_rewarded = np.sum(trials_df.loc[first_block_indices, 'outer_success'].values)
            # make an exception if it's == 16 because that's impossible and should actually be 15 (likely one extra due to overlap of goals with the next goal block)
            if n_rewarded == 16:
                n_rewarded = 15
            trials_df.loc[first_block_indices, 'outer_reps'] = n_rewarded

        # check whether the assigned outerreps match the ones found through outer_success, print a warning if not
        goal_blocks = trials_df['n_goal_block'].unique()
        for b, goal_block in enumerate(goal_blocks):
            block_df = trials_df[trials_df['n_goal_block'] == goal_block]
            if not block_df['complete_goal_block'].values[0]:  # skip any incomplete goal blocks
                continue
            assigned_outerreps = block_df['outer_reps'].values[0]
            found_outerreps = np.sum(block_df['outer_success'].values)
            if assigned_outerreps != found_outerreps:
                num_goals = self.key['descriptors']['num_goals']
                curr_goal = block_df['goal_well'].values[0]
                next_goal = trials_df.loc[trials_df['n_goal_block'] == goal_blocks[b + 1], 'goal_well'].values[1]
                last_outer_well = block_df['outer_well'].values[-1]
                print(f'WARNING: assigned outerreps ({assigned_outerreps}) != found outerreps ({found_outerreps}) for goal block {goal_block} | num_goals = {num_goals} | current goal is {curr_goal} while next goal is {next_goal} and last outerwell of curr block is {last_outer_well}')
        
        return trials_df

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

def goal_format(goal):
    # helper function for formatting weird string issues when currentgoal is printed out in statescript
    if goal[0] == '[':
        new_goal = []
        new_goal.append(goal[0] + goal[1])
        for g in goal[2:]:
            new_goal.append(g)
        return new_goal

    else:
        return goal

def get_beam_to_arm(dio_map):
    # helper function for getting dictionary to translate beam numbers to arm numbers
    beam_to_arm = {}
    for arm in np.arange(1, 9):
        beam_key = f'arm{arm}beam'
        beam = int(dio_map[beam_key])
        beam_to_arm[beam] = str(arm)
    return beam_to_arm

def detect_goal_info(currentgoals, currentgoal_times, descriptors, home, dio_map):
    # Goal: get current goals and current goal times via StateScriptLog 'CURRENTGOAL' printouts
    # Side goal: in the process, find descriptors forage_num, num_goal if they don't exist in the descriptors

    forage_num = None
    num_goal = None
    # If forage_num was set in the StateScript, retrieve:
    if "forage_num" in descriptors.keys():
        forage_num = descriptors['forage_num']
    # If num_goal was set in the StateScript, retrieve:
    if "num_goals" in descriptors.keys():
        num_goal = descriptors['num_goals']
    
    # exception for some epochs where foragenum was set to 3 when numgoals was set to 3
    if ('forage_num' in descriptors.keys()) & ('num_goals' in descriptors.keys()):
        if (num_goal > 1):
            forage_num = 0

    # initialize detected descriptors
    detect_forage_num = None
    detect_num_goal = None
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
            detect_num_goal = 1
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
            if num_goal is not None:
                if (num_goal != len(prev_goal)) | (num_goal != len(curr_goal)):
                    if not array_equal(np.asarray(curr_goal), np.asarray(prev_goal)):
                        warnings.warn(f'num_goal was set in state script as {num_goal} while the current goal has {len(curr_goal)} goals and the previous goal has {len(prev_goal)} goals!')
                        goals[-1] = '0'
                        goals.append(curr_goal)
                        trial_start_time = home[home <= curr_goal_time][-1]  # most recent trial start
                        if forage_num is not None:
                            if forage_num > 1:
                                forage_trial_start_time = trial_start_time
                        continue

            if not array_equal(np.asarray(curr_goal), np.asarray(prev_goal)):
                goals.append(curr_goal)
                trial_start_time = home[home <= curr_goal_time][-1]  # most recent trial start
                goal_switch_times.append(trial_start_time)
                detect_num_goal = len(curr_goal)

        if g == len(currentgoals) - 1:
            # if the last goal has multiple goals and detectforageassist is not None:
            #   - assign last goal to be 0
            if (len(curr_goal) > 1) & (detect_forage_num is not None):
                if len(curr_goal) != detect_forage_num:
                    raise RuntimeError(f'length of curr goal is {len(curr_goal)} does not match detected forage num {detect_forage_num}!')              
                goals[-1] = '0'
                goal_switch_times.append(forage_trial_start_time)

            # if the last goal has multiple goals and forageassist is None:
            #   - check that num_goals = len(prev_goal)
            #   - check that forage_num = 0
            if (len(curr_goal) > 1) & (detect_forage_num is None):
                if num_goal is not None:
                    if num_goal != len(curr_goal):
                        raise RuntimeError(f'statescript-derived num_goal ({num_goal}) does not match length of the current goal ({len(curr_goal)})!')
                detect_num_goal = len(curr_goal)

    if detect_forage_num is not None:
        detect_num_goal = 1

    if detect_num_goal > 1:
        detect_forage_num = 0
    
    # translate goals to be in the 1-8 range given the dio_mapping
    beam_to_arm = get_beam_to_arm(dio_map)
    for g, goal in enumerate(goals):
        if isinstance(goal, list):
            for i, beam in enumerate(goal):
                goal[i] = beam_to_arm[int(beam)]
        else:
            # exception for the case where we don't know what the goal is (if forage assist is on but they haven't chosen one of the goals yet)
            if goals[g] == '0':
                continue
            else:
                goals[g] = beam_to_arm[int(goal)]

    return goals, goal_switch_times, detect_num_goal, detect_forage_num


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

def create_trial_df(t, start_time, end_time, leave_home, trial_type, center_start, center_end, leave_center, center_success,
                    outer_well, outer_start, leave_outer, outer_success, lockout_start, lockout_end, during_lockout, 
                    lockout_type, lockout_desc, bug_trial):
    # helper function for creating a dataframe for a single trial
    trial = {
        'trial_num': t + 1,
        'start_time': start_time,
        'end_time': end_time,
        'leave_home': leave_home,
        'lockout_type': lockout_type,  # 0 if normal trial, 1 if lock1, 2 if lock2, 3 if lock3
        'lockout_desc': lockout_desc,  # quick description of what kind of lockout trial, mostly as a sanity check
        'bug_trial': bug_trial,  # 0 if normal trial, 1 if it's a bug trial
        'trial_type': trial_type,  # 1 if rip_complete, 2 if wait_complete
        'center_start': center_start,
        'center_end': center_end,
        'leave_center': leave_center,
        'center_success': center_success,
        'outer_well': outer_well,
        'goal_well': np.array(['NaN'], dtype='object'),
        'outer_start': outer_start,
        'leave_outer': leave_outer,
        'outer_success': outer_success,
        'lockout_start': lockout_start,
        'lockout_end': lockout_end,
        'during_lockout': during_lockout,
        'goal_block_phase': ''
    }

    return pd.DataFrame(trial, index=[0])

def valid_trial(t, start_time, end_time, home_label, center_well, 
                center_starts, center_ends, downtimesall, downwellsall, outer, goalrec, bug_trial=0.0, during_lockout=np.array([], dtype='str')):
    # helper function for creating a trial dataframe for a valid trial
    # collect relevant info
    if center_well == 'rip':
        trial_type = 1
    elif center_well == 'wait':
        trial_type = 2
    else:
        trial_type = 0
    center_start = center_starts[valid_indices(center_starts, [start_time, end_time])][0]
    center_end = center_ends[valid_indices(center_ends, [start_time, end_time])][0]
    outer_start = outer[0, valid_indices(outer[0, :], [start_time, end_time])[0]]
    leave_center = downtimesall[valid_indices(downtimesall, [center_start, outer_start])][-1]
    leave_home_time = downtimesall[(downwellsall == home_label) & (downtimesall >= start_time) & (downtimesall < leave_center)][-1]
    center_success = 1.0
    outer_well = outer[1, valid_indices(outer[0, :], [start_time, end_time])][0]

    leave_outer = downtimesall[valid_indices(downtimesall, [outer_start, end_time])][-1]
    outer_success = 1.0 if (len(valid_indices(goalrec, [start_time, end_time])) > 0) else 0.0

    if len(during_lockout) == 0:
        during_lockout=[[]]
    else:
        during_lockout = [during_lockout]
    
    # set lockout_type 
    lockout_type = 0
    lockout_desc = 'NaN'

    lockout_start_time = float('nan')
    lockout_end_time = float('nan')

    # populate trial dataframe
    trial_df = create_trial_df(t, start_time, end_time, leave_home_time, trial_type, center_start, center_end,
                    leave_center, center_success, outer_well, outer_start, leave_outer, outer_success,
                    lockout_start_time, lockout_end_time, during_lockout, lockout_type, lockout_desc, bug_trial)

    return trial_df

def lockout_center_outer_complete(t, start_time, end_time, lockout_start_time, lockout_end_time, home_label, center_well, 
                                  center_starts, center_ends, downtimesall, downwellsall, outer, goalrec, bug_trial=0.0, during_lockout=np.array([], dtype='str'), dio_map={}):
    # collect relevant info
    leave_home_time = downtimesall[(downwellsall == home_label) & (downtimesall >= start_time) & (downtimesall < lockout_start_time)][-1]
    if center_well == 'rip':
        trial_type = 1
    elif center_well == 'wait':
        trial_type = 2
    else:
        trial_type = 0
    center_start = center_starts[valid_indices(center_starts, [start_time, end_time])][0]
    center_end = center_ends[valid_indices(center_ends, [start_time, end_time])][0]
    outer_start = outer[0, valid_indices(outer[0, :], [start_time, lockout_start_time-.1])[0]]
    leave_center = downtimesall[valid_indices(downtimesall, [center_start, outer_start])][-1]
    center_success = 1.0
    outer_well = outer[1, valid_indices(outer[0, :], [start_time, lockout_start_time-.1])][0]

    dio_outer = int(dio_map[f'arm{int(outer_well)}beam'])
    leave_outer = downtimesall[(downtimesall >= outer_start) & (downtimesall < lockout_start_time) & (downwellsall == dio_outer)][-1]
    outer_success = 1.0 if (len(valid_indices(goalrec, [start_time, lockout_start_time])) > 0) else 0.0

    if len(during_lockout) == 0:
        during_lockout=[[]]
    else:
        during_lockout = [during_lockout]
    
    # set lockout_type 
    lockout_type = 2
    lockout_desc = 'center_outer_complete'

    # populate trial dataframe
    trial_df = create_trial_df(t, start_time, end_time, leave_home_time, trial_type, center_start, center_end,
                    leave_center, center_success, outer_well, outer_start, leave_outer, outer_success,
                    lockout_start_time, lockout_end_time, during_lockout, lockout_type, lockout_desc, bug_trial)

    return trial_df

def lockout_center_then_home(t, start_time, end_time, lockout_start_time, lockout_end_time, home_label, center_well, 
                             center_starts, center_ends, downtimesall, downwellsall, goalrec, bug_trial=0.0, during_lockout=np.array([], dtype='str')):
    # collect relevant info
    leave_home_time = downtimesall[(downwellsall == home_label) & (downtimesall >= start_time) & (downtimesall < lockout_start_time)][-1]
    if center_well == 'rip':
        trial_type = 1
    elif center_well == 'wait':
        trial_type = 2
    else:
        trial_type = 0
    center_start = center_starts[valid_indices(center_starts, [start_time, end_time])][0]
    center_end = center_ends[valid_indices(center_ends, [start_time, end_time])][0]
    outer_start = float('nan')
    leave_center = downtimesall[valid_indices(downtimesall, [center_start, lockout_start_time])][-1]
    center_success = 1.0
    outer_well = float('nan')
    leave_outer = float('nan')
    outer_success = 1.0 if (len(valid_indices(goalrec, [start_time, lockout_start_time])) > 0) else 0.0

    if len(during_lockout) == 0:
        during_lockout=[[]]
    else:
        during_lockout = [during_lockout]
    
    # set lockout_type 
    lockout_type = 2
    lockout_desc = 'center_then_home'

    # populate trial dataframe
    trial_df = create_trial_df(t, start_time, end_time, leave_home_time, trial_type, center_start, center_end,
                    leave_center, center_success, outer_well, outer_start, leave_outer, outer_success,
                    lockout_start_time, lockout_end_time, during_lockout, lockout_type, lockout_desc, bug_trial)

    return trial_df

def lockout_center_then_opp(t, start_time, end_time, lockout_start_time, lockout_end_time, home_label, center_well, 
                             center_starts, center_ends, downtimesall, downwellsall, goalrec, bug_trial=0.0, during_lockout=np.array([], dtype='str')):
    # collect relevant info
    leave_home_time = downtimesall[(downwellsall == home_label) & (downtimesall >= start_time) & (downtimesall < lockout_start_time)][-1]
    if center_well == 'rip':
        trial_type = 1
    elif center_well == 'wait':
        trial_type = 2
    else:
        trial_type = 0
    center_start = center_starts[valid_indices(center_starts, [start_time, end_time])][0]
    center_end = center_ends[valid_indices(center_ends, [start_time, end_time])][0]
    outer_start = float('nan')
    leave_center = downtimesall[valid_indices(downtimesall, [center_start, lockout_start_time])][-1]
    center_success = 1.0
    outer_well = float('nan')
    leave_outer = float('nan')
    outer_success = 1.0 if (len(valid_indices(goalrec, [start_time, lockout_start_time])) > 0) else 0.0

    if len(during_lockout) == 0:
        during_lockout=[[]]
    else:
        during_lockout = [during_lockout]
    
    # set lockout_type 
    lockout_type = 2
    lockout_desc = 'center_then_opp'

    # populate trial dataframe
    trial_df = create_trial_df(t, start_time, end_time, leave_home_time, trial_type, center_start, center_end,
                    leave_center, center_success, outer_well, outer_start, leave_outer, outer_success,
                    lockout_start_time, lockout_end_time, during_lockout, lockout_type, lockout_desc, bug_trial)

    return trial_df

def lockout_center_mismatch(t, start_time, end_time, lockout_start_time, lockout_end_time, home_label, center_well,
                            downtimesall, downwellsall, goalrec, bug_trial=0.0, during_lockout=np.array([], dtype='str')):
    # collect relevant info
    leave_home_time = downtimesall[(downwellsall == home_label) & (downtimesall >= start_time) & (downtimesall < lockout_start_time)][-1]
    if center_well == 'rip':
        trial_type = 1
    elif center_well == 'wait':
        trial_type = 2
    else:
        trial_type = 0
    center_start = float('nan')
    center_end = float('nan')
    outer_start = float('nan')
    leave_center = float('nan')
    center_success = 0.0
    outer_well = float('nan')
    leave_outer = float('nan')
    outer_success = 1.0 if (len(valid_indices(goalrec, [start_time, lockout_start_time])) > 0) else 0.0

    if len(during_lockout) == 0:
        during_lockout=[[]]
    else:
        during_lockout = [during_lockout]
    
    # set lockout_type 
    lockout_type = 1
    lockout_desc = 'center_mismatch'

    # populate trial dataframe
    trial_df = create_trial_df(t, start_time, end_time, leave_home_time, trial_type, center_start, center_end,
                    leave_center, center_success, outer_well, outer_start, leave_outer, outer_success,
                    lockout_start_time, lockout_end_time, during_lockout, lockout_type, lockout_desc, bug_trial)

    return trial_df

def lockout_outer_first(t, start_time, end_time, lockout_start_time, lockout_end_time, home_label, center_well,
                            downtimesall, downwellsall, goalrec, bug_trial=0.0, during_lockout=np.array([], dtype='str')):
    # collect relevant info
    leave_home_time = downtimesall[(downwellsall == home_label) & (downtimesall >= start_time) & (downtimesall < lockout_start_time)][-1]
    if center_well == 'rip':
        trial_type = 1
    elif center_well == 'wait':
        trial_type = 2
    else:
        trial_type = 0
    center_start = float('nan')
    center_end = float('nan')
    outer_start = float('nan')
    leave_center = float('nan')
    center_success = 0.0
    outer_well = float('nan')
    leave_outer = float('nan')
    outer_success = 1.0 if (len(valid_indices(goalrec, [start_time, lockout_start_time])) > 0) else 0.0

    if len(during_lockout) == 0:
        during_lockout=[[]]
    else:
        during_lockout = [during_lockout]
    
    # set lockout_type 
    lockout_type = 2
    lockout_desc = 'outer_first'

    # populate trial dataframe
    trial_df = create_trial_df(t, start_time, end_time, leave_home_time, trial_type, center_start, center_end,
                    leave_center, center_success, outer_well, outer_start, leave_outer, outer_success,
                    lockout_start_time, lockout_end_time, during_lockout, lockout_type, lockout_desc, bug_trial)

    return trial_df

def lockout_center_impatience_error(t, start_time, end_time, lockout_start_time, lockout_end_time, home_label, center_well,
                                    downtimesall, downwellsall, goalrec, bug_trial=0.0, during_lockout=np.array([], dtype='str')):
    # collect relevant info
    leave_home_time = downtimesall[(downwellsall == home_label) & (downtimesall >= start_time) & (downtimesall < lockout_start_time)][-1]
    if center_well == 'rip':
        trial_type = 1
    elif center_well == 'wait':
        trial_type = 2
    else:
        trial_type = 0
    center_start = float('nan')
    center_end = float('nan')
    outer_start = float('nan')
    leave_center = float('nan')
    center_success = 0.0
    outer_well = float('nan')
    leave_outer = float('nan')
    outer_success = 1.0 if (len(valid_indices(goalrec, [start_time, lockout_start_time])) > 0) else 0.0

    if len(during_lockout) == 0:
        during_lockout=[[]]
    else:
        during_lockout = [during_lockout]
    
    # set lockout_type 
    lockout_type = 3
    lockout_desc = 'center_impatience_error'

    # populate trial dataframe
    trial_df = create_trial_df(t, start_time, end_time, leave_home_time, trial_type, center_start, center_end,
                    leave_center, center_success, outer_well, outer_start, leave_outer, outer_success,
                    lockout_start_time, lockout_end_time, during_lockout, lockout_type, lockout_desc, bug_trial)

    return trial_df

def lockout_no_center_lights_bug(t, start_time, end_time, lockout_start_time, lockout_end_time, home_label, center_well,
                                    downtimesall, goalrec, bug_trial=0.0, during_lockout=np.array([], dtype='str')):
    # collect relevant info
    leave_home_time = float('nan')
    if center_well == 'rip':
        trial_type = 1
    elif center_well == 'wait':
        trial_type = 2
    else:
        trial_type = 0
    center_start = float('nan')
    center_end = float('nan')
    outer_start = float('nan')
    leave_center = float('nan')
    center_success = 0.0
    outer_well = float('nan')
    leave_outer = float('nan')
    outer_success = 1.0 if (len(valid_indices(goalrec, [start_time, lockout_start_time])) > 0) else 0.0

    if len(during_lockout) == 0:
        during_lockout=[[]]
    else:
        during_lockout = [during_lockout]
    
    # set lockout_type 
    lockout_type = 1
    lockout_desc = 'no_center_lights_bug'

    # populate trial dataframe
    trial_df = create_trial_df(t, start_time, end_time, leave_home_time, trial_type, center_start, center_end,
                    leave_center, center_success, outer_well, outer_start, leave_outer, outer_success,
                    lockout_start_time, lockout_end_time, during_lockout, lockout_type, lockout_desc, bug_trial)

    return trial_df

def lockout_untracked(t, start_time, end_time, lockout_start_time, lockout_end_time, home_label, center_well,
                        downtimesall, goalrec, bug_trial=0.0, during_lockout=np.array([], dtype='str'), lockout=True):
    # collect relevant info
    leave_home_time = float('nan')
    if center_well == 'rip':
        trial_type = 1
    elif center_well == 'wait':
        trial_type = 2
    else:
        trial_type = 0
    center_start = float('nan')
    center_end = float('nan')
    outer_start = float('nan')
    leave_center = float('nan')
    center_success = 0.0
    outer_well = float('nan')
    leave_outer = float('nan')
    if lockout:
        outer_success = 1.0 if (len(valid_indices(goalrec, [start_time, lockout_start_time])) > 0) else 0.0
    else:
        outer_success = 1.0 if (len(valid_indices(goalrec, [start_time, end_time])) > 0) else 0.0


    if len(during_lockout) == 0:
        during_lockout=[[]]
    else:
        during_lockout = [during_lockout]
    
    # set lockout_type 
    lockout_type = float('nan')
    lockout_desc = 'unexpected bug trial'

    # populate trial dataframe
    trial_df = create_trial_df(t, start_time, end_time, leave_home_time, trial_type, center_start, center_end,
                    leave_center, center_success, outer_well, outer_start, leave_outer, outer_success,
                    lockout_start_time, lockout_end_time, during_lockout, lockout_type, lockout_desc, bug_trial)

    return trial_df

def get_goal_switch_trials(trials_df, descriptors):
    # retrieve trials where the goal blocks switch from one to another, helper function for assigning other goal block info
    if descriptors['num_goals'] == 1:
        goal_switch_trials = np.where(np.diff(np.array(trials_df['goal_well'].values, dtype='int')) != 0)[0] + 1 + 1
    elif descriptors['num_goals'] > 1:
        goal_wells = np.array(trials_df['goal_well'].values)
        diff_vals = np.asarray([int(np.any(np.diff([np.array(goal_wells[g], dtype='int'), np.array(goal_wells[g - 1], dtype='int')], axis=0) != 0)) for g in np.arange(2, len(goal_wells))])
        goal_switch_trials = np.where(diff_vals != 0)[0] + 1 + 1
    
    return goal_switch_trials

def assign_search_repeat(goal_block_df):
    # determine which parts of each goal block are search vs repeat
    if goal_block_df.loc[goal_block_df['outer_success'] == 1].empty:
        last_search = goal_block_df['trial_num'].values[-1]
    else:
        last_search = goal_block_df.loc[goal_block_df['outer_success'] == 1, 'trial_num'].values[0]
    first_repeat = last_search + 1
    goal_block_df.loc[goal_block_df['trial_num'] < first_repeat, 'goal_block_phase'] = 'search'
    goal_block_df.loc[goal_block_df['trial_num'] >= first_repeat, 'goal_block_phase'] = 'repeat'
    return goal_block_df


def get_dio_event_times(key, nwbf, dio_event_name):

    dio_obj_id = (sgc.DIOEvents & {'nwb_file_name': key['nwb_file_name'], 'dio_event_name': dio_event_name}).fetch1('dio_object_id')
    dios = nwbf.objects[dio_obj_id]
    dio_times_all = np.asarray(dios.timestamps)
    dio_data_all = np.asarray(dios.data)

    # get start and end times of the epoch
    epoch_name = (sgc.TaskEpoch & {"nwb_file_name": key["nwb_file_name"], "epoch": key["epoch"]}).fetch1("interval_list_name")
    epoch_valid_times = ( # gets time bounds of epoch
        sgc.IntervalList & {"nwb_file_name" : key["nwb_file_name"], "interval_list_name": epoch_name}
    ).fetch1("valid_times")
    epoch_start_time, epoch_end_time = epoch_valid_times.squeeze()

    # constrain the extracted data to times within this epoch
    epoch_time_mask = (dio_times_all >= epoch_start_time) & (dio_times_all < epoch_end_time)
    dio_times = dio_times_all[epoch_time_mask]
    dio_data = dio_data_all[epoch_time_mask]

    # get start and end indices of the dio event being on
    n_dio_events = len(np.where(dio_data == 1)[0])
    dio_events_start_idx = np.where(dio_data == 1)[0].astype('int')
    dio_events_end_idx = []
    for start in dio_events_start_idx:
        end = np.where(dio_data[start + 1:] == 0)[0]
        if end.size > 0:
            dio_events_end_idx.append(start + 1 + end[0])

    # get start and end times of dio event being on
    dio_events_start_times = dio_times[dio_events_start_idx]
    dio_events_end_times = dio_times[dio_events_end_idx]

    # pad end times with end time of the epoch data to account for the case where the homelight was left on after the run session
    len_diff = len(dio_events_start_times) - len(dio_events_end_times)
    for i in range(len_diff):
        dio_events_end_times = np.append(dio_events_end_times, epoch_end_time)

    dio_events_intervals = np.vstack([dio_events_start_times, dio_events_end_times]).T

    return dio_events_start_times, dio_events_end_times, dio_events_intervals

def find_phase_bounds(phases):
    
    diff = np.diff(phases.astype('int'))
    start_indices = np.where(diff == 1)[0] + 1
    end_indices = np.where(diff == -1)[0] + 1

    # edge cases
    if phases[0]:
        start_indices = np.insert(start_indices, 0, 0)
    if phases[-1]:
        end_indices = np.append(end_indices, len(phases))

    bounds = list(zip(start_indices, end_indices))

    return bounds