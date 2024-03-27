import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from abc import ABC, abstractclassmethod


class TrialParser(ABC):
    @abstractclassmethod
    def parse_trials(self, epoch_key):
        pass


class V8TrialParser(TrialParser):
    def __init__(self, text, diomap, time_offset, desc):
        """
        text (str): raw text from statescript log
        diomap (dict): map event names to their dio channel (e.g. {'homebeam': 'Din1', ...})
        firsthometime (float): timestamp of 1st time homebeam was triggered. used for calculating timestamp offset.
        desc (dict): TaskEpoch key detailing session name and epoch number
        """
        self.text = text
        self.diomap = diomap
        self.time_offset = time_offset
        self.desc = desc
        self.trials_df = None

    def parse_trials(self):
        """Returne a dataframe containing the following trial metrics
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

        # text: contents of the epoch's statescript file
        parsed_events = self.__parse_statescript()
        events_df = self.__filter_events(*parsed_events)
        self.trials_df = events_df
        return events_df

    @staticmethod
    def plot_trials(df, session, epoch_num, return_fig=False):

        trialtype = df["trial_type"].to_numpy()
        RWstart = df["rw_start"].to_numpy()
        RWend = df["rw_end"].to_numpy()
        outertime = df["outer_time"].to_numpy()
        outerwell = df["outer_well"].to_numpy()
        goalwell = df["goal_well"].to_numpy()

        lockstarts = []
        lockends = []
        for t in range(len(df)):
            lockstarts.extend(df["lockout_starts"].iat[t])
            lockends.extend(df["lockout_ends"].iat[t])

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

        ax1.set_title("Landmark times")
        ax1.set_xlim(df["start_time"].iat[0], df["end_time"].iat[-1])
        ax1.set_xlabel("Time (ms)")
        ax1.set_ylim(-1, 3)
        ax1.set_yticks([])
        ax1.plot(df["start_time"], np.zeros(len(df["start_time"])), "b|")
        ax1.plot(df["leave_home"], np.zeros(len(df["start_time"])), "r|")
        ax1.plot(
            RWstart[np.where(trialtype == 1)[0]], np.ones(sum(trialtype == 1)), "g*"
        )
        ax1.plot(
            RWstart[np.where(trialtype == 2)[0]], np.ones(sum(trialtype == 2)), "b*"
        )
        ax1.plot(RWend[np.where(trialtype == 1)[0]], np.ones(sum(trialtype == 1)), "g.")
        ax1.plot(RWend[np.where(trialtype == 2)[0]], np.ones(sum(trialtype == 2)), "b.")
        ax1.plot(
            outertime[np.equal(outerwell, goalwell).nonzero()[0]],
            np.ones(sum(outerwell == goalwell)),
            "m*",
        )
        ax1.plot(
            outertime[np.not_equal(outerwell, goalwell).nonzero()[0]],
            np.ones(sum(outerwell != goalwell)),
            "mo",
        )
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
                "lockends",
            ],
            loc="upper right",
            ncol=2,
        )

        ax2.set_title("Goal well visits")
        ax2.set_xlim(df["start_time"].iat[0], df["end_time"].iat[-1])
        ax2.set_xlabel("Time (ms)")
        ax2.set_ylabel("Well number")
        ax2.plot(
            outertime[np.equal(outerwell, goalwell).nonzero()[0]],
            outerwell[np.equal(outerwell, goalwell).nonzero()[0]],
            "m*",
        )
        ax2.plot(
            outertime[np.not_equal(outerwell, goalwell).nonzero()[0]],
            outerwell[np.not_equal(outerwell, goalwell).nonzero()[0]],
            "mo",
        )
        ax2.plot(df["leave_outer"], df["outer_well"], "m|")
        ax2.legend(["goal well arms", "non-goal well arms"])
        fig.suptitle(f"{session}, epoch {epoch_num}", fontsize=16)

        if return_fig:
            return plt.gcf()

    def __parse_statescript(self):

        lines = self.text.split("\n")
        data = [line.split(" ") for line in lines if len(line) > 0 and line[0] != "#"]
        dataArray = np.array([d + [""] * (6 - len(d)) for d in data])

        # initialize uptimesall, downtimesall, lockends, lockstarts, goalcount, goalcounttimes, waitends, ripends
        uptimesall = (
            dataArray[np.where(dataArray[:, 1] == "UP"), 0][0].astype(int) / 1000
        )
        upwellsall = dataArray[np.where(dataArray[:, 1] == "UP"), 2][0].astype(int)
        # DIO time = unix time (s since 1970), SC times = ms since Trodes began
        offset = self.time_offset - uptimesall[upwellsall == 1][0]

        uptimesall = uptimesall + offset

        downtimesall = (
            dataArray[np.where(dataArray[:, 1] == "DOWN"), 0][0].astype(int) / 1000
            + offset
        )
        downwellsall = dataArray[np.where(dataArray[:, 1] == "DOWN"), 2][0].astype(int)

        lockends = (
            dataArray[np.where(dataArray[:, 1] == "LOCKEND"), 0][0].astype(int) / 1000
            + offset
        )

        lockstarts = lockends - 30  # !! lockoutPeriod= 30000

        goalcount = dataArray[np.where(dataArray[:, 1] == "goalTotal"), 3][0].astype(
            int
        )
        goalcounttimes = (
            dataArray[np.where(dataArray[:, 1] == "goalTotal"), 0][0].astype(int) / 1000
            + offset
        )

        # CHANGE BASED ON RAT
        waitends = (
            dataArray[np.where(dataArray[:, 1] == "CLICK1"), 0][0].astype(int) / 1000
            + offset
        )
        ripends = (
            dataArray[np.where(dataArray[:, 1] == "BEEP1"), 0][0].astype(int) / 1000
            + offset
        )

        # TODO filter to timestamps that weren't repeat pokes
        nonrepinds = (
            np.where(np.diff(upwellsall) != 0)[0] + 1
        )  # use +1 to get first value in string of duplicates
        homeindsall = np.where(upwellsall == int(self.diomap["homelight"][-1]))[0]
        # home pokes that followed a lockout
        afterlockinds = np.intersect1d(lookup(lockends, uptimesall), homeindsall)
        mask = np.concatenate((nonrepinds, afterlockinds))
        uptimes = uptimesall[mask]
        upwells = upwellsall[mask]

        # TODO: shift well numbers for upwells/times and downwells/times to be backward compatible
        home = uptimes[upwells == int(self.diomap["homebeam"][-1])]
        rip = uptimes[upwells == int(self.diomap["Rbeam"][-1])]
        wait = uptimes[upwells == int(self.diomap["Wbeam"][-1])]
        outer = np.array([uptimes[upwells > 3], upwells[upwells > 3]])
        # filter goal records to only count times where goalcount increased
        goalrec = goalcounttimes[np.where(np.diff(goalcount) > 0)[0] + 1]
        return (
            home,
            rip,
            wait,
            outer,
            uptimes,
            upwells,
            downtimesall,
            downwellsall,
            lockstarts,
            lockends,
            waitends,
            ripends,
            goalrec,
        )

    def __filter_events(
        self,
        home,
        rip,
        wait,
        outer,
        uptimes,
        upwells,
        downtimesall,
        downwellsall,
        lockstarts,
        lockends,
        waitends,
        ripends,
        goalrec,
    ):
        """
        Generates a dataframe containing behavior metrics for the given run epoch.
        """
        # only use start times that are NOT within 0.3 s of a lockstart
        goodhome = home[goodhome_filter(home, lockstarts)]

        # initialize dataframe to be populated
        tmp = pd.DataFrame()
        tmp.index += 1
        tmp["start_time"] = goodhome[
            :-1
        ]  # starttime (all homepokes excluding the last)
        tmp["end_time"] = goodhome[1:]  # endtime (all homepokes excluding the first)
        tmp["lockout_starts"] = np.empty((len(tmp["start_time"]), 0)).tolist()
        tmp["lockout_ends"] = np.empty((len(tmp["start_time"]), 0)).tolist()
        tmp["during_lockout"] = np.empty(
            (len(tmp), 0)
        ).tolist()  # tuple of uptimes and waittimes lists?
        tmp["lockout_type"] = np.zeros(len(tmp["start_time"]), dtype=np.int8)
        tmp["rw_start"] = np.zeros(len(tmp["start_time"]))
        tmp["rw_end"] = np.zeros(len(tmp["start_time"]))
        tmp["leave_home"] = np.zeros(len(tmp["start_time"]))
        tmp["leave_rw"] = np.zeros(len(tmp["start_time"]))
        tmp["trial_type"] = np.zeros(len(tmp["start_time"]), dtype=np.int8)
        tmp["outer_well"] = np.zeros(len(tmp["start_time"]), dtype=np.int8)
        tmp["outer_time"] = np.zeros(len(tmp["start_time"]))
        tmp["leave_outer"] = np.zeros(len(tmp["start_time"]))
        tmp["outer_success"] = np.zeros(len(tmp["start_time"]))
        tmp["goal_well"] = np.zeros(len(tmp["start_time"]), dtype=np.int8)
        tmp["rw_success"] = np.zeros(len(tmp["start_time"]), dtype=np.int8)

        for t in range(len(goodhome) - 1):
            try:
                # error trials
                start_time = tmp["start_time"].iat[t]
                end_time = tmp["end_time"].iat[t]
                if len(valid_indices(lockstarts, [start_time, end_time])) > 0:
                    tmp["lockout_starts"].iat[t] = lockstarts[
                        valid_indices(lockstarts, [start_time, end_time])
                    ].tolist()
                    tmp["lockout_ends"].iat[t] = lockends[
                        valid_indices(lockstarts, [start_time, end_time])
                    ].tolist()
                    tmp["during_lockout"].iat[t] = upwells[
                        valid_indices(
                            uptimes, [tmp["lockout_starts"].iat[t][0], end_time]
                        )
                    ].tolist()  # TODO: add uptimes in addition to wells
                    # completed rip or wait well succesfully
                    if (
                        len(valid_indices(ripends, [start_time, end_time])) > 0
                        or len(valid_indices(waitends, [start_time, end_time])) > 0
                    ):
                        tmp["lockout_type"].iat[t] = 1
                        tmp["rw_success"].iat[t] = 1
                        if len(
                            valid_indices(
                                rip, [start_time, tmp["lockout_starts"].iat[t][0] - 0.1]
                            )
                        ):
                            tmp["trial_type"].iat[t] = 1  # type=rip
                            tmp["rw_start"].iat[t] = rip[
                                valid_indices(
                                    rip,
                                    [start_time, tmp["lockout_starts"].iat[t][0] - 0.1],
                                )
                            ][
                                0
                            ]  # list??
                            tmp["leave_home"].iat[t] = downtimesall[
                                valid_indices(
                                    downtimesall, [start_time, tmp["rw_start"].iat[t]]
                                )
                            ][-1]
                            tmp["rw_end"].iat[t] = ripends[
                                valid_indices(ripends, [start_time, end_time])
                            ][0]
                            tmp["leave_rw"].iat[t] = downtimesall[
                                valid_indices(
                                    downtimesall,
                                    [
                                        tmp["rw_start"].iat[t],
                                        tmp["lockout_starts"].iat[t][0],
                                    ],
                                )
                            ][-1]
                            # those trials when he gets click/beep just as he leaves
                            if (
                                tmp["rw_end"].iat[t] - tmp["leave_rw"].iat[t]
                            ) < 0.3 and (
                                tmp["rw_end"].iat[t] - tmp["leave_rw"].iat[t]
                            ) > 0:
                                tmp["leave_rw"].iat[t] = tmp["rw_end"].iat[t]
                        elif len(
                            valid_indices(
                                wait,
                                [start_time, tmp["lockout_starts"].iat[t][0] - 0.1],
                            )
                        ):
                            tmp["trial_type"].iat[t] = 2
                            # type=wait
                            tmp["rw_start"].iat[t] = wait[
                                valid_indices(
                                    wait,
                                    [start_time, tmp["lockout_starts"].iat[t][0] - 0.1],
                                )
                            ][0]
                            tmp["leave_home"].iat[t] = downtimesall[
                                valid_indices(
                                    downtimesall, [start_time, tmp["rw_start"].iat[t]]
                                )
                            ][-1]
                            tmp["rw_end"].iat[t] = waitends[
                                valid_indices(waitends, [start_time, end_time])
                            ][0]
                            tmp["leave_rw"].iat[t] = downtimesall[
                                valid_indices(
                                    downtimesall,
                                    [
                                        tmp["rw_start"].iat[t],
                                        tmp["lockout_starts"].iat[t][0],
                                    ],
                                )
                            ][-1]
                            # those trials when he gets click/beep just as he leaves
                            if (
                                tmp["rw_end"].iat[t] - tmp["leave_rw"].iat[t]
                            ) < 0.3 and (
                                tmp["rw_end"].iat[t] - tmp["leave_rw"].iat[t]
                            ) > 0:
                                tmp["leave_rw"].iat[t] = tmp["rw_end"].iat[t]

                        # also completed outer successfully (lockedout on way home, ie by going to r/w), still considered locktype1, order error
                        if (
                            len(
                                valid_indices(
                                    outer[:, 0],
                                    [start_time, tmp["lockout_starts"].iat[t][0] - 0.1],
                                )
                            )
                            > 0
                        ):
                            tmp["outer_time"].iat[t] = outer[
                                valid_indices(
                                    outer[:, 0],
                                    [start_time, tmp["lockout_starts"].iat[t][0] - 0.1],
                                )[0],
                                0,
                            ]
                            tmp["outer_well"].iat[t] = outer[
                                valid_indices(
                                    outer[:, 0],
                                    [start_time, tmp["lockout_starts"].iat[t][0] - 0.1],
                                )[0],
                                1,
                            ]
                            tmp["leave_outer"].iat[t] = downtimesall[
                                (downtimesall >= tmp["outer_time"].iat[t])
                                & (downtimesall < tmp["lockout_starts"].iat[t][0])
                                & (downwellsall == tmp["outer_well"].iat[t])
                            ][0]
                            if (
                                len(
                                    valid_indices(
                                        goalrec,
                                        [
                                            tmp["start_time"].iat[t],
                                            tmp["lockout_starts"].iat[t][0],
                                        ],
                                    )
                                )
                                > 0
                            ):  # received outer reward
                                tmp["goal_well"].iat[t] = tmp["outer_well"].iat[t]
                                tmp["outer_success"].iat[t] = 1

                    # did not complete rip/wait successfully
                    else:
                        tmp["rw_success"].iat[t] = 0
                        # if he locks out by going straight out (locktype1 order error)
                        if len(
                            valid_indices(
                                outer[0], [start_time, tmp["lockout_starts"].iat[t][0]]
                            )
                        ):
                            tmp["leave_home"].iat[t] = downtimesall[
                                (downtimesall == 1)
                                & (downtimesall >= start_time)
                                & downtimesall
                                < tmp["lockout_starts"].iat[t]
                            ][-1]
                            tmp["lockout_type"].iat[t] = 1
                            tmp["trial_type"].iat[
                                t
                            ] = 0  # type=error, cannot define r or w
                        else:
                            # lockout bc visited rip on a wait trial
                            if (
                                len(
                                    valid_indices(
                                        rip,
                                        [
                                            tmp["lockout_starts"].iat[t][0] - 0.01,
                                            tmp["lockout_starts"].iat[t][0],
                                        ],
                                    )
                                )
                                > 0
                            ):  # rip visit was the lock cause
                                tmp["trial_type"].iat[t] = 2
                                tmp["lockout_type"].iat[t] = 2
                                tmp["leave_home"].iat[t] = downtimesall[
                                    (downwellsall == 1)
                                    & (downtimesall >= start_time)
                                    & (downtimesall < tmp["lockout_starts"].iat[t][0])
                                ][-1]
                            # lockout bc visited wait on a rip trial
                            elif (
                                len(
                                    valid_indices(
                                        wait,
                                        [
                                            tmp["lockout_starts"].iat[t][0] - 0.01,
                                            tmp["lockout_starts"].iat[t][0],
                                        ],
                                    )
                                )
                                > 0
                            ):  # wait visit was the lock cause
                                tmp["trial_type"].iat[t] = 1
                                tmp["lockout_type"].iat[t] = 2
                                tmp["leave_home"].iat[t] = downtimesall[
                                    (downwellsall == 1)
                                    & (downtimesall >= start_time)
                                    & (downtimesall < tmp["lockout_starts"].iat[t][0])
                                ][-1]
                            # correctly visited rip but was impatient
                            elif (
                                len(
                                    valid_indices(
                                        rip,
                                        [start_time, tmp["lockout_starts"].iat[t][0]],
                                    )
                                )
                                > 0
                            ):
                                tmp["trial_type"].iat[t] = 1
                                tmp["lockout_type"].iat[t] = 3
                                tmp["rw_start"].iat[t] = rip[
                                    valid_indices(
                                        rip,
                                        [start_time, tmp["lockout_starts"].iat[t][0]],
                                    )
                                ][0]
                                tmp["leave_rw"].iat[t] = tmp["lockout_starts"].iat[t][0]
                                tmp["rw_end"].iat[t] = tmp["lockout_starts"].iat[t][0]
                                tmp["leave_home"].iat[t] = downtimesall[
                                    valid_indices(
                                        downtimesall,
                                        [start_time, tmp["rw_start"].iat[t]],
                                    )
                                ][-1]
                            # correctly visited wait but was impatient
                            elif (
                                len(
                                    valid_indices(
                                        wait,
                                        [start_time, tmp["lockout_starts"].iat[t][0]],
                                    )
                                )
                                > 0
                            ):
                                tmp["trial_type"].iat[t] = 2
                                tmp["lockout_type"].iat[t] = 3
                                tmp["rw_start"].iat[t] = wait[
                                    valid_indices(
                                        wait,
                                        [start_time, tmp["lockout_starts"].iat[t][0]],
                                    )
                                ][0]
                                tmp["leave_rw"].iat[t] = tmp["lockout_starts"].iat[t][0]
                                tmp["rw_end"].iat[t] = tmp["lockout_starts"].iat[t][0]
                                tmp["leave_home"].iat[t] = downtimesall[
                                    valid_indices(
                                        downtimesall,
                                        [start_time, tmp["rw_start"].iat[t]],
                                    )
                                ][-1]
                # COMPLETE TRIAL no lockouts
                else:
                    tmp["rw_success"].iat[t] = 1
                    tmp["outer_time"].iat[t] = outer[
                        0, valid_indices(outer[0], [start_time, end_time])
                    ][0]
                    tmp["outer_well"].iat[t] = outer[
                        1, valid_indices(outer[0], [start_time, end_time])
                    ][0]
                    tmp["leave_outer"].iat[t] = downtimesall[
                        valid_indices(
                            downtimesall, [tmp["outer_time"].iat[t], end_time]
                        )
                    ][-1]
                    if len(
                        valid_indices(goalrec, [start_time, end_time])
                    ):  # received outer reward
                        tmp["goal_well"].iat[t] = tmp["outer_well"].iat[t]
                        tmp["outer_success"].iat[t] = 1

                    if len(
                        valid_indices(rip, [start_time, end_time - 0.001])
                    ):  # rip trial -.001 to catch trodes freeze trials
                        tmp["trial_type"].iat[t] = 1  # type=rip
                        tmp["rw_start"].iat[t] = rip[
                            valid_indices(rip, [start_time, end_time])
                        ][
                            0
                        ]  # PROBLEM LINE!! RWstart > RWend
                        tmp["rw_end"].iat[t] = ripends[
                            valid_indices(ripends, [start_time, end_time])
                        ][0]
                        tmp["leave_rw"].iat[t] = downtimesall[
                            valid_indices(
                                downtimesall,
                                [tmp["rw_start"].iat[t], tmp["outer_time"].iat[t]],
                            )
                        ][-1]
                        tmp["leave_home"].iat[t] = downtimesall[
                            valid_indices(
                                downtimesall, [start_time, tmp["rw_start"].iat[t]]
                            )
                        ][-1]
                        # those trials when he gets click/beep just as he leaves
                        if (tmp["rw_end"].iat[t] - tmp["leave_rw"].iat[t] < 0.3) and (
                            tmp["rw_end"].iat[t] - tmp["leave_rw"].iat[t] > 0
                        ):
                            tmp["leave_rw"].iat[t] = tmp["rw_end"].iat[t]

                    elif len(
                        valid_indices(wait, [start_time, end_time - 0.001])
                    ):  # wait trial
                        tmp["trial_type"].iat[t] = 2  # type=wait
                        tmp["rw_start"].iat[t] = wait[
                            valid_indices(wait, [start_time, end_time])
                        ][
                            0
                        ]  # PROBLEM LINE!! RWstart > RWend
                        tmp["rw_end"].iat[t] = waitends[
                            valid_indices(waitends, [start_time, end_time])
                        ][0]
                        tmp["leave_rw"].iat[t] = downtimesall[
                            valid_indices(
                                downtimesall,
                                [tmp["rw_start"].iat[t], tmp["outer_time"].iat[t]],
                            )
                        ][-1]
                        tmp["leave_home"].iat[t] = downtimesall[
                            valid_indices(
                                downtimesall, [start_time, tmp["rw_start"].iat[t]]
                            )
                        ][-1]
                        # those trials when he gets click/beep just as he leaves
                        if (tmp["rw_end"].iat[t] - tmp["leave_rw"].iat[t] < 0.3) and (
                            tmp["rw_end"].iat[t] - tmp["leave_rw"].iat[t] > 0
                        ):
                            tmp["leave_rw"].iat[t] = tmp["rw_end"].iat[t]

                # sanity checks:
                assert tmp["start_time"].iat[t] < tmp["leave_home"].iat[t]
                assert tmp["rw_start"].iat[t] <= tmp["rw_end"].iat[t]
                assert tmp["rw_end"].iat[t] <= tmp["leave_rw"].iat[t]
                assert tmp["outer_time"].iat[t] <= tmp["leave_outer"].iat[t]

            except Exception as e:
                _, _, e_traceback = sys.exc_info()
                e_line = e_traceback.tb_lineno

                print(
                    "bug trial #%d, epoch %d! line %d: %s"
                    % (t, self.desc["epoch"], e_line, str(e))
                )
                # zero out all measures for bug trials!
                tmp["lockout_starts"].iat[t] = []
                tmp["lockout_ends"].iat[t] = []
                tmp["during_lockout"].iat[t] = []
                tmp["lockout_type"].iat[t] = 0
                tmp["rw_start"].iat[t] = 0
                tmp["rw_end"].iat[t] = 0
                tmp["leave_home"].iat[t] = 0
                tmp["leave_rw"].iat[t] = 0
                tmp["trial_type"].iat[t] = 0
                tmp["outer_well"].iat[t] = 0
                tmp["outer_time"].iat[t] = 0
                tmp["leave_outer"].iat[t] = 0
                tmp["goal_well"].iat[t] = 0
                tmp["rw_success"].iat[t] = 0

        # work backwards to fill in goal info based on rewarded locations (only know once he gets goal for the first time)
        # this also works whne the end of the ep ends in zeros ( will just overwrite 0 with 0), just can't know goal for those trials
        for t in range(len(tmp["goal_well"]) - 1, 0, -1):
            if tmp["goal_well"].iat[t - 1] == 0:
                tmp["goal_well"].iat[t - 1] = tmp["goal_well"].iat[t]
        tmp.index += 1

        return tmp


# HELPER FUNCTIONS


def valid_indices(values, bounds):
    """
    values (array/list of ints)
    bounds (tuple): [lowerbound, upperbound]

    Returns new array containing indices i such that
    where lowerbound <= values[i] < upperbound
    """
    return np.nonzero((values >= bounds[0]) & (values < bounds[1]))[0]


def lookup(reference, target):
    """
    For each timestamp t in reference, returns the list of indices i where target[i] is
    the lowest timsteamp greater than reference[t].

    Assumes reference and target are 1D ndarrays containing monotonically increasing values (timestamps).
    """
    indices = []
    left = 0
    for ref in reference:
        while left < target.size and target[left] <= ref:
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
        if (
            np.max(lockstarts > home[i]) == 0
        ):  # case where no lockstarts occurred after hometime
            return np.array(indices)
        j = np.argmax(lockstarts > home[i])  # first index where lockstart > hometime
        if lockstarts[j] - 0.3 > home[i]:  # include i if hometime[i] < lockstart-3
            indices.append(i)
    return np.array(indices)
