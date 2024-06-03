import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def get_goal_fractions(df):
    
    num_trials = len(df)
    goals = df["goal_well"].to_numpy()
    goal_change_trials = np.where(np.diff(goals, prepend=0, append=num_trials) != 0)[0]

    # boolean mask: is_goal[i] = 1 iff goal arm was visited on trial i
    goal_trials = df.index[df.outer_well == df.goal_well].to_numpy()

    is_goal = np.zeros(num_trials, dtype=int)
    is_goal[goal_trials] = 1

    # map each trial to the number of trials that have passed since last goal-change / reward
    since_goal_change = np.zeros_like(goals)
    since_goal_found = np.zeros_like(goals)
    since_goal_found.fill(-1)
    first_goals = []
    for i in range(1, goal_change_trials.size):
        block_start = goal_change_trials[i-1]
        block_end = goal_change_trials[i]

        first_goal = block_start
        while first_goal < block_end and not is_goal[first_goal]:
            first_goal += 1 # increment until we find first goal trial within block
        if first_goal < block_end:
            first_goals.append(first_goal)
        
        counts_since_change = np.arange(1, block_end - block_start + 1)
        counts_since_found = counts_since_change - (first_goal - block_start) # align to first goal trial of block
        since_goal_change[block_start:block_end] = counts_since_change
        since_goal_found[block_start:block_end] = counts_since_found

    # calculating goal fractions wrt trials since goal-change / reward
    cumulative_goals = np.cumsum(is_goal) # cg[i] = cumulative # of goals up to and including trial i
    # cg[j] - cg[i] = cumulative goals at trial j made since trial i 

    goal_fraction_since_change = np.zeros(num_trials)
    goal_fraction_since_found = np.zeros(num_trials)
    goal_fraction_since_found.fill(-1)

    for i in range(1, goal_change_trials.size):
        block_start = goal_change_trials[i-1]
        block_end = goal_change_trials[i]
        if i == 1: # fenceposting
            goal_fraction_since_change[block_start:block_end] = cumulative_goals[block_start:block_end] / since_goal_change[block_start:block_end]
        else:
            goal_fraction_since_change[block_start:block_end] = (cumulative_goals[block_start:block_end] - cumulative_goals[block_start-1]) / since_goal_change[block_start:block_end]

        first_goal = goal_change_trials[i-1]
        while first_goal < block_end and not is_goal[first_goal]:
            first_goal += 1 # increment until we find first goal trial within block
        if first_goal < block_end:
            goal_fraction_since_found[first_goal:block_end] = (cumulative_goals[first_goal:block_end] - cumulative_goals[first_goal-1]) / since_goal_change[first_goal:block_end]

    # handle divide by 0 instances on 1st trials of each block and reward trials
    goal_fraction_since_change = np.nan_to_num(goal_fraction_since_change)
    goal_fraction_since_found = np.nan_to_num(goal_fraction_since_found)

    data = {}
    data["goal_fraction_since_change"] = goal_fraction_since_change
    data["trials_since_change"] = since_goal_change
    data["goal_fraction_since_found"] = goal_fraction_since_found
    data["trials_since_found"] = since_goal_found
    return data

def plot_goal_fractions(df):
    res = get_goal_fractions(df)
    rr_since_change = res["goal_fraction_since_change"]
    trials_since_change = res["trials_since_change"]
    rr_since_found = res["goal_fraction_since_found"]
    trials_since_found = res["trials_since_found"]
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4), sharex=True, sharey=True)

    ax1.scatter(trials_since_change, rr_since_change)
    ax1.set_xlabel("Trials since last goal change")
    ax1.set_ylabel("Frac. goal arm trials (since start of block)")
    ax1.set_ylim(0, 1)
    ax1.grid()
    # averaging goal fraction across trials of the same distance from the last goal change
    tmp = pd.DataFrame(data={"since_goal_change": trials_since_change, "goal_fraction": rr_since_change})
    avg_goal_fraction = tmp.groupby("since_goal_change").mean().to_numpy().flatten()

    ax1.plot(np.unique(trials_since_change), avg_goal_fraction, "r")

    mask = trials_since_found >= 0

    ax2.scatter(trials_since_found[mask], rr_since_found[mask])
    ax2.set_xlabel("Trials since goal was found")
    ax2.set_ylabel("Frac. goal arm trials (since start of block)")
    ax2.set_ylim(0, 1)
    ax2.grid()
    # averaging goal fraction across trials of the same distance from the last goal change
    tmp = pd.DataFrame(data={"since_goal_found": trials_since_found[mask], "goal_fraction": rr_since_found[mask]})
    avg_goal_fraction = tmp.groupby("since_goal_found").mean().to_numpy().flatten()

    ax2.plot(np.unique(trials_since_found[mask]), avg_goal_fraction, "r")