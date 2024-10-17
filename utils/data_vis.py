import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def get_goal_fractions(df):
    # Cut off any incomplete trial data from the end
    filtered_df = df.query("goal_well > 0")
    num_trials = len(filtered_df)
    goals = filtered_df["goal_well"].to_numpy()

    # Get indices of trials that were the 1st in their block
    goal_change_trials = np.where(np.diff(goals, prepend=0, append=num_trials) != 0)[0]
    # Get indices of goal trials
    goal_trials = filtered_df.index[filtered_df.outer_well == filtered_df.goal_well].to_numpy()
    is_goal = np.zeros(num_trials, dtype=int)
    is_goal[goal_trials] = 1

    # Map each trial to the number of trials that have passed since:
    # - the 1st trial of the current block
    # - the 1st rewarded trial of the current block
    since_goal_change = np.zeros(num_trials, dtype=np.int64)
    since_goal_found = np.zeros(num_trials, dtype=np.int64)
    goal_fraction_since_change = np.zeros(num_trials)

    # Use prefix sum to calculate cumulative goals since the 1st trial/goal of each block,
    # i.e. cg[i] = num goals accumulated from trial 0 to i (inclusive)
    cumulative_goals = np.cumsum(is_goal)

    # Calculating goal fraction history for each block
    for i in range(1, goal_change_trials.size):
        block_start = goal_change_trials[i-1]
        block_end = goal_change_trials[i]
        block_len = block_end - block_start
        # Get index of first goal trial within the block
        slice = is_goal[block_start:block_end]
        if not np.any(slice):
            # We only consider complete trials, so we shouldn't hit this case
            raise Exception("Invalid input: found incomplete trial")
        first_goal = np.argmax(slice) + block_start
        
        counts_since_change = np.arange(block_len)
        # pre-goal trials will have a negative count_since_found value
        counts_since_found = counts_since_change - (first_goal - block_start)
        since_goal_change[block_start:block_end] = counts_since_change + 1
        since_goal_found[block_start:block_end] = counts_since_found + 1

        # Get array of accumulated goals at each trial in the current block
        prev_goals = 0 if block_start == 0 else cumulative_goals[block_start-1]
        curr_block_goal_hist = cumulative_goals[block_start:block_end] - prev_goals
        trials_since_change = since_goal_change[block_start:block_end]

        goal_fraction_since_change[block_start:block_end] = \
            curr_block_goal_hist / trials_since_change

    # handle divide by 0 instances (i.e. trial before a goal)
    goal_fraction_since_change = np.nan_to_num(goal_fraction_since_change)

    data = {}
    data["goal_fraction_since_change"] = goal_fraction_since_change
    data["trials_since_change"] = since_goal_change
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