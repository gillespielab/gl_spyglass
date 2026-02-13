import numpy as np
import pandas as pd
import itertools

from gl_spyglass.custom_spyglass_tables.trial_info import TrialInfo8Arm

def get_task_descriptors(restr):
    '''
    Helper function for pulling out all the task descriptors for TrialInfo8Arm when restricted to a particular session entry
    
    :param restr: a dictionary which includes 'nwb_file_name' and 'epoch'
    '''
    # get relevant descriptors
    desc = (TrialInfo8Arm() & restr).fetch1('descriptors')
    lockout_period = desc['lockout_period']
    try:
        outer_reps = desc['outer_reps']
        if isinstance(outer_reps, list):
            outer_reps_type = 'variable'
        else:
            outer_reps_type = outer_reps
        num_goals = desc['num_goals']
        forage_num = desc['forage_num']
    except Exception as e:
        print(f'an error occurred: {e}')

    return lockout_period, outer_reps_type, num_goals, forage_num

def get_valid_trials_mask(df):
    '''
    Isolate valid trials, including:
    - lockout type == 0 (ensuring it's not a lockout trial)
    - bug trial == 0 (ensuring it's not a known bug trial)
    - goal well != 0 (sometimes would come up in bug trials, just making sure in case it didn't get caught by 'bug_trial')
    '''
    valid_trials_mask = (df['lockout_type'] == 0) & (df['bug_trial'] == 0) & (df['goal_well'] != '0')
    return valid_trials_mask

def get_epoch_trial_counts(df):
    '''
    Get all the trial counts for a particular epoch, mainly to compare between these found values and the ones in the behavior spreadsheets
    
    :param df: trial df that gets output from fetch1_dataframe() for TrialInfo8Arm()
    '''
    # retrieve relevant metadata
    n_trials = len(df)
    n_wait_trials = len(df[df['trial_type'] == 2])
    n_rip_trials = len(df[df['trial_type'] == 1])
    n_lock1s = len(df[df['lockout_type'] == 1])
    n_lock2s = len(df[df['lockout_type'] == 2])
    n_lock3s = len(df[df['lockout_type'] == 3])
    valid_trials_mask = get_valid_trials_mask(df)
    n_valid_trials = len(df[valid_trials_mask])
    n_goals = np.sum(df.loc[valid_trials_mask, 'outer_success'].values)
    n_others = n_valid_trials - n_goals
    n_goal_blocks = len(df['n_goal_block'].unique())
    n_complete_goal_blocks = len(df.loc[df['complete_goal_block'], 'n_goal_block'].unique())
    n_complete_search_phases = n_complete_goal_blocks

    # check if last goal block has a complete search phase
    if len(df[(df['n_goal_block'] == n_goal_blocks) & (df['goal_block_phase'] == 'repeat')]) > 0:
        n_partial_goal_switches = n_goal_blocks - 1
        n_complete_search_phases = n_goal_blocks
    else:
        n_partial_goal_switches = n_goal_blocks - 2
    if n_partial_goal_switches < 0:
        n_partial_goal_switches = 0

    # check if last goal block is complete
    if df.loc[(df['n_goal_block'] == n_goal_blocks), 'complete_goal_block'].values[0]:
        n_complete_goal_switches = n_goal_blocks - 1
    else:
        n_complete_goal_switches = n_goal_blocks - 2
    if n_complete_goal_switches < 0:
        n_complete_goal_switches = 0

    epoch_trial_counts_df = pd.DataFrame({
        'n_trials': n_trials,
        'n_wait_trials': n_wait_trials,
        'n_rip_trials': n_rip_trials,
        'n_lock1s': n_lock1s,
        'n_lock2s': n_lock2s,
        'n_lock3s': n_lock3s,
        'n_goals': n_goals,
        'n_valid_trials': n_valid_trials,
        'n_others': n_others,
        'n_goal_blocks': n_goal_blocks,
        'n_complete_goal_blocks': n_complete_goal_blocks,
        'n_complete_search_phases': n_complete_search_phases,
        'n_partial_goal_switches': n_partial_goal_switches,
        'n_complete_goal_switches': n_complete_goal_switches,
    }, index=[0])

    return epoch_trial_counts_df

def get_recorded_arm_visits(df):
    '''
    Calculates arm coverage dataframe (which arms were visited during the epoch and the number of times each one was visited).
    This is particularly useful for identifying epochs where the rats didn't visit each arm a significant number of times.
    
    :param df: from (TrialInfo8Arm() & restr).fetch1_dataframe(), one session one epoch
    '''
    # get arm coverage dataframe
    all_arm_lockout_visits = []
    for s in df['during_lockout'].to_numpy():
        s = s.replace('H', '').replace('R', '').replace('W', '')  # ignoring all non-outer arm visits for this
        if len(s) == 0:
            continue 
        elif len(s) == 1:
            all_arm_lockout_visits.append(np.array([int(s)]))
        else:
            all_arm_lockout_visits.append(np.array(list(s), dtype='int')) 
    if len(all_arm_lockout_visits) == 0:
        arm_lockout_visits = []
    else:
        arm_lockout_visits = list(itertools.chain.from_iterable(all_arm_lockout_visits))
    arm_valid_visits = df['outer_well'].values
    all_arm_visits = np.concatenate([arm_lockout_visits, arm_valid_visits])
    arm_visit_counts = [len(all_arm_visits[all_arm_visits == n_arm]) for n_arm in np.arange(1, 9)]
    arm_visits_df = pd.DataFrame({'n_arm': np.arange(1, 9), 'n_recorded_visits': arm_visit_counts})
    arm_visits_df = arm_visits_df[['n_arm', 'n_recorded_visits']]
    return arm_visits_df

def get_trial_offsets(goal_block_df):
    '''
    Find n_trial_since_goal_switch and n_trial_since_goal_found for the trial dataframe (isolated to one single goal block)
    '''
    complete_goal_block = goal_block_df['complete_goal_block'].values[0]
    switch_offset = goal_block_df['trial_num'].values[0]
    goal_block_df['n_trial_since_goal_switch'] = goal_block_df['trial_num'].values - switch_offset

    if complete_goal_block:
        found_offset = goal_block_df.loc[goal_block_df['goal_block_phase'] == 'repeat', 'trial_num'].values[0]
        goal_block_df['n_trial_since_goal_found'] = goal_block_df['trial_num'].values - found_offset
    
    return goal_block_df

def add_session_info(df, subj, date, epoch, lockout_period, outer_reps_type, num_goals, forage_num):
    '''
    helper function for adding a bunch of session-specific metadata to your dataframe
    '''
    df['subj'] = subj
    df['date'] = date 
    df['epoch'] = epoch 
    df['lockout_period'] = lockout_period
    df['outer_reps_type'] = str(outer_reps_type)
    df['num_goals'] = num_goals 
    df['forage_num'] = forage_num

    return df

def collect_beh_dfs(nwb_file_name, subj, date, epoch, plot_trials=False):
    '''
    fetch and collect the behavior dataframes for a particular epoch
    parses them into:
    1. session_info_df (overall summary stats about the epochs, number of trials, etc.)
    2. arm_visits_df (for arm coverage and visits to each arm)
    3. trials_df (from fetch1_dataframe() but with some added processing to uniquely identify each epoch's trials)
    '''

    # load in trial info dataframes
    restr = {'nwb_file_name': nwb_file_name,
            'epoch': epoch}

    # load in trial dataframe
    df = (TrialInfo8Arm() & restr).fetch1_dataframe()

    if plot_trials:
            # plot trials for sanity check
            (TrialInfo8Arm() & restr).plot_trials()

    # get general session information
    lockout_period, outer_reps_type, num_goals, forage_num = get_task_descriptors(restr)
    epoch_trial_counts_df = get_epoch_trial_counts(df)

    # get which arms were visited, to get a sense of arm coverage
    arm_visits_df = get_recorded_arm_visits(df)

    # for all future metrics, use valid_df (which excludes lockout trials)
    valid_trials_mask = get_valid_trials_mask(df)
    valid_df = df.loc[valid_trials_mask].copy()
    valid_df = valid_df.reset_index(drop=True)  # make sure the index is continuous and updated
    valid_df['trial_num'] = valid_df.index + 1  # reset the trial num based on the updated index 
    valid_df = valid_df.drop(columns=['level_1'])

    valid_df = valid_df.groupby('n_goal_block').apply(get_trial_offsets, include_groups=False).reset_index()
    trials_df = valid_df

    # some exceptions
    if isinstance(outer_reps_type, list) | isinstance(outer_reps_type, np.ndarray):
            outer_reps = trials_df['outer_reps'].unique()
            outer_reps = outer_reps[~np.isnan(outer_reps)]
            if len(outer_reps) == 0:  # if we still don't know, check manually-filled in exceptions list
                outer_reps = [check_unknown_outerreps(nwb_file_name, epoch)]

            if len(outer_reps) == 1:
                    outer_reps_type = outer_reps[0]
                    trials_df['outer_reps'] = outer_reps[0]
            if len(outer_reps) > 1:
                    outer_reps_type = 'variable'

    session_info_df = add_session_info(epoch_trial_counts_df, subj, date, epoch, lockout_period, outer_reps_type, num_goals, forage_num)
    arm_visits_df = add_session_info(arm_visits_df, subj, date, epoch, lockout_period, outer_reps_type, num_goals, forage_num)
    trials_df = add_session_info(trials_df, subj, date, epoch, lockout_period, outer_reps_type, num_goals, forage_num)
    trials_df['outer_reps'] = trials_df['outer_reps'].values.astype('str')

    return session_info_df, arm_visits_df, trials_df

def check_epoch_exclusion(nwb_file_name, epoch):
    '''
    Running list of weird epochs that have errors that I haven't been able to get past yet
    Convenient function for checking if a particular epoch is in this running list
    '''
    excluded_epochs = {'pippin20210404_.nwb': [2],  # cannot access statescript log contents?
                       'reginald20241010_.nwb': [4],  # cannot access statescript log contents, need to rerun nwb processing
                       'pippin20210407_.nwb': [2],  # something wrong with dio recordings, cannot find small enough offset?
                       'herman20211112_.nwb': [2],  # something wrong home dio ordering? data streaming disconnections noted in the spreadsheet
                       'chip20211220_.nwb': [2],  # 2 min session with some weird bugs
                       'tony20250408_.nwb': [4],
                       'tony20250415_.nwb': [2],
                       'tony20250506_.nwb': [2],
                       }  
    if nwb_file_name in excluded_epochs.keys():
        if epoch in excluded_epochs[nwb_file_name]:
            return True 
    
    else:
        return False

def check_unknown_outerreps(nwb_file_name, epoch):
    '''
    Some of the outerreps numbers couldn't be inferred from the statescriptlog due to
    slight differences in how the behavior code was run that day (if the python script
    wasn't appended at the beginning of it for some reason), so these are values that
    were hand-pulled from the behavior spreadsheets for those cases.
    If the outerreps could be inferred from the statescriptlog, it just gets automatically
    populated from there.
    '''
    outer_reps_info = {
        'hugo20210823_.nwb': {2: 10.0},
        'herman20211025_.nwb': {2: 15.0},
        'herman20211106_.nwb': {2: 15.0},
        'reginald20241025_.nwb': {2: 'variable'},
        }
    outer_reps = 'unknown'
    if nwb_file_name in outer_reps_info.keys():
        if epoch in outer_reps_info[nwb_file_name].keys():
            outer_reps = outer_reps_info[nwb_file_name][epoch]
    else:
        print(f'unknown outer reps type for {nwb_file_name} epoch {epoch}')
    return outer_reps