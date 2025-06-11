Features of TrialInfo8Arm
-

- For active/task epochs of 8-arm task 
- Currently includes TrialInfo8Arm which holds extracted context information about the 8-arm task.
    - Assigns trial numbers
    - Finds timestamps of relevant landmarks, and adjusts to universal time
    - Labels trials as search or repeat
    - Finds lockout trials, and removes bug trials
- Extendable design that will eventually support other tasks.
    - (If a new table needs to be created (ex: 6-arm), a new schema needs to be implemented in `trial_info.py`, and a new parser in `parse_trials_helper.py`)
- Wrapper function for plotting the trial context for some epoch
    - Option to display change of mind trials, or mark any subset of trials in the plot
 
Example Usage:
-
Example use case outlined in `notebooks/TrialInfoExample.ipynb`

Legacy Table TrialInfo
-

- The TrialInfo table still exists for backward compatability, but shouldn't be inserted into anymore
