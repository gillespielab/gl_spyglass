import datajoint as dj

dj.config['database.host'] = "gl-ash.biostr.washington.edu"
dj.config['database.user'] = "gabby"
dj.config['database.port'] = 3306

dj.conn()

import spyglass.common as sgc

'''

This is a template script to run any function across all subjects, dates, and epochs.
Fill out all the TODOs below to customize for your function and the specific subjects/dates/epochs you want to run it on.

To run your customized script in your terminal, navigate to the directory you have it saved in
and run `python example_all_subjs_script_outline.py` or once you've renamed it: 
`python {your_script_name}.py`

'''

# To set the dates to be processed for each subject, either: 
# TODO: comment out one of the options below and update with your own subjects (and dates)
# 1) manually specify dates for each of your subjects or
subj_date_dict = {
    'pippin': [20210421, 20210423, 20210424],
    'archibald': [20210711, 20210714, 20210721],
    'hugo': [20210818, 20210819, 20210822],
    'herman': [20211111, 20211112, 20211113],
    'chip': [20220104, 20220108, 20220109],
    'bobrick': [20231204, 20231205, 20231206],
    'reginald': [20241023, 20241017, 20241015],
    'timothy': [20241111, 20241113, 20241118],
    'tony': [20250430, 20250505, 20250501],
    'teddy': [20250626, 20250604, 20250620],
}

# 2) automically load in all accessible dates from sgc.Session() for each subject
subj_date_dict = {}
subjs = ['pippin', 'archibald', 'hugo', 'herman', 'chip', 'bobrick', 'reginald', 'timothy', 'tony', 'teddy']
for subj in subjs:
    nwb_file_names = (sgc.Session() & {"subject_id": subj}).fetch("nwb_file_name")
    dates = [nwb_file_name[-13:-5] for nwb_file_name in nwb_file_names]
    subj_date_dict[subj] = dates    


# iterate through your function for all subjects, dates, and epochs
# TODO: fill in the blanks in the print statements below to add more descriptions of what you're doing
subjs = subj_date_dict.keys()
for subj in subjs:
    print(f'STARTING _________ FOR {subj}')
    dates = subj_date_dict[subj]
    for date in dates:
        nwb_file_name = f'{subj}{date}_.nwb'
        print(f'processing {nwb_file_name}...')

        # automatically find all the relevant epochs for this subject and date
        all_epochs = (sgc.TaskEpoch() & {'nwb_file_name'}).fetch('epoch')

        # if you want to specify just run epochs or just sleep epochs, can specify by the name of your task
        # for example:
        #
        # all_run_epochs = (sgc.TaskEpoch() & {'nwb_file_name': nwb_file_name, 'task_name': 'Eight arm flexible spatial task'}).fetch('epoch')
        # all_epochs = all_run_epochs
        #
        #  or 
        #
        # all_sleep_epochs = (sgc.TaskEpoch() & {'nwb_file_name': nwb_file_name, 'task_name': 'Sleep'}).fetch('epoch')
        # all_epochs = all_sleep_epochs
        # 
        # you can also just manually specify certain epochs
        # 
        # all_epochs = [1, 2]
    
        for epoch in all_epochs:
            print(f'processing epoch {epoch}...')

            # if you need to find the corresponding interval list name and/or position interval list name, use the code below
            interval_list_name = (
                sgc.TaskEpoch() & {"nwb_file_name": nwb_file_name, "epoch": epoch}
            ).fetch1("interval_list_name")
            pos_interval_list_name = (
                sgc.IntervalList() & {'nwb_file_name': nwb_file_name, 'pipeline': 'position'}
            ).fetch('interval_list_name')[epoch - 1]

            # TODO: add your per-epoch function here, written so that it takes nwb_file_name and epoch (or interval_list_name or pos_interval_list_name) as inputs
            # for example:
            # 
            # single_interval_clusterless_pipeline(
            #     nwb_file_name,
            #     interval_list_name,
            #     team_name='Gabby Shvartsman',
            #     interval_type='single',
            # )
