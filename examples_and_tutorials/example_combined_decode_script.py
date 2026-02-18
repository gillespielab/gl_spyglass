import datajoint as dj

dj.config['database.host'] = "gl-ash.biostr.washington.edu"
dj.config['database.user'] = "gabby"
dj.config['database.port'] = 3306
# dj.config['database.reconnect'] = True
dj.config['connection_init_function'] = lambda conn: conn.ping(reconnect=True)

dj.conn()

import spyglass.common as sgc

import sys
sys.path.append("../..")
from gl_spyglass.utils.clusterless_decoding_functions import single_interval_decoding_pipeline

if __name__ == "__main__":

    subj_date_dict = {
        'pippin': [20210423],
    }

    subjs = subj_date_dict.keys()

    for subj in subjs:
        print(f"STARTING POPULATION FOR {subj}")
        dates = subj_date_dict[subj]

        for date in dates:
            nwb_file_name = f'{subj}{date}_.nwb'
            print(nwb_file_name)
            
            combined_interval_list_name = '04_r2 and 05_s3'
            interval_list_name = combined_interval_list_name

            single_interval_decoding_pipeline(
                nwb_file_name,
                interval_list_name,
                team_name='Gabby Shvartsman',
                interval_type='combined',
            )