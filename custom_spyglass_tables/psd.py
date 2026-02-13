import datajoint as dj
import numpy as np
import pandas as pd
from scipy import signal
from tqdm import tqdm
from typing import List

from spyglass.utils import SpyglassMixin, logger
import spyglass.lfp as lfp
from spyglass.lfp.v1.lfp import LFPV1
from spyglass.lfp import LFPOutput
from spyglass.common.common_nwbfile import AnalysisNwbfile

from mne.time_frequency import psd_array_multitaper

from gabby_analysis_gillespie.utils.preprocessing_helper_functions import validate_references

schema = dj.schema("gl_psd")

@schema
class PSDSelection(SpyglassMixin, dj.Manual):
    definition = """
     -> LFPV1
     -> LFPOutput.proj(lfp_merge_id='merge_id')
     referenced: bool
     method = 'welch': varchar(80)
     """
    
@schema
class PSD(SpyglassMixin, dj.Computed):
    """Computed PSDs based on the LFP data"""

    definition = """
    -> PSDSelection
    ---
    -> AnalysisNwbfile
    psd_df_object_id : varchar(40)
    """

    def make(self, key):
        """Populate PSD table.
        Fetches...
            - 
            - 
        """
        nwb_file_name = key['nwb_file_name']
        lfp_electrode_group_name = key['lfp_electrode_group_name']
        lfp_merge_id = key['lfp_merge_id']
        lfp_key = {'merge_id': lfp_merge_id}
        referenced = key['referenced']
        method = key['method']

        lfp_sampling_rate = (lfp.v1.LFPSelection & key).fetch1('target_sampling_rate')

        lfp_elect_list = (lfp.LFPElectrodeGroup.LFPElectrode() & {'nwb_file_name': nwb_file_name, 'lfp_electrode_group_name': lfp_electrode_group_name}).fetch('electrode_id')
        lfp_elect_list = np.sort(lfp_elect_list)

        # load in lfp and remove ref index
        print('loading in lfp...')
        lfp_df = (lfp.LFPOutput() & lfp_key).fetch1_dataframe()
        lfp_df.columns = lfp_elect_list

        # retrieve validated reference for each cannula and which ref goes with which electrodes
        electrodes_df, val_can_refs = validate_references(nwb_file_name, is_copy=True)

        # pull up the ca1 elec list to cross-check with
        ca1_elecs = electrodes_df.loc[electrodes_df['region_name'] == 'ca1', 'electrode_id'].values

        if referenced:
            # apply referencing: subtract reference from each channel
            for elec in lfp_df.columns:
                val_ref = electrodes_df.loc[electrodes_df['electrode_id'] == elec, 'val_ref'].values[0]
                lfp_df[elec] -= lfp_df[val_ref]

            # remove the reference from the lfp after referencing
            lfp_df = lfp_df.drop(columns=set(val_can_refs))
        
        # calculate psd for each channel using welch's method
        psd_dfs = []
        for elec in tqdm(lfp_df.columns):

            
            if method == 'welch':
                freq, Pxx = signal.welch(lfp_df[elec], fs=lfp_sampling_rate)
            if method == 'multitaper':
                Pxx, freqs = psd_array_multitaper(lfp_df[elec].values, fs=lfp_sampling_rate)
            elec_psd_df = pd.DataFrame({'electrode': elec, 'freq': freq, 'Pxx': Pxx, 'log_Pxx': np.log(Pxx)})

            if elec in ca1_elecs:
                elec_psd_df['elec_location'] = 'ca1'
            else:
                elec_psd_df['elec_location'] = 'other'

            psd_dfs.append(elec_psd_df)

        psd_df = pd.concat(psd_dfs).reset_index(drop=True)
        psd_df = psd_df[psd_df['freq'] < 300]  # drop the 300-500Hz frequency range

        # Insert into analysis nwb file
        nwb_analysis_file = AnalysisNwbfile()
        key["analysis_file_name"] = nwb_analysis_file.create(nwb_file_name)
        key["psd_df_object_id"] = nwb_analysis_file.add_nwb_object(
            analysis_file_name=key['analysis_file_name'],
            nwb_object=psd_df,
        )
        nwb_analysis_file.add(
            nwb_file_name=nwb_file_name,
            analysis_file_name=key['analysis_file_name'],
        )
        
        self.insert1(key)
    
    def fetch1_dataframe(self) -> pd.DataFrame:
        """Convenience function for returning the marks in a readable format"""
        _ = self.ensure_single_entry()
        return self.fetch_dataframe()[0]

    def fetch_dataframe(self) -> List[pd.DataFrame]:
        """Convenience function for returning all marks in a readable format"""
        return [data["psd_df"] for data in self.fetch_nwb()]



    