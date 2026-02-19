import pandas as pd 

import spyglass.common as sgc
from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename


def validate_references(nwb_file_name, is_copy=False, verbose=False):
    '''
    The 'original_reference_electrode' column in sgc.Electrode does not necessarily have the correct reference set,
    especially if we changed the reference after the recording happened (within the spreadsheet / yaml). This function
    returns an electrodes dataframe with 'val_ref' column updated to reflect the most up-to-date reference designation
    from the yaml.
    '''

    # if is_copy == True, nwb_file_name ends in _.nwb instead of just .nwb
    # compare the original_reference_electrode values (trodes config) and the can1ref and can2ref labels (specified in yaml), then decide which ones to use
    if is_copy:
        nwb_copy_file_name = nwb_file_name
    else:
        nwb_copy_file_name = get_nwb_copy_filename(nwb_file_name)

    # create dataframe with all relevant electrode info
    electrodes_df = (
        pd.DataFrame(
            (sgc.Electrode() & {"nwb_file_name": nwb_copy_file_name}) * sgc.BrainRegion
        )
        .loc[
            :,
            [
                "nwb_file_name",
                "electrode_group_name",
                "electrode_id",
                "probe_electrode",
                "region_name",
                "original_reference_electrode",
                "bad_channel",
            ],
        ]
        .sort_values(by="electrode_id")
    )

    # update formatting
    electrodes_df["electrode_group_name"] = electrodes_df[
        "electrode_group_name"
    ].astype("int")
    electrodes_df["original_reference_electrode"] = electrodes_df[
        "original_reference_electrode"
    ].astype("int")
    electrodes_df["bad_channel"] = electrodes_df["bad_channel"].astype("str")

    # calculate can 1 vs can 2 cutoff using half the detected tetrodes
    n_tetrodes = len(electrodes_df["electrode_group_name"].unique())
    if (n_tetrodes != 32) & (n_tetrodes != 64):
        raise Exception(
            f"Counted {n_tetrodes} tetrodes, not 32 or 64... revisit the yaml."
        )
    can_cutoff = n_tetrodes / 2

    # assign which tetrode/electrode is in which cannula
    electrodes_df.loc[electrodes_df["electrode_group_name"] < can_cutoff, "can"] = 1
    electrodes_df.loc[electrodes_df["electrode_group_name"] >= can_cutoff, "can"] = 2

    # identify original_reference_electrode values (set in trodes config)
    orig_ref_mapping = (
        electrodes_df[["original_reference_electrode", "can"]]
        .groupby(["original_reference_electrode", "can"])
        .mean()
        .reset_index()
        .set_index("can", drop=True)
        .to_dict()["original_reference_electrode"]
    )
    if verbose:
        print(f"original references mapping (cannula : electrode_id): {orig_ref_mapping}")

    # identify if yaml-assigned references exist and create dataframes for them
    yaml_can1ref_df = electrodes_df.loc[
        (electrodes_df["region_name"] == "can1ref")
        & (electrodes_df["bad_channel"] == "False")
    ]
    yaml_can2ref_df = electrodes_df.loc[
        (electrodes_df["region_name"] == "can2ref")
        & (electrodes_df["bad_channel"] == "False")
    ]

    # get validated referencing
    val_can_refs = [None, None]
    for can in [1, 2]:
        can_idx = can - 1
        if can == 1:
            yaml_can_df = yaml_can1ref_df
        if can == 2:
            yaml_can_df = yaml_can2ref_df

        # if yaml_can1ref_df is empty, set can1ref = None, continue
        if yaml_can_df.empty:
            if verbose:
                print(f"no can{can}ref in the yaml! will default to the other one")

        yaml_ref = yaml_can_df["electrode_id"].values
        if verbose:
            print(f"yaml_can{can}ref: {yaml_ref}")

        # if yaml_can1ref is length 1, then:
        if len(yaml_ref) == 1:
            # if they match, continue with original_reference_electrode
            orig_ref = orig_ref_mapping[can]
            if orig_ref == yaml_ref:
                val_can_refs[can_idx] = orig_ref
            # if not, set the first yaml_can1ref channel as the validated reference channel for can1
            else:
                val_can_refs[can_idx] = yaml_ref[0]  # default to yaml ref

        # if yaml_can1ref is length > 1 (usually 4, but sometimes less if there's bad channels), then:
        if len(yaml_ref) > 1:
            # if the can1 orig ref is in yaml-identified can1ref, continue with original_reference_electrode
            orig_ref = orig_ref_mapping[can]
            if orig_ref in yaml_ref:
                val_can_refs[can_idx] = orig_ref
            # if not, set the first yaml_can1ref channel as the validated reference channel for can1
            else:
                val_can_refs[can_idx] = yaml_ref[0]

    # if can1ref is None, set it to can2ref and vice versa; if both None, throw an error
    if (val_can_refs[0] is None) & (val_can_refs[1] is not None):
        val_can_refs[0] = val_can_refs[1]
    if (val_can_refs[0] is not None) & (val_can_refs[1] is None):
        val_can_refs[1] = val_can_refs[0]
    if (val_can_refs[0] is None) & (val_can_refs[1] is None):
        raise Exception(
            "No validated reference found for either cannula. Investigate the corresponding adjusting spreadsheet and yaml file."
        )

    # create and assigned validated references to electrode df
    electrodes_df.loc[electrodes_df["can"] == 1, "val_ref"] = val_can_refs[0]
    electrodes_df.loc[electrodes_df["can"] == 2, "val_ref"] = val_can_refs[1]
    if verbose:
        print(f"validated can1ref = {val_can_refs[0]}, can2ref = {val_can_refs[1]}")

    # can return updated electrodes_df list so that this can be pulled from to get the validated reference for the specific electrode_ids you want
    return electrodes_df, val_can_refs


def apply_referencing(df, electrodes_df):
    ref_df = pd.DataFrame()
    elecs = df.columns
    for elec in elecs:
        # find valid reference to apply
        val_ref = electrodes_df.loc[electrodes_df['electrode_id'] == elec, 'val_ref'].values[0]
        ref_vals = df[val_ref].values 
        # apply referencing
        ref_df[elec] = df[elec].values - ref_vals
    
    ref_df.index = df.index
    
    return ref_df