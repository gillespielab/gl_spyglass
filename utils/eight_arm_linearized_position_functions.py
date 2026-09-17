import numpy as np

import spyglass.common as sgc
import spyglass.position.v1 as sgpv1
import spyglass.position as sgp
import spyglass.linearization.v1 as sgpl
from spyglass.linearization.merge import LinearizedPositionOutput

# LINEARIZATION FUNCTIONS
ucsf_verts = np.array([[155.1164669 ,  78.10645453],
       [132.91368841,  45.47505638],
       [ 76.46594647,  15.81014897],
       [126.13995938,  54.3745286 ],
       [ 66.68167121,  33.18530903],
       [122.37677658,  61.15507887],
       [ 58.02635078,  49.2891159 ],
       [119.36623035,  71.32590426],
       [ 52.38157658,  66.66427596],
       [117.48463895,  81.07294527],
       [ 53.8868497 ,  84.03943601],
       [119.36623035,  89.97241749],
       [ 56.52107766, 102.68594924],
       [122.37677658,  97.17675215],
       [ 63.67112497, 117.94218733],
       [129.15050561, 104.3810868 ],
       [ 73.07908196, 131.92707225]])

# I think the v2 one is the one that I'm using from now on
ucsf_verts_v2 = np.array([[153.83092581,  75.86638095],
       [127.3304    ,  44.4525    ],
       [ 69.3364957 ,  18.1327619 ],
       [120.41721935,  50.39566667],
       [ 58.19859355,  33.83970238],
       [113.50403871,  59.31041667],
       [ 53.20574086,  50.39566667],
       [109.6633828 ,  69.92321429],
       [ 48.59695376,  67.37614286],
       [110.04744839,  80.5360119 ],
       [ 50.13321613,  84.35661905],
       [112.35184194,  89.4507619 ],
       [ 53.20574086, 101.33709524],
       [116.96062903,  97.5164881 ],
       [ 61.27111828, 117.46854762],
       [126.17820323, 103.03514286],
       [ 70.10462688, 131.90195238]])

uw_verts = np.array([[169.9373957 ,  67.89811905],
       [135.44735699, 106.08766667],
       [104.50776344, 174.08369048],
       [144.06986667, 114.0050119 ],
       [123.7816086 , 185.72684524],
       [154.2139957 , 118.66227381],
       [146.09869247, 192.2470119 ],
       [167.40136344, 124.71671429],
       [164.86533118, 196.43854762],
       [179.06711183, 123.7852619 ],
       [189.71844731, 195.50709524],
       [190.22565376, 118.66227381],
       [209.49949892, 188.05547619],
       [199.35536989, 112.14210714],
       [229.28055054, 178.27522619],
       [207.47067312, 103.29330952],
       [247.03277634, 166.63207143]])

def get_8arm_map_info(verts):
    '''
    Convert your vertices into the correct format to pass into the Linearization functions
    '''
    vert_labels = ['base', 
                'arm1_start', 'arm1_end',
                'arm2_start', 'arm2_end',
                'arm3_start', 'arm3_end',
                'arm4_start', 'arm4_end',
                'arm5_start', 'arm5_end',
                'arm6_start', 'arm6_end',
                'arm7_start', 'arm7_end',
                'arm8_start', 'arm8_end']
    vert_dict = {}
    vert_indices = {}
    for v, (vert, label) in enumerate(zip(verts, vert_labels)):
        vert_dict[label] = vert
        vert_indices[label] = v

    edge_labels = [('base', 'arm1_start'),
                ('base', 'arm2_start'),
                ('base', 'arm3_start'),
                ('base', 'arm4_start'),
                ('base', 'arm5_start'),
                ('base', 'arm6_start'),
                ('base', 'arm7_start'),
                ('base', 'arm8_start'),
                ('arm1_start', 'arm1_end'),
                ('arm2_start', 'arm2_end'),
                ('arm3_start', 'arm3_end'),
                ('arm4_start', 'arm4_end'),
                ('arm5_start', 'arm5_end'),
                ('arm6_start', 'arm6_end'),
                ('arm7_start', 'arm7_end'),
                ('arm8_start', 'arm8_end')]
    edge_indices = []
    for edge in edge_labels:
        edge_indices.append((vert_indices[edge[0]], vert_indices[edge[1]]))

    node_positions = np.asarray(verts)
    edges = np.asarray(edge_indices)
    linear_edge_order = edges
    linear_edge_spacing = 15

    edge_map = {0: 0, 1: 0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:1, 9:2, 10:3, 11:4, 12:5, 13:6, 14:7, 15:8}

    return node_positions, edges, linear_edge_order, linear_edge_spacing, edge_map

def load_pos_df(nwb_copy_file_name, interval_list_name, trodes_pos_params_name):
    # pair nwb_file_name, interval_list, params into trodes pos selection
    trodes_s_key = {
        'nwb_file_name': nwb_copy_file_name,
        'interval_list_name': interval_list_name,
        'trodes_pos_params_name': trodes_pos_params_name,
    }
    sgpv1.TrodesPosSelection.insert1(trodes_s_key, skip_duplicates=True)
    trodes_key = (sgpv1.TrodesPosSelection() & trodes_s_key).fetch1("KEY")

    # populate trodes pos v1 table using trodes key
    sgpv1.TrodesPosV1.populate(trodes_key)

    # get merge id corresponding to our inserted trodes_key
    merge_key = (sgp.PositionOutput.merge_get_part(trodes_key)).fetch1("KEY")
    # use merge_key to select relevant entry from the PositionOutput mergetable and fetch the dataframe
    pos_df = (sgp.PositionOutput & merge_key).fetch1_dataframe()

    return pos_df, trodes_key