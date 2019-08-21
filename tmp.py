import pickle as pkl
import numpy as np
import wfdb
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import defaultdict

def dynamic_replace(s):
    curr_aux = df['aux'].loc[0]
    for idx,x in enumerate(df['aux']):
        if x != '':
            curr_aux = x
        df.loc[idx, 'aux'] = curr_aux
    return df


def create_index_df(desired_segment_len=3600, basic_arr_path="data/mit-bih-arrhythmia-database-1.0.0"):
    desired_segment_len = 3600
    basic_arr_path = "data/mit-bih-arrhythmia-database-1.0.0"
    arr_db = wfdb.get_record_list('mitdb')
    num_samples_in_record = 30 * 60 * 360

    # for selection and sampling 
    segment_dict_ann = {}
    record_count = 0

    for _, record_id in enumerate(arr_db):
        record_path = os.path.join(basic_arr_path, str(record_id))

        ann = wfdb.rdann(record_path, 'atr', sampto=num_samples_in_record, return_label_elements=['description',
                                                                                                  'symbol',
                                                                                                  'label_store'])
        df = pd.DataFrame({'description': ann.description, 'sample': ann.sample, 'symbol': ann.symbol,
                           'label_store': ann.label_store, 'aux': ann.aux_note})
        df = dynamic_replace(df)
        counter = 0
        reset_flag = True
        allowed_labels = ['Normal beat']
        allowed_symbols = ['N']

        normal_counter = 0
        for i in range(1, df.shape[0] - 1):
            curr_label, curr_sample, curr_symbol = df.loc[i, ['description', 'sample', 'symbol']]
            if curr_label == 'Normal beat':
                normal_counter += 1
            if reset_flag:
                start_sample = curr_sample
                ann_num_start = i
                normal_counter = 0
                allowed_labels = ['Normal beat']
                allowed_symbols = ['N']
            next_label, next_sample, next_symbol = df.loc[i + 1, ['description', 'sample', 'symbol']]
            if curr_label == next_label or next_label in allowed_labels or len(allowed_labels) < 2:

                if next_label not in allowed_labels:
                    allowed_labels.append(next_label)
                    allowed_symbols.append(next_symbol)
                ann_num_end = i + 1
                counter += next_sample - curr_sample
                reset_flag = False
                if counter > desired_segment_len:
                    counter = 0
                    reset_flag = True
                    signal = wfdb.rdsamp(record_path, sampfrom=start_sample, sampto=start_sample + 3600)[0][:, 0]
                    normal_ratio = normal_counter / (ann_num_end - ann_num_start)

                    if df.loc[ann_num_start:ann_num_end]['aux'].unique().shape[0] == 1:
                        aux_seg = df.loc[ann_num_start]['aux']
                    else:
                        aux_seg = 'invalid'

                    segment_dict_ann[record_count] = [record_id, allowed_labels[-1], signal, normal_ratio,
                                                      allowed_symbols[-1], aux_seg]
                    record_count = record_count + 1
            else:
                counter = 0
                normal_counter = 0
                reset_flag = True
                allowed_labels = ['Normal beat']
                allowed_symbols = ['N']

    return segment_dict_ann

d_dict = create_index_df()