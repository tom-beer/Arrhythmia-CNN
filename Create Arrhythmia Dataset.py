import numpy as np
import wfdb
import os
import pandas as pd
import random
random.seed(42)


def dynamic_replace(df):
    curr_aux = df['aux'].loc[0]
    for idx, x in enumerate(df['aux']):
        if x != '':
            curr_aux = x
        df.loc[idx, 'aux'] = curr_aux
    return df


def create_index_df(desired_segment_len=3600, basic_arr_path="data/mit-bih-arrhythmia-database-1.0.0"):
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
seg_df = pd.DataFrame.from_dict(d_dict, orient='index')
seg_df.rename(columns={0: 'record_id', 1: 'label', 2: 'signal', 3: 'normal_ratio', 4: 'symbol', 5: 'aux'}, inplace=True)

num_labels_dict = {
    'Normal beat': 283,  # N
    'Left bundle branch block beat': 103,  # L
    'Atrial premature beat': 66,  # A
    'Atrial flutter': 20,  # (AFL (aux)
    'Atrial fibrillation': 135,  # (AFIB (aux)
    'Pre-excitation (WPW)': 21,  # (PREX (aux)
    'Premature ventricular contraction': 133,  # V
    'Ventricular bigeminy': 55,  # (B (aux)
    'Ventricular trigeminy': 13,  # (T (aux)
    'Ventricular tachycardia': 10,  # (VT (aux)
    'Idioventricular rhythm': 10,  # (IVR (aux)
    'Ventricular flutter': 10,  # (VFL (aux)
    'Fusion of ventricular and normal beat': 11,  # F
    'Second-degree heart block': 10,
    'Pacemaker rhythm': 45,  # /
    'Supraventricular tachyarrhythmia': 13,  # (SVTA (aux)
    'Right bundle branch block beat': 62,  # R
}

num_labels_we_have = {
    'N': 283,  # 'Normal beat'
    'L': 103,  # 'Left bundle branch block beat'
    'A': 66,  # 'Atrial premature beat':
    # 'V': 133, #'Premature ventricular contraction'
    # '!': 10,   #'Ventricular flutter'
    # 'F': 11, # 'Fusion of ventricular and normal beat'
    'R': 62,  # 'Right bundle branch block beat'
    'AFL': 20,
    'AFIB': 135,
    'PREX': 21,
    'B': 55,
    'T': 13,
    # '(VT'   : 10,
    'IVR': 10,
    'VFL': 10,
    'P': 45
    # '(SVTA' : 13
}

seg_df['target'] = 'invalid'
seg_df['aux'] = seg_df['aux'].str.lstrip('(')
seg_df['aux'] = seg_df['aux'].str.rstrip('\x00')

normal_aux_idx = (seg_df.aux == 'N') & (
    seg_df.symbol.apply(lambda x: True if x in num_labels_we_have.keys() else False))
seg_df.loc[normal_aux_idx, 'target'] = seg_df.loc[normal_aux_idx, 'symbol']

normal_label_idx = (seg_df.symbol == 'N') & (
    seg_df.aux.apply(lambda x: True if x in num_labels_we_have.keys() else False))
seg_df.loc[normal_label_idx, 'target'] = seg_df.loc[normal_label_idx, 'aux']

seg_df.loc[seg_df['aux'] == 'B', 'target'] = 'B'
seg_df.loc[seg_df['aux'] == 'T', 'target'] = 'T'
seg_df.loc[seg_df['aux'] == 'IVR', 'target'] = 'IVR'
seg_df.loc[seg_df['aux'] == 'P', 'target'] = 'P'
seg_df.loc[seg_df['aux'] == 'VFL', 'target'] = 'VFL'
seg_df['target'].value_counts()


def sample_per_sym(seg_df, cls, num):
    # filter
    target_df = seg_df.loc[seg_df[seg_df['target'] == cls].index]
    # sample
    print(f' len df: {len(target_df)}')
    sample_df = target_df.sample(n=num)

    # add the index at the begining of the signal
    signals = np.stack(sample_df.signal.values)
    idx = 0  # TODO: tmp fix. Bar - what is this?
    return idx, sample_df, signals


sample_df = []
signals = []
idx_arr = []
for cls, num in num_labels_we_have.items():
    print(f'target:{cls}, num: {num}')
    idx, sample_df_target, signals_target = sample_per_sym(seg_df, cls, num)
    sample_df.append(sample_df_target)
    signals.append(signals_target)
    idx_arr = np.append(idx_arr, idx)
sample_df = pd.concat(sample_df)

sample_df.reset_index()
label_numeric_dict = {
    'N': 0,
    'A': 1,
    'AFL': 2,
    'AFIB': 3,
    'PREX': 4,
    'B': 5,
    'T': 6,
    'IVR': 7,
    'VFL': 8,
    'L': 9,
    'R': 10,
    'P': 11
}
sample_df['target_str'] = sample_df['target']
sample_df = sample_df.replace({'target': label_numeric_dict})
sample_df.to_pickle("data/Arrhythmia_dataset.pkl")
