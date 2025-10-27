import numpy as np

from preprocess.parse_csv import EHRParser


def split_patients(patient_admission, admission_codes, code_map, train_num, test_num, seed=6669):
    np.random.seed(seed)
    common_pids = set()
    for i, code in enumerate(code_map):
        print('\r\t%.2f%%' % ((i + 1) * 100 / len(code_map)), end='')
        for pid, admissions in patient_admission.items():
            for admission in admissions:
                codes = admission_codes[admission[EHRParser.adm_id_col]]
                if code in codes:
                    common_pids.add(pid)
                    break
            else:
                continue
            break
    print('\r\t100%')
    max_admission_num = 0
    pid_max_admission_num = 0
    for pid, admissions in patient_admission.items():
        if len(admissions) > max_admission_num:
            max_admission_num = len(admissions)
            pid_max_admission_num = pid
    common_pids.add(pid_max_admission_num)
    remaining_pids = np.array(list(set(patient_admission.keys()).difference(common_pids)))
    np.random.shuffle(remaining_pids)

    valid_num = len(patient_admission) - train_num - test_num
    train_pids = np.array(list(common_pids.union(set(remaining_pids[:(train_num - len(common_pids))].tolist()))))
    valid_pids = remaining_pids[(train_num - len(common_pids)):(train_num + valid_num - len(common_pids))]
    test_pids = remaining_pids[(train_num + valid_num - len(common_pids)):]
    return train_pids, valid_pids, test_pids


def build_code_xy(pids, patient_admission, admission_codes_encoded, max_admission_num, code_num):
    n = len(pids)
    x = np.zeros((n, max_admission_num, code_num), dtype=bool)
    y = np.zeros((n, code_num), dtype=int)
    lens = np.zeros((n,), dtype=int)
    for i, pid in enumerate(pids):
        print('\r\t%d / %d' % (i + 1, len(pids)), end='')
        admissions = patient_admission[pid]
        for k, admission in enumerate(admissions[:-1]):
            codes = admission_codes_encoded[admission[EHRParser.adm_id_col]]
            x[i, k, codes] = 1
        codes = np.array(admission_codes_encoded[admissions[-1][EHRParser.adm_id_col]])
        y[i, codes] = 1
        lens[i] = len(admissions) - 1
    print('\r\t%d / %d' % (len(pids), len(pids)))
    return x, y, lens


def build_heart_failure_y(hf_prefix, codes_y, code_map):
    hf_list = np.array([cid for code, cid in code_map.items() if code.startswith(hf_prefix)])
    hfs = np.zeros((len(code_map),), dtype=int)
    hfs[hf_list] = 1
    hf_exist = np.logical_and(codes_y, hfs)
    y = (np.sum(hf_exist, axis=-1) > 0).astype(int)
    return y


def split_patient_cohorts(patient_admission, patient_cohorts, cohort_feature, test_num):
    if cohort_feature not in ['age', 'gender', 'ethnicity', 'insurance', 'language', 'marital_status', 'year', 'region']:
        raise ValueError(f'Invalid cohort feature: {cohort_feature}')

    feature_support = {pid: cohorts.get(cohort_feature) for pid, cohorts in patient_cohorts.items()}

    has_true = any(feature_support.values())
    has_false = any(not v for v in feature_support.values())

    if not has_true and not has_false:
        raise ValueError(f'Feature {cohort_feature} is not available in this dataset (all values are False)')

    if not has_false:
        raise ValueError(f'Feature {cohort_feature} has no False values, cannot split dataset')

    true_pids = [pid for pid, value in feature_support.items() if value is True]
    false_pids = [pid for pid, value in feature_support.items() if value is False]

    print(f'Cohort split on {cohort_feature}: {len(true_pids)} True, {len(false_pids)} False')

    train_pids = set(true_pids)

    max_admission_num = 0
    pid_max_admission_num = None
    for pid in false_pids:
        admissions = patient_admission[pid]
        if len(admissions) > max_admission_num:
            max_admission_num = len(admissions)
            pid_max_admission_num = pid

    if pid_max_admission_num is not None and pid_max_admission_num not in train_pids:
        false_pids.remove(pid_max_admission_num)
        test_pids_set = {pid_max_admission_num}
    else:
        test_pids_set = set()

    remaining_false_pids = np.array(false_pids)
    np.random.shuffle(remaining_false_pids)

    num_needed_for_test = test_num - len(test_pids_set)
    if num_needed_for_test > 0:
        test_pids_set.update(remaining_false_pids[:num_needed_for_test])
        valid_pids = remaining_false_pids[num_needed_for_test:]
    else:
        valid_pids = remaining_false_pids

    train_pids = np.array(list(train_pids))
    test_pids = np.array(list(test_pids_set))
    valid_pids = np.array(valid_pids)

    return train_pids, valid_pids, test_pids