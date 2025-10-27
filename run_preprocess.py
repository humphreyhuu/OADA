import os
import _pickle as pickle

from preprocess import save_data
from preprocess.parse_csv import Mimic3Parser, Mimic4Parser, EICUParser
from preprocess.encode import encode_code
from preprocess.build_dataset import split_patients, split_patient_cohorts, build_code_xy, build_heart_failure_y
from preprocess.auxiliary import identify_patient_cohorts

if __name__ == '__main__':
    conf = {
        'mimic3': {
            'parser': Mimic3Parser,
            'train_num': 6000,
            'test_num': 1000,
            'threshold': 0.01,
            'cohort_feature': 'age'
        },
        'mimic4': {
            'parser': Mimic4Parser,
            'train_num': 8000,
            'test_num': 1000,
            'threshold': 0.01,
            'sample_num': 10000,
            'cohort_feature': 'year'
        },
        'eicu': {
            'parser': EICUParser,
            'train_num': 8000,
            'test_num': 1000,
            'threshold': 0.01,
            'cohort_feature': 'region'
        }
    }
    from_saved = False
    data_path = 'data'
    dataset = 'eicu'  # mimic3, eicu, or mimic4
    dataset_path = os.path.join(data_path, dataset)
    if dataset in ['mimic3', 'mimic4']:
        raw_path = os.path.join(dataset_path, 'raw')
    elif dataset == 'eicu':
        raw_path = os.path.join(dataset_path, 'raw/physionet.org/files/eicu-crd/2.0')
    else:
        raise ValueError('Invalid dataset: %s' % dataset)
    if not os.path.exists(raw_path):
        os.makedirs(raw_path)
        print('please put the CSV files in `data/%s/raw`' % dataset)
        exit()
    parsed_path = os.path.join(dataset_path, 'parsed')
    if from_saved:
        patient_admission = pickle.load(open(os.path.join(parsed_path, 'patient_admission.pkl'), 'rb'))
        admission_codes = pickle.load(open(os.path.join(parsed_path, 'admission_codes.pkl'), 'rb'))
        patient_info = pickle.load(open(os.path.join(parsed_path, 'patient_info.pkl'), 'rb'))
    else:
        parser = conf[dataset]['parser'](raw_path)
        sample_num = conf[dataset].get('sample_num', None)
        patient_admission, admission_codes, patient_info = parser.parse(sample_num)
        print('saving parsed data ...')
        if not os.path.exists(parsed_path):
            os.makedirs(parsed_path)
        pickle.dump(patient_admission, open(os.path.join(parsed_path, 'patient_admission.pkl'), 'wb'))
        pickle.dump(admission_codes, open(os.path.join(parsed_path, 'admission_codes.pkl'), 'wb'))
        pickle.dump(patient_info, open(os.path.join(parsed_path, 'patient_info.pkl'), 'wb'))

    patient_num = len(patient_admission)
    max_admission_num = max([len(admissions) for admissions in patient_admission.values()])
    avg_admission_num = sum([len(admissions) for admissions in patient_admission.values()]) / patient_num
    max_visit_code_num = max([len(codes) for codes in admission_codes.values()])
    avg_visit_code_num = sum([len(codes) for codes in admission_codes.values()]) / len(admission_codes)
    print('patient num: %d' % patient_num)
    print('max admission num: %d' % max_admission_num)
    print('mean admission num: %.2f' % avg_admission_num)
    print('max code num in an admission: %d' % max_visit_code_num)
    print('mean code num in an admission: %.2f' % avg_visit_code_num)

    print('encoding code ...')
    admission_codes_encoded, code_map = encode_code(patient_admission, admission_codes)
    code_num = len(code_map)
    print('There are %d codes' % code_num)

    print('identifying patient cohorts ...')
    patient_cohorts = identify_patient_cohorts(patient_info)

    features = ['age', 'gender', 'ethnicity', 'insurance', 'language', 'marital_status', 'year', 'region']
    for feature in features:
        true_count = sum(1 for cohort in patient_cohorts.values() if cohort.get(feature) is True)
        false_count = sum(1 for cohort in patient_cohorts.values() if cohort.get(feature) is False)
        print(f'{feature}: True={true_count}, False={false_count}')

    cohort_feature = conf[dataset].get('cohort_feature')
    if cohort_feature:
        print(f'splitting patients by cohort feature: {cohort_feature} ...')
        train_pids, valid_pids, test_pids = split_patient_cohorts(
            patient_admission=patient_admission,
            patient_cohorts=patient_cohorts,
            cohort_feature=cohort_feature,
            test_num=conf[dataset]['test_num']
        )
    else:
        print('splitting patients using default method ...')
        train_pids, valid_pids, test_pids = split_patients(
            patient_admission=patient_admission,
            admission_codes=admission_codes,
            code_map=code_map,
            train_num=conf[dataset]['train_num'],
            test_num=conf[dataset]['test_num']
        )
    print('There are %d train, %d valid, %d test samples' % (len(train_pids), len(valid_pids), len(test_pids)))

    common_args = [patient_admission, admission_codes_encoded, max_admission_num, code_num]
    print('building train codes features and labels ...')
    (train_code_x, train_codes_y, train_visit_lens) = build_code_xy(train_pids, *common_args)
    print('building valid codes features and labels ...')
    (valid_code_x, valid_codes_y, valid_visit_lens) = build_code_xy(valid_pids, *common_args)
    print('building test codes features and labels ...')
    (test_code_x, test_codes_y, test_visit_lens) = build_code_xy(test_pids, *common_args)

    print('building train heart failure labels ...')
    train_hf_y = build_heart_failure_y('428', train_codes_y, code_map)
    print('building valid heart failure labels ...')
    valid_hf_y = build_heart_failure_y('428', valid_codes_y, code_map)
    print('building test heart failure labels ...')
    test_hf_y = build_heart_failure_y('428', test_codes_y, code_map)

    encoded_path = os.path.join(dataset_path, 'encoded')
    if not os.path.exists(encoded_path):
        os.makedirs(encoded_path)
    print('saving encoded data ...')
    pickle.dump(patient_admission, open(os.path.join(encoded_path, 'patient_admission.pkl'), 'wb'))
    pickle.dump(admission_codes_encoded, open(os.path.join(encoded_path, 'codes_encoded.pkl'), 'wb'))
    pickle.dump(code_map, open(os.path.join(encoded_path, 'code_map.pkl'), 'wb'))
    pickle.dump(patient_cohorts, open(os.path.join(encoded_path, 'patient_cohorts.pkl'), 'wb'))
    pickle.dump({
        'train_pids': train_pids,
        'valid_pids': valid_pids,
        'test_pids': test_pids
    }, open(os.path.join(encoded_path, 'pids.pkl'), 'wb'))

    print('saving standard data ...')
    standard_path = os.path.join(dataset_path, 'standard')
    train_path = os.path.join(standard_path, 'train')
    valid_path = os.path.join(standard_path, 'valid')
    test_path = os.path.join(standard_path, 'test')
    if not os.path.exists(standard_path):
        os.makedirs(standard_path)
    if not os.path.exists(train_path):
        os.makedirs(train_path)
        os.makedirs(valid_path)
        os.makedirs(test_path)

    print('\tsaving training data')
    save_data(train_path, train_code_x, train_visit_lens, train_codes_y, train_hf_y)
    print('\tsaving valid data')
    save_data(valid_path, valid_code_x, valid_visit_lens, valid_codes_y, valid_hf_y)
    print('\tsaving test data')
    save_data(test_path, test_code_x, test_visit_lens, test_codes_y, test_hf_y)