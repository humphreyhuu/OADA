"""
Analyze heart failure label distribution in full MIMIC-IV dataset BEFORE train/test split
This checks the raw HF positive/negative ratio across all patients
"""
import os
import _pickle as pickle
import numpy as np

from preprocess.parse_csv import Mimic3Parser, Mimic4Parser, EICUParser
from preprocess.encode import encode_code
from preprocess.build_dataset import build_heart_failure_y


def analyze_presplit_hf_distribution(dataset='mimic4'):
    """
    Analyze HF label distribution before any train/test splitting
    This follows the same preprocessing steps as run_preprocess.py
    """
    print(f"\n{'='*80}")
    print(f"Analyzing PRE-SPLIT HF Distribution for {dataset.upper()}")
    print(f"{'='*80}\n")

    data_path = 'data'
    dataset_path = os.path.join(data_path, dataset)

    if dataset in ['mimic3', 'mimic4']:
        raw_path = os.path.join(dataset_path, 'raw')
    elif dataset == 'eicu':
        raw_path = os.path.join(dataset_path, 'raw/physionet.org/files/eicu-crd/2.0')
    else:
        raise ValueError('Invalid dataset: %s' % dataset)

    parsed_path = os.path.join(dataset_path, 'parsed')

    # Check if parsed data exists
    if not os.path.exists(parsed_path):
        print(f"ERROR: Parsed path not found at {parsed_path}")
        print("Please run run_preprocess.py first to parse the raw data.")
        return

    # Load parsed data (same as run_preprocess.py with from_saved=True)
    print("Loading parsed data...")
    try:
        patient_admission = pickle.load(open(os.path.join(parsed_path, 'patient_admission.pkl'), 'rb'))
        admission_codes = pickle.load(open(os.path.join(parsed_path, 'admission_codes.pkl'), 'rb'))
        patient_info = pickle.load(open(os.path.join(parsed_path, 'patient_info.pkl'), 'rb'))
        print(f"Successfully loaded parsed data")
    except Exception as e:
        print(f"ERROR loading parsed data: {e}")
        return

    # Basic statistics
    patient_num = len(patient_admission)
    max_admission_num = max([len(admissions) for admissions in patient_admission.values()])
    avg_admission_num = sum([len(admissions) for admissions in patient_admission.values()]) / patient_num
    max_visit_code_num = max([len(codes) for codes in admission_codes.values()])
    avg_visit_code_num = sum([len(codes) for codes in admission_codes.values()]) / len(admission_codes)

    print(f"\nDataset Statistics:")
    print(f"  Total patients: {patient_num}")
    print(f"  Max admission num: {max_admission_num}")
    print(f"  Mean admission num: {avg_admission_num:.2f}")
    print(f"  Max code num in an admission: {max_visit_code_num}")
    print(f"  Mean code num in an admission: {avg_visit_code_num:.2f}")

    # Encode codes (same as run_preprocess.py)
    print('\nEncoding codes...')
    admission_codes_encoded, code_map = encode_code(patient_admission, admission_codes)
    code_num = len(code_map)
    print(f'  Total unique codes: {code_num}')

    # Build labels for ALL patients (no splitting)
    print('\nBuilding heart failure labels for ALL patients...')
    all_pids = list(patient_admission.keys())

    # Build code labels (we need this to extract HF labels)
    from preprocess.build_dataset import build_code_xy
    max_admission_num_for_build = max([len(admissions) for admissions in patient_admission.values()])

    print(f'  Processing {len(all_pids)} patients...')
    code_x, codes_y, visit_lens = build_code_xy(
        all_pids,
        patient_admission,
        admission_codes_encoded,
        max_admission_num_for_build,
        code_num
    )

    # Build HF labels using ICD-9 code 428 (Heart Failure)
    print('  Extracting heart failure labels (ICD-9: 428.x)...')
    hf_y = build_heart_failure_y('428', codes_y, code_map)

    # Analyze HF distribution
    print(f"\n{'='*80}")
    print("HEART FAILURE LABEL DISTRIBUTION (FULL DATASET - PRE-SPLIT)")
    print(f"{'='*80}\n")

    positive_count = np.sum(hf_y == 1)
    negative_count = np.sum(hf_y == 0)
    total_count = len(hf_y)

    print(f"Total patients: {total_count}")
    print(f"Positive (HF=1): {positive_count} ({positive_count/total_count*100:.2f}%)")
    print(f"Negative (HF=0): {negative_count} ({negative_count/total_count*100:.2f}%)")

    if positive_count > 0 and negative_count > 0:
        imbalance_ratio = max(positive_count, negative_count) / min(positive_count, negative_count)
        print(f"Class imbalance ratio: {imbalance_ratio:.2f}:1")
    elif positive_count == 0:
        print("ERROR: No positive samples found!")
    elif negative_count == 0:
        print("ERROR: No negative samples found!")

    # Additional analysis: Check which codes are related to HF
    print(f"\n{'='*80}")
    print("HEART FAILURE RELATED CODES IN DATASET")
    print(f"{'='*80}\n")

    hf_codes = [code for code in code_map.keys() if code.startswith('428')]
    print(f"Heart failure ICD-9 codes found in dataset:")
    print(f"Note: HF label is based on LAST admission codes only (codes_y)\n")

    for code in sorted(hf_codes):
        code_idx = code_map[code]
        # Count in LAST admission (codes_y) - this is what determines HF label
        count_last_admission = np.sum(codes_y[:, code_idx])
        print(f"  {code}: {count_last_admission} patients ({count_last_admission/total_count*100:.2f}%)")

    # Summary
    print(f"\n  TOTAL patients with HF in last admission:")
    hf_code_indices = [code_map[code] for code in hf_codes]
    total_last = 0
    for i in range(len(codes_y)):
        if np.any(codes_y[i, hf_code_indices]):
            total_last += 1

    print(f"    {total_last} patients ({total_last/total_count*100:.2f}%)")

    print(f"\n{'='*80}\n")

    # Return statistics for comparison
    return {
        'dataset': dataset,
        'total_patients': total_count,
        'positive': positive_count,
        'negative': negative_count,
        'positive_rate': positive_count / total_count * 100,
        'imbalance_ratio': max(positive_count, negative_count) / max(min(positive_count, negative_count), 1) if positive_count > 0 and negative_count > 0 else None
    }


if __name__ == '__main__':
    # Analyze all three datasets
    datasets = ['mimic3', 'mimic4', 'eicu']
    results = {}

    for dataset in datasets:
        dataset_path = os.path.join('data', dataset)
        if os.path.exists(dataset_path):
            result = analyze_presplit_hf_distribution(dataset)
            if result:
                results[dataset] = result
        else:
            print(f"\nSkipping {dataset} - dataset path not found at {dataset_path}\n")

    # Print comparison table
    if results:
        print(f"\n{'='*80}")
        print("COMPARATIVE SUMMARY (PRE-SPLIT)")
        print(f"{'='*80}\n")
        print(f"{'Dataset':<12} {'Total':<10} {'Positive':<10} {'Negative':<10} {'Pos%':<10} {'Imbalance':<12}")
        print("-" * 72)
        for dataset_name, stats in results.items():
            imbalance_str = f"{stats['imbalance_ratio']:.2f}" if stats['imbalance_ratio'] else "N/A"
            print(f"{dataset_name:<12} {stats['total_patients']:<10} {stats['positive']:<10} {stats['negative']:<10} "
                  f"{stats['positive_rate']:<10.2f} {imbalance_str:<12}")
        print(f"\n{'='*80}\n")
