import os
import _pickle as pickle
from collections import Counter
import numpy as np

def analyze_patient_info(dataset_name):
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name.upper()}")
    print(f"{'='*60}")

    parsed_path = os.path.join('data', dataset_name, 'parsed')
    patient_info = pickle.load(open(os.path.join(parsed_path, 'patient_info.pkl'), 'rb'))

    total_patients = len(patient_info)
    print(f"\nTotal patients: {total_patients}")

    features = ['age', 'gender', 'ethnicity', 'insurance', 'language', 'marital_status', 'year', 'region']

    feature_data = {feature: [] for feature in features}
    feature_nulls = {feature: 0 for feature in features}

    for pid, info in patient_info.items():
        for feature in features:
            value = info.get(feature)
            if value is None:
                feature_nulls[feature] += 1
            else:
                feature_data[feature].append(value)

    for feature in features:
        print(f"\n{'-'*60}")
        print(f"{feature.upper()} Distribution:")
        print(f"{'-'*60}")

        null_count = feature_nulls[feature]
        non_null_count = len(feature_data[feature])

        print(f"Null values: {null_count} ({null_count/total_patients*100:.2f}%)")
        print(f"Non-null values: {non_null_count} ({non_null_count/total_patients*100:.2f}%)")

        if non_null_count > 0:
            if feature == 'age':
                ages = feature_data[feature]
                print(f"\nAge statistics:")
                print(f"  Mean: {np.mean(ages):.2f}")
                print(f"  Median: {np.median(ages):.2f}")
                print(f"  Std: {np.std(ages):.2f}")
                print(f"  Min: {np.min(ages)}")
                print(f"  Max: {np.max(ages)}")

                age_bins = [0, 18, 30, 40, 50, 60, 70, 80, 200]
                age_labels = ['<18', '18-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+']
                age_counts = {label: 0 for label in age_labels}

                for age in ages:
                    for i in range(len(age_bins) - 1):
                        if age_bins[i] <= age < age_bins[i+1]:
                            age_counts[age_labels[i]] += 1
                            break

                print(f"\nAge distribution by bins:")
                for label in age_labels:
                    count = age_counts[label]
                    if count > 0:
                        print(f"  {label}: {count} ({count/total_patients*100:.2f}%)")

            elif feature == 'year':
                year_counts = Counter(feature_data[feature])
                print(f"\nYear value counts (sorted by year):")
                for year in sorted(year_counts.keys()):
                    count = year_counts[year]
                    print(f"  {year}: {count} ({count/total_patients*100:.2f}%)")

            else:
                value_counts = Counter(feature_data[feature])
                print(f"\n{feature.capitalize()} value counts (top 20):")
                for value, count in value_counts.most_common(20):
                    print(f"  {value}: {count} ({count/total_patients*100:.2f}%)")

if __name__ == '__main__':
    datasets = ['mimic3', 'mimic4', 'eicu']

    for dataset in datasets:
        dataset_path = os.path.join('data', dataset, 'parsed', 'patient_info.pkl')
        if os.path.exists(dataset_path):
            analyze_patient_info(dataset)
        else:
            print(f"\n{'='*60}")
            print(f"Dataset: {dataset.upper()}")
            print(f"{'='*60}")
            print(f"Skipping {dataset}: patient_info.pkl not found")

    print(f"\n{'='*60}")
    print("Analysis Complete")
    print(f"{'='*60}")
