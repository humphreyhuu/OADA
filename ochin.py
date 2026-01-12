"""
OCHIN Patient Observation Data Analysis Script
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')


class OCHINDataAnalyzer:

    def __init__(self, data_path, icd_map_path=None):
        self.data_path = data_path
        self.icd_map_path = icd_map_path
        self.df = None
        self.merged_df = None
        self.icd9_to_icd10_map = None
        self.facilities_df = None
        self.demographics_df = None
        self.visit_df = None

    def _load_icd9_to_icd10_map(self):
        if self.icd_map_path is None or not os.path.exists(self.icd_map_path):
            print("Warning: ICD mapping file not provided or not found. ICD9 codes will remain unchanged.")
            return {}

        print("Loading ICD-9 to ICD-10 mapping...")
        icd_df = pd.read_csv(self.icd_map_path, usecols=['ICD10', 'ICD9'],
                            converters={'ICD10': str, 'ICD9': str})

        icd9_to_icd10 = {}
        for _, row in icd_df.iterrows():
            icd9_code = row['ICD9']
            icd10_code = row['ICD10']
            if pd.notna(icd9_code) and pd.notna(icd10_code) and icd9_code != 'NoDx':
                if icd9_code not in icd9_to_icd10:
                    icd9_to_icd10[icd9_code] = icd10_code

        print(f"Loaded {len(icd9_to_icd10)} ICD9-to-ICD10 mappings")
        return icd9_to_icd10

    def load_data(self):
        print("Loading data...")
        self.df = pd.read_csv(self.data_path)

        initial_rows = len(self.df)
        print(f"Initial data loaded: {initial_rows:,} rows")

        self.df.rename(columns={
            'PATIENT_NUM': 'patient_id',
            'ENCOUNTER_NUM': 'admission_id',
            'CONCEPT_CD': 'concept_cd',
            'START_DATE': 'start_date',
            'END_DATE': 'end_date'
        }, inplace=True)

        print("\nFiltering data...")
        self.df = self.df[self.df['admission_id'] != 0]
        print(f"  Removed {initial_rows - len(self.df):,} rows with admission_id=0")
        print(f"  Remaining: {len(self.df):,} rows")

        initial_after_filter1 = len(self.df)
        self.df = self.df[self.df['concept_cd'].str.startswith('ICD10CM:') |
                         self.df['concept_cd'].str.startswith('ICD9CM:')]
        print(f"  Removed {initial_after_filter1 - len(self.df):,} rows without ICD10CM or ICD9CM prefix")
        print(f"  Remaining: {len(self.df):,} rows")

        self.df['start_date'] = pd.to_datetime(self.df['start_date'], errors='coerce')
        self.df['end_date'] = pd.to_datetime(self.df['end_date'], errors='coerce')

        self.icd9_to_icd10_map = self._load_icd9_to_icd10_map()

        print("\nProcessing ICD codes...")
        self.df['icd10_code'] = self.df['concept_cd'].apply(self._extract_and_convert_to_icd10)

        print(f"\nData loaded and filtered successfully: {len(self.df):,} rows")
        return self

    def _extract_and_convert_to_icd10(self, concept_cd):
        if pd.isna(concept_cd):
            return None

        concept_cd = str(concept_cd)

        if concept_cd.startswith('ICD10CM:'):
            return concept_cd.split(':', 1)[1] if ':' in concept_cd else None

        elif concept_cd.startswith('ICD9CM:'):
            icd9_code = concept_cd.split(':', 1)[1] if ':' in concept_cd else None
            if icd9_code is None or icd9_code == '':
                return None

            if self.icd9_to_icd10_map and icd9_code in self.icd9_to_icd10_map:
                return self.icd9_to_icd10_map[icd9_code]
            else:
                icd9_no_dot = icd9_code.replace('.', '')
                if self.icd9_to_icd10_map and icd9_no_dot in self.icd9_to_icd10_map:
                    return self.icd9_to_icd10_map[icd9_no_dot]
                return icd9_code

        return None

    def basic_info(self):
        print("\n" + "="*80)
        print("BASIC DATASET INFORMATION")
        print("="*80)

        print(f"\nDataset shape: {self.df.shape}")
        print(f"Total rows: {len(self.df):,}")

        print("\nMissing values for key columns:")
        print(f"  end_date: {self.df['end_date'].isna().sum():,} ({self.df['end_date'].isna().sum()/len(self.df)*100:.2f}%)")
        print(f"  icd10_code: {self.df['icd10_code'].isna().sum():,} ({self.df['icd10_code'].isna().sum()/len(self.df)*100:.2f}%)")

        return self

    def patient_analysis(self):
        print("\n" + "="*80)
        print("PATIENT ANALYSIS")
        print("="*80)

        total_patients = self.df['patient_id'].nunique()
        print(f"\nTotal unique patients: {total_patients:,}")

        admissions_per_patient = self.df.groupby('patient_id')['admission_id'].nunique()

        print(f"\nAdmissions per patient statistics:")
        print(f"  Maximum admissions: {admissions_per_patient.max()}")
        print(f"  Average admissions: {admissions_per_patient.mean():.2f}")
        print(f"  Median admissions: {admissions_per_patient.median():.0f}")
        print(f"  Min admissions: {admissions_per_patient.min()}")
        print(f"  Std admissions: {admissions_per_patient.std():.2f}")

        patients_1_admission = (admissions_per_patient == 1).sum()
        pct_1_admission = (patients_1_admission / total_patients) * 100
        print(f"\nPatients with only 1 admission:")
        print(f"  Count: {patients_1_admission:,}")
        print(f"  Percentage: {pct_1_admission:.2f}%")

        print(f"\nTop 10 patients by number of admissions:")
        top_patients = admissions_per_patient.nlargest(10)
        for idx, (patient, count) in enumerate(top_patients.items(), 1):
            print(f"  {idx}. Patient {patient}: {count} admissions")

        return self

    def admission_analysis(self):
        print("\n" + "="*80)
        print("ADMISSION ANALYSIS")
        print("="*80)

        total_admissions = self.df['admission_id'].nunique()
        print(f"\nTotal unique admissions: {total_admissions:,}")

        icd10_df = self.df[self.df['icd10_code'].notna()].copy()

        if len(icd10_df) > 0:
            print(f"\nUnique ICD10CM codes: {icd10_df['icd10_code'].nunique():,}")

            codes_per_admission = icd10_df.groupby('admission_id')['icd10_code'].nunique()

            print(f"\nUnique ICD10CM codes per admission:")
            print(f"  Maximum codes: {codes_per_admission.max()}")
            print(f"  Average codes: {codes_per_admission.mean():.2f}")
            print(f"  Median codes: {codes_per_admission.median():.0f}")
            print(f"  Min codes: {codes_per_admission.min()}")
            print(f"  Std codes: {codes_per_admission.std():.2f}")

        return self

    def concept_analysis(self):
        print("\n" + "="*80)
        print("CONCEPT CODE ANALYSIS")
        print("="*80)

        icd10_df = self.df[self.df['icd10_code'].notna()].copy()
        if len(icd10_df) > 0:
            print(f"\nTop 10 most common ICD10CM codes:")
            top_icd10 = icd10_df['icd10_code'].value_counts().head(10)
            for idx, (code, count) in enumerate(top_icd10.items(), 1):
                pct = (count / len(icd10_df)) * 100
                print(f"  {idx}. {code}: {count:,} ({pct:.2f}%)")

        return self

    def date_validation(self):
        print("\n" + "="*80)
        print("DATE VALIDATION")
        print("="*80)

        print("\nChecking date consistency within admissions...")

        start_date_check = self.df.groupby('admission_id')['start_date'].apply(
            lambda x: x.nunique()
        )
        inconsistent_start = (start_date_check > 1).sum()

        end_date_check = self.df[self.df['end_date'].notna()].groupby('admission_id')['end_date'].apply(
            lambda x: x.nunique()
        )
        inconsistent_end = (end_date_check > 1).sum()

        print(f"\nSTART_DATE consistency:")
        if inconsistent_start == 0:
            print(f"  All admissions have consistent start_date")
        else:
            print(f"  {inconsistent_start:,} admissions have inconsistent start_date")

        print(f"\nEND_DATE consistency:")
        if inconsistent_end == 0:
            print(f"  All admissions have consistent end_date (where not null)")
        else:
            print(f"  {inconsistent_end:,} admissions have inconsistent end_date")

        return self

    def analyze_additional_files(self, base_path):
        print("\n" + "="*80)
        print("ANALYZING ADDITIONAL DATA FILES")
        print("="*80)

        # [1/3] Analyzing facilities.csv
        print("\n[1/3] Analyzing facilities.csv...")
        facilities_df = pd.read_csv(base_path + 'facilities.csv')
        print(f"Columns: {list(facilities_df.columns)}")

        print(f"\nMissing values:")
        missing = facilities_df.isnull().sum()
        missing_pct = (missing / len(facilities_df)) * 100
        for col in facilities_df.columns:
            print(f"  {col}: {missing[col]:,} ({missing_pct[col]:.2f}%)")

        for col in ['FACILITY_TYPE', 'HEALTHSYSTEMID', 'DELIVERY_SITE_ID', 'STATE_ABBR']:
            if col in facilities_df.columns:
                print(f"\n{col}:")
                print(facilities_df[col].value_counts())

        facilities_keep_cols = ['LOCATION_CD', 'FACILITY_TYPE', 'HEALTHSYSTEMID', 'DELIVERY_SITE_ID', 'STATE_ABBR']
        facilities_keep_cols = [col for col in facilities_keep_cols if col in facilities_df.columns]
        facilities_df = facilities_df[facilities_keep_cols]
        print(f"\nFiltered shape: {facilities_df.shape}")

        # [2/3] Analyzing patient_demographics_full.csv
        print("\n" + "-"*80)
        print("[2/3] Analyzing patient_demographics_full.csv...")
        demographics_df = pd.read_csv(base_path + 'patient_demographics_full.csv')
        print(f"Columns: {list(demographics_df.columns)}")

        print(f"\nMissing values:")
        missing = demographics_df.isnull().sum()
        missing_pct = (missing / len(demographics_df)) * 100
        for col in demographics_df.columns:
            print(f"  {col}: {missing[col]:,} ({missing_pct[col]:.2f}%)")

        for col in ['BIRTH_DATE', 'RACE_CD', 'GENDER_CD']:
            if col in demographics_df.columns:
                print(f"\n{col}:")
                print(demographics_df[col].value_counts())

        demographics_keep_cols = ['PATIENT_NUM', 'BIRTH_DATE', 'RACE_CD', 'GENDER_CD']
        demographics_keep_cols = [col for col in demographics_keep_cols if col in demographics_df.columns]
        demographics_df = demographics_df[demographics_keep_cols]
        print(f"\nFiltered shape: {demographics_df.shape}")

        # [3/3] Analyzing visit_dimension.csv
        print("\n" + "-"*80)
        print("[3/3] Analyzing visit_dimension.csv...")
        chunk_size = 500000
        visit_chunks = []
        columns_printed = False

        for chunk in pd.read_csv(base_path + 'visit_dimension.csv', chunksize=chunk_size):
            if not columns_printed:
                print(f"Columns: {list(chunk.columns)}")
                columns_printed = True

            visit_keep_cols = ['ENCOUNTER_NUM', 'PATIENT_NUM', 'LOCATION_CD']
            visit_keep_cols = [col for col in visit_keep_cols if col in chunk.columns]
            chunk = chunk[visit_keep_cols]
            visit_chunks.append(chunk)

        visit_df = pd.concat(visit_chunks, ignore_index=True)

        print(f"\nMissing values:")
        missing = visit_df.isnull().sum()
        missing_pct = (missing / len(visit_df)) * 100
        for col in visit_df.columns:
            print(f"  {col}: {missing[col]:,} ({missing_pct[col]:.2f}%)")

        print(f"\nFiltered shape: {visit_df.shape}")

        print("\n" + "="*80)
        print("INTEGRATION FEASIBILITY ANALYSIS")
        print("="*80)

        obs_patients = set(self.df['patient_id'].unique())
        obs_encounters = set(self.df['admission_id'].unique())

        demo_patients = set(demographics_df['PATIENT_NUM'].unique())
        visit_patients = set(visit_df['PATIENT_NUM'].unique())
        visit_encounters = set(visit_df['ENCOUNTER_NUM'].unique())
        visit_locations = set(visit_df['LOCATION_CD'].dropna().unique())
        facility_locations = set(facilities_df['LOCATION_CD'].unique())

        print("\n1. Demographics linkage via PATIENT_NUM:")
        print(f"   Patients in observation: {len(obs_patients):,}")
        print(f"   Patients in demographics: {len(demo_patients):,}")
        overlap = obs_patients & demo_patients
        print(f"   Overlap: {len(overlap):,} ({len(overlap)/len(obs_patients)*100:.2f}%)")

        print("\n2. Visit dimension linkage via ENCOUNTER_NUM:")
        print(f"   Encounters in observation: {len(obs_encounters):,}")
        print(f"   Encounters in visit_dimension: {len(visit_encounters):,}")
        overlap = obs_encounters & visit_encounters
        print(f"   Overlap: {len(overlap):,} ({len(overlap)/len(obs_encounters)*100:.2f}%)")

        print("\n3. Visit dimension linkage via PATIENT_NUM:")
        print(f"   Patients in observation: {len(obs_patients):,}")
        print(f"   Patients in visit_dimension: {len(visit_patients):,}")
        overlap = obs_patients & visit_patients
        print(f"   Overlap: {len(overlap):,} ({len(overlap)/len(obs_patients)*100:.2f}%)")

        # Print first 10 unmatched patient_nums
        unmatched_patients = obs_patients - visit_patients
        if len(unmatched_patients) > 0:
            print(f"   First 10 unmatched patient_nums in observation: {list(unmatched_patients)[:10]}")

        print("\n4. Facility linkage via LOCATION_CD:")
        print(f"   Locations in visit_dimension: {len(visit_locations):,}")
        print(f"   Locations in facilities: {len(facility_locations):,}")
        overlap = visit_locations & facility_locations
        print(f"   Overlap: {len(overlap):,} ({len(overlap)/len(visit_locations)*100:.2f}%)")

        # Store the filtered dataframes for potential future use
        self.facilities_df = facilities_df
        self.demographics_df = demographics_df
        self.visit_df = visit_df

        return self

    def merge_data(self):
        print("\n" + "="*80)
        print("MERGING DATA")
        print("="*80)

        if self.df is None:
            print("Error: Observation data not loaded. Please run load_data() first.")
            return self

        if self.demographics_df is None or self.visit_df is None or self.facilities_df is None:
            print("Error: Additional files not loaded. Please run analyze_additional_files() first.")
            return self

        # Start with observation data
        merged = self.df.copy()
        merged['comment'] = ''

        print("\n[1/3] Merging with demographics...")
        print(f"Before merge: {merged.shape}")

        # Merge with demographics (100% overlap expected)
        demo_cols = ['PATIENT_NUM', 'BIRTH_DATE', 'RACE_CD', 'GENDER_CD']
        demo_subset = self.demographics_df[demo_cols].copy()

        merged = merged.merge(
            demo_subset,
            left_on='patient_id',
            right_on='PATIENT_NUM',
            how='left',
            suffixes=('', '_demo')
        )
        merged.drop('PATIENT_NUM', axis=1, inplace=True)

        # Track missing demographics
        demo_missing = merged[['BIRTH_DATE', 'RACE_CD', 'GENDER_CD']].isna().any(axis=1)
        merged.loc[demo_missing, 'comment'] = merged.loc[demo_missing, 'comment'] + 'demographics_missing;'

        print(f"After merge: {merged.shape}")
        print(f"Missing demographics: {demo_missing.sum()} rows")

        print("\n[2/3] Merging with visit_dimension (most frequent LOCATION_CD)...")
        print(f"Before merge: {merged.shape}")

        # Find most frequent LOCATION_CD for each PATIENT_NUM
        visit_location = self.visit_df.groupby('PATIENT_NUM')['LOCATION_CD'].agg(
            lambda x: x.mode()[0] if not x.mode().empty else np.nan
        ).reset_index()
        visit_location.columns = ['PATIENT_NUM', 'LOCATION_CD']

        merged = merged.merge(
            visit_location,
            left_on='patient_id',
            right_on='PATIENT_NUM',
            how='left',
            suffixes=('', '_visit')
        )
        merged.drop('PATIENT_NUM', axis=1, inplace=True)

        # Track missing location
        location_missing = merged['LOCATION_CD'].isna()
        merged.loc[location_missing, 'comment'] = merged.loc[location_missing, 'comment'] + 'location_missing_from_visit;'

        print(f"After merge: {merged.shape}")
        print(f"Missing LOCATION_CD: {location_missing.sum()} rows")

        print("\n[3/3] Merging with facilities...")
        print(f"Before merge: {merged.shape}")

        # Merge with facilities
        facility_cols = ['LOCATION_CD', 'FACILITY_TYPE', 'HEALTHSYSTEMID', 'STATE_ABBR']
        facility_subset = self.facilities_df[facility_cols].copy()

        merged = merged.merge(
            facility_subset,
            on='LOCATION_CD',
            how='left',
            suffixes=('', '_facility')
        )

        # Track missing facility info
        facility_missing = merged[['FACILITY_TYPE', 'HEALTHSYSTEMID', 'STATE_ABBR']].isna().any(axis=1)
        # Only mark as facility_missing if LOCATION_CD is not missing
        facility_missing_with_location = facility_missing & ~merged['LOCATION_CD'].isna()
        merged.loc[facility_missing_with_location, 'comment'] = merged.loc[facility_missing_with_location, 'comment'] + 'facility_missing;'

        # For rows where LOCATION_CD was already missing, the facility info should also be missing
        merged.loc[location_missing & facility_missing, 'comment'] = merged.loc[location_missing & facility_missing, 'comment'].str.replace('facility_missing;', '')

        print(f"After merge: {merged.shape}")
        print(f"Missing facility info (with valid LOCATION_CD): {facility_missing_with_location.sum()} rows")

        # Clean up comment column (remove trailing semicolons)
        merged['comment'] = merged['comment'].str.rstrip(';')

        self.merged_df = merged

        print("\n" + "="*80)
        print("MERGED DATA SUMMARY")
        print("="*80)
        print(f"\nFinal merged data shape: {self.merged_df.shape}")
        print(f"\nComment value counts:")
        print(self.merged_df['comment'].value_counts())

        print(f"\nMissing value summary for merged columns:")
        merged_cols = ['BIRTH_DATE', 'RACE_CD', 'GENDER_CD', 'LOCATION_CD', 'FACILITY_TYPE', 'HEALTHSYSTEMID', 'STATE_ABBR']
        for col in merged_cols:
            if col in self.merged_df.columns:
                missing_count = self.merged_df[col].isna().sum()
                missing_pct = (missing_count / len(self.merged_df)) * 100
                print(f"  {col}: {missing_count:,} ({missing_pct:.2f}%)")

        return self

    def run_full_analysis(self, analyze_additional=False, base_path=None, merge=False):
        print("\n" + "="*80)
        print("STARTING COMPREHENSIVE OCHIN DATA ANALYSIS")
        print("="*80)

        self.load_data()
        self.basic_info()
        self.patient_analysis()
        self.admission_analysis()
        self.concept_analysis()
        self.date_validation()

        if analyze_additional and base_path:
            self.analyze_additional_files(base_path)

            if merge:
                self.merge_data()

        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)

        return self


def main():
    base_path = '/home/CAMPUS/phu9/data/OADA/data/OCHIN/raw/'
    data_path = base_path + 'patient_observation_subset500000.csv'
    icd_map_path = '../data/mimic4/raw/icd10-icd9.csv'

    if not os.path.exists(icd_map_path):
        icd_map_path = None

    analyzer = OCHINDataAnalyzer(data_path, icd_map_path)
    analyzer.run_full_analysis(analyze_additional=True, base_path=base_path, merge=True)


if __name__ == "__main__":
    main()
