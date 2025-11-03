from collections import OrderedDict


def identify_patient_cohorts(patient_info):
    patient_cohorts = OrderedDict()

    for pid, info in patient_info.items():
        cohort = {}

        age = info.get('age')
        cohort['age'] = age > 60 if age is not None else False

        gender = info.get('gender')
        if gender is not None:
            gender_upper = str(gender).upper()
            cohort['gender'] = gender_upper in ['M', 'MALE']
        else:
            cohort['gender'] = False

        ethnicity = info.get('ethnicity')
        if ethnicity is not None:
            ethnicity_upper = str(ethnicity).upper()
            cohort['ethnicity'] = 'WHITE' in ethnicity_upper or 'CAUCASIAN' in ethnicity_upper
        else:
            cohort['ethnicity'] = False

        insurance = info.get('insurance')
        if insurance is not None:
            insurance_upper = str(insurance).upper()
            cohort['insurance'] = 'MEDICARE' in insurance_upper
        else:
            cohort['insurance'] = False

        language = info.get('language')
        if language is not None:
            language_upper = str(language).upper()
            cohort['language'] = 'ENGL' in language_upper or language_upper == 'ENGLISH'
        else:
            cohort['language'] = False

        marital_status = info.get('marital_status')
        if marital_status is not None and str(marital_status).strip() != '':
            marital_upper = str(marital_status).upper()
            cohort['marital_status'] = 'SINGLE' not in marital_upper
        else:
            cohort['marital_status'] = False

        year = info.get('year')
        cohort['year'] = year <= 2017 if year is not None else False

        region = info.get('region')
        if region is None:
            cohort['region'] = False
        else:
            region_str = str(region).strip()
            if region_str == '':
                cohort['region'] = True
            else:
                region_upper = region_str.upper()
                cohort['region'] = region_upper != 'WEST'

        patient_cohorts[pid] = cohort

    return patient_cohorts
