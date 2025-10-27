# OADA Project

## Patient Cohort Binarization Rules

This document describes how patient demographic and spatiotemporal features are converted to binary values in the `identify_patient_cohorts()` function.

### Binarization Rules Table

| Feature | True Condition | False Condition | Applicable Dataset | Notes |
|---------|---------------|-----------------|-------------------|-------|
| **age** | > 60 | d 60 or NULL | All | Age in years |
| **gender** | M or Male | Female or NULL | All | Case-insensitive |
| **ethnicity** | Contains "WHITE" or "Caucasian" | Other values or NULL | All | Partial match, case-insensitive |
| **insurance** | Contains "Medicare" | Other values or NULL | MIMIC-III, MIMIC-IV | Partial match, case-insensitive |
| **language** | Contains "ENGL" or equals "English" | Other values or NULL | MIMIC-III, MIMIC-IV | Partial match, case-insensitive |
| **marital_status** | Any non-"SINGLE" value | "SINGLE" or NULL/empty | MIMIC-III, MIMIC-IV | Case-insensitive |
| **year** | e 2017 | < 2017 or NULL | MIMIC-IV | Temporal feature |
| **region** | Any non-"West" value | "West" or NULL/empty | eICU | Spatial feature |


