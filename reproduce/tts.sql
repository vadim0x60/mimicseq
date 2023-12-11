SELECT hadm_id, CASE WHEN ROW_NUMBER() OVER (ORDER BY priority) < 10000 THEN 'test' ELSE 'train' END AS fold FROM (
  SELECT
    hadm_id,
    ROW_NUMBER() OVER (PARTITION BY admission_type, admission_location, discharge_location, ethnicity, anchor_age, gender ORDER BY RAND()) AS priority
  FROM `physionet-data.mimic_core.admissions` AS a LEFT JOIN `physionet-data.mimic_core.patients` AS p ON a.subject_id = p.subject_id
)