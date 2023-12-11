CREATE TABLE `graphsim.mimic.mimic` AS (
  WITH prolonged_events AS (
    SELECT hadm_id, starttime, endtime, label, intensity FROM (
      SELECT hadm_id, itemid, starttime, endtime, amount AS intensity FROM `physionet-data.mimic_icu.inputevents` UNION ALL  
      SELECT hadm_id, itemid, starttime, endtime, value AS intensity FROM `physionet-data.mimic_icu.procedureevents`
      ) AS prolonged_icu_events 
    JOIN `physionet-data.mimic_icu.d_items` AS d_icuitems ON prolonged_icu_events.itemid = d_icuitems.itemid
    UNION ALL
    SELECT hadm_id, starttime, stoptime AS endtime, CONCAT(drug, ' ', prod_strength, ' prescription, ', route) AS label, doses_per_24_hrs AS intensity
    FROM `physionet-data.mimic_hosp.prescriptions`
  ), patientprofiles AS (
    SELECT hadm_id, admittime, ethnicity, gender, anchor_age
    FROM `physionet-data.mimic_core.admissions` AS adm JOIN `physionet-data.mimic_core.patients` AS pat
    ON adm.subject_id = pat.subject_id
  )
  SELECT hadm_id, eventtime, label, intensity FROM (
    SELECT hadm_id, admittime AS eventtime, CONCAT(admission_type, ' ADMISSION') AS label, NULL as intensity FROM `physionet-data.mimic_core.admissions` 
    UNION ALL
    SELECT hadm_id, dischtime AS eventtime, CONCAT('DISCHARGE TO ', discharge_location) AS label, NULL as intensity FROM `physionet-data.mimic_core.admissions` 
    UNION ALL
    SELECT hadm_id, deathtime AS eventtime, 'DEATH' AS label, NULL as intensity FROM `physionet-data.mimic_core.admissions` 
    UNION ALL
    SELECT hadm_id, starttime AS eventtime, CONCAT('Start ', label) AS label, intensity FROM prolonged_events
    UNION ALL
    SELECT hadm_id, endtime AS eventtime, CONCAT('Stop ', label) AS label, intensity FROM prolonged_events
    UNION ALL
    SELECT hadm_id, admittime as eventtime, ethnicity AS label, NULL as intensity FROM patientprofiles
    UNION ALL
    SELECT hadm_id, admittime as eventtime, 'MALE' AS label, NULL as intensity FROM patientprofiles WHERE gender = 'M'
    UNION ALL
    SELECT hadm_id, admittime as eventtime, 'FEMALE' AS label, NULL as intensity FROM patientprofiles WHERE gender = 'F'
    UNION ALL
    SELECT hadm_id, admittime as eventtime, 'AGE' AS label, anchor_age as intensity FROM patientprofiles
    UNION ALL
    SELECT hadm_id, eventtime, label, intensity FROM (
      SELECT hadm_id, itemid, charttime AS eventtime, valuenum AS intensity FROM `physionet-data.mimic_icu.chartevents` UNION ALL 
      SELECT hadm_id, itemid, charttime AS eventtime, value AS intensity FROM `physionet-data.mimic_icu.outputevents`
      ) AS icu_events 
    JOIN `physionet-data.mimic_icu.d_items` AS d_icuitems ON icu_events.itemid = d_icuitems.itemid
    UNION ALL
    SELECT hadm_id, charttime AS eventtime, label, valuenum AS intensity
    FROM `physionet-data.mimic_hosp.labevents` AS labevents JOIN `physionet-data.mimic_hosp.d_labitems` AS d_labitems 
    ON labevents.itemid = d_labitems.itemid
    UNION ALL
    SELECT hadm_id, charttime AS eventtime, CONCAT(ab_name, ' resistance') AS label, IFNULL(dilution_value, 256) as intensity 
    FROM `physionet-data.mimic_hosp.microbiologyevents` 
    UNION ALL
    SELECT hadm_id, chartdate as eventtime, p_icd.long_title as label, null as intensity 
    FROM `physionet-data.mimic_hosp.procedures_icd` AS procedures JOIN `physionet-data.mimic_hosp.d_icd_procedures` AS p_icd 
    ON procedures.icd_code = p_icd.icd_code
    UNION ALL
    SELECT hadm_id, chartdate AS eventtime, d_hcpcs.short_description AS label, null AS intensity
    FROM `physionet-data.mimic_hosp.hcpcsevents` AS hcpcevents JOIN `physionet-data.mimic_hosp.d_hcpcs` AS d_hcpcs
    ON hcpcevents.hcpcs_cd = d_hcpcs.code
    UNION ALL
    SELECT hadm_id, charttime AS eventtime, CONCAT(medication, ' ', event_txt) AS label, NULL as intensity 
    FROM `physionet-data.mimic_hosp.emar`
  )
  WHERE hadm_id IS NOT NULL AND eventtime IS NOT NULL AND label IS NOT NULL
)