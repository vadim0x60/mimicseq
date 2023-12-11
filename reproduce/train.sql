CREATE TABLE `graphsim.mimic.train` AS (
  SELECT sample_id, eventtime, event_id, intensity, intensity_norm FROM `graphsim.mimic.mimicseq` AS m JOIN `graphsim.mimic.patients` AS p ON m.hadm_id = p.hadm_id WHERE p.fold = 'train' ORDER BY sample_id, eventtime
)