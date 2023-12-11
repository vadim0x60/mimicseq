CREATE TABLE `graphsim.mimic.patients` AS (
  SELECT ROW_NUMBER() OVER (PARTITION BY fold ORDER BY len) AS sample_id, hadm_id, len, fold FROM (
    SELECT e.hadm_id, COUNT(e.event_id) AS len, f.fold from `graphsim.mimic.events` AS e JOIN `graphsim.mimic.folds` AS f ON e.hadm_id = f.hadm_id GROUP BY hadm_id, fold
  )
)