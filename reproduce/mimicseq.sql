CREATE TABLE `graphsim.mimic.mimicseq` AS (
  SELECT hadm_id, eventtime, e.event_id, intensity, COALESCE((intensity - intensity_mean) / (intensity_std + 1), 0) AS intensity_norm FROM `graphsim.mimic.events` AS e JOIN `graphsim.mimic.eventtypes` AS et ON e.event_id = et.event_id
)