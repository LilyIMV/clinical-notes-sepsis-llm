/*
Extract 55 hours of notes of sepsis case icustays (48 hours before onset and 7 hours after onset).
---------------------------------------------------------------------------------------------------------------------
- AUTHOR (of this version): Lily Voge, June 2025
---------------------------------------------------------------------------------------------------------------------
*/


-- This query pivots lab values 

SET search_path TO mimiciii;

-- Drop existing view if it exists
DROP MATERIALIZED VIEW IF EXISTS case_55h_notes_ex1c CASCADE;

-- Create materialized view with notes around sepsis onset
CREATE MATERIALIZED VIEW case_55h_notes_ex1c AS
SELECT
  ne.row_id AS note_id, 
  ie.icustay_id,
  ie.subject_id,
  ne.charttime AS chart_time,
  ne.text AS note_text,
  ne.category AS note_category, 
  CASE 
    WHEN ne.charttime < ch.sepsis_onset THEN 0 
    ELSE 1 
  END AS sepsis_target


-- Only include range from 
FROM cases_hourly_ex1c ch
LEFT JOIN icustays ie ON ch.icustay_id = ie.icustay_id
LEFT JOIN noteevents ne 
  ON ne.subject_id = ie.subject_id 
  AND ne.hadm_id = ie.hadm_id
  AND ne.charttime BETWEEN (ch.sepsis_onset - INTERVAL '48' HOUR) AND ch.sepsis_onset
  AND ne.category IN ('Nursing', 'Physician ')
  AND ne.text IS NOT NULL

-- Only include ICU stays with at least one valid note in the time window
WHERE EXISTS (
  SELECT 1
  FROM noteevents ne2
  WHERE ne2.subject_id = ie.subject_id
    AND ne2.hadm_id = ie.hadm_id
    AND ne2.charttime BETWEEN (ch.sepsis_onset - INTERVAL '48' HOUR) AND ch.sepsis_onset
    AND ne2.category IN ('Nursing', 'Physician ')
    AND ne2.text IS NOT NULL
)

ORDER BY ie.icustay_id, ne.charttime;



DROP MATERIALIZED VIEW IF EXISTS case_55h_notes_binned CASCADE;

CREATE MATERIALIZED VIEW case_55h_notes_binned AS
SELECT
  icustay_id,
  subject_id,
  chart_time,
  date_trunc('hour', chart_time) AS chart_hour,
  sepsis_target,
  note_id,
  note_text,
  note_category
FROM case_55h_notes_ex1c
ORDER BY icustay_id, subject_id, chart_time;

