/*
Extract 55 hours of notes of control case icustays (48 hours before onset and 7 hours after onset).
---------------------------------------------------------------------------------------------------------------------
- AUTHOR (of this version): Lily Voge, June 2025
---------------------------------------------------------------------------------------------------------------------
*/

SET search_path TO mimiciii;

-- Drop existing view if it exists
DROP MATERIALIZED VIEW IF EXISTS control_55h_notes_ex1c CASCADE;

-- Create materialized view for control group notes
CREATE MATERIALIZED VIEW control_55h_notes_ex1c AS
SELECT
  ne.row_id AS note_id, 
  ie.icustay_id,
  ie.subject_id,
  ne.charttime AS chart_time,
  ne.text AS note_text,
  ne.category AS note_category,
  CASE 
    WHEN ne.charttime < ch.control_onset_time THEN 0 
    ELSE 1 
  END AS pseudo_target


FROM matched_controls_hourly ch
LEFT JOIN icustays ie ON ch.icustay_id = ie.icustay_id
LEFT JOIN noteevents ne 
  ON ne.subject_id = ie.subject_id 
  AND ne.hadm_id = ie.hadm_id
  AND ne.charttime BETWEEN (ch.control_onset_time - INTERVAL '48' HOUR) AND ch.control_onset_time 
  AND ne.category IN ('Nursing', 'Physician ')
  AND ne.text IS NOT NULL

-- Only include ICU stays with at least one valid note in the time window

WHERE EXISTS (
  SELECT 1
  FROM noteevents ne2
  WHERE ne2.subject_id = ie.subject_id
    AND ne2.hadm_id = ie.hadm_id
    AND ne2.charttime BETWEEN (ch.control_onset_time - INTERVAL '48' HOUR) AND ch.control_onset_time 
    AND ne2.category IN ('Nursing', 'Physician ')
    AND ne2.text IS NOT NULL
)
    
ORDER BY ie.icustay_id, ne.charttime;




DROP MATERIALIZED VIEW IF EXISTS control_55h_notes_binned CASCADE;

CREATE MATERIALIZED VIEW control_55h_notes_binned AS
SELECT
  icustay_id,
  subject_id,
  chart_time,
  date_trunc('hour', chart_time) AS chart_hour,
  pseudo_target,
  note_id,
  note_text,
  note_category
FROM control_55h_notes_ex1c
ORDER BY icustay_id, subject_id, chart_time;
