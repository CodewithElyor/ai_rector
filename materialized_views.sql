cat > materialized_views.sql <<'SQL'
-- Student success
DROP MATERIALIZED VIEW IF EXISTS mv_student_success;
CREATE MATERIALIZED VIEW mv_student_success AS
SELECT * FROM vw_student_success
WITH NO DATA;
CREATE UNIQUE INDEX IF NOT EXISTS ux_mv_ss ON mv_student_success(faculty, term);

-- Teacher performance
DROP MATERIALIZED VIEW IF EXISTS mv_teacher_perf;
CREATE MATERIALIZED VIEW mv_teacher_perf AS
SELECT * FROM vw_teacher_perf
WITH NO DATA;
CREATE INDEX IF NOT EXISTS ix_mv_tp_fac_term ON mv_teacher_perf(faculty, term);

-- Finance
DROP MATERIALIZED VIEW IF EXISTS mv_fin_summary;
CREATE MATERIALIZED VIEW mv_fin_summary AS
SELECT * FROM vw_fin_summary
WITH NO DATA;
CREATE INDEX IF NOT EXISTS ix_mv_fin_month_fac ON mv_fin_summary(month, faculty);

-- Dastlabki to'ldirish
REFRESH MATERIALIZED VIEW CONCURRENTLY mv_student_success;
REFRESH MATERIALIZED VIEW CONCURRENTLY mv_teacher_perf;
REFRESH MATERIALIZED VIEW CONCURRENTLY mv_fin_summary;
SQL