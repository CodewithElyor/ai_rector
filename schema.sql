CREATE SCHEMA IF NOT EXISTS ai_rektor;
CREATE SCHEMA IF NOT EXISTS ai_rektor_stg;
SET search_path TO ai_rektor, public;

CREATE TABLE IF NOT EXISTS students (
  student_id    BIGINT PRIMARY KEY,
  faculty       TEXT NOT NULL,
  admitted_year INTEGER CHECK (admitted_year BETWEEN 2000 AND 2100),
  status        TEXT CHECK (status IN ('active','academic_leave','expelled'))
);

CREATE TABLE IF NOT EXISTS teachers (
  teacher_id   TEXT PRIMARY KEY,
  teacher_name TEXT NOT NULL,
  dept         TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS enrollments (
  enrollment_id BIGSERIAL PRIMARY KEY,
  student_id    BIGINT REFERENCES students(student_id),
  course_id     TEXT NOT NULL,
  course_name   TEXT NOT NULL,
  teacher_id    TEXT REFERENCES teachers(teacher_id),
  faculty       TEXT NOT NULL,
  term          TEXT NOT NULL,
  grade         NUMERIC(5,2) CHECK (grade BETWEEN 0 AND 100),
  attendance    NUMERIC(4,2) CHECK (attendance BETWEEN 0 AND 1)
);

CREATE UNIQUE INDEX IF NOT EXISTS ux_enroll_unique
  ON enrollments (student_id, course_id, term);

CREATE INDEX IF NOT EXISTS idx_enroll_term    ON enrollments(term);
CREATE INDEX IF NOT EXISTS idx_enroll_faculty ON enrollments(faculty);

CREATE TABLE IF NOT EXISTS finance (
  date     DATE NOT NULL,
  faculty  TEXT NOT NULL,
  revenue  INTEGER NOT NULL,
  expense  INTEGER NOT NULL,
  PRIMARY KEY (date, faculty)
);

CREATE OR REPLACE VIEW vw_student_success AS
SELECT
  e.faculty,
  e.term,
  COUNT(DISTINCT e.student_id) AS students,
  ROUND(AVG((e.grade/25.0)),3) AS avg_gpa,
  ROUND(AVG(CASE WHEN e.grade >= 60 THEN 1 ELSE 0 END),3) AS pass_rate,
  ROUND(AVG(e.attendance),3) AS attendance_avg
FROM enrollments e
GROUP BY 1,2;

CREATE OR REPLACE VIEW vw_fin_summary AS
SELECT
  date_trunc('month', f.date)::date AS month,
  f.faculty,
  SUM(f.revenue) AS revenue,
  SUM(f.expense) AS expense,
  SUM(f.revenue - f.expense) AS net
FROM finance f
GROUP BY 1,2
ORDER BY 1,2;

CREATE OR REPLACE VIEW vw_teacher_perf AS
SELECT
  e.teacher_id,
  COALESCE(t.teacher_name, e.teacher_id) AS teacher_name,
  e.faculty,
  e.term,
  ROUND(AVG(CASE WHEN e.grade >= 60 THEN 1 ELSE 0 END),3) AS pass_rate,
  ROUND(AVG(e.grade),2) AS avg_grade,
  ROUND(AVG(e.attendance),3) AS attendance,
  COUNT(*) AS n
FROM enrollments e
LEFT JOIN teachers t USING (teacher_id)
GROUP BY 1,2,3,4;