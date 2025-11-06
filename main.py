#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import hashlib
import logging
import os
import sys
import time
from pathlib import Path

import psycopg2

# ---------- .env (ishonchli yuklash) ----------
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(dotenv_path=find_dotenv(usecwd=True), override=False)
except Exception:
    pass

DEFAULT_SCHEMA = "ai_rektor"
STAGING_SCHEMA = "ai_rektor_stg"

TABLES = {
    "students": {
        "cols": [
            ("student_id", "BIGINT"),
            ("faculty", "TEXT"),
            ("admitted_year", "INTEGER"),
            ("status", "TEXT"),
        ],
        "conflict": ["student_id"],
        "csv": "students.csv",
    },
    "teachers": {
        "cols": [
            ("teacher_id", "TEXT"),
            ("teacher_name", "TEXT"),
            ("dept", "TEXT"),
        ],
        "conflict": ["teacher_id"],
        "csv": "teachers.csv",
    },
    "enrollments": {
        "cols": [
            ("student_id", "BIGINT"),
            ("course_id", "TEXT"),
            ("course_name", "TEXT"),
            ("teacher_id", "TEXT"),
            ("faculty", "TEXT"),
            ("term", "TEXT"),
            ("grade", "NUMERIC(5,2)"),
            ("attendance", "NUMERIC(4,2)"),
        ],
        "conflict": ["student_id", "course_id", "term"],
        "csv": "enrollments.csv",
    },
    "finance": {
        "cols": [
            ("date", "DATE"),
            ("faculty", "TEXT"),
            ("revenue", "INTEGER"),
            ("expense", "INTEGER"),
        ],
        "conflict": ["date", "faculty"],
        "csv": "finance.csv",
    },
}

# ---------- fallback schema.sql mazmuni ----------
FALLBACK_SCHEMA_SQL = f"""
CREATE SCHEMA IF NOT EXISTS {DEFAULT_SCHEMA};
CREATE SCHEMA IF NOT EXISTS {STAGING_SCHEMA};
SET search_path TO {DEFAULT_SCHEMA}, public;

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
CREATE INDEX IF NOT EXISTS idx_enroll_student ON enrollments(student_id);

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
"""

# ---------- helperlar ----------
def read_sql_file_if_exists(path: Path) -> str | None:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return None

def ensure_schema(cur: psycopg2.extensions.cursor, schema_sql_path: Path):
    # Avval search_path — keyingi CREATE’lar to‘g‘ri sxemaga ketsin
    cur.execute(f"CREATE SCHEMA IF NOT EXISTS {DEFAULT_SCHEMA};")
    cur.execute(f"CREATE SCHEMA IF NOT EXISTS {STAGING_SCHEMA};")
    cur.execute(f"SET search_path TO {DEFAULT_SCHEMA}, public;")

    sql = read_sql_file_if_exists(schema_sql_path)
    if sql:
        cur.execute(sql)
    else:
        cur.execute(FALLBACK_SCHEMA_SQL)

def detect_delimiter(csv_path: Path) -> str:
    """CSV delimiter auto-detect (fallback ',')."""
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            sample = f.read(8192)
        if not sample:
            return ","
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;|\t")
            return dialect.delimiter
        except Exception:
            header = sample.splitlines()[0] if sample.splitlines() else ""
            counts = {",": header.count(","), ";": header.count(";"), "|": header.count("|"), "\t": header.count("\t")}
            return max(counts, key=counts.get) if counts else ","
    except Exception:
        return ","

def make_staging_sql(table: str, cols):
    cols_sql = ", ".join([f"{c} {t}" for c, t in cols])
    return f"""
DROP TABLE IF EXISTS {STAGING_SCHEMA}.{table};
CREATE TABLE {STAGING_SCHEMA}.{table} ({cols_sql});
"""

def copy_into_staging(cur, table: str, csv_path: Path, delimiter: str | None):
    delim = delimiter or detect_delimiter(csv_path)
    cur.execute("SET client_encoding = 'UTF8';")
    sql = f"COPY {STAGING_SCHEMA}.{table} FROM STDIN WITH (FORMAT CSV, HEADER TRUE, DELIMITER '{delim}');"
    with open(csv_path, "r", encoding="utf-8") as f:
        cur.copy_expert(sql, f)

def upsert_sql(table: str, cols, conflict_cols):
    all_cols = [c for c, _ in cols]
    insert_cols = ", ".join(all_cols)
    set_cols = [c for c in all_cols if c not in conflict_cols]

    if not set_cols:
        # Hamma ustun conflict bo‘lsa — DO NOTHING
        return f"""
INSERT INTO {DEFAULT_SCHEMA}.{table} ({insert_cols})
SELECT {insert_cols} FROM {STAGING_SCHEMA}.{table}
ON CONFLICT ({", ".join(conflict_cols)}) DO NOTHING;
"""
    set_clause = ", ".join([f"{c}=EXCLUDED.{c}" for c in set_cols])
    return f"""
INSERT INTO {DEFAULT_SCHEMA}.{table} ({insert_cols})
SELECT {insert_cols} FROM {STAGING_SCHEMA}.{table}
ON CONFLICT ({", ".join(conflict_cols)}) DO UPDATE SET {set_clause};
"""

def ensure_materialized_views(cur):
    """MV’lar yo‘q bo‘lsa — WITH NO DATA bilan yaratamiz (birinchi marta xatosiz)."""
    cur.execute(f"SET search_path TO {DEFAULT_SCHEMA}, public;")
    cur.execute(
        """
        DO $$
        BEGIN
          IF NOT EXISTS (
            SELECT 1 FROM pg_matviews
            WHERE schemaname = current_schema() AND matviewname = 'mv_student_success'
          ) THEN
            EXECUTE 'CREATE MATERIALIZED VIEW mv_student_success AS SELECT * FROM vw_student_success WITH NO DATA';
          END IF;

          IF NOT EXISTS (
            SELECT 1 FROM pg_matviews
            WHERE schemaname = current_schema() AND matviewname = 'mv_teacher_perf'
          ) THEN
            EXECUTE 'CREATE MATERIALIZED VIEW mv_teacher_perf AS SELECT * FROM vw_teacher_perf WITH NO DATA';
          END IF;

          IF NOT EXISTS (
            SELECT 1 FROM pg_matviews
            WHERE schemaname = current_schema() AND matviewname = 'mv_fin_summary'
          ) THEN
            EXECUTE 'CREATE MATERIALIZED VIEW mv_fin_summary AS SELECT * FROM vw_fin_summary WITH NO DATA';
          END IF;
        END$$;
        """
    )

def refresh_materialized_views(cur):
    """MV refresh — avval CONCURRENTLY, bo‘lmasa oddiy REFRESH."""
    ensure_materialized_views(cur)
    for name in ["mv_student_success", "mv_teacher_perf", "mv_fin_summary"]:
        try:
            cur.execute(f"REFRESH MATERIALIZED VIEW CONCURRENTLY {name};")
        except Exception:
            try:
                cur.execute(f"REFRESH MATERIALIZED VIEW {name};")
            except Exception:
                pass

def run_refresh(dsn: str, data_dir: Path, schema_sql_path: Path, delimiter: str | None, refresh_mv: bool):
    # CSV mavjudmi?
    for t, cfg in TABLES.items():
        p = data_dir / cfg["csv"]
        if not p.exists():
            raise FileNotFoundError(f"CSV topilmadi: {p}")

    conn = psycopg2.connect(dsn)
    conn.autocommit = False
    try:
        with conn.cursor() as cur:
            # Sessiya boshida search_path
            cur.execute(f"SET search_path TO {DEFAULT_SCHEMA}, public;")
            logging.info("search_path = %s, public", DEFAULT_SCHEMA)

            logging.info("Ensure schema/tables/views...")
            ensure_schema(cur, schema_sql_path)

            # FK tartibi: avval parents
            for tname in ["students", "teachers", "enrollments", "finance"]:
                cfg = TABLES[tname]
                cols = cfg["cols"]
                csv_path = data_dir / cfg["csv"]

                logging.info("STAGING: %s", tname)
                cur.execute(make_staging_sql(tname, cols))

                logging.info("COPY CSV -> STAGING: %s", csv_path)
                copy_into_staging(cur, tname, csv_path, delimiter)

                cur.execute(f"SELECT COUNT(*) FROM {STAGING_SCHEMA}.{tname}")
                stg_cnt = cur.fetchone()[0]
                logging.info("STAGING COUNT %s = %s", tname, stg_cnt)

                # Target jadvali bo‘lmasa ham yaratiladi (fallback)
                col_def = ", ".join([f"{c} {t}" for c, t in cols])
                cur.execute(f"CREATE TABLE IF NOT EXISTS {DEFAULT_SCHEMA}.{tname} ({col_def});")

                cur.execute(f"SELECT COUNT(*) FROM {DEFAULT_SCHEMA}.{tname}")
                before = cur.fetchone()[0]
                cur.execute(upsert_sql(tname, cols, cfg["conflict"]))
                cur.execute(f"SELECT COUNT(*) FROM {DEFAULT_SCHEMA}.{tname}")
                after = cur.fetchone()[0]
                logging.info("TARGET COUNT %s: %s -> %s", tname, before, after)

            if refresh_mv:
                logging.info("Materialized view'larni yangilash...")
                refresh_materialized_views(cur)

            conn.commit()
            logging.info("✅ Refresh OK")
    except Exception:
        conn.rollback()
        logging.exception("❌ Xato! Transaction rollback qilindi.")
        raise
    finally:
        conn.close()

def hash_dir(data_dir: Path) -> str:
    """CSV fayl hash — o‘zgarishni aniqlash."""
    h = hashlib.sha256()
    for name in sorted([cfg["csv"] for cfg in TABLES.values()]):
        p = data_dir / name
        if not p.exists():
            continue
        h.update(p.name.encode())
        with open(p, "rb") as f:
            while True:
                chunk = f.read(1 << 20)
                if not chunk:
                    break
                h.update(chunk)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser(description="AI-Rektor CSV → Postgres ETL (refresh)")
    ap.add_argument("--dsn", default=os.getenv("DSN"),
                    help='Postgres DSN, masalan: postgresql://postgres:7778@localhost:5432/Start_Up')
    ap.add_argument("--data-dir", default=os.getenv("DATA_DIR", "./data"))
    ap.add_argument("--schema-file", default="schema.sql", help="schema.sql yo‘li (ixtiyoriy)")
    ap.add_argument("--delimiter", default=None, help="Majburiy CSV delimiter, masalan ';' (bo‘sh: auto-detect)")
    ap.add_argument("--once", action="store_true", help="Bir marta refresh")
    ap.add_argument("--interval", type=int, default=int(os.getenv("REFRESH_INTERVAL_MIN", "60")),
                    help="Daqiqadagi interval (watch yo‘q bo‘lsa)")
    ap.add_argument("--watch", action="store_true", help="CSV o‘zgarsa avtomatik refresh (hash bilan)")
    ap.add_argument("--refresh-mv", action="store_true", help="Materialized view’larni yangilashga urinish")
    ap.add_argument("--log-level", default="INFO", help="INFO/DEBUG/WARNING/ERROR")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(message)s")

    if not args.dsn:
        print("DSN talab qilinadi. --dsn yoki .env DSN ni bering", file=sys.stderr)
        sys.exit(2)

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Data papkasi topilmadi: {data_dir}", file=sys.stderr)
        sys.exit(2)

    schema_sql_path = Path(args.schema_file)

    if args.once:
        run_refresh(args.dsn, data_dir, schema_sql_path, args.delimiter, args.refresh_mv)
        return

    if args.watch:
        logging.info("Watch rejimi: CSV hash o‘zgarsa refresh (interval=%d daqiqa)", args.interval)
        last_hash = None
        try:
            while True:
                try:
                    cur_hash = hash_dir(data_dir)
                    if cur_hash != last_hash:
                        logging.info("O‘zgarish topildi → refresh boshlanadi")
                        run_refresh(args.dsn, data_dir, schema_sql_path, args.delimiter, args.refresh_mv)
                        last_hash = cur_hash
                    else:
                        logging.debug("O‘zgarish yo‘q")
                except Exception as e:
                    logging.error("Refreshda xato: %s", e)
                time.sleep(args.interval * 60)
        except KeyboardInterrupt:
            logging.info("⏹️ Watch to‘xtatildi (Ctrl+C)")
    else:
        logging.info("Interval rejimi: har %d daqiqada refresh", args.interval)
        try:
            while True:
                try:
                    run_refresh(args.dsn, data_dir, schema_sql_path, args.delimiter, args.refresh_mv)
                except Exception as e:
                    logging.error("Refreshda xato: %s", e)
                time.sleep(args.interval * 60)
        except KeyboardInterrupt:
            logging.info("⏹️ Interval to‘xtatildi (Ctrl+C)")

if __name__ == "__main__":
    main()