# app.py — AI-Rektor Dashboard (modern dizayn, gradient KPI, dark analytics ranglar)
import os, sys, subprocess
from typing import Dict, Any, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv

# =========================
# 1) Config & Theme
# =========================
load_dotenv()
DB_DSN    = os.getenv("DB_DSN", "postgresql://postgres:7778@localhost:5432/Start_Up")
DB_SCHEMA = os.getenv("DB_SCHEMA", "ai_rektor")
CACHE_TTL = int(os.getenv("CACHE_TTL_SEC", "300"))  # 5 min

# MV bor bo‘lsa undan, bo‘lmasa VIEW’dan foydalanamiz
USE_MV    = True
VIEW_SS = "mv_student_success" if USE_MV else "vw_student_success"
VIEW_TP = "mv_teacher_perf"    if USE_MV else "vw_teacher_perf"
VIEW_FN = "mv_fin_summary"     if USE_MV else "vw_fin_summary"

st.set_page_config(page_title="AI-Rektor Dashboard", layout="wide", initial_sidebar_state="expanded")

# Plotly umumiy ko‘rinishi (dark analytics palitra)
px.defaults.template = "plotly_dark"
px.defaults.color_discrete_sequence = [
    "#10b981", "#3b82f6", "#f59e0b", "#ef4444", "#8b5cf6", "#14b8a6", "#22c55e"
]

# Global CSS — gradient sidebar, fontlar, KPI effekt
st.markdown("""
<style>
/* Sidebar gradient */
[data-testid="stSidebar"] {
  background: linear-gradient(160deg, #1e293b 0%, #0f172a 100%) !important;
  color: #e2e8f0 !important;
  border-right: 1px solid rgba(255,255,255,0.06);
}
[data-testid="stHeader"] { background: none; }

/* Body font */
html, body, [class*="css"] { font-family: "Inter", system-ui, -apple-system, Segoe UI, Roboto, sans-serif !important; }
h1, h2, h3, h4 { font-family: "Poppins", Inter, sans-serif !important; }

/* Dataframe radius */
.block-container { padding-top: 1.5rem; }
div[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

/* Tabs underline accent */
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] { background: #0b1220; border-radius: 10px; }
.stTabs [aria-selected="true"] { background: #111827; border: 1px solid rgba(255,255,255,0.06); }

/* Subtle cards spacing fix on mobile */
@media (max-width: 900px) { .st-emotion-cache-ocqkz7 { row-gap: 10px !important; } }
</style>
""", unsafe_allow_html=True)

st.title("🎓 AI-Rektor Dashboard (boyitilgan)")

# Neon/Cloud DSN ogohlantirish (agar kimdir local docker host 'db' ni qo‘yib yuborsa)
if "://ai_ro" in DB_DSN and "@db:" in DB_DSN:
    st.warning("DSN ichida `@db:` bor — bu Docker compose ichida ishlaydi. Cloud yoki lokal uchun Neon DSN dan foydalaning.", icon="⚠️")

# =========================
# 2) Connection (cached)
# =========================
@st.cache_resource(show_spinner=False)
def get_engine():
    return create_engine(
        DB_DSN,
        pool_size=5,
        max_overflow=5,
        pool_pre_ping=True,
        pool_recycle=1800,
        future=True,
    )

# SQL helpers
def _exec_sql(sql: str, params: Dict[str, Any] | None = None) -> pd.DataFrame:
    engine = get_engine()
    with engine.connect() as conn:
        conn.exec_driver_sql(f"SET search_path TO {DB_SCHEMA}, public;")
        return pd.read_sql(text(sql), conn, params=params)

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def run_sql_cached(sql: str, params: Dict[str, Any] | None = None) -> pd.DataFrame:
    return _exec_sql(sql, params)

def run_sql(sql: str, params: Dict[str, Any] | None = None) -> pd.DataFrame:
    return _exec_sql(sql, params)

def invalidate_cache():
    run_sql_cached.clear()

# =========================
# 3) UI helperlar
# =========================
def kpi_card(title: str, value_str: str, sub: str, color: str, icon: str):
    st.markdown(f"""
    <div style="
        background:linear-gradient(135deg, {color}cc 0%, #0f172a 90%);
        padding:20px;border-radius:16px;
        box-shadow:0 0 10px #00000040;
        border:1px solid rgba(255,255,255,0.08);
    ">
        <div style="font-size:13px;color:#cbd5e1;display:flex;align-items:center;gap:8px">
            <span style="font-size:16px;">{icon}</span> {title}
        </div>
        <div style="font-size:34px;font-weight:800;color:#f8fafc;margin:8px 0">{value_str}</div>
        <div style="font-size:13px;color:#94a3b8">{sub}</div>
    </div>
    """, unsafe_allow_html=True)

def download_buttons(df: pd.DataFrame, base_name: str):
    # CSV
    ccsv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ CSV", ccsv, f"{base_name}.csv", "text/csv", use_container_width=True)

    # Excel (openpyxl yoki XlsxWriter bilan)
    try:
        xlsx_path = f"{base_name}.xlsx"
        with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
            df.to_excel(writer, sheet_name="data", index=False)
        with open(xlsx_path, "rb") as f:
            st.download_button("⬇️ Excel", f, f"{base_name}.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               use_container_width=True)
    except Exception as e:
        st.info("Excel generator o‘rnatilmagan ko‘rinadi (openpyxl yoki XlsxWriter). CSV yuklab olishdan foydalaning.")
    finally:
        try: os.remove(xlsx_path)
        except Exception: pass

def where_clause(term: str, faculty: str) -> Tuple[str, Dict[str, Any]]:
    w, p = [], {}
    if term != "Barchasi":
        w.append("term = :term"); p["term"] = term
    if faculty != "Barchasi":
        w.append("faculty = :faculty"); p["faculty"] = faculty
    return ("WHERE " + " AND ".join(w)) if w else "", p

# =========================
# 4) Ma’lumotlarni olishga tayyorlanish
# =========================
# MV yo‘qligini avtomatik aniqlab, VIEWga fallback qilamiz
try:
    _ = run_sql_cached(f"SELECT 1 FROM {VIEW_SS} LIMIT 1;")
except SQLAlchemyError:
    USE_MV = False
    VIEW_SS = "vw_student_success"
    VIEW_TP = "vw_teacher_perf"
    VIEW_FN = "vw_fin_summary"
    st.info("Materialized viewlar topilmadi — vaqtincha VIEW‘larga o‘tildi.")

st.caption(f"Manba: {'MV' if USE_MV else 'VIEW'} • Schema: `{DB_SCHEMA}` • Cache TTL: {CACHE_TTL}s")

# =========================
# 5) Sidebar — Filtrlar
# =========================
st.sidebar.header("⚙️ Filtrlar")

try:
    terms_df = run_sql_cached(f"SELECT DISTINCT term FROM {VIEW_SS} ORDER BY term;")
except SQLAlchemyError as e:
    st.error(f"❌ DB ulanishda xato: {e}")
    st.stop()

term = st.sidebar.selectbox("Term", ["Barchasi"] + terms_df["term"].tolist(), index=0)
if term != "Barchasi":
    facs_df = run_sql_cached(f"SELECT DISTINCT faculty FROM {VIEW_SS} WHERE term=:t ORDER BY faculty;", {"t": term})
else:
    facs_df = run_sql_cached(f"SELECT DISTINCT faculty FROM {VIEW_SS} ORDER BY faculty;")

faculty  = st.sidebar.selectbox("Fakultet", ["Barchasi"] + facs_df["faculty"].tolist(), index=0)
row_limit = st.sidebar.slider("Jadval limiti", 50, 3000, 300, 50)
risk_att = st.sidebar.slider("Risk chegarasi — Davomat %", 50, 90, 75, 1)
risk_grd = st.sidebar.slider("Risk chegarasi — O‘rtacha baho", 40, 100, 60, 1)

if st.sidebar.button("🔄 Keshni tozalash"):
    invalidate_cache()
    st.sidebar.success("Kesh tozalandi. Ma’lumotlar yangilanadi.")

where_sql, params = where_clause(term, faculty)

# =========================
# 6) Tabs
# =========================
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "🎓 Students", "👩‍🏫 Teachers", "💼 Finance", "🛠️ Admin"])

# ===== Overview =====
with tab1:
    # KPI
    kpi_sql = f"""
    SELECT
      SUM(students)::bigint AS students,
      ROUND(AVG(avg_gpa)::numeric, 2) AS avg_gpa,
      ROUND(AVG(pass_rate)*100, 1)   AS pass_pct,
      ROUND(AVG(attendance_avg)*100, 1) AS att_pct
    FROM {VIEW_SS}
    {where_sql}
    """
    kpdf = run_sql_cached(kpi_sql, params=params)
    k = kpdf.iloc[0] if not kpdf.empty else {"students":0,"avg_gpa":0,"pass_pct":0,"att_pct":0}

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card("Talabalar", f"{int(k['students'] or 0):,}", "Jami studentlar", "#22c55e", "🎓")
    with c2: kpi_card("O‘rtacha GPA", f"{k['avg_gpa'] or 0}", "Term / fakultetga bog‘liq", "#3b82f6", "📘")
    with c3: kpi_card("O‘tish ko‘rsatkichi", f"{k['pass_pct'] or 0}%", "AVG pass_rate", "#f59e0b", "✅")
    with c4: kpi_card("Davomat", f"{k['att_pct'] or 0}%", "AVG attendance", "#ec4899", "📅")

    st.divider()
    colA, colB = st.columns(2)

    prs = run_sql_cached(f"""
        SELECT faculty, term, ROUND(AVG(pass_rate)*100,1) AS pass_pct
        FROM {VIEW_SS}
        {where_sql}
        GROUP BY faculty, term
        ORDER BY term, faculty
    """, params=params)
    fig = px.bar(prs, x="faculty", y="pass_pct", color="term",
                 title="Pass rate (%) — term × fakultet", barmode="group")
    colA.plotly_chart(fig, use_container_width=True)

    att = run_sql_cached(f"""
        SELECT faculty, term, ROUND(AVG(attendance_avg)*100,1) AS att_pct
        FROM {VIEW_SS}
        {where_sql}
        GROUP BY faculty, term
        ORDER BY term, faculty
    """, params=params)
    fig2 = px.line(att, x="term", y="att_pct", color="faculty", markers=True,
                   title="Davomat (%) — term bo‘yicha")
    colB.plotly_chart(fig2, use_container_width=True)

    st.subheader("🏆 Top o‘qituvchilar (pass_rate)")
    tp = run_sql_cached(f"""
        SELECT teacher_name, faculty, term,
               ROUND(pass_rate*100,1) AS pass_pct,
               ROUND(avg_grade,2)     AS avg_grade,
               ROUND(attendance*100,1) AS att_pct,
               n
        FROM {VIEW_TP}
        {where_sql}
        ORDER BY pass_rate DESC
        LIMIT 20
    """, params=params)
    st.dataframe(tp, use_container_width=True, height=360)
    download_buttons(tp, "top_teachers")

# ===== Students =====
with tab2:
    st.subheader("Talaba natijalari (kesimlar)")
    ss = run_sql_cached(f"""
        SELECT faculty, term, students,
               ROUND(avg_gpa,2)               AS avg_gpa,
               ROUND(pass_rate*100,1)         AS pass_pct,
               ROUND(attendance_avg*100,1)    AS att_pct
        FROM {VIEW_SS}
        {where_sql}
        ORDER BY term, faculty
        LIMIT :lim
    """, params={**params, "lim": row_limit})
    st.dataframe(ss, use_container_width=True, height=420)
    download_buttons(ss, "students_view")

    st.divider()
    st.subheader("⚠️ Riskli talabalar")
    risk = run_sql_cached(f"""
        SELECT e.student_id,
               e.faculty,
               ROUND(AVG(e.attendance)*100,1) AS att_pct,
               ROUND(AVG(e.grade),2)          AS avg_grade,
               COUNT(*) AS n_courses
        FROM enrollments e
        WHERE 1=1
          {"AND e.term = :term" if "term" in params else ""}
          {"AND e.faculty = :faculty" if "faculty" in params else ""}
        GROUP BY e.student_id, e.faculty
        HAVING AVG(e.attendance) < :att_thr/100.0 OR AVG(e.grade) < :grd_thr
        ORDER BY att_pct ASC, avg_grade ASC
        LIMIT :lim
    """, params={**params, "lim": row_limit, "att_thr": risk_att, "grd_thr": risk_grd})
    st.dataframe(risk, use_container_width=True, height=380)
    download_buttons(risk, "risk_students")

# ===== Teachers =====
with tab3:
    st.subheader("O‘qituvchi performansi")
    tp_all = run_sql_cached(f"""
        SELECT teacher_name, faculty, term,
               ROUND(pass_rate*100,1) AS pass_pct,
               ROUND(avg_grade,2)     AS avg_grade,
               ROUND(attendance*100,1) AS att_pct,
               n
        FROM {VIEW_TP}
        {where_sql}
        ORDER BY faculty, teacher_name, term
        LIMIT :lim
    """, params={**params, "lim": row_limit})
    st.dataframe(tp_all, use_container_width=True, height=420)
    download_buttons(tp_all, "teachers_perf")

    st.markdown("### Fakultetlar bo‘yicha o‘rtacha ko‘rsatkichlar")
    tchart = run_sql_cached(f"""
        SELECT faculty,
               ROUND(AVG(pass_rate)*100,1) AS pass_pct,
               ROUND(AVG(avg_grade),2)     AS avg_grade,
               ROUND(AVG(attendance)*100,1) AS att_pct
        FROM {VIEW_TP}
        {where_sql}
        GROUP BY faculty
        ORDER BY faculty
    """, params=params)
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.bar(tchart, x="faculty", y="pass_pct",
                           title="Pass rate (%) — o‘rtacha, fakultet bo‘yicha"),
                    use_container_width=True)
    c2.plotly_chart(px.bar(tchart, x="faculty", y="avg_grade",
                           title="O‘rtacha baho — fakultet bo‘yicha"),
                    use_container_width=True)

    st.divider()
    st.markdown("### 👆 O‘qituvchi drilldown")
    colt1, colt2 = st.columns([2,1])
    tnames = run_sql_cached(f"SELECT DISTINCT teacher_name FROM {VIEW_TP} ORDER BY teacher_name;")
    t_sel = colt1.selectbox("O‘qituvchi", tnames["teacher_name"].tolist() if not tnames.empty else [])
    show_btn = colt2.button("Ko‘rish", use_container_width=True)

    if t_sel and show_btn:
        det = run_sql_cached(f"""
            SELECT e.term, e.course_id, e.course_name,
                   ROUND(AVG(e.grade),2)          AS avg_grade,
                   ROUND(AVG(e.attendance)*100,1) AS att_pct,
                   COUNT(*)                       AS n
            FROM enrollments e
            LEFT JOIN teachers t USING(teacher_id)
            WHERE t.teacher_name = :tn
              {"AND e.term = :term" if "term" in params else ""}
              {"AND e.faculty = :faculty" if "faculty" in params else ""}
            GROUP BY e.term, e.course_id, e.course_name
            ORDER BY e.term, e.course_id
        """, params={**params, "tn": t_sel})
        st.dataframe(det, use_container_width=True, height=380)
        download_buttons(det, f"teacher_detail_{t_sel.replace(' ','_')}")

# ===== Finance =====
with tab4:
    st.subheader("Moliya (oyma-oy)")
    fin = run_sql_cached(f"""
        SELECT month, faculty, revenue, expense, net
        FROM {VIEW_FN}
        {("WHERE faculty=:faculty" if faculty!="Barchasi" else "")}
        ORDER BY month, faculty
    """, params=({"faculty": faculty} if faculty!="Barchasi" else None))
    st.dataframe(fin, use_container_width=True, height=420)
    fig = px.line(fin, x="month", y="net", color="faculty", markers=True, title="Net (Revenue - Expense)")
    st.plotly_chart(fig, use_container_width=True)
    download_buttons(fin, "finance_monthly")

# ===== Admin =====
with tab5:
    st.subheader("Kesh & ETL")
    cA, cB, cC = st.columns([1,1,2])
    if cA.button("🧹 Keshni tozalash", use_container_width=True):
        invalidate_cache()
        st.success("Kesh tozalandi.")

    etl_path = cB.text_input("ETL (main.py) yo‘li", value="main.py")
    data_dir = cC.text_input("Data papka", value="./data")

    dsn_ui = st.text_input("DSN (bo‘sh qoldirsangiz .env/DB_DSN ishlatiladi)", value=DB_DSN)
    run_cols = st.columns([1,1,1,2])
    run_once = run_cols[0].button("🔄 Run ETL (once)", use_container_width=True)
    run_refresh_mv = run_cols[1].checkbox("MV refresh", value=True)
    show_cmd = run_cols[2].checkbox("Buyruqni ko‘rsat", value=False)

    if run_once:
        cmd = [sys.executable, etl_path, "--dsn", dsn_ui, "--data-dir", data_dir, "--once", "--log-level", "INFO"]
        if run_refresh_mv: cmd.append("--refresh-mv")
        if show_cmd: st.code(" ".join(cmd))
        with st.spinner("ETL ishga tushmoqda..."):
            cp = subprocess.run(cmd, capture_output=True, text=True)
        st.write("Return code:", cp.returncode)
        st.text_area("STDOUT", cp.stdout, height=180)
        st.text_area("STDERR", cp.stderr, height=180)
        if cp.returncode == 0:
            st.success("ETL OK — kesh tozalanmoqda…")
            invalidate_cache()
        else:
            st.error("ETL xato! Loglarni ko‘ring.")
