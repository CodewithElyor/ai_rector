# app.py — AI-Rektor (Auth + Admin + Rektor, Uzbek UI)
# Python 3.12+, streamlit 1.51.0, streamlit-authenticator 0.4.2

import os
from typing import Dict, Any, Tuple
from io import BytesIO
from datetime import datetime

import bcrypt
import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import create_engine, text
import streamlit_authenticator as stauth

# =========================
# Konfiguratsiya / Secrets
# =========================
st.set_page_config(page_title="🎓 AI-Rektor Dashboard", layout="wide", initial_sidebar_state="expanded")

DB_DSN    = st.secrets.get("DB_DSN", os.getenv("DB_DSN", "postgresql://postgres:7778@localhost:5432/Start_Up"))
DB_SCHEMA = st.secrets.get("DB_SCHEMA", os.getenv("DB_SCHEMA", "ai_rektor"))
CACHE_TTL = int(st.secrets.get("CACHE_TTL_SEC", os.getenv("CACHE_TTL_SEC", "300")))
ALLOW_ETL = str(st.secrets.get("ALLOW_ETL_CLOUD", os.getenv("ALLOW_ETL_CLOUD", "0"))).lower() in ("1","true","yes")

# MV bor deb boshlaymiz; bo‘lmasa VIEW’ga tushamiz
USE_MV  = True
VIEW_SS = "mv_student_success" if USE_MV else "vw_student_success"
VIEW_TP = "mv_teacher_perf"   if USE_MV else "vw_teacher_perf"
VIEW_FN = "mv_fin_summary"    if USE_MV else "vw_fin_summary"

# ==============
# Auth yordamchi
# ==============
def _ensure_hashed(pw_or_hash: str) -> str:
    """Berilgan parol/hashtan bcrypt hash qaytaradi (agar allaqachon hash bo‘lsa — o‘sha o‘zini)."""
    if pw_or_hash and pw_or_hash.startswith(("$2a$", "$2b$", "$2y$")):
        return pw_or_hash
    return bcrypt.hashpw((pw_or_hash or "admin").encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

def _prepare_auth():
    """
    Secrets formati (tavsiya):

    [auth]
    cookie_name = "ai_rektor_auth"
    cookie_key  = "supersecret"
    expiry_days = 7

    [[auth.users]]
    name     = "Admin"
    username = "admin"
    email    = "admin@uni.uz"
    # parol: admin
    password = "$2b$12$0.0O7mC8Vbq8J6Qw6r5f9u9yXk9J9Y6.1gE8eQd0dY8Y0zJk5tXGi"
    role     = "admin"
    """
    cfg = st.secrets.get("auth", {})
    users = cfg.get("users", [])

    # Dev fallback (agar secrets’da users bo‘sh bo‘lsa)
    if not users:
        users = [{
            "name": "Admin",
            "username": "admin",
            "email": "admin@uni.uz",
            "password_plain": "admin",
            "role": "admin",
        }]

    creds = {"usernames": {}}
    roles = {}
    for u in users:
        uname = u.get("username")
        if not uname:
            continue
        creds["usernames"][uname] = {
            "name": u.get("name", uname),
            "email": u.get("email", ""),
            "password": _ensure_hashed(u.get("password") or u.get("password_plain", "admin")),
        }
        roles[uname] = u.get("role", "teacher")

    cookie_name = cfg.get("cookie_name", "ai_rektor_auth")
    cookie_key  = cfg.get("cookie_key", "supersecret")
    expiry_days = int(cfg.get("expiry_days", 7))
    return creds, roles, cookie_name, cookie_key, expiry_days

creds, ROLES_MAP, cookie_name, cookie_key, expiry_days = _prepare_auth()

authenticator = stauth.Authenticate(
    credentials=creds,
    cookie_name=cookie_name,
    key=cookie_key,
    cookie_expiry_days=expiry_days,
)

# ==============
# Stil / CSS
# ==============
DARK_CSS = """
<style>
.block-container { padding-top: 1rem !important; }
.kpi { background: var(--background-secondary); border:1px solid rgba(148,163,184,.25); border-radius:16px; padding:14px 16px; }
.kpi h4{ margin:0 0 6px 0; font-size:12px; color:#7c8aa0; }
.kpi .val{ font-size:26px; font-weight:800; }
.kpi .sub{ font-size:12px; color:#94a3b8; }
hr{ border-color: rgba(148,163,184,.25); }
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)

# =========================
# DB ulanish (cached engine)
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

engine = get_engine()

# -----------
# SQL helper
# -----------
@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def run_sql_cached(sql: str, params: Dict[str, Any] | None = None) -> pd.DataFrame:
    with engine.connect() as conn:
        conn.exec_driver_sql(f"SET search_path TO {DB_SCHEMA}, public;")
        return pd.read_sql(text(sql), conn, params=params)

def run_sql(sql: str, params: Dict[str, Any] | None = None) -> pd.DataFrame:
    with engine.connect() as conn:
        conn.exec_driver_sql(f"SET search_path TO {DB_SCHEMA}, public;")
        return pd.read_sql(text(sql), conn, params=params)

def invalidate_cache():
    run_sql_cached.clear()

def download_buttons(df: pd.DataFrame, base_name: str):
    c1, c2 = st.columns(2)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    c1.download_button("⬇️ CSV", csv_bytes, f"{base_name}.csv", "text/csv", use_container_width=True)
    xlsx = BytesIO()
    with pd.ExcelWriter(xlsx, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="data")
    c2.download_button("⬇️ Excel", xlsx.getvalue(), f"{base_name}.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       use_container_width=True)

# =========================
# LOGIN (0.4.2 bilan mos)
# =========================
st.title("🎓 AI-Rektor Dashboard")

login_result = authenticator.login(location="main", key="login_form")
if login_result is None:
    # Forma chizilgan, foydalanuvchi hali submit qilmagan
    st.info("Iltimos, tizimga kiring.")
    st.stop()

# 0.4.x: login_result -> (name, auth_status, username)
name, auth_status, username = login_result

if auth_status is False:
    st.error("Login yoki parol noto‘g‘ri.")
    st.stop()
if auth_status is None:
    st.info("Iltimos, tizimga kiring.")
    st.stop()

role = ROLES_MAP.get(username, "teacher")
top_l, top_r = st.columns([3,1])
with top_l:
    st.caption(f"👤 {name} · rol: **{role}** · schema: `{DB_SCHEMA}` · cache: {CACHE_TTL}s")
with st.sidebar:
    authenticator.logout("Chiqish", key="logout_btn")

# =========================
# MV mavjudligini tekshirish
# =========================
def _table_ok(name: str) -> bool:
    try:
        _ = run_sql_cached(f"SELECT 1 FROM {name} LIMIT 1;")
        return True
    except Exception:
        return False

if USE_MV and not all(_table_ok(t) for t in (VIEW_SS, VIEW_TP, VIEW_FN)):
    VIEW_SS, VIEW_TP, VIEW_FN = "vw_student_success", "vw_teacher_perf", "vw_fin_summary"
    st.warning("Materialized view topilmadi. Oddiy VIEW’lardan foydalanildi.")

# =========================
# Sidebar — Filtrlar
# =========================
st.sidebar.header("⚙️ Filtrlar")

def _safe_terms() -> list[str]:
    try:
        return run_sql_cached(f"SELECT DISTINCT term FROM {VIEW_SS} ORDER BY term;")["term"].tolist()
    except Exception:
        return []

terms = _safe_terms()
term = st.sidebar.selectbox("Term", ["Barchasi"] + terms, index=0)

try:
    if term != "Barchasi":
        facs_df = run_sql_cached(f"SELECT DISTINCT faculty FROM {VIEW_SS} WHERE term=:t ORDER BY faculty;", {"t": term})
    else:
        facs_df = run_sql_cached(f"SELECT DISTINCT faculty FROM {VIEW_SS} ORDER BY faculty;")
    fac_list = facs_df["faculty"].tolist()
except Exception:
    fac_list = []

faculty  = st.sidebar.selectbox("Fakultet", ["Barchasi"] + fac_list, index=0)
row_limit = st.sidebar.slider("Jadval limiti", 50, 3000, 300, 50)
risk_att  = st.sidebar.slider("Risk chegarasi — Davomat %", 50, 95, 75, 1)
risk_grd  = st.sidebar.slider("Risk chegarasi — O‘rtacha baho", 40, 100, 60, 1)

def where_clause(term_: str, faculty_: str) -> Tuple[str, Dict[str, Any]]:
    w, p = [], {}
    if term_ != "Barchasi":
        w.append("term = :term"); p["term"] = term_
    if faculty_ != "Barchasi":
        w.append("faculty = :faculty"); p["faculty"] = faculty_
    return ("WHERE " + " AND ".join(w)) if w else "", p

where_sql, params = where_clause(term, faculty)

# =========================
# KPI helper
# =========================
def kpi_color(value: float, good: float, warn: float, reverse=False) -> str:
    if value is None: return "#0b1220"
    v = float(value)
    if reverse:
        if v <= good: return "#073b1c"
        if v <= warn: return "#3b3607"
        return "#3b0707"
    else:
        if v >= good: return "#073b1c"
        if v >= warn: return "#3b3607"
        return "#3b0707"

def kpi_card(title: str, value_str: str, sub: str, bg: str):
    st.markdown(
        f"""<div class="kpi" style="background:{bg};">
              <h4>{title}</h4>
              <div class="val">{value_str}</div>
              <div class="sub">{sub}</div>
            </div>""",
        unsafe_allow_html=True
    )

# =========================
# Tabs: Overview • Students • Teachers • Finance • Admin • Rektor
# =========================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["📊 Overview", "🎓 Students", "👩‍🏫 Teachers", "💼 Finance", "🛠️ Admin", "🏛️ Rektor"]
)

# ===== Overview =====
with tab1:
    try:
        kpi_sql = f"""
        SELECT
          SUM(students)::bigint AS students,
          ROUND(AVG(avg_gpa)::numeric, 2) AS avg_gpa,
          ROUND(AVG(pass_rate)*100, 1) AS pass_pct,
          ROUND(AVG(attendance_avg)*100, 1) AS att_pct
        FROM {VIEW_SS}
        {where_sql}
        """
        kpdf = run_sql_cached(kpi_sql, params=params)
        k = kpdf.iloc[0] if not kpdf.empty else {"students":0,"avg_gpa":0,"pass_pct":0,"att_pct":0}

        c1, c2, c3, c4 = st.columns(4)
        bg1 = kpi_color(k["students"] or 0, good=1500, warn=800)
        bg2 = kpi_color(k["avg_gpa"] or 0,   good=3.0,  warn=2.5)
        bg3 = kpi_color(k["pass_pct"] or 0,  good=85,   warn=70)
        bg4 = kpi_color(k["att_pct"] or 0,   good=85,   warn=75)

        with c1: kpi_card("Talabalar", f"{int(k['students'] or 0):,}", "Jami studentlar", bg1)
        with c2: kpi_card("O‘rtacha GPA", f"{k['avg_gpa'] or 0}", "Filtrlar ta’sir qiladi", bg2)
        with c3: kpi_card("O‘tish darajasi", f"{k['pass_pct'] or 0}%", "AVG pass_rate", bg3)
        with c4: kpi_card("Davomat", f"{k['att_pct'] or 0}%", "AVG attendance", bg4)

        st.divider()
        colA, colB = st.columns(2)

        prs = run_sql_cached(f"""
            SELECT faculty, term, ROUND(AVG(pass_rate)*100,1) AS pass_pct
            FROM {VIEW_SS}
            {where_sql}
            GROUP BY faculty, term
            ORDER BY term, faculty
        """, params=params)
        colA.plotly_chart(px.bar(prs, x="faculty", y="pass_pct", color="term",
                                 title="Pass rate (%) — term × fakultet", barmode="group"),
                          use_container_width=True)

        att = run_sql_cached(f"""
            SELECT faculty, term, ROUND(AVG(attendance_avg)*100,1) AS att_pct
            FROM {VIEW_SS}
            {where_sql}
            GROUP BY faculty, term
            ORDER BY term, faculty
        """, params=params)
        colB.plotly_chart(px.line(att, x="term", y="att_pct", color="faculty", markers=True,
                                  title="Davomat (%) — term bo‘yicha"),
                          use_container_width=True)

        st.subheader("🏆 Top o‘qituvchilar (pass_rate)")
        tp = run_sql_cached(f"""
            SELECT teacher_name, faculty, term, pass_rate, avg_grade, attendance, n
            FROM {VIEW_TP}
            {where_sql}
            ORDER BY pass_rate DESC
            LIMIT 20
        """, params=params)
        st.dataframe(tp, use_container_width=True, height=360)
        download_buttons(tp, "top_teachers")
    except Exception as e:
        st.error(f"Ma’lumot o‘qishda xato (Overview): {e}")

# ===== Students =====
with tab2:
    try:
        st.subheader("Talaba natijalari (kesimlar)")
        ss = run_sql_cached(f"""
            SELECT faculty, term, students,
                   ROUND(avg_gpa,2) AS avg_gpa,
                   ROUND(pass_rate*100,1) AS pass_pct,
                   ROUND(attendance_avg*100,1) AS att_pct
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
                   ROUND(AVG(e.grade),2) AS avg_grade,
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
    except Exception as e:
        st.error(f"Ma’lumot o‘qishda xato (Students): {e}")

# ===== Teachers =====
with tab3:
    try:
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
                               title="Pass rate (%) — o‘rtacha, fakultet kesimida"),
                        use_container_width=True)
        c2.plotly_chart(px.bar(tchart, x="faculty", y="avg_grade",
                               title="O‘rtacha baho — fakultet kesimida"),
                        use_container_width=True)

        st.divider()
        st.markdown("### 👆 O‘qituvchi drilldown")
        colt1, colt2 = st.columns([2,1])
        tnames = run_sql_cached(f"SELECT DISTINCT teacher_name FROM {VIEW_TP} ORDER BY teacher_name;")
        t_sel = colt1.selectbox("O‘qituvchi", (tnames["teacher_name"].tolist() if not tnames.empty else []))
        show_btn = colt2.button("Ko‘rish", use_container_width=True)

        if t_sel and show_btn:
            det = run_sql_cached(f"""
                SELECT e.term, e.course_id, e.course_name,
                       ROUND(AVG(e.grade),2) AS avg_grade,
                       ROUND(AVG(e.attendance)*100,1) AS att_pct,
                       COUNT(*) AS n
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
    except Exception as e:
        st.error(f"Ma’lumot o‘qishda xato (Teachers): {e}")

# ===== Finance =====
with tab4:
    try:
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
    except Exception as e:
        st.error(f"Ma’lumot o‘qishda xato (Finance): {e}")

# ===== Admin =====
with tab5:
    st.subheader("Admin")
    st.caption("Kesh tozalash, diagnostika va (ixtiyoriy) ETL.")

    cA, cB = st.columns([1,1])
    if cA.button("🧹 Keshni tozalash", use_container_width=True):
        invalidate_cache()
        st.success("Kesh tozalandi.")

    with cB:
        try:
            ping = run_sql("SELECT current_user, current_database(), current_schemas(true);")
            st.dataframe(ping, use_container_width=True)
        except Exception as e:
            st.error(f"DB ulanishda xato: {e}")

    st.divider()
    try:
        cnt = run_sql("""
            SELECT
              (SELECT COUNT(*) FROM students)    AS students,
              (SELECT COUNT(*) FROM enrollments) AS enrollments,
              (SELECT COUNT(*) FROM teachers)    AS teachers,
              (SELECT COUNT(*) FROM finance)     AS finance;
        """)
        st.dataframe(cnt, use_container_width=True)
    except Exception as e:
        st.error(f"Hisobda xato: {e}")

    st.divider()
    if not ALLOW_ETL:
        st.info("ETL tugmasi o‘chirilgan (ALLOW_ETL_CLOUD=false).")
    else:
        st.warning("Cloud konteynerda fayl yo‘llari va ruxsatlar to‘g‘ri bo‘lishi kerak.")
        st.code("python main.py --dsn \"$DB_DSN\" --data-dir ./data --once --refresh-mv", language="bash")

# ===== Rektor =====
with tab6:
    try:
        st.subheader("🏛️ Rektor paneli")
        k2 = run_sql_cached(f"""
            SELECT
              SUM(students)::bigint AS students,
              ROUND(AVG(avg_gpa)::numeric, 2) AS avg_gpa,
              ROUND(AVG(pass_rate)*100, 1) AS pass_pct,
              ROUND(AVG(attendance_avg)*100, 1) AS att_pct
            FROM {VIEW_SS}
        """).iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        with c1: kpi_card("Jami talabalar", f"{int(k2['students'] or 0):,}", datetime.now().strftime("Yangilanish: %Y-%m-%d"), "#0b1220")
        with c2: kpi_card("O‘rtacha GPA", f"{k2['avg_gpa'] or 0}", "Butun universitet kesimida", "#0b1220")
        with c3: kpi_card("O‘tish darajasi", f"{k2['pass_pct'] or 0}%", "Pass_rate (AVG)", "#0b1220")
        with c4: kpi_card("Davomat", f"{k2['att_pct'] or 0}%", "Attendance (AVG)", "#0b1220")

        st.divider()
        st.markdown("### Fakultetlar reytingi (o‘tish darajasiga ko‘ra)")
        rank = run_sql_cached(f"""
            SELECT faculty,
                   ROUND(AVG(pass_rate)*100,1) AS pass_pct,
                   ROUND(AVG(avg_grade),2)     AS avg_grade,
                   ROUND(AVG(attendance)*100,1) AS att_pct,
                   COUNT(*) AS n
            FROM {VIEW_TP}
            GROUP BY faculty
            ORDER BY pass_pct DESC
            LIMIT 10
        """)
        st.dataframe(rank, use_container_width=True, height=360)
        download_buttons(rank, "rektor_faculty_ranking")

        st.divider()
        st.markdown("### Net (oylar bo‘yicha, barcha fakultetlar)")
        fin_all = run_sql_cached(f"SELECT month, faculty, net FROM {VIEW_FN} ORDER BY month, faculty;")
        st.plotly_chart(px.line(fin_all, x="month", y="net", color="faculty", markers=True,
                                title="Net (Revenue - Expense) — universitet bo‘yicha"),
                        use_container_width=True)
    except Exception as e:
        st.error(f"Ma’lumot o‘qishda xato (Rektor): {e}")
