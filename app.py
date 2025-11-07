# app.py — AI-Rektor (Auth + Admin + Rektor) — Streamlit Cloud friendly
# -*- coding: utf-8 -*-
import os
from typing import Dict, Any, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import create_engine, text

# Auth
import streamlit_authenticator as stauth

# -------------------------
# Konfiguratsiya / Secrets
# -------------------------
# Streamlit Cloud: Secrets bo‘limida saqlanadi
# (Agar secrets bo‘lmasa .env dan oladi, yanada bo‘lmasa local default)
DB_DSN = st.secrets.get("DB_DSN", os.getenv("DB_DSN", "postgresql://postgres:7778@localhost:5432/Start_Up"))
DB_SCHEMA = st.secrets.get("DB_SCHEMA", os.getenv("DB_SCHEMA", "ai_rektor"))
CACHE_TTL = int(st.secrets.get("CACHE_TTL_SEC", os.getenv("CACHE_TTL_SEC", "300")))
ALLOW_ETL = bool(st.secrets.get("ALLOW_ETL_CLOUD", os.getenv("ALLOW_ETL_CLOUD", "0")))

# Auth konfiguratsiyasi (Streamlit secrets toml):
# [auth]
# cookie_name = "ai_rektor_auth"
# cookie_key = "supersecret"
# expiry_days = 7
# [[auth.users]]
# name = "Admin"
# username = "admin"
# email = "admin@uni.uz"
# password = "$2b$12$..."     # bcrypt hash (stauth.Hasher([...]).generate())
# role = "admin"
# [[auth.users]]
# name = "Rektor"
# username = "rektor"
# email = "rektor@uni.uz"
# password = "$2b$12$..."
# role = "rektor"
AUTH_CFG = st.secrets.get("auth", {
    "cookie_name": "ai_rektor_auth",
    "cookie_key": "supersecret",
    "expiry_days": 7,
    "users": [
        # Minimal fallback user (demo): login admin / pass admin
        # Cloud’da albatta secrets toml bilan haqiqiy bcrypt hash qo‘ying!
        {
            "name": "Admin",
            "username": "admin",
            "email": "admin@uni.uz",
            "password": stauth.Hasher(["admin"]).generate()[0],
            "role": "admin",
        }
    ],
})

def _prepare_auth_config(auth_cfg: Dict[str, Any]) -> tuple[dict, dict]:
    """streamlit-authenticator kutgan formatga aylantiramiz."""
    users = auth_cfg.get("users", [])
    creds = {"usernames": {}}
    roles = {}
    for u in users:
        uname = u.get("username")
        if not uname:
            continue
        creds["usernames"][uname] = {
            "name": u.get("name", uname),
            "password": u.get("password", ""),  # bcrypt hash
            "email": u.get("email", ""),
        }
        roles[uname] = u.get("role", "teacher")
    return creds, roles

CREDENTIALS, USER_ROLES = _prepare_auth_config(AUTH_CFG)

# UI — sahifa sozlamalari
st.set_page_config(
    page_title="AI-Rektor Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Kichik CSS (biroz “dark-ish” kayfiyat)
st.markdown("""
<style>
:root {
  --card-bg: #0f172a11; /* engil shaffof */
  --card-border: #e5e7eb33;
}
.block-container { padding-top: 1.2rem; }
.kpi-card { background: var(--card-bg); border:1px solid var(--card-border); border-radius:14px; padding:16px; }
.kpi-title { font-size:12px; color:#475569; margin-bottom:6px; }
.kpi-value { font-size:28px; font-weight:700; }
.kpi-sub { font-size:12px; color:#64748b; }
</style>
""", unsafe_allow_html=True)

st.title("🎓 AI-Rektor Dashboard")

# --------------------------
# Auth — login / logout
# --------------------------
authenticator = stauth.Authenticate(
    CREDENTIALS,
    AUTH_CFG.get("cookie_name", "ai_rektor_auth"),
    AUTH_CFG.get("cookie_key", "supersecret"),
    AUTH_CFG.get("expiry_days", 7),
)

# Headerda login form
if "authentication_status" not in st.session_state:
    st.session_state.authentication_status = None

col_login_l, col_login_r = st.columns([3, 1])
with col_login_r:
    name, auth_status, uname = authenticator.login("Kirish", "main")
    st.session_state.authentication_status = auth_status

if st.session_state.authentication_status is False:
    st.error("Login yoki parol xato.")
    st.stop()
elif st.session_state.authentication_status is None:
    st.info("Iltimos, tizimga kiring.")
    st.stop()

# Auth OK — foydalanuvchi va roli
current_user = uname
current_role = USER_ROLES.get(uname, "teacher")

with st.sidebar:
    st.markdown(f"**👤 Foydalanuvchi:** {name}  \n**Rol:** `{current_role}`")
    authenticator.logout("Chiqish", "sidebar")

# --------------------------
# Connection (pooled, cached)
# --------------------------
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

# Mat. viewlar mavjudmi? avtomatik aniqlash
@st.cache_data(ttl=600, show_spinner=False)
def detect_views() -> Dict[str, str]:
    """MV bo‘lsa mv_* ni, bo‘lmasa vw_* ni ishlatish."""
    mapping = {"ss": "vw_student_success", "tp": "vw_teacher_perf", "fn": "vw_fin_summary"}
    try:
        with engine.connect() as conn:
            conn.exec_driver_sql(f"SET search_path TO {DB_SCHEMA}, public;")
            q = text("SELECT matviewname FROM pg_matviews WHERE schemaname = :s")
            mv = pd.read_sql(q, conn, params={"s": DB_SCHEMA})
        mnames = mv["matviewname"].tolist() if not mv.empty else []
        if "mv_student_success" in mnames:
            mapping["ss"] = "mv_student_success"
        if "mv_teacher_perf" in mnames:
            mapping["tp"] = "mv_teacher_perf"
        if "mv_fin_summary" in mnames:
            mapping["fn"] = "mv_fin_summary"
    except Exception:
        pass
    return mapping

VIEW_MAP = detect_views()
VIEW_SS = VIEW_MAP["ss"]
VIEW_TP = VIEW_MAP["tp"]
VIEW_FN = VIEW_MAP["fn"]

# --------------------------
# SQL helpers
# --------------------------
@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def run_sql_cached(sql: str, params: Dict[str, Any] | None = None) -> pd.DataFrame:
    try:
        with engine.connect() as conn:
            conn.exec_driver_sql(f"SET search_path TO {DB_SCHEMA}, public;")
            return pd.read_sql(text(sql), conn, params=params)
    except Exception as e:
        st.error(f"❌ DB ulanishda xato: {e}")
        return pd.DataFrame()

def run_sql(sql: str, params: Dict[str, Any] | None = None) -> pd.DataFrame:
    with engine.connect() as conn:
        conn.exec_driver_sql(f"SET search_path TO {DB_SCHEMA}, public;")
        return pd.read_sql(text(sql), conn, params=params)

def invalidate_cache():
    run_sql_cached.clear()
    detect_views.clear()

def download_buttons(df: pd.DataFrame, base_name: str):
    ccsv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ CSV", ccsv, f"{base_name}.csv", "text/csv", use_container_width=True)
    try:
        xlsx_path = f"{base_name}.xlsx"
        with pd.ExcelWriter(xlsx_path) as writer:
            df.to_excel(writer, sheet_name="data", index=False)
        with open(xlsx_path, "rb") as f:
            st.download_button("⬇️ Excel", f, f"{base_name}.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               use_container_width=True)
    finally:
        try: os.remove(xlsx_path)
        except Exception: pass

# --------------------------
# Sidebar — filtrlash
# --------------------------
st.sidebar.header("⚙️ Filtrlar")
terms_df = run_sql_cached(f"SELECT DISTINCT term FROM {VIEW_SS} ORDER BY term;")
term = st.sidebar.selectbox("Term", ["Barchasi"] + (terms_df["term"].tolist() if not terms_df.empty else []), index=0)

if term != "Barchasi":
    facs_df = run_sql_cached(f"SELECT DISTINCT faculty FROM {VIEW_SS} WHERE term=:t ORDER BY faculty;", {"t": term})
else:
    facs_df = run_sql_cached(f"SELECT DISTINCT faculty FROM {VIEW_SS} ORDER BY faculty;")

faculty = st.sidebar.selectbox("Fakultet", ["Barchasi"] + (facs_df["faculty"].tolist() if not facs_df.empty else []), index=0)
row_limit = st.sidebar.slider("Jadval limiti", 50, 3000, 300, 50)
risk_att = st.sidebar.slider("Risk chegarasi — Davomat %", 50, 90, 75, 1)
risk_grd = st.sidebar.slider("Risk chegarasi — O‘rtacha baho", 40, 100, 60, 1)

if st.sidebar.button("🔄 Keshni tozalash"):
    invalidate_cache()
    st.sidebar.success("Kesh tozalandi. Ma’lumotlar yangilanadi.")

def where_clause(term: str, faculty: str) -> Tuple[str, Dict[str, Any]]:
    w, p = [], {}
    if term != "Barchasi":
        w.append("term = :term"); p["term"] = term
    if faculty != "Barchasi":
        w.append("faculty = :faculty"); p["faculty"] = faculty
    return ("WHERE " + " AND ".join(w)) if w else "", p

where_sql, params = where_clause(term, faculty)
st.caption(f"Schema: `{DB_SCHEMA}` • Cache TTL: {CACHE_TTL}s • Manba: `{VIEW_SS}/{VIEW_TP}/{VIEW_FN}`")

# --------------------------
# KPI cards (rangli)
# --------------------------
def kpi_color(value: float, good: float, warn: float, reverse=False) -> str:
    # reverse=False: katta -> yaxshi. True: kichik -> yaxshi
    if value is None: return "#f1f5f9"
    v = value
    if reverse:
        if v <= good: return "#dcfce7"
        if v <= warn: return "#fef9c3"
        return "#fee2e2"
    else:
        if v >= good: return "#dcfce7"
        if v >= warn: return "#fef9c3"
        return "#fee2e2"

def kpi_card(title: str, value_str: str, sub: str, bg: str):
    st.markdown(
        f"""
        <div class="kpi-card" style="background:{bg};">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value_str}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# --------------------------
# Tabs — Admin oldin, Rektor keyin (talab bo‘yicha)
# --------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["📊 Overview", "🎓 Students", "👩‍🏫 Teachers", "💼 Finance", "🛠️ Admin", "🏛️ Rektor"]
)

# ===== Overview =====
with tab1:
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
    bg1 = kpi_color(float(k["students"] or 0), good=1500, warn=800)
    bg2 = kpi_color(float(k["avg_gpa"] or 0), good=3.0, warn=2.5)
    bg3 = kpi_color(float(k["pass_pct"] or 0), good=85, warn=70)
    bg4 = kpi_color(float(k["att_pct"] or 0), good=85, warn=75)

    with c1: kpi_card("Talabalar", f"{int(k['students'] or 0):,}", "Jami studentlar", bg1)
    with c2: kpi_card("O‘rtacha GPA", f"{k['avg_gpa'] or 0}", "Term/Fakultet filtrlariga bog‘liq", bg2)
    with c3: kpi_card("O‘tish ko‘rsatkichi", f"{k['pass_pct'] or 0}%", "AVG pass_rate", bg3)
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
        SELECT teacher_name, faculty, term, pass_rate, avg_grade, attendance, n
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

# ===== Teachers =====
with tab3:
    st.subheader("O‘qituvchi performansi")
    tp_all = run_sql_cached(f"""
        SELECT teacher_name, faculty, term,
               ROUND(pass_rate*100,1) AS pass_pct,
               ROUND(avg_grade,2) AS avg_grade,
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
               ROUND(AVG(avg_grade),2) AS avg_grade,
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
    if not fin.empty:
        fig = px.line(fin, x="month", y="net", color="faculty", markers=True, title="Net (Revenue - Expense)")
        st.plotly_chart(fig, use_container_width=True)
    download_buttons(fin, "finance_monthly")

# ===== Admin =====  (faqat admin roliga ko‘rinadi)
with tab5:
    if current_role != "admin":
        st.info("Bu bo‘lim faqat adminlar uchun.")
    else:
        st.subheader("🛠️ Admin paneli")
        cA, cB, cC = st.columns([1,1,2])
        if cA.button("🧹 Keshni tozalash", use_container_width=True):
            invalidate_cache()
            st.success("Kesh tozalandi.")
        st.caption("Eslatma: Cloudda ETL ishga tushirish odatda tavsiya qilinmaydi (fayl ruxsatlari, vaqt limiti).")

        if ALLOW_ETL:
            st.warning("ALLOW_ETL_CLOUD=1 bo‘lgani uchun (tajribaviy) ETL tugmalari paydo bo‘ladi.")
            import sys, subprocess
            etl_path = st.text_input("ETL (main.py) yo‘li", value="main.py")
            data_dir = st.text_input("Data papka", value="./data")
            dsn_ui = st.text_input("DSN (.env/DB_DSN bo‘sh bo‘lsa ishlatiladi)", value=DB_DSN)
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
        else:
            st.info("Cloud muhiti: ETL tugmasi o‘chirib qo‘yilgan. Lokal ishga tushiring.")

# ===== Rektor ===== (rektor va admin ko‘radi)
with tab6:
    if current_role not in ("rektor", "admin"):
        st.info("Bu bo‘lim faqat rektor (va admin) uchun.")
    else:
        st.subheader("🏛️ Rektor ko‘rinishi — Topline metrikalar")
        topline = run_sql_cached(f"""
            WITH ss AS (
              SELECT faculty,
                     ROUND(AVG(avg_gpa)::numeric,2) AS avg_gpa,
                     ROUND(AVG(pass_rate)*100,1) AS pass_pct,
                     ROUND(AVG(attendance_avg)*100,1) AS att_pct,
                     SUM(students)::bigint AS students
              FROM {VIEW_SS}
              {where_sql}
              GROUP BY faculty
            ),
            fin AS (
              SELECT faculty,
                     SUM(net) AS net_sum
              FROM {VIEW_FN}
              GROUP BY faculty
            )
            SELECT ss.faculty, ss.students, ss.avg_gpa, ss.pass_pct, ss.att_pct, COALESCE(fin.net_sum,0) AS net
            FROM ss
            LEFT JOIN fin USING(faculty)
            ORDER BY ss.faculty;
        """, params=params)
        st.dataframe(topline, use_container_width=True, height=420)
        if not topline.empty:
            c1, c2 = st.columns(2)
            c1.plotly_chart(px.bar(topline, x="faculty", y="students", title="Talabalar soni — fakultetlar kesimida"),
                            use_container_width=True)
            c2.plotly_chart(px.bar(topline, x="faculty", y="net", title="Net (oylar yig‘indisi) — fakultetlar kesimida"),
                            use_container_width=True)
        download_buttons(topline, "rektor_topline")
