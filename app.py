# app.py — AI-Rektor (Auth + Admin + Rektor, Uzbek UI)
# -*- coding: utf-8 -*-

import os
from typing import Dict, Any, Tuple
from io import BytesIO
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import create_engine, text
import streamlit_authenticator as stauth  # 0.4.2

# =========================
# Konfiguratsiya / Secrets
# =========================
st.set_page_config(page_title="🎓 AI-Rektor Dashboard", layout="wide", initial_sidebar_state="expanded")

DB_DSN    = st.secrets.get("DB_DSN", os.getenv("DB_DSN", "postgresql://postgres:7778@localhost:5432/Start_Up"))
DB_SCHEMA = st.secrets.get("DB_SCHEMA", os.getenv("DB_SCHEMA", "ai_rektor"))
CACHE_TTL = int(st.secrets.get("CACHE_TTL_SEC", os.getenv("CACHE_TTL_SEC", "300")))
ALLOW_ETL = str(st.secrets.get("ALLOW_ETL_CLOUD", os.getenv("ALLOW_ETL_CLOUD", "False"))).lower() in ("1", "true", "yes")

# Materialized Viewlar mavjud bo‘lmasa — View’ga tushamiz
USE_MV = True
VIEW_SS = "mv_student_success" if USE_MV else "vw_student_success"
VIEW_TP = "mv_teacher_perf"   if USE_MV else "vw_teacher_perf"
VIEW_FN = "mv_fin_summary"    if USE_MV else "vw_fin_summary"

# =========================
# Auth konfiguratsiya
# =========================
AUTH_CREDENTIALS = st.secrets.get("auth", {})
USERS = AUTH_CREDENTIALS.get("users", [])
if not USERS:
    USERS = [{
        "name": "Admin",
        "username": "admin",
        "email": "admin@uni.uz",
        "password": "admin",  # auto_hash=True uchun
        "role": "admin",
    }]

CREDS = {"usernames": {}}
ROLES_MAP = {}
for u in USERS:
    uname = u.get("username", "")
    if not uname:
        continue
    CREDS["usernames"][uname] = {
        "name": u.get("name", uname),
        "email": u.get("email", ""),
        "password": u.get("password", "admin"),
    }
    ROLES_MAP[uname] = u.get("role", "teacher")

COOKIE_NAME = AUTH_CREDENTIALS.get("cookie_name", "ai_rektor_auth")
COOKIE_KEY  = AUTH_CREDENTIALS.get("cookie_key", "supersecret")
EXPIRY_DAYS = int(AUTH_CREDENTIALS.get("expiry_days", 7))

authenticator = stauth.Authenticate(
    credentials=CREDS,
    cookie_name=COOKIE_NAME,
    key=COOKIE_KEY,
    cookie_expiry_days=EXPIRY_DAYS,
    auto_hash=True,
)

# =========================
# LOGIN — main sahifada
# =========================
st.title("🎓 AI-Rektor Dashboard")
st.caption(f"auth={getattr(stauth,'__version__','?')} · py={os.sys.version.split()[0]}")

try:
    name, auth_status, username = authenticator.login("Kirish", "main", key="login_form")
except TypeError:
    # fallback (ba’zi minor versiyalar param nomini talab qiladi)
    name, auth_status, username = authenticator.login(form_name="Kirish", location="main", key="login_form")

if auth_status is False:
    st.error("Login yoki parol noto‘g‘ri.")
    st.stop()
elif auth_status is None:
    st.info("Iltimos, tizimga kiring.")
    st.stop()

authenticator.logout("Chiqish", "sidebar", key="logout_btn")

role = ROLES_MAP.get(username, "teacher")
st.caption(f"👤 {name} · rol: **{role}** · schema: `{DB_SCHEMA}` · cache: {CACHE_TTL}s")

# =========================
# Stil / CSS
# =========================
DARK_CSS = """
<style>
.block-container { padding-top: 1rem !important; }
.kpi {
  background: var(--background-secondary);
  border: 1px solid rgba(148,163,184,.25);
  border-radius: 16px;
  padding: 14px 16px;
}
.kpi h4 { margin: 0 0 6px 0; font-size: 12px; color: #7c8aa0; }
.kpi .val { font-size: 26px; font-weight: 800; }
.kpi .sub { font-size: 12px; color: #94a3b8; }
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)

# =========================
# DB ulanish
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
        df.to_excel(writer, sheet_name="data", index=False)
    c2.download_button("⬇️ Excel", xlsx.getvalue(), f"{base_name}.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       use_container_width=True)

# =========================
# Filtrlar
# =========================
st.sidebar.header("⚙️ Filtrlar")

def safe_terms():
    try:
        return run_sql_cached(f"SELECT DISTINCT term FROM {VIEW_SS} ORDER BY term;")["term"].tolist()
    except Exception:
        return []

terms = safe_terms()
term = st.sidebar.selectbox("Term", ["Barchasi"] + terms, index=0)
if term != "Barchasi":
    facs = run_sql_cached(f"SELECT DISTINCT faculty FROM {VIEW_SS} WHERE term=:t ORDER BY faculty;", {"t": term})
else:
    facs = run_sql_cached(f"SELECT DISTINCT faculty FROM {VIEW_SS} ORDER BY faculty;")

faculty = st.sidebar.selectbox("Fakultet", ["Barchasi"] + facs["faculty"].tolist() if not facs.empty else [])
row_limit = st.sidebar.slider("Jadval limiti", 50, 3000, 300, 50)

def where_clause(term_: str, faculty_: str):
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
def kpi_card(title, val, sub):
    st.markdown(
        f"<div class='kpi'><h4>{title}</h4><div class='val'>{val}</div><div class='sub'>{sub}</div></div>",
        unsafe_allow_html=True
    )

# =========================
# Tabs
# =========================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Overview", "🎓 Students", "👩‍🏫 Teachers", "💼 Finance", "🛠️ Admin", "🏛️ Rektor"])

# ===== Overview =====
with tab1:
    kpdf = run_sql_cached(f"""
    SELECT
      SUM(students)::bigint AS students,
      ROUND(AVG(avg_gpa)::numeric, 2) AS avg_gpa,
      ROUND(AVG(pass_rate)*100, 1) AS pass_pct,
      ROUND(AVG(attendance_avg)*100, 1) AS att_pct
    FROM {VIEW_SS}
    {where_sql}
    """, params=params)
    k = kpdf.iloc[0] if not kpdf.empty else {"students":0,"avg_gpa":0,"pass_pct":0,"att_pct":0}

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card("Talabalar", f"{int(k['students'] or 0):,}", "Jami")
    with c2: kpi_card("O‘rtacha GPA", f"{k['avg_gpa'] or 0}", "Filtrga qarab")
    with c3: kpi_card("O‘tish darajasi", f"{k['pass_pct'] or 0}%", "")
    with c4: kpi_card("Davomat", f"{k['att_pct'] or 0}%", "")

# ===== Admin =====
with tab5:
    st.subheader("Admin")
    if st.button("🧹 Keshni tozalash"):
        invalidate_cache()
        st.success("Kesh tozalandi.")

    try:
        ping = run_sql("SELECT current_user, current_database(), current_schemas(true);")
        st.dataframe(ping, use_container_width=True)
    except Exception as e:
        st.error(f"DB ulanishda xato: {e}")
