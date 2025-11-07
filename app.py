# app.py — AI-Rektor (Streamlit Cloud uchun)
import os
from typing import Dict, Any, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import create_engine, text

# Auth
import streamlit_authenticator as stauth

# PDF eksport (oddiy jadval PDF)
from io import BytesIO
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# -------------------------
# Konfiguratsiya / Secrets
# -------------------------
# Streamlit Cloud: Secrets bo‘limida saqlanadi
DB_DSN = st.secrets.get("DB_DSN", os.getenv("DB_DSN", "postgresql://postgres:7778@localhost:5432/Start_Up"))
DB_SCHEMA = st.secrets.get("DB_SCHEMA", os.getenv("DB_SCHEMA", "ai_rektor"))
CACHE_TTL = int(st.secrets.get("CACHE_TTL_SEC", os.getenv("CACHE_TTL_SEC", "300")))
ALLOW_ETL = bool(st.secrets.get("ALLOW_ETL_CLOUD", False))  # Cloud’da ETL tugmasi odatda OFF

# Auth konfiguratsiya
AUTH_CREDENTIALS = st.secrets.get("auth", {})  # secrets.toml ichidagi [auth] bo‘limi
# Tuzilma: 
# [auth]
# cookie_name = "ai_rektor_auth"
# cookie_key = "supersecret"
# expiry_days = 7
# [[auth.users]]
# name = "Admin"
# username = "admin"
# email = "admin@uni.uz"
# password = "$2b$12$...."        # bcrypt hash
# role = "admin"
# [[auth.users]]
# name = "Dekan"
# username = "dean"
# email = "dean@uni.uz"
# password = "$2b$12$...."
# role = "dekan"
# ...

def _prepare_auth():
    """streamlit-authenticator-ga mos shaklga o‘tkazamiz."""
    users = AUTH_CREDENTIALS.get("users", [])
    names = [u.get("name", u.get("username", "User")) for u in users]
    usernames = [u.get("username") for u in users]
    passwords = [u.get("password") for u in users]  # bu allaqachon bcrypt hash bo‘lishi kerak
    emails = [u.get("email", "") for u in users]
    roles = [u.get("role", "teacher") for u in users]

    # authenticator config
    creds = {"usernames": {}}
    for i, uname in enumerate(usernames):
        creds["usernames"][uname] = {
            "name": names[i],
            "email": emails[i],
            "password": passwords[i],
            "role": roles[i],
        }

    cookie_name = AUTH_CREDENTIALS.get("cookie_name", "ai_rektor_auth")
    cookie_key  = AUTH_CREDENTIALS.get("cookie_key",  "please_change_me")
    expiry_days = int(AUTH_CREDENTIALS.get("expiry_days", 7))

    authenticator = stauth.Authenticate(
        credentials=creds,
        cookie_name=cookie_name,
        key=cookie_key,
        cookie_expiry_days=expiry_days,
    )
    return authenticator

# -------------------------
# UI umumiy sozlamalar
# -------------------------
st.set_page_config(
    page_title="AI-Rektor Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Appning umumiy ranglari (light/dark ikkala rejimga mos)
PRIMARY = "#2563eb"   # ko‘k
SOFT_BG = "#f8fafc"
CARD_BORDER = "#e5e7eb"
TEXT_MUTED = "#64748b"
TEXT_DARK = "#0f172a"

def kpi_color(value: float, good: float, warn: float, reverse=False) -> str:
    if value is None:
        return "#f1f5f9"
    v = float(value)
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
        <div style="background:{bg};padding:16px;border-radius:14px;border:1px solid {CARD_BORDER}">
            <div style="font-size:12px;color:{TEXT_MUTED};margin-bottom:6px">{title}</div>
            <div style="font-size:28px;font-weight:700;color:{TEXT_DARK}">{value_str}</div>
            <div style="font-size:12px;color:{TEXT_MUTED}">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# -------------------------
# DB ulanish (cached)
# -------------------------
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

# -------------------------
# SQL helperlar (cached)
# -------------------------
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
    c1, c2, c3 = st.columns(3)
    # CSV
    ccsv = df.to_csv(index=False).encode("utf-8")
    c1.download_button("⬇️ CSV", ccsv, f"{base_name}.csv", "text/csv", use_container_width=True)
    # Excel (openpyxl/XlsxWriter orqali)
    xlsx_bytes = BytesIO()
    with pd.ExcelWriter(xlsx_bytes, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="data", index=False)
    c2.download_button("⬇️ Excel", xlsx_bytes.getvalue(), f"{base_name}.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       use_container_width=True)
    # PDF (reportlab bilan oddiy jadval)
    pdf_bytes = df_to_pdf_bytes(df, title=base_name)
    c3.download_button("⬇️ PDF", pdf_bytes, f"{base_name}.pdf", "application/pdf", use_container_width=True)

def df_to_pdf_bytes(df: pd.DataFrame, title: str = "Hisobot") -> bytes:
    buff = BytesIO()
    doc = SimpleDocTemplate(buff, pagesize=landscape(A4), leftMargin=24, rightMargin=24, topMargin=24, bottomMargin=24)
    styles = getSampleStyleSheet()
    elems = []
    elems.append(Paragraph(title, styles["Title"]))
    elems.append(Spacer(1, 12))
    # Jadval
    data = [df.columns.tolist()] + df.astype(str).values.tolist()
    table = Table(data, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#e2e8f0")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.HexColor("#0f172a")),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,0), 9),
        ("FONTSIZE", (0,1), (-1,-1), 8),
        ("BOTTOMPADDING", (0,0), (-1,0), 8),
        ("GRID", (0,0), (-1,-1), 0.25, colors.HexColor("#cbd5e1")),
    ]))
    elems.append(table)
    doc.build(elems)
    return buff.getvalue()

# -------------------------
# AUTH — login / logout
# -------------------------
authenticator = _prepare_auth()
if "authentication_status" not in st.session_state:
    st.session_state.authentication_status = None

with st.sidebar:
    st.image("https://img.icons8.com/external-flaticons-lineal-color-flat-icons/64/000000/external-university-college-major-flaticons-lineal-color-flat-icons.png", width=48)
    st.markdown("### 🎓 AI-Rektor")
    name, auth_status, username = authenticator.login("Kirish", "sidebar")
    if auth_status is False:
        st.error("Login yoki parol xato.")
    elif auth_status is None:
        st.info("Login va parolni kiriting.")
    else:
        st.success(f"Xush kelibsiz, **{name}**!")
        authenticator.logout("Chiqish", "sidebar")

# Rolni aniqlash
def current_role() -> str:
    try:
        if st.session_state.get("authentication_status"):
            u = authenticator._credentials["usernames"].get(username, {})
            return u.get("role", "teacher")
    except Exception:
        pass
    return "guest"

role = current_role()
if st.session_state.get("authentication_status") is not True:
    st.stop()

# -------------------------
# Constants: MV yoki VIEW
# -------------------------
USE_MV = True  # MV mavjud bo‘lsa tezroq
VIEW_SS = "mv_student_success" if USE_MV else "vw_student_success"
VIEW_TP = "mv_teacher_perf"   if USE_MV else "vw_teacher_perf"
VIEW_FN = "mv_fin_summary"    if USE_MV else "vw_fin_summary"

# -------------------------
# Sarlavha va xabarnoma
# -------------------------
st.markdown(
    f"""
    <div style="background:{SOFT_BG};padding:16px 20px;border-radius:14px;border:1px solid {CARD_BORDER};margin-bottom:12px">
        <div style="display:flex;align-items:center;gap:10px;">
            <span style="font-size:28px">🎓</span>
            <div>
                <div style="font-size:22px;font-weight:700;color:{TEXT_DARK}">AI-Rektor — boshqaruv paneli</div>
                <div style="color:{TEXT_MUTED};font-size:13px">
                    Ma’lumotlar bazasi: <b>{DB_SCHEMA}</b> • Cache TTL: <b>{CACHE_TTL}s</b> • Manba: <b>{'MV' if USE_MV else 'VIEW'}</b> • Rol: <b>{role}</b>
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# -------------------------
# Sidebar — filtrlar
# -------------------------
st.sidebar.header("⚙️ Filtrlar")
try:
    terms_df = run_sql_cached(f"SELECT DISTINCT term FROM {VIEW_SS} ORDER BY term;")
except Exception:
    # Agar MV hali yo‘q bo‘lsa VIEWga qaytamiz
    VIEW_SS = "vw_student_success"
    VIEW_TP = "vw_teacher_perf"
    VIEW_FN = "vw_fin_summary"
    terms_df = run_sql_cached(f"SELECT DISTINCT term FROM {VIEW_SS} ORDER BY term;")

term = st.sidebar.selectbox("Term", ["Barchasi"] + terms_df["term"].tolist(), index=0)

if term != "Barchasi":
    facs_df = run_sql_cached(f"SELECT DISTINCT faculty FROM {VIEW_SS} WHERE term=:t ORDER BY faculty;", {"t": term})
else:
    facs_df = run_sql_cached(f"SELECT DISTINCT faculty FROM {VIEW_SS} ORDER BY faculty;")

faculty = st.sidebar.selectbox("Fakultet", ["Barchasi"] + facs_df["faculty"].tolist(), index=0)
row_limit = st.sidebar.slider("Jadval limiti", 50, 3000, 300, 50)
risk_att = st.sidebar.slider("Risk — Davomat (%)", 50, 90, 75, 1)
risk_grd = st.sidebar.slider("Risk — O‘rtacha baho", 40, 100, 60, 1)

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

# -------------------------
# Tabs
# -------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Umumiy", "🎓 Talabalar", "👩‍🏫 O‘qituvchilar", "💼 Moliya", "🛠️ Admin"])

# ===== Umumiy =====
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
    with c1: kpi_card("Talabalar", f"{int(k['students'] or 0):,}", "Jami talaba", kpi_color(k["students"] or 0, 1500, 800))
    with c2: kpi_card("O‘rtacha GPA", f"{k['avg_gpa'] or 0}", "Filtrlarga bog‘liq", kpi_color(k["avg_gpa"] or 0, 3.0, 2.5))
    with c3: kpi_card("O‘tish ko‘rsatkichi", f"{k['pass_pct'] or 0}%", "AVG pass_rate", kpi_color(k["pass_pct"] or 0, 85, 70))
    with c4: kpi_card("Davomat", f"{k['att_pct'] or 0}%", "AVG attendance", kpi_color(k["att_pct"] or 0, 85, 75))

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
                 title="O‘tish ko‘rsatkichi (%) — term × fakultet", barmode="group")
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
    download_buttons(tp, "top_o‘qituvchilar")

# ===== Talabalar =====
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
    download_buttons(ss, "talabalar_kesim")

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
    download_buttons(risk, "riskli_talabalar")

# ===== O‘qituvchilar =====
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
    download_buttons(tp_all, "o‘qituvchi_perf")

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
                           title="O‘tish ko‘rsatkichi (%) — o‘rtacha, fakultet bo‘yicha"),
                    use_container_width=True)
    c2.plotly_chart(px.bar(tchart, x="faculty", y="avg_grade",
                           title="O‘rtacha baho — fakultet bo‘yicha"),
                    use_container_width=True)

    st.divider()
    st.markdown("### 👆 O‘qituvchi bo‘yicha drilldown")
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
        download_buttons(det, f"o‘qituvchi_det_{t_sel.replace(' ','_')}")

# ===== Moliya =====
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
    download_buttons(fin, "moliya_oymaoy")

# ===== Admin (faqat admin & dekan) =====
with tab5:
    if role not in ("admin", "dekan"):
        st.warning("Bu bo‘lim faqat admin/dekan uchun.")
        st.stop()

    st.subheader("Admin / ETL / Kesh")
    cA, cB, cC = st.columns([1,1,2])
    if cA.button("🧹 Keshni tozalash", use_container_width=True):
        invalidate_cache()
        st.success("Kesh tozalandi.")

    st.caption("Cloud-da ETL ishga tushirish odatda o‘chirib qo‘yiladi. Kerak bo‘lsa ALLOW_ETL_CLOUD = true.")
    if ALLOW_ETL:
        st.info("ETL tugmasi yoqilgan. `main.py` ichida Neon DSN bilan ishlaydi.")
        import subprocess, sys
        etl_path = cB.text_input("ETL (main.py) yo‘li", value="main.py")
        data_dir = cC.text_input("Data papka", value="./data")
        dsn_ui = st.text_input("DSN (.secrets DB_DSN bo‘sh bo‘lsa)", value=DB_DSN)
        run_cols = st.columns([1,1,1,2])
        run_once = run_cols[0].button("🔄 Run ETL (once)", use_container_width=True)
        refresh_mv = run_cols[1].checkbox("MV refresh", value=True)
        show_cmd = run_cols[2].checkbox("Buyruqni ko‘rsat", value=False)

        if run_once:
            cmd = [sys.executable, etl_path, "--dsn", dsn_ui, "--data-dir", data_dir, "--once", "--log-level", "INFO"]
            if refresh_mv: cmd.append("--refresh-mv")
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
        st.info("ETL tugmasi o‘chirilgan. (ALLOW_ETL_CLOUD=false) — faqat lokal serverda ishlatish tavsiya qilinadi.")
