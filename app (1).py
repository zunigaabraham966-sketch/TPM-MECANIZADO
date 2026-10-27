import numpy as np, pandas as pd, plotly.express as px, plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(page_title="Dashboard TPM · OEE", page_icon="🏭", layout="wide")

# ───────────── Parámetros base (de tu Excel) ─────────────
REAL = [78,250,193,168,97,90,46,76,80,230,114,195,70,70,47,105,145,90,130,75,
        160,85,125,110,155,95,140,80,150,151]          # producción real días 1-30
TEO, TP, CT = 400, 480, 1.2                            # pzas teóricas/día, min planificados, min/pza
# distribución de la brecha usada en tu base: fallas, setup, defectos, otros
SH = dict(f=.4834, s=.0687, d=.3479, o=.10)
# máquina: (frac. fallas, frac. setup, frac. velocidad/otros, tasa defectos, vib0, temp0, amp0)
MAQ = {"M1 · GP4 (base real)": (.34,.048,.07,.45,4.5,68,32),
       "M2 · Torno CNC":       (.12,.05,.06,.04,3.0,62,28),
       "M3 · Fresadora":       (.09,.07,.05,.06,2.6,58,24),
       "M4 · Prensa":          (.15,.04,.08,.03,3.4,66,40)}
CAUSAS = ["Eléctrica","Mecánica (rodamientos/transmisión)","Hidráulica/Neumática",
          "Sensores/Control","Falta de lubricación","Desgaste de herramental","Error de operación"]
SETUPS = ["Cambio de formato","Ajuste de herramienta","Calibración"]
VIB_ALM, TEMP_ALM = 7.0, 85.0
SENS = {"Vibración (mm/s)":"vib","Temperatura (°C)":"temp","Corriente (A)":"amp"}

# ───────────── Simulación (reemplazable por tus datos reales) ─────────────
@st.cache_data
def make_data(tpm: bool):
    rng = np.random.default_rng(7)
    days = pd.date_range("2026-04-01", "2026-09-30"); n = len(days); g = 1 if tpm else 0
    D, E, T = [], [], []
    for mi, (m, (f, s, o, dr, *_)) in enumerate(MAQ.items()):
        w = rng.dirichlet(np.ones(len(CAUSAS)) * 3)
        for i, day in enumerate(days):
            p = i / (n - 1)
            real = (mi == 0 and i < 30)
            if real:
                gap = TEO - REAL[i]; good = REAL[i]
                fal, stp, oth, dfc = SH["f"]*gap*CT, SH["s"]*gap*CT, SH["o"]*gap*CT, SH["d"]*gap
            else:
                stp = TP*s*(1-.3*g*p)*rng.gamma(6, 1/6)
                oth = TP*o*(1-.3*g*p)*rng.gamma(6, 1/6)
                fal = TP*f*(1-.55*g*p)*rng.gamma(4, .25)
                fal = max(0, min(fal, TP - stp - oth - 10))
                tot = (TP - fal - stp - oth) / CT
                dfc = tot * min(.9, dr*(1-.6*g*p)*rng.gamma(5, .2)); good = tot - dfc
            nf = 0 if fal < 1 else 1 + rng.poisson(fal / 40)
            D.append(dict(fecha=day, maq=m, plan=TP, teorica=TEO, falla=fal, setup=stp, otros=oth,
                          defectos=dfc, buenas=good, total=good+dfc, n_fallas=nf, dato="Real" if real else "Simulado"))
            for mins in (rng.dirichlet(np.ones(nf)) * fal if nf else []):
                E.append((day, m, "Falla", rng.choice(CAUSAS, p=w), mins))
            for mins in rng.dirichlet(np.ones(k := 1+rng.poisson(.8))) * stp:
                E.append((day, m, "Setup", rng.choice(SETUPS), mins))
    D = pd.DataFrame(D); E = pd.DataFrame(E, columns=["fecha","maq","tipo","causa","min"])
    # sensores: suben antes/durante las fallas
    st_ = D.falla / D.plan
    st2 = .5*st_ + .5*st_.groupby(D.maq).shift(-1).fillna(st_)
    base = pd.DataFrame(MAQ, index=["f","s","o","d","v","t","a"]).T
    v0, t0, a0 = D.maq.map(base.v), D.maq.map(base.t), D.maq.map(base.a)
    D["vib"]  = v0*(1+1.6*st2) + rng.normal(0, .3, len(D))
    D["temp"] = t0 + 30*st2 + rng.normal(0, 2, len(D))
    D["amp"]  = a0*(1+.5*st2) + rng.normal(0, 1.2, len(D))
    for m in MAQ:
        for mi in range(6):
            p = mi / 5; plan = int(rng.integers(10, 15)); det = int(rng.poisson(22 - 9*g*p))
            T.append(dict(mes=pd.Timestamp(2026, 4+mi, 1), maq=m, pm_plan=plan,
                          pm_ok=int(rng.binomial(plan, .68+.24*g*p)), tarj_det=det,
                          tarj_cer=int(rng.binomial(det, .55+.35*g*p)),
                          cli=float(np.clip(.62+.3*g*p+rng.normal(0, .03), 0, 1))))
    return D, E, pd.DataFrame(T)

SUMC = ["plan","falla","setup","otros","defectos","buenas","total","teorica","n_fallas"]
def agg(df, keys):
    g = df.groupby(keys)[SUMC].sum().reset_index()
    op = g.plan - g.falla - g.setup
    g["Disponibilidad"] = op / g.plan
    g["Rendimiento"] = g.total*CT / op.replace(0, np.nan)
    g["Calidad"] = g.buenas / g.total.replace(0, np.nan)
    g["OEE"] = g.Disponibilidad * g.Rendimiento * g.Calidad
    g["MTBF (h)"] = op / g.n_fallas.replace(0, np.nan) / 60
    g["MTTR (min)"] = g.falla / g.n_fallas.replace(0, np.nan)
    return g

def pct(fig, axis="y"):
    getattr(fig, f"update_{axis}axes")(tickformat=".0%"); return fig

def pareto(names, vals, title, unit):
    o = np.argsort(vals)[::-1]; names = [names[i] for i in o]; vals = np.array(vals)[o]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=names, y=vals, name=unit, marker_color="#2E75B6")
    fig.add_scatter(x=names, y=vals.cumsum()/max(vals.sum(), 1e-9), name="% acumulado",
                    mode="lines+markers", marker_color="#C00000", secondary_y=True)
    fig.update_layout(title=title, height=400, legend=dict(orientation="h", y=-.3))
    fig.update_yaxes(tickformat=".0%", range=[0, 1.05], secondary_y=True)
    return fig

# ───────────── Sidebar ─────────────
st.sidebar.title("🏭 Filtros")
esc = st.sidebar.radio("Escenario simulado (meses 2-6)", ["Con TPM (mejora gradual)", "Línea base (sin TPM)"])
D, E, T = make_data(esc.startswith("Con"))
maqs = st.sidebar.multiselect("Máquinas", list(MAQ), default=list(MAQ))
rango = st.sidebar.date_input("Rango de fechas", (D.fecha.min().date(), D.fecha.max().date()),
                              min_value=D.fecha.min().date(), max_value=D.fecha.max().date())
gran = st.sidebar.selectbox("Agrupar por", ["Diario", "Semanal", "Mensual"], index=1)
meta = st.sidebar.slider("Meta OEE", 30, 95, 85, 5) / 100
causas = st.sidebar.multiselect("Causas de falla (tab Paradas)", CAUSAS, default=CAUSAS)
if len(rango) != 2 or not maqs:
    st.warning("Selecciona un rango de fechas completo y al menos una máquina."); st.stop()
a, b = pd.Timestamp(rango[0]), pd.Timestamp(rango[1])

d = D[D.maq.isin(maqs) & D.fecha.between(a, b)].copy()
d["periodo"] = d.fecha.dt.to_period({"Diario":"D","Semanal":"W","Mensual":"M"}[gran]).dt.start_time
d["mes"] = d.fecha.dt.strftime("%Y-%m")
ev = E[E.maq.isin(maqs) & E.fecha.between(a, b)]
evf = ev[(ev.tipo == "Falla") & ev.causa.isin(causas)]
nd = (b - a).days + 1
prev = D[D.maq.isin(maqs) & (D.fecha >= a - pd.Timedelta(days=nd)) & (D.fecha < a)]

# ───────────── Encabezado y KPIs ─────────────
st.title("Dashboard TPM · Eficiencia Global de los Equipos (OEE)")
st.caption("Días 1-30 de M1 = tu base (producción real; motivos simulados en tu Excel). "
           "Resto de días, máquinas, paros, sensores y pilares TPM = **datos simulados**.")
k = agg(d.assign(g=1), "g").iloc[0]
kp = agg(prev.assign(g=1), "g").iloc[0] if len(prev) else None
dl = lambda c, rel=False: None if kp is None or pd.isna(kp[c]) else (
    f"{(k[c]/kp[c]-1)*100:+.1f}%" if rel else f"{(k[c]-kp[c])*100:+.1f} pp")
c = st.columns(6)
c[0].metric("OEE", f"{k.OEE:.1%}", dl("OEE")); c[1].metric("Disponibilidad", f"{k.Disponibilidad:.1%}", dl("Disponibilidad"))
c[2].metric("Rendimiento", f"{k.Rendimiento:.1%}", dl("Rendimiento")); c[3].metric("Calidad", f"{k.Calidad:.1%}", dl("Calidad"))
c[4].metric("MTBF (h)", f"{k['MTBF (h)']:.1f}", dl("MTBF (h)", True))
c[5].metric("MTTR (min)", f"{k['MTTR (min)']:.0f}", dl("MTTR (min)", True), delta_color="inverse")
st.progress(float(np.clip(k.OEE / meta, 0, 1)), text=f"OEE vs meta {meta:.0%} · brecha {(meta-k.OEE)*100:.1f} pp "
            f"· delta = vs período anterior de igual duración")

t1, t2, t3, t4, t5, t6 = st.tabs(["📈 OEE", "🧮 6 grandes pérdidas", "🔧 Paradas y fiabilidad",
                                  "📡 Sensores", "🏅 Pilares TPM", "🗂️ Datos"])

with t1:
    tr = agg(d, ["periodo"])
    fig = px.line(tr, x="periodo", y=["OEE","Disponibilidad","Rendimiento","Calidad"], markers=gran != "Diario",
                  title=f"Evolución {gran.lower()} de OEE y factores")
    fig.add_hline(y=meta, line_dash="dash", line_color="red", annotation_text="Meta OEE")
    st.plotly_chart(pct(fig), use_container_width=True)
    l, r = st.columns(2)
    pm = agg(d, ["maq"]).melt("maq", ["Disponibilidad","Rendimiento","Calidad","OEE"], "Indicador", "valor")
    l.plotly_chart(pct(px.bar(pm, x="maq", y="valor", color="Indicador", barmode="group",
                              title="Indicadores por máquina")), use_container_width=True)
    hm = agg(d, ["maq", "mes"]).pivot(index="maq", columns="mes", values="OEE")
    r.plotly_chart(px.imshow(hm, text_auto=".0%", color_continuous_scale="RdYlGn", zmin=0, zmax=1,
                             aspect="auto", title="OEE por máquina y mes"), use_container_width=True)

# pérdidas en minutos
LOSS = {"1 · Fallas de equipo": d.falla.sum(), "2 · Preparación y ajustes": d.setup.sum(),
        "3 · Paradas menores/microparos": .55*d.otros.sum(), "4 · Velocidad reducida": .45*d.otros.sum(),
        "5 · Defectos y retrabajo": .85*d.defectos.sum()*CT, "6 · Pérdidas de arranque": .15*d.defectos.sum()*CT}
with t2:
    st.caption("Mapeo TPM: fallas y setup = tal cual tu base; 'otros' se reparte 55% microparos / 45% velocidad; "
               "defectos 85% proceso / 15% arranque (supuesto, ajustable en el código).")
    l, r = st.columns(2)
    wf = go.Figure(go.Waterfall(x=["Tiempo planificado"] + list(LOSS) + ["Tiempo valioso"],
        measure=["absolute"] + ["relative"]*6 + ["total"],
        y=[d.plan.sum()/60] + [-v/60 for v in LOSS.values()] + [d.buenas.sum()*CT/60],
        text=[f"{d.plan.sum()/60:,.0f}"] + [f"{v/60:,.0f}" for v in LOSS.values()] + [f"{d.buenas.sum()*CT/60:,.0f}"]))
    wf.update_layout(title="Cascada de pérdidas (horas)", height=420, showlegend=False)
    l.plotly_chart(wf, use_container_width=True)
    r.plotly_chart(pareto(list(LOSS), [v/60 for v in LOSS.values()], "Pareto de las 6 grandes pérdidas", "Horas"),
                   use_container_width=True)
    lp = d.groupby("periodo")[["falla","setup","otros","defectos","plan"]].sum()
    lp = pd.DataFrame({n: v for n, v in zip(LOSS, [lp.falla, lp.setup, .55*lp.otros, .45*lp.otros,
                                                   .85*lp.defectos*CT, .15*lp.defectos*CT])}).div(lp.plan, axis=0)
    st.plotly_chart(pct(px.bar(lp.reset_index().melt("periodo"), x="periodo", y="value", color="variable",
                    title="Pérdidas como % del tiempo planificado", labels={"value":"% tiempo","variable":""})),
                    use_container_width=True)

with t3:
    l, r = st.columns(2)
    cz = evf.groupby("causa")["min"].sum()
    if len(cz): l.plotly_chart(pareto(list(cz.index), list(cz.values/60), "Pareto de causas de falla", "Horas"),
                               use_container_width=True)
    mm = agg(d, ["mes"])
    f2 = make_subplots(specs=[[{"secondary_y": True}]])
    f2.add_bar(x=mm.mes, y=mm["MTBF (h)"], name="MTBF (h)", marker_color="#2E75B6")
    f2.add_scatter(x=mm.mes, y=mm["MTTR (min)"], name="MTTR (min)", mode="lines+markers",
                   marker_color="#C00000", secondary_y=True)
    f2.update_layout(title="MTBF y MTTR mensual", height=400, legend=dict(orientation="h", y=-.3))
    r.plotly_chart(f2, use_container_width=True)
    l, r = st.columns(2)
    l.plotly_chart(px.histogram(evf, x="min", nbins=30, title="Distribución de duración de paradas (min)"),
                   use_container_width=True)
    bm = agg(d, ["maq"])
    r.dataframe(bm[["maq","n_fallas","MTBF (h)","MTTR (min)"]].rename(columns={"maq":"Máquina","n_fallas":"N° fallas"})
                .round(1), hide_index=True, use_container_width=True)
    st.markdown(f"**Paradas por falla:** {len(evf):,} · **Horas perdidas:** {evf['min'].sum()/60:,.1f} · "
                f"**Setups:** {(ev.tipo=='Setup').sum():,}")

with t4:
    ms = st.selectbox("Máquina", maqs); sn = st.selectbox("Sensor", list(SENS)); col = SENS[sn]
    sd = d[d.maq == ms]; thr = {"vib": VIB_ALM, "temp": TEMP_ALM}.get(col)
    fg = px.line(sd, x="fecha", y=col, title=f"{sn} · {ms}")
    bad = sd[sd.falla > 120]
    fg.add_scatter(x=bad.fecha, y=bad[col], mode="markers", marker=dict(color="red", size=8), name="Día con falla >120 min")
    if thr: fg.add_hline(y=thr, line_dash="dash", line_color="orange", annotation_text="Umbral de alarma")
    st.plotly_chart(fg, use_container_width=True)
    l, r = st.columns(2)
    l.plotly_chart(px.scatter(sd, x=col, y="falla", title=f"{sn} vs minutos de falla",
                              labels={"falla": "Min de falla"}), use_container_width=True)
    r.metric("Correlación sensor ↔ falla", f"{sd[col].corr(sd.falla):.2f}")
    if thr: r.metric("Días sobre umbral de alarma", int((sd[col] > thr).sum()))
    r.caption("Un sensor con alta correlación es buen candidato para mantenimiento predictivo.")

with t5:
    tm = T[T.maq.isin(maqs) & (T.mes >= a.to_period("M").start_time) & (T.mes <= b)]
    tg = tm.groupby("mes")[["pm_plan","pm_ok","tarj_det","tarj_cer"]].sum().reset_index()
    pc = tm.pm_ok.sum() / max(tm.pm_plan.sum(), 1)
    c = st.columns(3)
    c[0].metric("Cumplimiento mant. planificado", f"{pc:.1%}")
    c[1].metric("Tarjetas rojas cerradas", f"{tm.tarj_cer.sum()/max(tm.tarj_det.sum(),1):.1%}")
    c[2].metric("Cumplimiento limpieza-inspección-lubricación", f"{tm.cli.mean():.1%}")
    l, r = st.columns(2)
    pp = tm.assign(c=tm.pm_ok / tm.pm_plan)
    l.plotly_chart(pct(px.line(pp, x="mes", y="c", color="maq", markers=True,
                  title="Cumplimiento del plan de mantenimiento preventivo")), use_container_width=True)
    r.plotly_chart(px.bar(tg.melt("mes", ["tarj_det","tarj_cer"]), x="mes", y="value", color="variable",
                  barmode="group", title="Mantenimiento autónomo: tarjetas rojas detectadas vs cerradas"),
                  use_container_width=True)
    st.plotly_chart(pct(px.line(tm, x="mes", y="cli", color="maq", markers=True,
                  title="Cumplimiento de estándares de limpieza, inspección y lubricación")), use_container_width=True)

with t6:
    st.write("Cada tabla se puede descargar como plantilla: reemplaza `make_data()` en `app.py` por la lectura de tus datos reales.")
    cols = st.columns(4)
    for cc, (nm, df) in zip(cols, [("diario", D), ("paradas", E), ("pilares_tpm", T)]):
        cc.download_button(f"⬇️ {nm}.csv", df.to_csv(index=False).encode("utf-8-sig"), f"{nm}.csv", "text/csv")
    st.dataframe(d.drop(columns=["periodo","mes"]).round(2), use_container_width=True, hide_index=True)
