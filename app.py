"""
app.py
======
Aplicativos interactivos para la tesis sobre valoración de opciones europeas
usando la distribución AGB2 (transformación afín de la GB2) como función de
densidad neutral al riesgo.

Se organizan en dos pestañas:
    1. Explorador de densidades GB2 / AGB2 y aproximación de una lognormal por MCE.
    2. Valoración de opciones europeas (call y put) según asimetría y curtosis.

Ejecutar localmente:   streamlit run app.py
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

import gb2

st.set_page_config(
    page_title="Tesis AGB2 · Valoración de opciones",
    page_icon="📈",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Encabezado
# ---------------------------------------------------------------------------
st.title("Distribución AGB2 y valoración de opciones europeas")
st.caption(
    "Aplicativos interactivos de la tesis: la distribución AGB2 (transformación "
    "afín de la GB2) como función de densidad neutral al riesgo."
)

tab_dens, tab_val = st.tabs(
    [
        "1 · Explorador de densidades",
        "2 · Valoración de opciones",
    ]
)


# ===========================================================================
# PESTAÑA 1 — Explorador de densidades
# ===========================================================================
with tab_dens:
    st.header("Explorador de densidades GB2 / AGB2")

    # -----------------------------------------------------------------------
    # PARTE 1 — Exploración libre de la densidad
    # -----------------------------------------------------------------------
    st.subheader("1 · Exploración de la densidad")
    st.write(
        "Ingresa manualmente los parámetros. Todos deben ser positivos, excepto "
        "$\\alpha$ (localización), que puede ser negativo. Los parámetros de forma "
        "$a$, $p$ y $q$ controlan la asimetría y la curtosis."
    )

    usar_agb2 = st.toggle("Usar transformación afín (AGB2)", value=False, key="toggle_agb2")

    # Parámetros en fila horizontal, con entrada manual (number_input)
    if usar_agb2:
        ca, cp, cq, cphi, calpha = st.columns(5)
        a = ca.number_input("a (forma)", min_value=0.01, value=2.0, step=0.1,
                            format="%.2f", key="dens_a")
        p = cp.number_input("p (forma)", min_value=0.01, value=1.5, step=0.1,
                            format="%.2f", key="dens_p")
        q = cq.number_input("q (forma)", min_value=0.01, value=3.0, step=0.1,
                            format="%.2f", key="dens_q")
        phi = cphi.number_input("φ (escala)", min_value=0.01, value=1.0, step=0.1,
                                format="%.2f", key="dens_phi")
        alpha = calpha.number_input("α (localización)", value=0.0, step=0.5,
                                    format="%.2f", key="dens_alpha")
        b = phi
    else:
        ca, cp, cq, cb = st.columns(4)
        a = ca.number_input("a (forma)", min_value=0.01, value=2.0, step=0.1,
                            format="%.2f", key="dens_a")
        p = cp.number_input("p (forma)", min_value=0.01, value=1.5, step=0.1,
                            format="%.2f", key="dens_p")
        q = cq.number_input("q (forma)", min_value=0.01, value=3.0, step=0.1,
                            format="%.2f", key="dens_q")
        b = cb.number_input("b (escala)", min_value=0.01, value=1.0, step=0.1,
                            format="%.2f", key="dens_b")
        alpha = 0.0
        phi = b

    # Momentos / coeficientes de forma
    if usar_agb2:
        mean, var, beta1, beta2 = gb2.agb2_central_moments(alpha, phi, a, p, q)
    else:
        mean, var, beta1, beta2 = gb2.gb2_central_moments(a, b, p, q)

    # Rango del eje
    try:
        lo = gb2.agb2_quantile(0.005, alpha, phi, a, p, q) if usar_agb2 \
            else gb2.gb2_quantile(0.005, a, b, p, q)
        hi = gb2.agb2_quantile(0.995, alpha, phi, a, p, q) if usar_agb2 \
            else gb2.gb2_quantile(0.995, a, b, p, q)
    except Exception:
        lo, hi = 0.0, 10.0
    lo = min(lo, alpha) if usar_agb2 else 0.0
    x = np.linspace(lo, hi, 600)

    if usar_agb2:
        pdf = gb2.agb2_pdf(x, alpha, phi, a, p, q)
        cdf = gb2.agb2_cdf(x, alpha, phi, a, p, q)
        etiqueta = f"AGB2(α={alpha:g}, φ={phi:g}, a={a:g}, p={p:g}, q={q:g})"
    else:
        pdf = gb2.gb2_pdf(x, a, b, p, q)
        cdf = gb2.gb2_cdf(x, a, b, p, q)
        etiqueta = f"GB2(a={a:g}, b={b:g}, p={p:g}, q={q:g})"

    col_pdf, col_cdf = st.columns(2)
    with col_pdf:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=pdf, name="Densidad", line=dict(width=3)))
        fig.update_layout(
            title=etiqueta, xaxis_title="x", yaxis_title="Densidad",
            height=360, margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)
    with col_cdf:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=x, y=cdf, name="CDF", line=dict(width=3),
                                  line_color="seagreen"))
        fig2.update_layout(
            title="Función de distribución acumulada",
            xaxis_title="x", yaxis_title="F(x)",
            height=360, margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Panel de coeficientes
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Media", f"{mean:.4f}" if not np.isnan(mean) else "no existe")
    m2.metric("Varianza", f"{var:.4f}" if not np.isnan(var) else "no existe")
    m3.metric("Asimetría β₁", f"{beta1:.4f}" if not np.isnan(beta1) else "no existe",
              help="Normal: 0")
    m4.metric("Curtosis β₂", f"{beta2:.4f}" if not np.isnan(beta2) else "no existe",
              help="Normal: 3")
    if a * q <= 4:
        st.warning(
            f"Con a·q = {a * q:.2f} ≤ 4, algunos momentos no existen "
            "(la GB2 solo tiene momentos de orden menor que a·q)."
        )

    st.divider()

    # -----------------------------------------------------------------------
    # PARTE 2 — Aproximación de una lognormal por una GB2
    # -----------------------------------------------------------------------
    st.subheader("2 · Aproximación de una lognormal mediante la GB2")
    st.write(
        "Propón una distribución lognormal (columna 1). Los parámetros de la GB2 "
        "asociada se **generan automáticamente** (columna 2) mediante la cadena de "
        "reparametrizaciones, de modo que la GB2 se aproxime a esa lognormal. "
        "La aproximación mejora cuando $a \\to 0$ y $q \\to \\infty$."
    )

    ccol1, ccol2, ccol3 = st.columns([1, 1, 2])

    # Columna 1: el usuario propone la lognormal
    with ccol1:
        st.markdown("**Lognormal propuesta**")
        mu_ln = st.number_input("μ (media log)", value=0.0, step=0.1,
                                format="%.2f", key="ln_mu")
        sigma_ln = st.number_input("σ (desviación log)", min_value=0.01,
                                   value=0.50, step=0.05, format="%.2f", key="ln_sigma")
        sigma2_ln = sigma_ln ** 2

    # Estimación MCE: se ajustan (a, p, q) igualando beta1, beta2, beta3 de la
    # lognormal, y b para reproducir su varianza. Se cachea para no recalcular
    # mientras el usuario no cambie mu o sigma2.
    @st.cache_data(show_spinner="Estimando GB2 por MCE...")
    def _estimar_mce(mu, s2):
        return gb2.mce_fit_from_lognormal(mu, s2)

    a_est, b_est, p_est, q_est, loss = _estimar_mce(mu_ln, sigma2_ln)
    b1_obj, b2_obj, b3_obj = gb2.lognormal_standardized_moments(sigma2_ln)

    # Columna 2: parámetros GB2 estimados (solo lectura)
    with ccol2:
        st.markdown("**GB2 estimada (MCE)**")
        if np.isnan(a_est):
            st.error("El método MCE no convergió para estos parámetros.")
        else:
            st.metric("a", f"{a_est:.4f}")
            st.metric("b", f"{b_est:.4g}")
            st.metric("p", f"{p_est:.4f}")
            st.metric("q", f"{q_est:.4f}")

    # Columna 3: gráfica comparativa (sin título)
    with ccol3:
        x_hi = np.exp(mu_ln + 3.5 * sigma_ln)
        x_lo = max(1e-4, np.exp(mu_ln - 3.5 * sigma_ln))
        xx = np.linspace(x_lo, x_hi, 600)

        ln_pdf = gb2.lognormal_pdf(xx, mu_ln, sigma2_ln)
        if not np.isnan(a_est):
            try:
                gb2_approx = gb2.gb2_pdf(xx, a_est, b_est, p_est, q_est)
            except Exception:
                gb2_approx = np.full_like(xx, np.nan)
        else:
            gb2_approx = np.full_like(xx, np.nan)

        figc = go.Figure()
        figc.add_trace(go.Scatter(
            x=xx, y=ln_pdf, name=f"Lognormal(μ={mu_ln:g}, σ={sigma_ln:g})",
            line=dict(width=3, color="royalblue")
        ))
        figc.add_trace(go.Scatter(
            x=xx, y=gb2_approx, name="GB2 estimada (MCE)",
            line=dict(width=2, dash="dash", color="crimson")
        ))
        figc.update_layout(
            xaxis_title="x", yaxis_title="Densidad",
            height=420, margin=dict(l=10, r=10, t=20, b=10),
            legend=dict(orientation="h", y=1.12),
        )
        st.plotly_chart(figc, use_container_width=True)

    # Momentos objetivo vs ajustados y calidad del ajuste
    if not np.isnan(a_est):
        b1_fit, b2_fit, b3_fit = gb2.gb2_standardized_moments(a_est, p_est, q_est)
        mm1, mm2, mm3 = st.columns(3)
        mm1.metric("β₁ (asimetría)", f"{b1_fit:.4f}",
                   delta=f"obj {b1_obj:.4f}", delta_color="off")
        mm2.metric("β₂ (curtosis)", f"{b2_fit:.4f}",
                   delta=f"obj {b2_obj:.4f}", delta_color="off")
        mm3.metric("β₃", f"{b3_fit:.4f}",
                   delta=f"obj {b3_obj:.4f}", delta_color="off")
        st.caption(
            f"Pérdida MCE (suma de cuadrados de las diferencias de momentos): "
            f"{loss:.2e}. Los parámetros de forma se estiman igualando "
            f"β₁, β₂, β₃; b reproduce la varianza de la lognormal."
        )


# ===========================================================================
# PESTAÑA 2 — Valoración de opciones
# ===========================================================================
with tab_val:
    st.header("Valoración de opciones europeas: asimetría y curtosis")
    st.write(
        "Ajusta manualmente la asimetría (β₁) y la curtosis (β₂) de la "
        "distribución neutral al riesgo. La AGB2 se calibra por MCE a esos "
        "momentos y se valoran call y put europeas. El eje horizontal es la "
        "*moneyness* $F/K$, con $F = S_t\\,e^{r\\tau}$; valores menores a 1 son "
        "OTM y mayores a 1 son ITM (para la call)."
    )

    # Parámetros del ejercicio (valores de la tesis, editables)
    st.markdown("**Parámetros del ejercicio**")
    p1, p2, p3, p4 = st.columns(4)
    K_ex = p1.number_input("Strike K", 1.0, 1000.0, 100.0, 1.0,
                           format="%.4f", key="K_ex")
    r_ex = p2.number_input("Tasa r", 0.0, 0.5, 0.10, 0.01,
                           format="%.4f", key="r_ex")
    tau_ex = p3.number_input("Vencimiento τ", 0.01, 5.0, 0.25, 0.05,
                             format="%.4f", key="tau_ex")
    sigma2_ex = p4.number_input("Varianza σ²", 0.01, 2.0, 0.40, 0.05,
                                format="%.4f", key="sigma2_ex")

    # Momentos de referencia (lognormal) para anclar los sliders
    b1_ref, b2_ref, b3_ref = gb2.mcdonald_baseline_moments(sigma2_ex, tau_ex)

    st.markdown(
        f"**Momentos objetivo** (la lognormal de referencia tiene "
        f"β₁ = {b1_ref:.3f}, β₂ = {b2_ref:.3f})"
    )
    s1, s2 = st.columns(2)
    beta1_obj = s1.number_input(
        "Asimetría β₁", value=float(round(b1_ref, 4)), step=0.0001,
        format="%.4f", key="beta1_obj",
    )
    beta2_obj = s2.number_input(
        "Curtosis β₂", min_value=0.0001, value=float(round(b2_ref, 4)),
        step=0.0001, format="%.4f", key="beta2_obj",
    )

    # Cálculo cacheado: calibra y valora en la grilla
    @st.cache_data(show_spinner="Calibrando por MCE y valorando...")
    def _valorar(b1, b2, K, r, tau, s2):
        return gb2.price_grid_from_moments(b1, b2, None, K=K, r=r, tau=tau, sigma2=s2)

    res = _valorar(beta1_obj, beta2_obj, K_ex, r_ex, tau_ex, sigma2_ex)
    mny = res["moneyness"]
    a_c, p_c, q_c, loss_c = res["params"]

    precios_validos = (
        res["params_ok"]
        and np.all(np.isfinite(res["call"]))
        and np.all(np.isfinite(res["put"]))
    )

    if not precios_validos:
        st.warning(
            "Para estos momentos la calibración cae en un régimen numéricamente "
            "inestable de la AGB2 (asimetría negativa fuerte suele requerir un "
            "parámetro *a* muy grande, cerca del caso degenerado). Prueba con una "
            "asimetría menos extrema o una curtosis algo mayor."
        )
    else:
        col_call, col_put = st.columns(2)
        with col_call:
            figc = go.Figure()
            figc.add_trace(go.Scatter(
                x=mny, y=res["bs_call"], name="Black-Scholes",
                line=dict(width=3, color="#1f77b4"),
            ))
            figc.add_trace(go.Scatter(
                x=mny, y=res["call"], name="AGB2",
                line=dict(width=2.5, color="#d62728", dash="dot"),
            ))
            figc.add_vline(x=1.0, line_dash="dash", line_color="gray")
            figc.update_layout(
                title="Precio de la Call",
                xaxis_title="Moneyness F/K", yaxis_title="Precio Call",
                height=420, margin=dict(l=10, r=10, t=40, b=10),
                legend=dict(orientation="h", y=-0.2),
            )
            st.plotly_chart(figc, use_container_width=True)

        with col_put:
            figp = go.Figure()
            figp.add_trace(go.Scatter(
                x=mny, y=res["bs_put"], name="Black-Scholes",
                line=dict(width=3, color="#1f77b4"),
            ))
            figp.add_trace(go.Scatter(
                x=mny, y=res["put"], name="AGB2",
                line=dict(width=2.5, color="#d62728", dash="dot"),
            ))
            figp.add_vline(x=1.0, line_dash="dash", line_color="gray")
            figp.update_layout(
                title="Precio de la Put",
                xaxis_title="Moneyness F/K", yaxis_title="Precio Put",
                height=420, margin=dict(l=10, r=10, t=40, b=10),
                legend=dict(orientation="h", y=-0.2),
            )
            st.plotly_chart(figp, use_container_width=True)

        st.caption(
            f"AGB2 calibrada: a = {a_c:.3f}, p = {p_c:.2f}, q = {q_c:.2f} "
            f"(pérdida MCE = {loss_c:.1e}). La línea gris marca el ATM (F/K = 1). "
            "La curva BS es independiente de β₁ y β₂ por construcción."
        )

