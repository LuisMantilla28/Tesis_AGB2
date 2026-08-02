"""
app.py
======
Aplicativos interactivos para la tesis sobre valoración de opciones europeas
usando la distribución AGB2 (transformación afín de la GB2) como función de
densidad neutral al riesgo.

Se organizan en cuatro pestañas:
    1. Explorador de densidades GB2 / AGB2 (con asimetría, curtosis y límite lognormal).
    2. Comparación de valoración: AGB2 vs Black-Scholes vs Corrado-Su.
    3. Visualización de las griegas.
    4. Laboratorio de estimación de parámetros.

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

tab_dens, tab_val, tab_greeks, tab_est = st.tabs(
    [
        "1 · Explorador de densidades",
        "2 · Valoración de opciones",
        "3 · Griegas",
        "4 · Estimación de parámetros",
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
        a = ca.number_input("a (forma)", min_value=0.0001, value=2.0, step=0.1,
                            format="%.4f", key="dens_a")
        p = cp.number_input("p (forma)", min_value=0.0001, value=1.5, step=0.1,
                            format="%.4f", key="dens_p")
        q = cq.number_input("q (forma)", min_value=0.0001, value=3.0, step=0.1,
                            format="%.4f", key="dens_q")
        phi = cphi.number_input("φ (escala)", min_value=0.0001, value=1.0, step=0.1,
                                format="%.4f", key="dens_phi")
        alpha = calpha.number_input("α (localización)", value=0.0, step=0.5,
                                    format="%.4f", key="dens_alpha")
        b = phi
    else:
        ca, cp, cq, cb = st.columns(4)
        a = ca.number_input("a (forma)", min_value=0.0001, value=2.0, step=0.1,
                            format="%.4f", key="dens_a")
        p = cp.number_input("p (forma)", min_value=0.0001, value=1.5, step=0.1,
                            format="%.4f", key="dens_p")
        q = cq.number_input("q (forma)", min_value=0.0001, value=3.0, step=0.1,
                            format="%.4f", key="dens_q")
        b = cb.number_input("b (escala)", min_value=0.0001, value=1.0, step=0.1,
                            format="%.4f", key="dens_b")
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
                                format="%.4f", key="ln_mu")
        sigma2_ln = st.number_input("σ² (varianza log)", min_value=0.0001,
                                    value=0.25, step=0.05, format="%.4f", key="ln_s2")

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
        sigma_ln = np.sqrt(sigma2_ln)
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
            x=xx, y=ln_pdf, name=f"Lognormal(μ={mu_ln:g}, σ²={sigma2_ln:g})",
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
    st.header("Valoración de una opción europea: AGB2 vs Black-Scholes vs Corrado-Su")

    c1, c2, c3 = st.columns(3)
    with c1:
        S = st.number_input("Precio subyacente S", 1.0, 1000.0, 100.0, 1.0)
        K = st.number_input("Strike K", 1.0, 1000.0, 100.0, 1.0)
    with c2:
        r = st.number_input("Tasa libre de riesgo r", 0.0, 0.5, 0.05, 0.005, format="%.3f")
        tau = st.number_input("Tiempo al vencimiento τ (años)", 0.01, 5.0, 0.5, 0.05)
    with c3:
        sigma = st.number_input("Volatilidad σ", 0.01, 2.0, 0.20, 0.01)

    st.markdown("**Parámetros de forma de la AGB2** (debe cumplirse a·q > 2)")
    g1, g2, g3 = st.columns(3)
    a_v = g1.slider("a ", 0.2, 10.0, 1.5, 0.1, key="a_val")
    p_v = g2.slider("p ", 0.2, 20.0, 3.0, 0.1, key="p_val")
    q_v = g3.slider("q ", 0.2, 40.0, 5.0, 0.1, key="q_val")

    tipo = st.radio("Tipo de opción", ["Call", "Put"], horizontal=True)

    if a_v * q_v <= 2:
        st.error("Se requiere a·q > 2 para que exista la varianza de la GB2 base.")
    else:
        c_bs, p_bs = gb2.bs_call_put(S, K, r, tau, sigma)
        c_agb2, p_agb2 = gb2.agb2_call_put(S, K, r, tau, sigma, a_v, p_v, q_v)
        # Corrado-Su usando la asimetría/curtosis implícitas de la AGB2
        _, _, sk, ku = gb2.gb2_central_moments(a_v, 1.0, p_v, q_v)
        sk = 0.0 if np.isnan(sk) else sk
        exc = 0.0 if np.isnan(ku) else (ku - 3.0)
        c_cs, p_cs = gb2.cs_call_put(S, K, r, tau, sigma, sk, exc)

        idx = 0 if tipo == "Call" else 1
        val_bs = (c_bs, p_bs)[idx]
        val_agb2 = (c_agb2, p_agb2)[idx]
        val_cs = (c_cs, p_cs)[idx]

        r1, r2, r3 = st.columns(3)
        r1.metric("Black-Scholes", f"{val_bs:.4f}")
        r2.metric("AGB2 (propuesto)", f"{val_agb2:.4f}",
                  delta=f"{val_agb2 - val_bs:+.4f} vs BS")
        r3.metric("Corrado-Su", f"{val_cs:.4f}",
                  delta=f"{val_cs - val_bs:+.4f} vs BS")

        st.divider()
        st.subheader("Precio vs moneyness (K/S)")
        Ks = np.linspace(0.7 * S, 1.3 * S, 40)
        precios_bs, precios_agb2, precios_cs = [], [], []
        for Ki in Ks:
            cb, pb = gb2.bs_call_put(S, Ki, r, tau, sigma)
            ca, pa = gb2.agb2_call_put(S, Ki, r, tau, sigma, a_v, p_v, q_v)
            cc, pc = gb2.cs_call_put(S, Ki, r, tau, sigma, sk, exc)
            precios_bs.append((cb, pb)[idx])
            precios_agb2.append((ca, pa)[idx])
            precios_cs.append((cc, pc)[idx])

        figv = go.Figure()
        figv.add_trace(go.Scatter(x=Ks / S, y=precios_bs, name="Black-Scholes"))
        figv.add_trace(go.Scatter(x=Ks / S, y=precios_agb2, name="AGB2"))
        figv.add_trace(go.Scatter(x=Ks / S, y=precios_cs, name="Corrado-Su"))
        figv.update_layout(
            xaxis_title="Moneyness K/S", yaxis_title=f"Precio {tipo}",
            height=380, margin=dict(l=10, r=10, t=20, b=10),
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(figv, use_container_width=True)
        st.caption(
            "Cuando q → ∞ y a → 0, el precio AGB2 converge al de Black-Scholes "
            "(caso asintótico lognormal). Prueba subiendo q y bajando a."
        )


# ===========================================================================
# PESTAÑA 3 — Griegas
# ===========================================================================
with tab_greeks:
    st.header("Griegas de una call europea")
    st.write(
        "Se muestran las sensibilidades de Black-Scholes en función del "
        "*moneyness*. Sirven de referencia para comparar con las expresiones "
        "AGB2 desarrolladas en la tesis."
    )

    gc1, gc2, gc3 = st.columns(3)
    S_g = gc1.number_input("S ", 1.0, 1000.0, 100.0, 1.0, key="S_g")
    r_g = gc2.number_input("r ", 0.0, 0.5, 0.05, 0.005, key="r_g", format="%.3f")
    tau_g = gc3.number_input("τ ", 0.01, 5.0, 0.5, 0.05, key="tau_g")
    sigma_g = st.slider("σ ", 0.05, 1.0, 0.20, 0.01, key="sigma_g")

    Ks = np.linspace(0.7 * S_g, 1.3 * S_g, 60)
    griegas = {"delta": [], "gamma": [], "vega": [], "theta": [], "rho": []}
    for Ki in Ks:
        g = gb2.bs_greeks(S_g, Ki, r_g, tau_g, sigma_g)
        for k in griegas:
            griegas[k].append(g[k])

    seleccion = st.multiselect(
        "Griegas a mostrar",
        ["delta", "gamma", "vega", "theta", "rho"],
        default=["delta", "gamma", "vega"],
    )
    figg = go.Figure()
    for k in seleccion:
        figg.add_trace(go.Scatter(x=Ks / S_g, y=griegas[k], name=k.capitalize()))
    figg.update_layout(
        xaxis_title="Moneyness K/S", yaxis_title="Valor de la griega",
        height=420, margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(figg, use_container_width=True)


# ===========================================================================
# PESTAÑA 4 — Estimación de parámetros
# ===========================================================================
with tab_est:
    st.header("Laboratorio de estimación de parámetros AGB2")
    st.write(
        "Genera una muestra simulada con parámetros conocidos y observa cómo la "
        "densidad AGB2 se ajusta al histograma. Se reportan las pruebas de "
        "bondad de ajuste de Kolmogorov-Smirnov y Anderson-Darling vía la "
        "transformación integral de probabilidad (PIT)."
    )

    e1, e2, e3 = st.columns(3)
    with e1:
        a_e = st.slider("a real", 0.3, 8.0, 2.0, 0.1, key="a_e")
        p_e = st.slider("p real", 0.3, 10.0, 2.0, 0.1, key="p_e")
    with e2:
        q_e = st.slider("q real", 0.3, 10.0, 3.0, 0.1, key="q_e")
        alpha_e = st.slider("α real", -20.0, 20.0, 0.0, 0.5, key="alpha_e")
    with e3:
        phi_e = st.slider("φ real", 0.2, 5.0, 1.0, 0.1, key="phi_e")
        n = st.select_slider("Tamaño de muestra", [100, 250, 500, 1000, 2000], value=500)

    semilla = st.number_input("Semilla", 0, 9999, 42, 1)

    if st.button("Simular y ajustar", type="primary"):
        muestra = gb2.agb2_rvs(alpha_e, phi_e, a_e, p_e, q_e, size=n, random_state=semilla)

        # Densidad "ajustada" = densidad con los parámetros verdaderos (demo).
        # En la app real aquí iría el estimador MCE / MLE de la tesis.
        xs = np.linspace(muestra.min(), np.percentile(muestra, 99), 400)
        dens = gb2.agb2_pdf(xs, alpha_e, phi_e, a_e, p_e, q_e)

        fig_e = go.Figure()
        fig_e.add_trace(go.Histogram(
            x=muestra, histnorm="probability density",
            name="Muestra", opacity=0.6, nbinsx=40
        ))
        fig_e.add_trace(go.Scatter(x=xs, y=dens, name="Densidad AGB2", line=dict(width=3)))
        fig_e.update_layout(
            height=400, margin=dict(l=10, r=10, t=20, b=10),
            legend=dict(orientation="h", y=1.1),
            barmode="overlay",
        )
        st.plotly_chart(fig_e, use_container_width=True)

        # PIT: si el modelo es correcto, U = F(Y) ~ Uniforme(0,1)
        u = gb2.agb2_cdf(muestra, alpha_e, phi_e, a_e, p_e, q_e)
        u = np.clip(u, 1e-9, 1 - 1e-9)

        ks_stat, ks_p = stats.kstest(u, "uniform")
        ad = stats.anderson(stats.norm.ppf(u))  # AD sobre normalidad de Phi^{-1}(U)

        c_ks, c_ad = st.columns(2)
        c_ks.metric("KS p-valor", f"{ks_p:.4f}",
                    "ajuste OK" if ks_p > 0.05 else "posible falta de ajuste")
        c_ad.metric("AD estadístico", f"{ad.statistic:.4f}",
                    help="Compara contra valores críticos; menor es mejor.")
        st.caption(
            "Nota: en esta demo la densidad ajustada usa los parámetros reales. "
            "Al integrar tus estimadores (MCE / EPCA / MLE) el ajuste reflejará "
            "el error real de estimación."
        )
