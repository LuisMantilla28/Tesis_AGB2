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
    st.write(
        "Modifica los parámetros y observa cómo cambian la densidad, la función "
        "de distribución y los coeficientes de forma. Los parámetros $a$, $p$ y "
        "$q$ controlan la asimetría y la curtosis; $b$ (o $\\phi$) es de escala y "
        "$\\alpha$ de localización."
    )

    col_ctrl, col_plot = st.columns([1, 2])

    with col_ctrl:
        usar_agb2 = st.toggle("Usar transformación afín (AGB2)", value=False)

        a = st.slider("a  (forma)", 0.2, 10.0, 2.0, 0.1)
        p = st.slider("p  (forma)", 0.2, 10.0, 1.5, 0.1)
        q = st.slider("q  (forma)", 0.2, 10.0, 3.0, 0.1)

        if usar_agb2:
            alpha = st.slider("α  (localización)", -10.0, 10.0, 0.0, 0.5)
            phi = st.slider("φ  (escala)", 0.2, 5.0, 1.0, 0.1)
            b = phi
        else:
            b = st.slider("b  (escala)", 0.2, 5.0, 1.0, 0.1)
            alpha = 0.0
            phi = b

        mostrar_ln = st.checkbox(
            "Superponer límite lognormal", value=False,
            help="Aplica la cadena de reparametrizaciones que lleva la GB2 a la "
                 "lognormal cuando q → ∞ y a → 0."
        )

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

    with col_plot:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=pdf, name="Densidad", line=dict(width=3)))

        if mostrar_ln and not usar_agb2:
            # límite lognormal con mu, sigma2 elegidos para comparar formas
            mu_ln, sigma2_ln = 0.0, 0.25
            ln = gb2.lognormal_pdf(x, mu_ln, sigma2_ln)
            fig.add_trace(go.Scatter(
                x=x, y=ln, name="Lognormal (referencia)",
                line=dict(width=2, dash="dash")
            ))

        fig.update_layout(
            title=etiqueta,
            xaxis_title="x",
            yaxis_title="Densidad",
            height=380,
            margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(orientation="h", y=1.12),
        )
        st.plotly_chart(fig, use_container_width=True)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=x, y=cdf, name="CDF", line=dict(width=3)))
        fig2.update_layout(
            title="Función de distribución acumulada",
            xaxis_title="x", yaxis_title="F(x)",
            height=260, margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Panel de coeficientes
    st.subheader("Coeficientes de forma")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Media", f"{mean:.4f}" if not np.isnan(mean) else "no existe")
    m2.metric("Varianza", f"{var:.4f}" if not np.isnan(var) else "no existe")
    m3.metric(
        "Asimetría β₁", f"{beta1:.4f}" if not np.isnan(beta1) else "no existe",
        help="Normal: 0"
    )
    m4.metric(
        "Curtosis β₂", f"{beta2:.4f}" if not np.isnan(beta2) else "no existe",
        help="Normal: 3"
    )
    if not np.isnan(beta2):
        if beta2 > 3:
            st.info("β₂ > 3 → distribución **leptocúrtica** (colas más pesadas que la normal).")
        elif beta2 < 3:
            st.info("β₂ < 3 → distribución **platicúrtica** (colas más ligeras que la normal).")
    if a * q <= 4:
        st.warning(
            f"Con a·q = {a * q:.2f} ≤ 4, algunos momentos no existen "
            "(la GB2 solo tiene momentos de orden menor que a·q)."
        )

    with st.expander("💡 Curiosidad: b no afecta la curtosis"):
        st.write(
            "Dado que $b$ es un parámetro de escala, dos densidades con el mismo "
            "$a, p, q$ pero distinto $b$ tienen **idéntica** asimetría y curtosis, "
            "aunque visualmente una parezca más apuntada que la otra."
        )
        xx = np.linspace(0.001, 6, 500)
        figb = go.Figure()
        for bb, color in zip([0.5, 1.0, 2.0], ["crimson", "seagreen", "royalblue"]):
            figb.add_trace(go.Scatter(
                x=xx, y=gb2.gb2_pdf(xx, 2.0, bb, 1.0, 1.0),
                name=f"b={bb}", line=dict(color=color)
            ))
        _, _, b1_demo, b2_demo = gb2.gb2_central_moments(2.0, 1.0, 1.0, 1.0)
        figb.update_layout(
            title=f"GB2(a=2, p=q=1): misma β₁ y β₂ para todo b",
            height=320, margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(figb, use_container_width=True)


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
