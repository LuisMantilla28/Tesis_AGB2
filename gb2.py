"""
gb2.py
======
Módulo matemático central para la distribución GB2 y su transformación afín AGB2.

Contiene:
    - Densidad, CDF, cuantil y muestreo de la GB2 y la AGB2.
    - Momentos, coeficientes de asimetría (beta_1) y curtosis (beta_2).
    - Convergencia a la lognormal (cadena de reparametrizaciones).
    - Valoración de opciones europeas (call y put) bajo AGB2, Black-Scholes
      y una aproximación tipo Corrado-Su.

Notación (consistente con la tesis):
    GB2(a, b, p, q)   con a, b, p, q > 0
        f(x) = a x^{ap-1} / ( b^{ap} B(p,q) [1 + (x/b)^a]^{p+q} ),  x > 0
    AGB2(alpha, phi, a, p, q):  Y = alpha + phi * X,  con X ~ GB2(a, 1, p, q)

Se usa la función beta incompleta regularizada I_z(p,q) para la CDF,
disponible directamente en scipy como betainc / betaincinv.

Autor: (tu nombre)
Trabajo de tesis - distribución AGB2 aplicada a valoración de opciones.
"""

from __future__ import annotations

import numpy as np
from scipy import special
from scipy import integrate


# ---------------------------------------------------------------------------
# Utilidades internas
# ---------------------------------------------------------------------------

def _check_positive(**kwargs) -> None:
    """Verifica que los parámetros indicados sean estrictamente positivos."""
    for name, value in kwargs.items():
        if np.any(np.asarray(value) <= 0):
            raise ValueError(f"El parámetro '{name}' debe ser positivo (recibido {value}).")


def log_beta(p: float, q: float) -> float:
    """Logaritmo de la función beta B(p, q), estable numéricamente."""
    return special.betaln(p, q)


# ---------------------------------------------------------------------------
# Distribución GB2(a, b, p, q)
# ---------------------------------------------------------------------------

def gb2_pdf(x, a, b, p, q):
    """
    Densidad de X ~ GB2(a, b, p, q).

    f(x) = a x^{ap-1} / ( b^{ap} B(p,q) [1 + (x/b)^a]^{p+q} ),  x > 0.

    Se calcula en escala logarítmica para evitar overflow/underflow.
    """
    _check_positive(a=a, b=b, p=p, q=q)
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    mask = x > 0
    if np.any(mask):
        xm = x[mask]
        z = (xm / b) ** a
        log_f = (
            np.log(a)
            + (a * p - 1.0) * np.log(xm)
            - a * p * np.log(b)
            - log_beta(p, q)
            - (p + q) * np.log1p(z)
        )
        out[mask] = np.exp(log_f)
    return out if out.shape else float(out)


def gb2_cdf(x, a, b, p, q):
    """
    Función de distribución acumulada de X ~ GB2(a, b, p, q).

    F(x) = I_z(p, q),  con z = (x/b)^a / (1 + (x/b)^a),
    donde I_z es la beta incompleta regularizada.
    """
    _check_positive(a=a, b=b, p=p, q=q)
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    mask = x > 0
    if np.any(mask):
        z = (x[mask] / b) ** a
        z = z / (1.0 + z)
        out[mask] = special.betainc(p, q, z)
    return out if out.shape else float(out)


def gb2_quantile(u, a, b, p, q):
    """
    Cuantil (inversa de la CDF) de X ~ GB2(a, b, p, q).

    x_u = b * ( I^{-1}_u(p, q) / (1 - I^{-1}_u(p, q)) )^{1/a}.
    """
    _check_positive(a=a, b=b, p=p, q=q)
    u = np.asarray(u, dtype=float)
    w = special.betaincinv(p, q, u)  # I^{-1}_u(p, q)
    return b * (w / (1.0 - w)) ** (1.0 / a)


def gb2_rvs(a, b, p, q, size=1, random_state=None):
    """Muestreo por transformación inversa de GB2(a, b, p, q)."""
    rng = np.random.default_rng(random_state)
    u = rng.uniform(size=size)
    return gb2_quantile(u, a, b, p, q)


def gb2_moment(m, a, b, p, q):
    """
    Momento m-ésimo (no central) de X ~ GB2(a, b, p, q):

        E[X^m] = b^m * B(p + m/a, q - m/a) / B(p, q),   válido si a*q > m.

    Devuelve np.nan si el momento no existe.
    """
    _check_positive(a=a, b=b, p=p, q=q)
    if a * q <= m:
        return np.nan
    log_val = m * np.log(b) + log_beta(p + m / a, q - m / a) - log_beta(p, q)
    return np.exp(log_val)


# ---------------------------------------------------------------------------
# Momentos centrales, asimetría y curtosis
# ---------------------------------------------------------------------------

def gb2_central_moments(a, b, p, q):
    """
    Devuelve (media, varianza, beta1, beta2) de GB2(a, b, p, q).

    beta1 = mu3 / mu2^{3/2}  (asimetría estandarizada, con signo)
    beta2 = mu4 / mu2^2      (curtosis, 3 para la normal)

    Si no existen los momentos necesarios (a*q <= 4), los faltantes son np.nan.
    Nota: beta1 y beta2 no dependen de b (parámetro de escala).
    """
    m1 = gb2_moment(1, a, b, p, q)
    m2 = gb2_moment(2, a, b, p, q)
    m3 = gb2_moment(3, a, b, p, q)
    m4 = gb2_moment(4, a, b, p, q)

    mean = m1
    if np.isnan(m2):
        return mean, np.nan, np.nan, np.nan

    var = m2 - m1 ** 2

    beta1 = np.nan
    if not np.isnan(m3) and var > 0:
        mu3 = m3 - 3 * m1 * m2 + 2 * m1 ** 3
        beta1 = mu3 / var ** 1.5

    beta2 = np.nan
    if not np.isnan(m4) and var > 0:
        mu4 = m4 - 4 * m1 * m3 + 6 * m1 ** 2 * m2 - 3 * m1 ** 4
        beta2 = mu4 / var ** 2

    return mean, var, beta1, beta2


# ---------------------------------------------------------------------------
# Distribución AGB2(alpha, phi, a, p, q):  Y = alpha + phi * X,  X ~ GB2(a,1,p,q)
# ---------------------------------------------------------------------------

def agb2_pdf(y, alpha, phi, a, p, q):
    """
    Densidad de Y ~ AGB2(alpha, phi, a, p, q).

    f(y) = a (y-alpha)^{ap-1} / ( phi^{ap} B(p,q) [1 + ((y-alpha)/phi)^a]^{p+q} ),
    para y > alpha, y 0 en otro caso.

    Equivale a f_{GB2(a, phi, p, q)}(y - alpha).
    """
    _check_positive(phi=phi, a=a, p=p, q=q)
    y = np.asarray(y, dtype=float)
    return gb2_pdf(y - alpha, a, phi, p, q)


def agb2_cdf(y, alpha, phi, a, p, q):
    """CDF de Y ~ AGB2(alpha, phi, a, p, q) = F_{GB2(a, phi, p, q)}(y - alpha)."""
    _check_positive(phi=phi, a=a, p=p, q=q)
    y = np.asarray(y, dtype=float)
    return gb2_cdf(y - alpha, a, phi, p, q)


def agb2_quantile(u, alpha, phi, a, p, q):
    """Cuantil de Y ~ AGB2: y_u = alpha + phi * quantile_GB2(a,1,p,q)(u)."""
    return alpha + phi * gb2_quantile(u, a, 1.0, p, q)


def agb2_rvs(alpha, phi, a, p, q, size=1, random_state=None):
    """Muestreo de AGB2 por transformación inversa."""
    return alpha + phi * gb2_rvs(a, 1.0, p, q, size=size, random_state=random_state)


def agb2_central_moments(alpha, phi, a, p, q):
    """
    (media, varianza, beta1, beta2) de AGB2.
    Los momentos centrales estandarizados (beta1, beta2) coinciden con los de
    GB2(a, 1, p, q): son invariantes ante la transformación afín.
    """
    mean0, var0, beta1, beta2 = gb2_central_moments(a, 1.0, p, q)
    mean = alpha + phi * mean0
    var = phi ** 2 * var0 if not np.isnan(var0) else np.nan
    return mean, var, beta1, beta2


# ---------------------------------------------------------------------------
# Convergencia GB2 -> gamma generalizada -> lognormal
# ---------------------------------------------------------------------------

def lognormal_reparam(a, q, mu, sigma2):
    """
    Reparametrización de la Proposición de convergencia a la lognormal:

        beta = (sigma2 * a^2)^{1/a}
        b    = beta * q^{1/a}
        p    = (a*mu + 1) / (sigma2 * a^2)

    Bajo q -> infinito y luego a -> 0+, GB2(a, b, p, q) converge a LN(mu, sigma2).
    Devuelve (a, b, p, q) listos para pasar a las funciones GB2.
    """
    beta = (sigma2 * a ** 2) ** (1.0 / a)
    b = beta * q ** (1.0 / a)
    p = (a * mu + 1.0) / (sigma2 * a ** 2)
    return a, b, p, q


def lognormal_pdf(x, mu, sigma2):
    """Densidad lognormal LN(mu, sigma2) para comparación."""
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    mask = x > 0
    sigma = np.sqrt(sigma2)
    xm = x[mask]
    out[mask] = np.exp(-((np.log(xm) - mu) ** 2) / (2 * sigma2)) / (
        xm * sigma * np.sqrt(2 * np.pi)
    )
    return out if out.shape else float(out)


def lognormal_standardized_moments(sigma2):
    """
    Momentos centrales estandarizados (beta1, beta2, beta3) de LN(mu, sigma2).
    No dependen de mu (es parámetro de escala en la lognormal).

        w = exp(sigma2)
        beta1 = (w + 2) * sqrt(w - 1)                       (asimetría)
        beta2 = w^4 + 2 w^3 + 3 w^2 - 3                     (curtosis)
        beta3 = (asimetría de orden 5, momento central estandarizado mu5/mu2^{5/2})

    beta1 y beta2 son estándar. beta3 se calcula desde los momentos crudos.
    """
    w = np.exp(sigma2)
    beta1 = (w + 2.0) * np.sqrt(w - 1.0)
    beta2 = w ** 4 + 2.0 * w ** 3 + 3.0 * w ** 2 - 3.0

    # beta3 = mu5 / mu2^{5/2}, con mu=0 (invariante a mu). Usar X = exp(sigma*Z).
    # Momentos crudos E[X^k] = exp(k^2 sigma2 / 2) para LN(0, sigma2).
    def raw(k):
        return np.exp(k ** 2 * sigma2 / 2.0)
    m1, m2, m3, m4, m5 = raw(1), raw(2), raw(3), raw(4), raw(5)
    var = m2 - m1 ** 2
    mu5 = (m5 - 5 * m1 * m4 + 10 * m1 ** 2 * m3
           - 10 * m1 ** 3 * m2 + 4 * m1 ** 5)
    beta3 = mu5 / var ** 2.5
    return beta1, beta2, beta3


# ---------------------------------------------------------------------------
# Momentos centrales estandarizados de la GB2 y estimación por MCE
# ---------------------------------------------------------------------------

def gb2_standardized_moments(a, p, q):
    """
    Momentos centrales estandarizados (beta1, beta2, beta3) de GB2(a, 1, p, q).
    Corresponden a mu3/mu2^{3/2}, mu4/mu2^2 y mu5/mu2^{5/2}.
    Son invariantes al parámetro de escala b (y a la transformación afín).

    Requiere a*q > 5 para que exista el quinto momento (necesario para beta3).
    Devuelve np.nan en los que no existan.
    """
    m1 = gb2_moment(1, a, 1.0, p, q)
    m2 = gb2_moment(2, a, 1.0, p, q)
    m3 = gb2_moment(3, a, 1.0, p, q)
    m4 = gb2_moment(4, a, 1.0, p, q)
    m5 = gb2_moment(5, a, 1.0, p, q)

    if np.isnan(m2):
        return np.nan, np.nan, np.nan
    var = m2 - m1 ** 2
    if var <= 0:
        return np.nan, np.nan, np.nan

    beta1 = np.nan
    if not np.isnan(m3):
        mu3 = m3 - 3 * m1 * m2 + 2 * m1 ** 3
        beta1 = mu3 / var ** 1.5

    beta2 = np.nan
    if not np.isnan(m4):
        mu4 = m4 - 4 * m1 * m3 + 6 * m1 ** 2 * m2 - 3 * m1 ** 4
        beta2 = mu4 / var ** 2

    beta3 = np.nan
    if not np.isnan(m5) and var > 0:
        mu5 = (m5 - 5 * m1 * m4 + 10 * m1 ** 2 * m3
               - 10 * m1 ** 3 * m2 + 4 * m1 ** 5)
        beta3 = mu5 / var ** 2.5

    return beta1, beta2, beta3


def mce_fit_shape(target_beta1, target_beta2, target_beta3,
                  a0=0.9, p0=1.0, q0=1.0, multistart=True):
    """
    Estima los parámetros de forma (a, p, q) de la GB2 por el método de
    momentos centrales estandarizados (MCE): minimiza la pérdida

        L(a, p, q) = sum_{m=1}^{3} (beta_m^{muestral} - beta_m^{GB2}(a,p,q))^2

    igualando beta1, beta2, beta3 objetivo con los de GB2(a,1,p,q).

    La optimización se hace en escala logarítmica (para forzar a,p,q > 0).
    Como la pérdida es no convexa y sensible a los valores iniciales (según se
    documenta en la tesis), se usa un esquema multi-arranque: se lanza el
    optimizador desde el punto dado por el usuario y desde varios puntos
    adicionales, devolviendo la mejor solución encontrada.

    Parámetros
    ----------
    target_beta1, target_beta2, target_beta3 : momentos objetivo.
    a0, p0, q0 : valores iniciales sugeridos.
    multistart : si True, añade puntos de arranque adicionales.

    Devuelve
    --------
    (a, p, q, loss) : parámetros estimados y pérdida residual.
    """
    from scipy.optimize import minimize

    target = np.array([target_beta1, target_beta2, target_beta3])

    def loss(theta):
        a, p, q = np.exp(theta)
        b1, b2, b3 = gb2_standardized_moments(a, p, q)
        if np.isnan(b1) or np.isnan(b2) or np.isnan(b3):
            # penalización SUAVE que empuja hacia aq > 5 (existencia de momentos)
            deficit = max(0.0, 5.5 - a * q)
            return 1e3 + 1e2 * deficit
        pred = np.array([b1, b2, b3])
        return float(np.sum((pred - target) ** 2))

    # puntos de arranque (pocos y bien elegidos para mantener la app ágil)
    starts = [(a0, p0, q0)]
    if multistart:
        starts += [
            (0.9, 20.0, 30.0),   # cercano a la calibración lognormal de la tesis
            (2.0, 3.0, 5.0),
        ]

    best = (np.nan, np.nan, np.nan, np.inf)
    for s in starts:
        x0 = np.log(np.array(s, dtype=float))
        try:
            res = minimize(loss, x0, method="Nelder-Mead",
                           options={"xatol": 1e-6, "fatol": 1e-9,
                                    "maxiter": 2000})
            a, p, q = np.exp(res.x)
            if res.fun < best[3]:
                best = (a, p, q, res.fun)
        except Exception:
            continue

    return best


def mce_fit_from_lognormal(mu, sigma2, a0=0.9, p0=1.0, q0=1.0):
    """
    Estima una GB2(a, b, p, q) que reproduce los momentos centrales
    estandarizados de una lognormal LN(mu, sigma2) mediante MCE.

    Los parámetros de forma (a, p, q) se estiman igualando beta1, beta2, beta3.
    El parámetro de escala b se ajusta para reproducir además la varianza
    (segundo momento central) de la lognormal, análogo a la estimación de phi
    en la tesis: b = sqrt( Var_LN / mu2(GB2(a,1,p,q)) ).

    Devuelve
    --------
    (a, b, p, q, loss)
    """
    b1_t, b2_t, b3_t = lognormal_standardized_moments(sigma2)
    a, p, q, loss = mce_fit_shape(b1_t, b2_t, b3_t, a0, p0, q0)
    if np.isnan(a):
        return np.nan, np.nan, np.nan, np.nan, np.nan

    # varianza de la lognormal
    w = np.exp(sigma2)
    var_ln = np.exp(2 * mu) * w * (w - 1.0)

    # segundo momento central de GB2(a,1,p,q)
    m1 = gb2_moment(1, a, 1.0, p, q)
    m2 = gb2_moment(2, a, 1.0, p, q)
    var_x = m2 - m1 ** 2 if not np.isnan(m2) else np.nan
    if np.isnan(var_x) or var_x <= 0:
        return a, np.nan, p, q, loss

    b = np.sqrt(var_ln / var_x)
    return a, b, p, q, loss


# ---------------------------------------------------------------------------
# Valoración de opciones europeas
# ---------------------------------------------------------------------------

def bs_call_put(S, K, r, tau, sigma):
    """
    Precio Black-Scholes de call y put europeas.
    Devuelve (call, put).
    """
    if tau <= 0 or sigma <= 0:
        call = max(S - K, 0.0)
        put = max(K - S, 0.0)
        return call, put
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    Nd1 = special.ndtr(d1)
    Nd2 = special.ndtr(d2)
    call = S * Nd1 - K * np.exp(-r * tau) * Nd2
    put = call - S + K * np.exp(-r * tau)
    return call, put


def bs_greeks(S, K, r, tau, sigma):
    """Griegas de una call europea bajo Black-Scholes."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    Nd1 = special.ndtr(d1)
    Nd2 = special.ndtr(d2)
    phi_d1 = np.exp(-0.5 * d1 ** 2) / np.sqrt(2 * np.pi)
    delta = Nd1
    gamma = phi_d1 / (S * sigma * np.sqrt(tau))
    vega = S * phi_d1 * np.sqrt(tau)
    theta = -S * phi_d1 * sigma / (2 * np.sqrt(tau)) - r * K * np.exp(-r * tau) * Nd2
    rho = K * tau * np.exp(-r * tau) * Nd2
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


def _agb2_risk_neutral_params(S, r, tau, sigma, a, p, q):
    """
    Determina alpha(sigma) y phi(sigma) de la parametrización neutral al riesgo
    del capítulo de valoración, de modo que:
        - la varianza de log-retornos coincida con la del modelo BS
        - se cumpla la condición de no arbitraje E[S_T] = S e^{r tau}

    Requiere a*q > 2 (existencia de la varianza de la GB2 base).
    Devuelve (alpha, phi).
    """
    if a * q <= 2:
        return np.nan, np.nan

    X_mean = np.exp(log_beta(p + 1.0 / a, q - 1.0 / a) - log_beta(p, q))
    X_m2 = np.exp(log_beta(p + 2.0 / a, q - 2.0 / a) - log_beta(p, q))
    X_var = X_m2 - X_mean ** 2
    if X_var <= 0:
        return np.nan, np.nan
    sigma_X = np.sqrt(X_var)

    target_var = np.exp(sigma ** 2 * tau) - 1.0  # varianza relativa lognormal
    phi = S * np.exp(r * tau) * np.sqrt(target_var) / sigma_X
    alpha = S * np.exp(r * tau) - phi * X_mean
    return alpha, phi


def agb2_call_put(S, K, r, tau, sigma, a, p, q):
    """
    Precio de call y put europeas usando AGB2 como densidad neutral al riesgo.

    C = e^{-r tau} E[(S_T - K)^+],  con S_T ~ AGB2(alpha, phi, a, p, q).
    Se evalúa numéricamente la esperanza por integración sobre la densidad AGB2.
    Devuelve (call, put).
    """
    alpha, phi = _agb2_risk_neutral_params(S, r, tau, sigma, a, p, q)
    if np.isnan(alpha):
        return np.nan, np.nan

    def integrand(s):
        return (s - K) * agb2_pdf(s, alpha, phi, a, p, q)

    upper = agb2_quantile(0.99999, alpha, phi, a, p, q)
    lower = max(K, alpha)
    if upper <= lower:
        call = 0.0
    else:
        val, _ = integrate.quad(integrand, lower, upper, limit=200)
        call = np.exp(-r * tau) * val
        call = max(call, 0.0)

    put = call - S + K * np.exp(-r * tau)
    return call, max(put, 0.0)


def cs_call_put(S, K, r, tau, sigma, skew, kurt):
    """
    Aproximación Corrado-Su (expansión de Gram-Charlier) para call y put.

    skew : coeficiente de asimetría de los log-retornos (mu3)
    kurt : exceso de curtosis (mu4 - 3)

    Fórmula estándar de Corrado-Su (1996) con correcciones Q3 y Q4.
    Devuelve (call, put).
    """
    if tau <= 0 or sigma <= 0:
        return max(S - K, 0.0), max(K - S, 0.0)

    sqrt_tau = np.sqrt(tau)
    d = (np.log(S / K) + (r + 0.5 * sigma ** 2) * tau) / (sigma * sqrt_tau)
    Nd = special.ndtr(d)
    nd = np.exp(-0.5 * d ** 2) / np.sqrt(2 * np.pi)

    c_bs = S * Nd - K * np.exp(-r * tau) * special.ndtr(d - sigma * sqrt_tau)

    Q3 = (1.0 / 6.0) * S * sigma * sqrt_tau * (
        (2 * sigma * sqrt_tau - d) * nd
        + sigma ** 2 * tau * Nd
    )
    Q4 = (1.0 / 24.0) * S * sigma * sqrt_tau * (
        (d ** 2 - 1 - 3 * sigma * sqrt_tau * (d - sigma * sqrt_tau)) * nd
        + sigma ** 3 * tau ** 1.5 * Nd
    )

    call = c_bs + skew * Q3 + kurt * Q4
    call = max(call, 0.0)
    put = call - S + K * np.exp(-r * tau)
    return call, max(put, 0.0)


# ---------------------------------------------------------------------------
# Prueba rápida al ejecutar el módulo directamente
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Verificación de que la densidad integra 1 y la CDF es coherente.
    from scipy.integrate import quad

    a, b, p, q = 2.0, 1.0, 1.5, 3.0
    area, _ = quad(lambda x: gb2_pdf(x, a, b, p, q), 0, np.inf)
    print(f"Integral de la densidad GB2 (debe ser ~1): {area:.6f}")

    # La CDF en el cuantil 0.5 debe dar 0.5
    med = gb2_quantile(0.5, a, b, p, q)
    print(f"Mediana={med:.4f}, CDF(mediana)={gb2_cdf(med, a, b, p, q):.6f}")

    # Momentos por integración vs fórmula cerrada
    m1_num, _ = quad(lambda x: x * gb2_pdf(x, a, b, p, q), 0, np.inf)
    m1_form = gb2_moment(1, a, b, p, q)
    print(f"E[X] numérico={m1_num:.4f}, fórmula={m1_form:.4f}")

    mean, var, beta1, beta2 = gb2_central_moments(a, b, p, q)
    print(f"media={mean:.4f}, var={var:.4f}, beta1={beta1:.4f}, beta2={beta2:.4f}")

    # Valoración: la AGB2 debe acercarse a BS cuando q y a llevan al límite lognormal
    S, K, r, tau, sigma = 100.0, 100.0, 0.05, 0.5, 0.2
    c_bs, p_bs = bs_call_put(S, K, r, tau, sigma)
    c_agb2, p_agb2 = agb2_call_put(S, K, r, tau, sigma, a=0.5, p=20.0, q=40.0)
    print(f"Call BS={c_bs:.4f}, Call AGB2={c_agb2:.4f}")
