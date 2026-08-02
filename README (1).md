# Tesis_AGB2

Aplicativos interactivos en **Streamlit** que acompañan el trabajo de tesis sobre
valoración de opciones europeas usando la distribución **AGB2** (transformación
afín de la GB2) como función de densidad neutral al riesgo, comparada con el
modelo de Black-Scholes.

La aplicación permite explorar de forma visual e interactiva la flexibilidad de la
distribución GB2/AGB2 en asimetría y curtosis, y cómo esa flexibilidad se traduce
en diferencias de precio respecto al modelo lognormal clásico.

## Contenido del repositorio

| Archivo | Descripción |
|---|---|
| `app.py` | Interfaz Streamlit con las dos pestañas de la aplicación. |
| `gb2.py` | Módulo matemático: densidad, CDF, cuantiles, momentos, coeficientes de forma, convergencia a la lognormal, estimación por MCE y valoración de opciones (AGB2 y Black-Scholes). |
| `requirements.txt` | Dependencias del proyecto. |
| `.streamlit/config.toml` | Configuración de tema. |

## Las dos pestañas de la aplicación

### 1 · Explorador de densidades

Se divide en dos partes.

La primera permite ingresar manualmente los parámetros de la distribución y
observar en vivo la densidad y la función de distribución acumulada, junto con los
coeficientes de media, varianza, asimetría (β₁) y curtosis (β₂). Se puede
alternar entre la GB2 (parámetros `a`, `b`, `p`, `q`) y su transformación afín
AGB2 (con localización `α` y escala `φ`).

La segunda parte reproduce el resultado de convergencia central del trabajo: el
usuario propone una distribución lognormal mediante `μ` y `σ`, y la aplicación
estima automáticamente, mediante el método de **momentos centrales estandarizados
(MCE)**, los parámetros de la GB2 que mejor reproducen esos momentos. Se grafica
la lognormal propuesta junto a la GB2 estimada para visualizar la calidad de la
aproximación.

### 2 · Valoración de opciones europeas

Réplica interactiva del ejercicio de sensibilidad de McDonald (1991) y Madan
(1990). Con los parámetros del ejercicio (strike, tasa, vencimiento y varianza),
el usuario fija de forma explícita la asimetría (β₁) y la curtosis (β₂) de la
densidad neutral al riesgo. La AGB2 se calibra por MCE a esos momentos y se
valoran opciones call y put europeas bajo la parametrización de no arbitraje.

Se muestran dos gráficas, precio de la call y de la put, en función de la
*moneyness* F/K (con F = Sₜ e^{rτ}), comparando el modelo AGB2 con Black-Scholes.
Esto permite ver cómo la asimetría y la curtosis modifican los precios,
especialmente fuera del dinero, algo que el modelo lognormal ignora por
construcción.

## Ejecutar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

La aplicación se abre en `http://localhost:8501`.

## Despliegue en Streamlit Community Cloud

1. Sube este repositorio a GitHub (público).
2. Entra a [share.streamlit.io](https://share.streamlit.io) y conecta tu cuenta de GitHub.
3. Selecciona el repositorio, la rama `main` y el archivo `app.py`.
4. En un par de minutos obtienes una URL pública para compartir.

Cada `git push` a la rama seleccionada redespliega la aplicación automáticamente.

## Notas sobre la implementación

- La CDF de la GB2 se calcula con la función beta incompleta regularizada `I_z(p,q)`
  (`scipy.special.betainc`) y el cuantil con su inversa (`scipy.special.betaincinv`),
  evitando implementar la función hipergeométrica ₂F₁ a mano.
- El método MCE resuelve un sistema no lineal cuya solución no es única; la
  superficie de pérdida es no convexa, por lo que la estimación depende de los
  valores iniciales y del optimizador. Distintas implementaciones (por ejemplo R
  y Python) pueden converger a parámetros `(a, p, q)` diferentes que, sin embargo,
  reproducen los mismos momentos objetivo y prácticamente los mismos precios.
- Todas las funciones matemáticas fueron verificadas: la densidad integra 1, la
  CDF es consistente con el cuantil, los momentos por integración coinciden con
  las fórmulas cerradas, y el precio Black-Scholes de referencia se reproduce
  exactamente.

## Referencias

- McDonald, J. B. (1991). Parametric models for partially adaptive estimation with skewed and leptokurtic residuals.
- Madan, D. B., & Seneta, E. (1990). The Variance Gamma model for share market returns.

## Licencia

Uso académico.
