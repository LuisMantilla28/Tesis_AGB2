# Tesis_AGB2

Aplicativos interactivos en **Streamlit** para la valoración de opciones europeas
usando la distribución **AGB2** (transformación afín de la GB2) como función de
densidad neutral al riesgo. Incluye un explorador de densidades, la comparación
con Black-Scholes y Corrado-Su, la visualización de las griegas y un laboratorio
de estimación de parámetros.

Este repositorio acompaña el trabajo de tesis sobre un modelo de valoración de
opciones europeas en el que el precio terminal del activo subyacente se modela
como una transformación afín de una variable con colas pesadas.

## Contenido

| Archivo | Descripción |
|---|---|
| `gb2.py` | Módulo matemático: densidad, CDF, cuantiles, momentos, asimetría/curtosis, convergencia a la lognormal y valoración de opciones (AGB2, Black-Scholes, Corrado-Su). |
| `app.py` | Interfaz Streamlit con cuatro pestañas. |
| `requirements.txt` | Dependencias. |

## Las cuatro pestañas

1. **Explorador de densidades** — Sliders para `a, b, p, q` (y `α, φ` en modo AGB2),
   con densidad y CDF en vivo, coeficientes de asimetría (β₁) y curtosis (β₂),
   y el límite lognormal como referencia.
2. **Valoración de opciones** — Precio de call/put bajo AGB2, Black-Scholes y
   Corrado-Su, con la curva precio vs *moneyness*.
3. **Griegas** — Delta, Gamma, Vega, Theta y Rho en función del *moneyness*.
4. **Estimación de parámetros** — Simulación de muestras y bondad de ajuste
   (Kolmogorov-Smirnov y Anderson-Darling vía la transformación PIT).

## Ejecutar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

La app abre en `http://localhost:8501`.

## Desplegar en Streamlit Community Cloud

1. Sube este repositorio a GitHub (público).
2. Entra a [share.streamlit.io](https://share.streamlit.io) y conecta tu cuenta de GitHub.
3. Selecciona el repositorio `Tesis_AGB2`, la rama `main` y el archivo `app.py`.
4. En un par de minutos obtienes una URL pública para compartir con los jurados.

Cada `git push` a la rama seleccionada redespliega la app automáticamente.

## Nota matemática

La CDF de la GB2 se calcula con la función beta incompleta regularizada
`I_z(p,q)` (`scipy.special.betainc`) y el cuantil con su inversa
(`scipy.special.betaincinv`), evitando implementar la función hipergeométrica
₂F₁ a mano. Todas las funciones fueron verificadas: la densidad integra 1, la
CDF es consistente con el cuantil, y los momentos por integración coinciden con
las fórmulas cerradas.

## Licencia

Uso académico. (Ajusta esta sección según prefieras.)
