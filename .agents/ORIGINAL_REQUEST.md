# Original User Request

## Initial Request — 2026-08-31T13:09:00Z

Construir una aplicación web interactiva en Python (Streamlit + Plotly) para la optimización cuantitativa de portafolios de inversión y cálculo de la Frontera Eficiente de Markowitz, integrando modelos de covarianza robusta (Shrinkage Ledoit-Wolf, EWMA), simulaciones de Monte Carlo multidimensionales y análisis de riesgo avanzado.

Working directory: c:/Nico/Antigravity/Frontera Eficiente
Integrity mode: development

## Requirements

### R1. Ingesta y Manejo Híbrido de Datos Financieros
La aplicación debe permitir la obtención automática de precios históricos ajustados mediante `yfinance` para cualquier activo global (acciones, ETFs, criptoactivos, CEDEARs con terminación `.BA`, etc.) y soportar la carga manual de archivos CSV/Excel con series temporales de precios o retornos. Debe incluir limpieza de datos (alineación de fechas y tratamiento de valores nulos) y caché para optimizar las descargas.

### R2. Modelado Estadístico y Estimación Robusta de Riesgo
Implementar estimadores de retornos esperados (media histórica anualizada, EWMA y CAPM) y estimadores de matriz de covarianza: covarianza muestral clásica, **Shrinkage Ledoit-Wolf** (para mitigar el error muestral y estabilizar ponderaciones out-of-sample) y EWMA. Proporcionar matrices de correlación y covarianza interactivas.

### R3. Motor de Optimización Cuantitativa y Frontera Eficiente
Calcular con precisión matemática:
- Cartera de **Máximo Sharpe** (Tangencia) sujeta a una tasa libre de riesgo ($R_f$) editable.
- Cartera de **Mínima Varianza Global (GMV)**.
- Curva continua de la **Frontera Eficiente** de Markowitz.
- Línea de Asignación de Capital (**CAL**).
Soportar restricciones configurables de inversión: posiciones solo largas (*Long-Only*, $0 \le w_i \le 1$, $\sum w_i = 1$), ventas en corto (*Short-Selling*), y límites mínimos/máximos por activo ($w_{min} \le w_i \le w_{max}$).

### R4. Simulaciones de Monte Carlo Dobles
- **Monte Carlo de Ponderaciones**: Generar miles de asignaciones aleatorias uniformes sobre el simplex (distribución Dirichlet) para mapear el espacio riesgo-retorno coloreado por Ratio de Sharpe.
- **Monte Carlo de Trayectorias Futuras**: Simular la evolución estocástica patrimonial a 1–5 años (Movimiento Browniano Geométrico o bootstrapping histórico) con conos de probabilidad (percentiles 5%, 25%, 50%, 75%, 95%).

### R5. Interfaz de Usuario, Asignación Interactiva y Reportes
Interfaz moderna en Streamlit con:
- Sliders interactivos de ponderación con botones de **"Normalizar a 100%"** y **"Aplicar Cartera Óptima Sharpe"**.
- Portafolios predefinidos de 1-click (Clásico 60/40, All-Weather Ray Dalio, Big Tech, CEDEARs Argentina, Cripto + TradFi).
- Gráficos interactivos Plotly (Frontera con CAL y activos individuales, heatmaps, comparativa de pesos por barras/donut, backtest histórico de $10,000 USD y conos de proyección).
- Tabla de métricas de riesgo completas: Retorno Anualizado, Volatilidad, Sharpe, Sortino, Calmar, Max Drawdown, VaR 95% y CVaR 95%.
- Exportación de resultados y matrices a CSV / Excel.

## Acceptance Criteria

### Exactitud Matemática y Optimización
- [ ] La suma de las ponderaciones de cualquier cartera optimizada o rebalanceada es exactamente $1.0 \pm 10^{-5}$ y respeta los límites $[w_{min}, w_{max}]$.
- [ ] El Ratio de Sharpe de la cartera de Máximo Sharpe es matemáticamente mayor o igual al de cualquier activo individual o cartera aleatoria de Monte Carlo bajo los mismos parámetros.
- [ ] La volatilidad de la cartera GMV es matemáticamente menor o igual a la de cualquier otra combinación factible.
- [ ] Las matrices de covarianza (muestral, Ledoit-Wolf y EWMA) son simétricas y semidefinidas positivas.

### Funcionalidad y Robustez
- [ ] La descarga por tickers de `yfinance` y la carga por CSV/Excel procesan correctamente los datos sin fallar ante feriados o desalineación de fechas.
- [ ] La simulación de Monte Carlo de 10,000 carteras se ejecuta de manera vectorizada en menos de 2 segundos.
- [ ] Todos los gráficos interactivos de Plotly se renderizan correctamente sin errores en la consola.
- [ ] El conjunto de pruebas automatizadas (`pytest` o script de verificación) valida todos los cálculos analíticos y optimizadores con 100% de éxito.
