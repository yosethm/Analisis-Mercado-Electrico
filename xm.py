
import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import io
from PIL import Image
from datetime import datetime
import matplotlib.dates as mdates
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
import joblib

# =========================
# Configuración inicial de la app
# =========================
# Título de la pestaña y layout ancho.
st.set_page_config(page_title="Precios XM", layout="wide")
# Título visible y descripción corta.
st.title("⚡ Análisis del Precio del Mercado Eléctrico Colombiano 📈")
st.caption("Estudio histórico y predicciones del precio de la energía (COP/kWh) con modelos avanzados de Machine Learning y mucho mas")

# Tema visual por defecto para seaborn
sns.set_theme(style="whitegrid")

# =========================
# Sidebar (parámetros de usuario)
# =========================
st.sidebar.header("Parámetros de consulta")
# Rango de fechas para consultar los datos en la API
fecha_inicio = st.sidebar.date_input("Fecha inicial")
fecha_fin = st.sidebar.date_input("Fecha final")
# Bandera para activar el consumo de la API
usar_api = st.sidebar.checkbox("Conectar a API", value=False)

# =========================
# Estilos CSS y logo (renderizado con Markdown)
# =========================
st.markdown("""
<style>
    :root {
        --primary-color: #4e89ae;
        --secondary-color: #43658b;
        --text-color: #1e3d59;
        --highlight-color: #ff6e40;
        --background-color: #f5f0e1;
    }
    

    h1, h2, h3 {
        color: var(--text-color);
        font-weight: 700;
        border-bottom: 2px solid var(--highlight-color);
        padding-bottom: 10px;
        margin-bottom: 20px;
        animation: fadeIn 0.8s ease-in-out;
    }

    [data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.8);
        padding: 15px 10px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
        animation: fadeIn 0.8s ease-in-out;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-5px);
    }

    [data-testid="stTable"] {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        animation: fadeIn 0.8s ease-in-out;
    }

    .stSelectbox, .stSlider, .stNumberInput, .stTextInput {
        background-color: white !important;
        border-radius: 8px !important;
        padding: 10px !important;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05) !important;
        animation: fadeIn 0.8s ease-in-out;
    }

    button[data-baseweb="tab"] {
        font-weight: bold;
        border-radius: 5px 5px 0 0;
        padding: 10px 15px;
        background-color: rgba(255, 255, 255, 0.9);
        transition: all 0.3s;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        border-bottom: 3px solid var(--highlight-color);
        color: var(--text-color);
        animation: pulse 1.5s infinite;
    }

    .footer {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 8px;
        text-align: center;
        margin-top: 30px;
        font-size: 0.8em;
        color: #555;
    }

    .stPlotlyChart {
        background-color: white;
        padding: 10px;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        animation: fadeIn 0.8s ease-in-out;
    }

    .stProgress > div > div > div > div {
        background-color: var(--highlight-color);
    }

    [title]:hover::after {
        content: attr(title);
        background: #444;
        color: #fff;
        padding: 6px 8px;
        border-radius: 4px;
        position: absolute;
        top: 100%;
        white-space: nowrap;
        z-index: 1000;
    }

    @keyframes fadeIn {
        0% {opacity: 0; transform: translateY(10px);}
        100% {opacity: 1; transform: translateY(0);}
    }

    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(255,110,64, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(255,110,64, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255,110,64, 0); }
    }

    /* Logo adaptado */
    .logo-container {
        position: absolute;
        top: 10px;
        right: 15px;
        display: flex;
        gap: 12px;
        z-index: 1000;
        background: rgba(255,255,255,0.9);
        padding: 4px 8px;
        border-radius: 8px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    }
    .logo-container img {
        height: 54px;
        max-width: 100%;
        opacity: 0.9;
        transition: transform 0.3s ease-in-out, opacity 0.3s ease-in-out;
    }
    .logo-container img:hover {
        transform: scale(1.08);
        opacity: 1;
    }

    @media (max-width: 768px) {
        .logo-container {
            top: 5px;
            right: 5px;
            gap: 6px;
            padding: 2px 6px;
        }
        .logo-container img {
            height: 42px;
        }
    }
</style>

<div class="logo-container">
    <a href="https://www.udea.edu.co" target="_blank">
        <img src="https://raw.githubusercontent.com/Emma-Ok/BootcampTalentoTech/main/Escudo-UdeA.svg.png" alt="Escudo UdeA">
    </a>
</div>
""", unsafe_allow_html=True)

# Validación de fechas: evita que el usuario ponga inicio > fin
if fecha_inicio > fecha_fin:
    st.sidebar.error("La fecha inicial no puede ser mayor a la final.")
    st.stop()

# =========================
# Funciones auxiliares
# =========================
@st.cache_data(show_spinner=True)  # Cachea resultados de la API para no repetir llamadas
def obtener_datos_por_rango(f_ini, f_fin):
    # ID del dataset en el backend de SIMEM
    dataset_id = '96D56E'
    # Normalización: ajusta los días al primer día del mes para recorrer mes a mes
    f_ini = pd.to_datetime(f_ini).replace(day=1)
    f_fin = pd.to_datetime(f_fin).replace(day=1)
    meses = pd.date_range(f_ini, f_fin, freq='MS')  # 'MS' = Month Start

    dfs = []
    for fecha in meses:
        # Determina el inicio y fin de cada mes
        f_inicio_mes = fecha.date()
        f_fin_mes = (fecha + pd.offsets.MonthEnd(0)).date()
        # Construye la URL con parámetros
        url = f"https://www.simem.co/backend-files/api/PublicData?startDate={f_inicio_mes}&enddate={f_fin_mes}&datasetId={dataset_id}"

        try:
            # Llamada HTTP con timeout de 30s
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                payload = r.json()
                # Extrae registros dentro del JSON anidado
                datos = payload.get("result", {}).get("records", [])
                if datos:
                    df_mes = pd.DataFrame(datos)
                    # Parseo de tipos
                    df_mes["Fecha"] = pd.to_datetime(df_mes["Fecha"])
                    df_mes["Valor"] = pd.to_numeric(df_mes["Valor"], errors="coerce")
                    # Quita filas sin valor numérico
                    df_mes = df_mes.dropna(subset=["Valor"])
                    dfs.append(df_mes)
            else:
                # Notifica si la API responde con error HTTP
                st.error(f"Error en {fecha.strftime('%B %Y')}: Código {r.status_code}")
        except Exception as e:
            # Captura errores de red/parseo
            st.error(f"Error en {fecha.strftime('%B %Y')}: {e}")

    # Concatena todos los meses y ordena por fecha
    if dfs:
        out = pd.concat(dfs).sort_values("Fecha").reset_index(drop=True)
        return out
    else:
        return pd.DataFrame()

def generar_gif(df):
    """
    Genera un GIF animado por mes con:
    - Serie diaria
    - Promedio, máximo y mínimo del mes
    - Media móvil de 5 periodos como 'tendencia'
    """
    df = df.copy()
    df["Mes"] = df["Fecha"].dt.to_period("M")  # Agrupa por periodo mensual (YYYY-MM)
    imgs = []
    fixed_size = (1200, 600)  # Tamaño uniforme para el GIF

    for mes in df["Mes"].unique():
        data = df[df["Mes"] == mes]
        mes_txt = datetime.strptime(str(mes), "%Y-%m").strftime("%B %Y")

        # Figura por mes
        fig, ax = plt.subplots(figsize=(12, 6))
        # Serie diaria
        ax.plot(data["Fecha"], data["Valor"], color="black", marker="o", markersize=4,
                markerfacecolor="blue", linewidth=1.5, label="Datos")
        # Líneas de referencia (promedio, máximo, mínimo)
        ax.axhline(data["Valor"].mean(), color="purple", linestyle='-', linewidth=1, label="Promedio")
        ax.axhline(data["Valor"].max(), color="red", linestyle='--', linewidth=1, label="Máximo")
        ax.axhline(data["Valor"].min(), color="blue", linestyle='--', linewidth=1, label="Mínimo")
        # Media móvil como señal de tendencia
        ax.plot(data["Fecha"], data["Valor"].rolling(5, min_periods=1).mean(),
                linestyle="--", color="black", linewidth=2, label="Tendencia")

        # Etiquetas y formato de fechas
        ax.set_title(f"Precio Energía - {mes_txt}")
        ax.set_xlabel("Fecha")
        ax.set_ylabel("Precio (COP/kWh)")
        ax.legend()
        ax.grid(True)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

        fig.tight_layout()

        # Convierte la figura a imagen y acumula para el GIF
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        buf.seek(0)
        plt.close(fig)

        img_pil = Image.open(buf).convert("RGB")
        img_pil = img_pil.resize(fixed_size)
        imgs.append(img_pil)

    # Ensambla el GIF con duración de 1s por frame
    gif_buf = io.BytesIO()
    imgs[0].save(
        gif_buf,
        format="GIF",
        save_all=True,
        append_images=imgs[1:],
        duration=1000,
        loop=0
    )
    gif_buf.seek(0)
    gif_buf.name = "grafico_precios_mes.gif"
    return gif_buf

# =========================
# Tabs de la interfaz
# =========================
tab,tab1,tab2 = st.tabs(["Consulta & Análisis",
                    "Modelo Predictivo",
                    "Graficas"])

# =========================
# Pestaña 1: Consulta & Análisis
# =========================
with tab:
    if usar_api:
        # Llama a la API según el rango dado
        df = obtener_datos_por_rango(fecha_inicio, fecha_fin)

        if not df.empty:
            st.subheader("Datos obtenidos")
            st.dataframe(df)

            # Exportación de CSV para descarga
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Descargar CSV", csv, file_name="precios_xm.csv", mime="text/csv")

            st.success(f"Datos obtenidos: {len(df)} registros")

            # KPIs básicos descriptivos
            st.subheader("Estadísticas descriptivas")
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Promedio", f"{df['Valor'].mean():.2f} COP")
            col2.metric("Máximo", f"{df['Valor'].max():.2f} COP")
            col3.metric("Mínimo", f"{df['Valor'].min():.2f} COP")
            col4.metric("Desviación", f"{df['Valor'].std():.2f} COP")
            col5.metric("Mediana", f"{df['Valor'].median():.2f} COP")
            

            # Generación y visualización de GIF mensual
            st.markdown("---")
            st.subheader("GIF de Precios Mensuales")
            gif_img = generar_gif(df)
            st.image(gif_img, caption="Evolución mensual del precio de energía", use_container_width=True)
            st.download_button("Descargar GIF", gif_img, file_name="precios_mes.gif", mime="image/gif")

    else:
        # Mensaje de ayuda si no se activa la API
        st.info("Activa **Conectar a API** para consultar y visualizar los datos.")
        

# =========================
# Función: Entrenamiento del modelo XGBoost optimizado
# =========================
def entrenar_xgb_optimizado(df):
    # --- Ingeniería de características temporales
    df["Fecha"] = pd.to_datetime(df["Fecha"])
    df["year"] = df["Fecha"].dt.year
    df["month"] = df["Fecha"].dt.month
    df["day"] = df["Fecha"].dt.day
    df["dayofweek"] = df["Fecha"].dt.dayofweek
    df["dayofyear"] = df["Fecha"].dt.dayofyear

    # --- Lags y medias móviles para capturar dinámica de series de tiempo
    for lag in [1, 2, 3, 7, 14]:
        df[f"Valor_lag{lag}"] = df["Valor"].shift(lag)
    df["rolling_mean_3"] = df["Valor"].rolling(window=3).mean()
    df["rolling_mean_7"] = df["Valor"].rolling(window=7).mean()
    # Elimina las filas iniciales con NaN por los lags/rolling
    df = df.dropna()

    # Si hay muy pocos datos, aborta
    if len(df) < 20:
        return None, None, None, None, None

    # --- Codificación One-Hot de variables categóricas
    cat_features = ["CodigoVariable", "CodigoDuracion", "UnidadMedida"]
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoded = encoder.fit_transform(df[cat_features])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_features))
    # Guarda el encoder para usarlo luego en predicción futura
    joblib.dump(encoder, "encoder_xgb.pkl")

    # --- Features finales (numéricas + categóricas codificadas) y target
    X = pd.concat([
        df[["year", "month", "day", "dayofweek", "dayofyear",
            "Valor_lag1", "Valor_lag2", "Valor_lag3", "Valor_lag7", "Valor_lag14",
            "rolling_mean_3", "rolling_mean_7"]].reset_index(drop=True),
        encoded_df.reset_index(drop=True)
    ], axis=1)
    y = df["Valor"]

    # --- Split temporal (sin barajar) para respetar el orden de la serie
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # --- Búsqueda de hiperparámetros con GridSearchCV
    xgb_base = XGBRegressor(random_state=42)
    param_grid = {
        "n_estimators": [200, 500],
        "max_depth": [4, 6],
        "learning_rate": [0.03, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0]
    }
    grid = GridSearchCV(
        estimator=xgb_base,
        param_grid=param_grid,
        scoring="neg_mean_absolute_error",  # minimiza MAE
        cv=3,
        verbose=0,
        n_jobs=-1
    )
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    # --- Predicción y métricas en el set de prueba
    y_pred = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # --- Persistencia del modelo para uso posterior
    joblib.dump(best_model, "modelo_xgb.pkl")

    return best_model, mae, r2, y_test, y_pred

# =========================
# Pestaña 2: Modelo Predictivo
# =========================
with tab1:
    st.subheader("🤖 Modelo Predictivo")

    # Introducción didáctica al modelo
    st.markdown("""
    **📌 Introducción al Modelo**
    
    El modelo implementado utiliza **XGBoost (Extreme Gradient Boosting)**, 
    un algoritmo de machine learning basado en árboles de decisión optimizados mediante boosting.
    
    Este método combina múltiples modelos débiles (árboles) de forma secuencial, 
    corrigiendo los errores cometidos por los anteriores y logrando un alto poder predictivo.
    
    En este caso, se entrenó el modelo con datos históricos de precios de energía en Colombia (COP/kWh), 
    incluyendo variables temporales (año, mes, día, día de la semana) y características de series de tiempo 
    como retardos (*lags*) y medias móviles.
    """)
    
        # Advertencia/sugerencia antes de la predicción
    st.info(
        "🔔 Recomendación: Para obtener mejores resultados (errores entre 0 % y 10 %), utiliza al menos 2 a 3 años de datos históricos y limita la predicción a un máximo de 15 días. Valores superiores podrían ocasionar overfitting"
    )
    
    if usar_api and not df.empty:
        # Resumen rápido del dataset cargado
        st.subheader("📊 Resumen de datos actuales")
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Filas", len(df))
        col_b.metric("Promedio", f"{df['Valor'].mean():.2f} COP/kWh")
        col_c.metric("Máximo", f"{df['Valor'].max():.2f} COP/kWh")
        col_d.metric("Mínimo", f"{df['Valor'].min():.2f} COP/kWh")

        # Guarda versión anterior del df para detectar cambios
        if "df_anterior" not in st.session_state:
            st.session_state.df_anterior = df.copy()

        # Compara df actual con el anterior para avisar al usuario
        if not st.session_state.df_anterior.equals(df):
            st.info("🔄 El dataset ha cambiado con respecto a la última vista.")
            cambios_filas = len(df) - len(st.session_state.df_anterior)
            cambios_promedio = df["Valor"].mean() - st.session_state.df_anterior["Valor"].mean()
            st.write(f"- Cambio en número de filas: **{cambios_filas:+}**")
            st.write(f"- Cambio en promedio de valor: **{cambios_promedio:+.2f} COP/kWh**")
            st.session_state.df_anterior = df.copy()

        # Selección del horizonte de predicción
        dias_a_predecir = st.number_input(
            "¿Cuántos días quieres predecir?",
            min_value=1,
            max_value=365,
            value=60,
            step=1
        )

        # Botón para ejecutar entrenamiento + proyección
        # Botón para ejecutar la predicción
        if st.button("🚀 Predecir"):
        
            model, mae, r2, y_test, y_pred = entrenar_xgb_optimizado(df)
        
            if model is not None:
                # Muestra métricas de desempeño en test
                col1, col2 = st.columns(2)
                col1.metric("MAE", f"{mae:.2f} COP/kWh")
                col2.metric("R²", f"{r2:.4f}")
        
                # Breve interpretación de métricas
                st.markdown(f"""
                **📊 Interpretación de Resultados**  
                - **MAE:** {mae:.2f} COP/kWh → Error medio absoluto de las predicciones.  
                - **R²:** {r2:.4f} → Explica el {r2*100:.1f}% de la variabilidad.
                """)
        
                # Carga del encoder guardado para generar features en línea
                encoder = joblib.load("encoder_xgb.pkl")
                cat_features = ["CodigoVariable", "CodigoDuracion", "UnidadMedida"]
        
                # Copia del último df para ir agregando las predicciones día a día (método autoregresivo)
                ultimos_datos = df.copy()
                predicciones_futuras = []
        
                for i in range(dias_a_predecir):
                    # Parte de la última fila y avanza 1 día en la columna Fecha
                    nueva_fila = ultimos_datos.iloc[[-1]].copy()
                    nueva_fila["Fecha"] = nueva_fila["Fecha"] + pd.Timedelta(days=1)
                    # Recalcula variables temporales
                    nueva_fila["year"] = nueva_fila["Fecha"].dt.year
                    nueva_fila["month"] = nueva_fila["Fecha"].dt.month
                    nueva_fila["day"] = nueva_fila["Fecha"].dt.day
                    nueva_fila["dayofweek"] = nueva_fila["Fecha"].dt.dayofweek
                    nueva_fila["dayofyear"] = nueva_fila["Fecha"].dt.dayofyear
        
                    # Construye lags a partir de la serie extendida
                    for lag in [1, 2, 3, 7, 14]:
                        nueva_fila[f"Valor_lag{lag}"] = ultimos_datos["Valor"].shift(lag).iloc[-1]
                    # Medias móviles a partir de lo último disponible
                    nueva_fila["rolling_mean_3"] = ultimos_datos["Valor"].rolling(3).mean().iloc[-1]
                    nueva_fila["rolling_mean_7"] = ultimos_datos["Valor"].rolling(7).mean().iloc[-1]
        
                    # Codifica categóricas con el encoder entrenado
                    encoded_new = encoder.transform(nueva_fila[cat_features])
                    encoded_new_df = pd.DataFrame(encoded_new, columns=encoder.get_feature_names_out(cat_features))
        
                    # Ensambla el vector de entrada para el modelo
                    X_nuevo = pd.concat([
                        nueva_fila[["year", "month", "day", "dayofweek", "dayofyear",
                                    "Valor_lag1", "Valor_lag2", "Valor_lag3", "Valor_lag7", "Valor_lag14",
                                    "rolling_mean_3", "rolling_mean_7"]].reset_index(drop=True),
                        encoded_new_df.reset_index(drop=True)
                    ], axis=1)
        
                    # Predice el valor del día siguiente y lo agrega a la serie
                    y_nuevo = model.predict(X_nuevo)[0]
                    nueva_fila["Valor"] = y_nuevo
        
                    predicciones_futuras.append([nueva_fila["Fecha"].values[0], y_nuevo])
                    ultimos_datos = pd.concat([ultimos_datos, nueva_fila], ignore_index=True)
        
                # DataFrame con predicciones acumuladas
                pred_df = pd.DataFrame(predicciones_futuras, columns=["Fecha", "Predicción"])
                pred_df["Fecha"] = pd.to_datetime(pred_df["Fecha"])
        
                # Métricas descriptivas de la predicción futura
                promedio_pred = pred_df["Predicción"].mean()
                maximo_pred = pred_df["Predicción"].max()
                minimo_pred = pred_df["Predicción"].min()
        
                # Métricas de los datos reales (para comparación)
                promedio_orig = df["Valor"].mean()
                maximo_orig = df["Valor"].max()
                minimo_orig = df["Valor"].min()
        
                # Señal de tendencia y variación porcentual entre el primer y último día predicho
                tendencia = "⬆️ Al alza" if pred_df["Predicción"].iloc[-1] > pred_df["Predicción"].iloc[0] else "⬇️ A la baja"
                variacion_pct = ((pred_df["Predicción"].iloc[-1] - pred_df["Predicción"].iloc[0]) / pred_df["Predicción"].iloc[0]) * 100
        
                # --- Visualización combinando histórico + predicción
                st.subheader("📈 Proyección General")
                hist = df[["Fecha", "Valor"]].rename(columns={"Valor": "Precio"})
                hist["Serie"] = "Histórico"
        
                pred = pred_df.rename(columns={"Predicción": "Precio"})
                pred["Serie"] = "Predicción"
        
                pred_start = df["Fecha"].max() + pd.Timedelta(days=1)
                combi = pd.concat([hist, pred], ignore_index=True)
        
                # Colores y elementos de guía visual
                palette = {"Histórico": "#2E86DE", "Predicción": "#E74C3C"}
                fig, ax = plt.subplots(figsize=(13, 6))
                ax.axvspan(pred_start, pred["Fecha"].max(), color="#FAD02E", alpha=0.18, label="Periodo de predicción")
                sns.lineplot(data=combi, x="Fecha", y="Precio", hue="Serie", linewidth=2.2, palette=palette, ax=ax)
                # Puntos de los primeros 15 días para resaltar el inicio de la proyección
                sns.scatterplot(data=pred.head(15), x="Fecha", y="Precio", s=45, color="#E74C3C", edgecolor="white", ax=ax)
                ax.axvline(pred_start, ls="--", color="#7f8c8d", lw=1.5)
                ax.set_xlabel("Fecha")
                ax.set_ylabel("Precio (COP/kWh)")
                ax.set_title("Evolución histórica y proyección de precios (COP/kWh)")
                ax.legend(title="Serie", loc="upper left")
                plt.tight_layout()
                st.pyplot(fig)
        
                # Resumen comparativo entre predicción y original
                st.markdown("### 📌 Resumen Estadístico de la Predicción")
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Promedio", f"{promedio_pred:.2f} COP/kWh", f"{(promedio_pred - promedio_orig):+.2f} COP")
                col2.metric("Máximo", f"{maximo_pred:.2f} COP/kWh", f"{(maximo_pred - maximo_orig):+.2f} COP")
                col3.metric("Mínimo", f"{minimo_pred:.2f} COP/kWh", f"{(minimo_pred - minimo_orig):+.2f} COP")
                col4.metric("Tendencia", tendencia)
                col5.metric("Variación %", f"{variacion_pct:.2f}%")
        
        
                # ── Resumen narrativo de la proyección ───────────────────────────────────
                inicio_pred = pred["Fecha"].min().date()
                fin_pred = pred["Fecha"].max()
                cambio_inmediato = pred["Precio"].iloc[0] - df["Valor"].iloc[-1]
                cambio_total = pred["Precio"].iloc[-1] - df["Valor"].iloc[-1]
                variacion_pct_total = (pred["Precio"].iloc[-1] / pred["Precio"].iloc[0] - 1) * 100
                pendiente_media = (pred["Precio"].iloc[-1] - pred["Precio"].iloc[0]) / max(1, len(pred))  # COP/día
                
                # Promedio de últimos 30 días reales vs. promedio de la proyección
                ventana = 30
                hist_30 = df[df["Fecha"] >= df["Fecha"].max() - pd.Timedelta(days=ventana)]
                prom_hist_30 = hist_30["Valor"].mean() if not hist_30.empty else float("nan")
                prom_pred = pred["Precio"].mean()
                
                # Detección de máximos y mínimos con sus fechas
                idx_max = pred["Precio"].idxmax()
                idx_min = pred["Precio"].idxmin()
                max_pred_val = pred.loc[idx_max, "Precio"]
                max_pred_fecha = pred.loc[idx_max, "Fecha"].date()
                min_pred_val = pred.loc[idx_min, "Precio"]
                min_pred_fecha = pred.loc[idx_min, "Fecha"].date()
                
                st.markdown("### 🧾 Resumen de la proyección")
                st.markdown(
                    f"""
                - **Horizonte**: del **{inicio_pred}** al **{fin_pred}**  
                - **Cambio inmediato** (primer predicho vs último real): **{cambio_inmediato:+.2f} COP/kWh**  
                - **Cambio total** (último predicho vs último real): **{cambio_total:+.2f} COP/kWh** (*{variacion_pct_total:+.2f}%* sobre el primer predicho)  
                - **Pendiente media en el periodo de predicción**: **{pendiente_media:+.2f} COP/día**  
                - **Promedio histórico (últimos 30 días)**: **{prom_hist_30:.2f} COP/kWh**  
                - **Promedio en la predicción**: **{prom_pred:.2f} COP/kWh** (**{prom_pred - prom_hist_30:+.2f} COP** vs historial 30d)  
                - **Máximo predicho**: **{max_pred_val:.2f} COP/kWh** el **{max_pred_fecha}**  
                - **Mínimo predicho**: **{min_pred_val:.2f} COP/kWh** el **{min_pred_fecha}**
                """
                )
                st.markdown("### 📌 Resumen Estadístico de la Predicción")

                # KPIs repetidos (como en el bloque anterior) — se muestran nuevamente según tu estructura original
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Promedio", f"{promedio_pred:.2f} COP/kWh", f"{(promedio_pred - promedio_orig):+.2f} COP")
                col2.metric("Máximo", f"{maximo_pred:.2f} COP/kWh", f"{(maximo_pred - maximo_orig):+.2f} COP")
                col3.metric("Mínimo", f"{minimo_pred:.2f} COP/kWh", f"{(minimo_pred - minimo_orig):+.2f} COP")
                col4.metric("Tendencia", tendencia)
                col5.metric("Variación %", f"{variacion_pct:.2f}%")
                
                # (Aquí termina la sección de cambio respecto al último valor real en tu código)

    else:
        # Si no hay datos (no se activó API), se indica al usuario
        st.info("Activa **Conectar a API** en la pestaña anterior para usar el modelo.")
# =========================
# Nueva pestaña: Solo Gráficas con explicación dinámica
# =========================

with tab2:
    st.subheader("📊 Visualizaciones clave (sin modelo)")

    if usar_api:
        # Verifica que 'df' exista (se crea en la pestaña 1 al cargar datos)
        try:
            df  # noqa: F821
        except NameError:
            st.warning("Primero ve a **Consulta & Análisis**, activa **Conectar a API** y carga los datos.")
        else:
            if df.empty:
                st.info("No hay datos para graficar todavía.")
            else:
                # Imports locales para ayuda de análisis y numérica
                import calendar
                import numpy as np

                # ---- Helpers de explicación dinámica ----
                def trend_text(series_vals, freq_label):
                    """Describe tendencia simple (pendiente), cambio % y fuerza (R²) de una regresión lineal."""
                    s = pd.Series(series_vals).dropna()
                    if len(s) < 3:
                        return "Serie muy corta para evaluar tendencia."
                    x = np.arange(len(s))
                    coef = np.polyfit(x, s.values, 1)
                    yhat = coef[0]*x + coef[1]
                    # Cálculo de R² manual
                    ss_res = np.sum((s.values - yhat)**2)
                    ss_tot = np.sum((s.values - s.mean())**2) if np.sum((s.values - s.mean())**2) != 0 else 0
                    r2 = 0.0 if ss_tot == 0 else 1 - ss_res/ss_tot
                    change_pct = (s.iloc[-1]/s.iloc[0]-1)*100 if s.iloc[0] != 0 else np.nan
                    dir_txt = "al alza 📈" if change_pct > 0 else ("a la baja 📉" if change_pct < 0 else "estable ➖")
                    # Clasificación verbal de la fuerza de la señal
                    if r2 >= 0.7: fuerza = "fuerte"
                    elif r2 >= 0.4: fuerza = "moderada"
                    else: fuerza = "débil"
                    return (f"Tendencia {dir_txt} en el periodo {freq_label.lower()} "
                            f"({change_pct:+.2f}%). Señal {fuerza} (R²={r2:.2f}).")

                def dist_text(s):
                    """Resumen de distribución: media, mediana, desviación, rango y sesgo."""
                    s = s.dropna()
                    if s.empty: return "Sin datos para distribución."
                    rango = (s.min(), s.max())
                    skew = s.skew()
                    if abs(skew) < 0.3: sesgo = "simétrica"
                    elif skew > 0: sesgo = "con cola a la derecha (picos altos poco frecuentes)"
                    else: sesgo = "con cola a la izquierda (picos bajos poco frecuentes)"
                    return (f"Media {s.mean():.2f}, mediana {s.median():.2f}, desviación {s.std():.2f}. "
                            f"Rango [{rango[0]:.2f}, {rango[1]:.2f}]. Distribución {sesgo}.")

                def box_text(df_box):
                    """Comentario dinámico para boxplot mensual: mes con mayor/menor mediana y mayor IQR."""
                    if df_box.empty: return "Sin datos mensuales suficientes."
                    med = df_box.groupby("Mes")["Valor"].median().sort_values(ascending=False)
                    iqr = df_box.groupby("Mes")["Valor"].apply(lambda x: x.quantile(0.75)-x.quantile(0.25)).sort_values(ascending=False)
                    top_mes = med.index[0]
                    bot_mes = med.index[-1]
                    var_mes = iqr.index[0]
                    return (f"Mes con mediana más alta: **{top_mes}**; más baja: **{bot_mes}**. "
                            f"Mayor variabilidad (IQR) en **{var_mes}**.")

                def heat_text(piv):
                    """Lee máximos y mínimos del mapa de calor Año-Mes ignorando NaN."""
                    if piv.isna().all().all(): return "Sin datos suficientes para mapa de calor."
                    # localizar máximos y mínimos ignorando NaN
                    max_val = np.nanmax(piv.values)
                    min_val = np.nanmin(piv.values)
                    max_pos = np.where(piv.values == max_val)
                    min_pos = np.where(piv.values == min_val)
                    # tomar primer hallazgo
                    y_max, m_max = piv.index[max_pos[0][0]], piv.columns[max_pos[1][0]]
                    y_min, m_min = piv.index[min_pos[0][0]], piv.columns[min_pos[1][0]]
                    return (f"Máximo promedio: **{max_val:.2f}** en **{m_max} {y_max}**. "
                            f"Mínimo promedio: **{min_val:.2f}** en **{m_min} {y_min}**.")

                def pers_text(corr):
                    """Traduce el coeficiente de correlación lag-1 a una interpretación cualitativa."""
                    if np.isnan(corr): return "No se puede calcular persistencia (datos insuficientes)."
                    if corr >= 0.8: lvl = "muy alta"
                    elif corr >= 0.6: lvl = "alta"
                    elif corr >= 0.4: lvl = "moderada"
                    elif corr >= 0.2: lvl = "baja"
                    else: lvl = "muy baja"
                    dirr = "positiva" if corr >= 0 else "negativa"
                    return f"Persistencia {lvl} ({dirr}), correlación lag-1 = {corr:.2f}."

                # ---- Preparar datos base para las gráficas ----
                df_vis = df.copy()
                df_vis["Fecha"] = pd.to_datetime(df_vis["Fecha"])
                df_vis["Valor"] = pd.to_numeric(df_vis["Valor"], errors="coerce")
                df_vis = df_vis.dropna(subset=["Valor"]).sort_values("Fecha")

                # Selector de frecuencia de agregación
                freq = st.radio("Frecuencia de agregación", ["Diaria", "Semanal", "Mensual"], index=0, horizontal=True)
                freq_map = {"Diaria": "D", "Semanal": "W", "Mensual": "MS"}
                res = (
                    df_vis.set_index("Fecha")
                    .resample(freq_map[freq])["Valor"].mean()
                    .reset_index()
                    .rename(columns={"Valor": "Precio"})
                )

                # ===== 1) Serie temporal con media móvil =====
                st.markdown("#### 1) Serie temporal con media móvil")
                win = 7 if freq == "Diaria" else (4 if freq == "Semanal" else 3)
                fig1, ax1 = plt.subplots(figsize=(12, 5))
                sns.lineplot(data=res, x="Fecha", y="Precio", linewidth=2, ax=ax1, label="Serie")
                ax1.plot(res["Fecha"], res["Precio"].rolling(win, min_periods=1).mean(),
                         linestyle="--", linewidth=2, label=f"Media móvil ({win})")
                ax1.set_xlabel("Fecha")
                ax1.set_ylabel("Precio (COP/kWh)")
                ax1.set_title(f"Evolución {freq.lower()} y media móvil")
                ax1.legend(loc="upper left")
                ax1.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig1)
                st.markdown(
                    f"**Explicación:** La línea azul es el precio promedio {freq.lower()} y la discontinua suaviza con una ventana de {win} periodos.\n\n"
                    f"**Análisis:** {trend_text(res['Precio'], freq)}"
                )

                # ===== 2) Distribución =====
                st.markdown("#### 2) Distribución de precios")
                fig2, ax2 = plt.subplots(figsize=(12, 5))
                sns.histplot(res["Precio"], bins=30, kde=True, ax=ax2)
                ax2.set_title("Distribución de precios")
                ax2.set_xlabel("Precio (COP/kWh)")
                ax2.set_ylabel("Frecuencia")
                ax2.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig2)
                st.markdown(
                    f"**Explicación:** Histograma con densidad (KDE) para conocer rangos típicos.\n\n"
                    f"**Análisis:** {dist_text(res['Precio'])}"
                )

                # ===== 3) Boxplot por mes =====
                st.markdown("#### 3) Estacionalidad por mes (boxplot)")
                df_box = df_vis.copy()
                df_box["MesN"] = df_box["Fecha"].dt.month
                df_box["Mes"] = df_box["MesN"].apply(lambda m: calendar.month_name[m])
                order_months = list(calendar.month_name)[1:]
                fig3, ax3 = plt.subplots(figsize=(14, 5))
                sns.boxplot(data=df_box, x="Mes", y="Valor", order=order_months, ax=ax3)
                ax3.set_xlabel("Mes")
                ax3.set_ylabel("Precio (COP/kWh)")
                ax3.set_title("Distribución de precios por mes")
                ax3.tick_params(axis="x", rotation=30)
                ax3.grid(True, axis="y", alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig3)
                st.markdown(
                    f"**Explicación:** Cada caja resume la variación mensual (mediana, cuartiles y atípicos).\n\n"
                    f"**Análisis:** {box_text(df_box)}"
                )

                # ===== 4) Mapa de calor Año-Mes =====
                st.markdown("#### 4) Mapa de calor Año vs Mes (promedio)")
                df_hm = df_vis.copy()
                df_hm["Año"] = df_hm["Fecha"].dt.year
                df_hm["MesN"] = df_hm["Fecha"].dt.month
                piv = df_hm.pivot_table(index="Año", columns="MesN", values="Valor", aggfunc="mean").reindex(columns=range(1, 13))
                piv.columns = [calendar.month_abbr[c] for c in piv.columns]
                fig4, ax4 = plt.subplots(figsize=(12, 6))
                sns.heatmap(piv, annot=False, fmt=".1f", linewidths=0.3, ax=ax4)
                ax4.set_title("Promedio de precios por Año y Mes")
                plt.tight_layout()
                st.pyplot(fig4)
                st.markdown(
                    f"**Explicación:** Colores más intensos indican promedios más altos.\n\n"
                    f"**Análisis:** {heat_text(piv)}"
                )

                # ===== 5) Persistencia (Valor vs. lag-1) =====
                st.markdown("#### 5) Persistencia (Valor vs. Valor anterior)")
                df_lag = df_vis.copy()
                df_lag["Valor_lag1"] = df_lag["Valor"].shift(1)
                df_lag = df_lag.dropna()
                fig5, ax5 = plt.subplots(figsize=(12, 5))
                sns.regplot(data=df_lag, x="Valor_lag1", y="Valor", ax=ax5, scatter_kws={"s": 25, "alpha": 0.6})
                ax5.set_xlabel("Precio periodo anterior (COP/kWh)")
                ax5.set_ylabel("Precio actual (COP/kWh)")
                ax5.set_title("Relación precio vs. rezago (lag-1)")
                ax5.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig5)
                corr = df_lag["Valor_lag1"].corr(df_lag["Valor"]) if not df_lag.empty else np.nan
                st.markdown(
                    f"**Explicación:** Compara el precio actual con el del periodo previo para medir inercia.\n\n"
                    f"**Análisis:** {pers_text(float(corr) if pd.notna(corr) else np.nan)}"
                )

                # ===== 6) Top picos y valles (últimos 12 meses) =====
                st.markdown("#### 6) Top 10 picos y valles (últimos 12 meses)")
                ult_12m = df_vis[df_vis["Fecha"] >= (df_vis["Fecha"].max() - pd.Timedelta(days=365))]
                if ult_12m.empty:
                    st.info("No hay suficientes datos en los últimos 12 meses para este resumen.")
                else:
                    top_max = ult_12m.nlargest(10, "Valor")[["Fecha", "Valor"]].rename(columns={"Valor": "Precio"})
                    top_min = ult_12m.nsmallest(10, "Valor")[["Fecha", "Valor"]].rename(columns={"Valor": "Precio"})
                    colm1, colm2 = st.columns(2)
                    with colm1:
                        st.write("**Máximos (Top 10)**")
                        st.dataframe(top_max.reset_index(drop=True))
                    with colm2:
                        st.write("**Mínimos (Top 10)**")
                        st.dataframe(top_min.reset_index(drop=True))
                    # análisis dinámico
                    r = ult_12m["Valor"].max() - ult_12m["Valor"].min()
                    st.markdown(
                        f"**Explicación:** Listado de los picos más altos y más bajos del último año.\n\n"
                        f"**Análisis:** Amplitud anual ≈ **{r:.2f}** COP/kWh. "
                        f"Último valor real: **{df_vis['Valor'].iloc[-1]:.2f}** COP/kWh."
                    )
    else:
        # Si no hay conexión a API, se avisa
        st.info("Activa **Conectar a API** para visualizar las gráficas.")


# =========================
# Pie de página (footer)
# =========================
st.markdown("""
<style>
.footer {
    position: relative;
    bottom: 0;
    width: 100%;
    background: linear-gradient(90deg, #4e89ae, #43658b);
    color: white;
    text-align: center;
    padding: 15px 10px;
    border-radius: 8px;
    font-size: 0.9rem;
    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
}
.footer p {
    margin: 4px 0;
}
</style>

<div class="footer">
    <p>⚡ Autor: <b>Yoseth Mosquera</b></p>
    <p>🎓 Universidad: <b>Universidad de Antioquia</b></p>
    <p>📊 Fuente: <b>Datos obtenidos de SIMEM</b></p>
    <p>© 2024</p>
</div>
""", unsafe_allow_html=True)
