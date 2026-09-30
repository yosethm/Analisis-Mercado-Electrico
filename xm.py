import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import io
from PIL import Image
from datetime import datetime
import matplotlib.dates as mdates

# =========================
# Configuración inicial de la app
# =========================
# Título de la pestaña y layout ancho.
st.set_page_config(page_title="Precios XM", layout="wide")

# Título visible y descripción corta.
st.title("Análisis del Precio del Mercado Eléctrico Colombiano")
st.caption(
    "Estudio histórico y predicciones del precio de la energía "
    "y mucho mas"
)

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

    /* Tarjetas compactas para estadísticas descriptivas */
    .stats-shell { max-width: 1080px; margin: 4px auto 22px auto; padding: 2px 4px; }
    .stats-grid { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 14px; }
    .stat-card { position: relative; min-width: 0; padding: 18px 16px 16px; border-radius: 22px; border: 1px solid rgba(67,101,139,.13); background: linear-gradient(145deg, rgba(255,255,255,.98), rgba(246,249,252,.96)); box-shadow: 0 8px 22px rgba(30,61,89,.08); overflow: hidden; transition: transform .25s ease, box-shadow .25s ease, border-color .25s ease; }
    .stat-card::before { content: ""; position: absolute; top: 0; left: 0; right: 0; height: 4px; background: linear-gradient(90deg, var(--highlight-color), var(--secondary-color)); opacity: .9; }
    .stat-card:hover { transform: translateY(-5px); box-shadow: 0 14px 30px rgba(30,61,89,.14); border-color: rgba(255,110,64,.30); }
    .stat-label { display: flex; align-items: center; gap: 8px; margin-bottom: 9px; color: #647487; font-size: .88rem; font-weight: 700; letter-spacing: .01em; }
    .stat-icon { width: 28px; height: 28px; display: inline-flex; align-items: center; justify-content: center; border-radius: 9px; background: rgba(78,137,174,.10); font-size: .90rem; }
    .stat-value { color: var(--text-color); font-size: clamp(1.28rem,1.6vw,1.72rem); font-weight: 800; line-height: 1.1; white-space: nowrap; font-variant-numeric: tabular-nums; }
    .stat-unit { margin-left: 4px; color: #7c8998; font-size: .72em; font-weight: 700; }
    .stat-note { margin-top: 7px; color: #98a3af; font-size: .72rem; line-height: 1.25; }
    @media (max-width: 1100px) { .stats-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); } }
    @media (max-width: 700px) { .stats-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
    @media (max-width: 460px) { .stats-grid { grid-template-columns: 1fr; } }

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
        0% {
            opacity: 0;
            transform: translateY(10px);
        }

        100% {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes pulse {
        0% {
            box-shadow: 0 0 0 0 rgba(255,110,64, 0.7);
        }

        70% {
            box-shadow: 0 0 0 10px rgba(255,110,64, 0);
        }

        100% {
            box-shadow: 0 0 0 0 rgba(255,110,64, 0);
        }
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
        <img
            src="https://raw.githubusercontent.com/Emma-Ok/BootcampTalentoTech/main/Escudo-UdeA.svg.png"
            alt="Escudo UdeA"
        >
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
@st.cache_data(show_spinner=True)
def obtener_datos_por_rango(f_ini, f_fin):

    # ID del dataset en el backend de SIMEM
    dataset_id = "96D56E"

    # Normalización: ajusta los días al primer día del mes
    # para recorrer mes a mes
    f_ini = pd.to_datetime(f_ini).replace(day=1)
    f_fin = pd.to_datetime(f_fin).replace(day=1)

    meses = pd.date_range(
        f_ini,
        f_fin,
        freq="MS"
    )

    dfs = []

    for fecha in meses:

        # Determina el inicio y fin de cada mes
        f_inicio_mes = fecha.date()
        f_fin_mes = (
            fecha + pd.offsets.MonthEnd(0)
        ).date()

        # Construye la URL con parámetros
        url = (
            "https://www.simem.co/backend-files/api/PublicData"
            f"?startDate={f_inicio_mes}"
            f"&enddate={f_fin_mes}"
            f"&datasetId={dataset_id}"
        )

        try:

            # Llamada HTTP con timeout de 30s
            r = requests.get(
                url,
                timeout=30
            )

            if r.status_code == 200:

                payload = r.json()

                # Extrae registros dentro del JSON anidado
                datos = payload.get(
                    "result",
                    {}
                ).get(
                    "records",
                    []
                )

                if datos:

                    df_mes = pd.DataFrame(datos)

                    # Parseo de tipos
                    df_mes["Fecha"] = pd.to_datetime(
                        df_mes["Fecha"]
                    )

                    df_mes["Valor"] = pd.to_numeric(
                        df_mes["Valor"],
                        errors="coerce"
                    )

                    # Quita filas sin valor numérico
                    df_mes = df_mes.dropna(
                        subset=["Valor"]
                    )

                    dfs.append(df_mes)

            else:

                # Notifica si la API responde con error HTTP
                st.error(
                    f"Error en {fecha.strftime('%B %Y')}: "
                    f"Código {r.status_code}"
                )

        except Exception as e:

            # Captura errores de red/parseo
            st.error(
                f"Error en {fecha.strftime('%B %Y')}: {e}"
            )

    # Concatena todos los meses y ordena por fecha
    if dfs:

        out = (
            pd.concat(dfs)
            .sort_values("Fecha")
            .reset_index(drop=True)
        )

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

    df["Mes"] = df["Fecha"].dt.to_period("M")

    imgs = []

    fixed_size = (1200, 600)

    for mes in df["Mes"].unique():

        data = df[
            df["Mes"] == mes
        ]

        mes_txt = datetime.strptime(
            str(mes),
            "%Y-%m"
        ).strftime("%B %Y")

        # Figura por mes
        fig, ax = plt.subplots(
            figsize=(12, 6)
        )

        # Serie diaria
        ax.plot(
            data["Fecha"],
            data["Valor"],
            color="black",
            marker="o",
            markersize=4,
            markerfacecolor="blue",
            linewidth=1.5,
            label="Datos"
        )

        # Líneas de referencia
        ax.axhline(
            data["Valor"].mean(),
            color="purple",
            linestyle="-",
            linewidth=1,
            label="Promedio"
        )

        ax.axhline(
            data["Valor"].max(),
            color="red",
            linestyle="--",
            linewidth=1,
            label="Máximo"
        )

        ax.axhline(
            data["Valor"].min(),
            color="blue",
            linestyle="--",
            linewidth=1,
            label="Mínimo"
        )

        # Media móvil como señal de tendencia
        ax.plot(
            data["Fecha"],
            data["Valor"]
            .rolling(
                5,
                min_periods=1
            )
            .mean(),
            linestyle="--",
            color="black",
            linewidth=2,
            label="Tendencia"
        )

        # Etiquetas y formato de fechas
        ax.set_title(
            f"Precio Energía - {mes_txt}"
        )

        ax.set_xlabel("Fecha")

        ax.set_ylabel(
            "Precio (COP/kWh)"
        )

        ax.legend()

        ax.grid(True)

        ax.xaxis.set_major_formatter(
            mdates.DateFormatter("%b %Y")
        )

        fig.tight_layout()

        # Convierte la figura a imagen
        buf = io.BytesIO()

        fig.savefig(
            buf,
            format="png",
            bbox_inches="tight",
            dpi=150
        )

        buf.seek(0)

        plt.close(fig)

        img_pil = (
            Image.open(buf)
            .convert("RGB")
        )

        img_pil = img_pil.resize(
            fixed_size
        )

        imgs.append(img_pil)

    # Ensambla el GIF
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

    gif_buf.name = (
        "grafico_precios_mes.gif"
    )

    return gif_buf


# =========================
# Tabs de la interfaz
# =========================
tab, tab2 = st.tabs(
    [
        "Consulta & Análisis",
        "Graficas"
    ]
)

# =========================
# Pestaña 1: Consulta & Análisis
# =========================
with tab:

    if usar_api:

        # Llama a la API según el rango dado
        df = obtener_datos_por_rango(
            fecha_inicio,
            fecha_fin
        )

        if not df.empty:

            st.subheader(
                "Datos obtenidos"
            )

            st.dataframe(df)

            # Exportación de CSV para descarga
            csv = (
                df.to_csv(index=False)
                .encode("utf-8")
            )

            st.download_button(
                "Descargar CSV",
                csv,
                file_name="precios_xm.csv",
                mime="text/csv"
            )

            st.success(
                f"Datos obtenidos: {len(df)} registros"
            )

            # KPIs básicos descriptivos
            st.subheader("Estadísticas descriptivas")

            promedio = df["Valor"].mean()
            maximo = df["Valor"].max()
            minimo = df["Valor"].min()
            desviacion = df["Valor"].std()
            mediana = df["Valor"].median()

            st.markdown(
                f"""
                <div class="stats-shell">
                    <div class="stats-grid">
                        <div class="stat-card" title="Precio promedio del periodo consultado"><div class="stat-label"><span class="stat-icon">◉</span>Promedio</div><div class="stat-value">{promedio:.2f}<span class="stat-unit">COP</span></div><div class="stat-note">Media del periodo</div></div>
                        <div class="stat-card" title="Precio máximo registrado en el periodo"><div class="stat-label"><span class="stat-icon">↗</span>Máximo</div><div class="stat-value">{maximo:.2f}<span class="stat-unit">COP</span></div><div class="stat-note">Mayor valor observado</div></div>
                        <div class="stat-card" title="Precio mínimo registrado en el periodo"><div class="stat-label"><span class="stat-icon">↘</span>Mínimo</div><div class="stat-value">{minimo:.2f}<span class="stat-unit">COP</span></div><div class="stat-note">Menor valor observado</div></div>
                        <div class="stat-card" title="Dispersión de los precios respecto al promedio"><div class="stat-label"><span class="stat-icon">σ</span>Desviación</div><div class="stat-value">{desviacion:.2f}<span class="stat-unit">COP</span></div><div class="stat-note">Variabilidad del periodo</div></div>
                        <div class="stat-card" title="Valor central de la distribución ordenada"><div class="stat-label"><span class="stat-icon">◆</span>Mediana</div><div class="stat-value">{mediana:.2f}<span class="stat-unit">COP</span></div><div class="stat-note">Punto medio de los datos</div></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Generación y visualización de GIF mensual
            st.markdown("---")

            st.subheader(
                "GIF de Precios Mensuales"
            )

            gif_img = generar_gif(df)

            st.image(
                gif_img,
                caption=(
                    "Evolución mensual "
                    "del precio de energía"
                ),
                use_container_width=True
            )

            st.download_button(
                "Descargar GIF",
                gif_img,
                file_name="precios_mes.gif",
                mime="image/gif"
            )

    else:

        st.info(
            "Activa **Conectar a API** "
            "para consultar y visualizar los datos."
        )


# =========================
# Nueva pestaña:
# Solo Gráficas con explicación dinámica
# =========================
with tab2:

    st.subheader(
        "📊 Visualizaciones clave (sin modelo)"
    )

    if usar_api:

        # Verifica que 'df' exista
        try:
            df

        except NameError:

            st.warning(
                "Primero ve a **Consulta & Análisis**, "
                "activa **Conectar a API** "
                "y carga los datos."
            )

        else:

            if df.empty:

                st.info(
                    "No hay datos para graficar todavía."
                )

            else:

                # Imports locales
                import calendar
                import numpy as np

                # =========================
                # Helpers de explicación dinámica
                # =========================
                def trend_text(
                    series_vals,
                    freq_label
                ):

                    """
                    Describe tendencia simple,
                    cambio % y fuerza (R²)
                    de una regresión lineal.
                    """

                    s = pd.Series(
                        series_vals
                    ).dropna()

                    if len(s) < 3:
                        return (
                            "Serie muy corta "
                            "para evaluar tendencia."
                        )

                    x = np.arange(
                        len(s)
                    )

                    coef = np.polyfit(
                        x,
                        s.values,
                        1
                    )

                    yhat = (
                        coef[0] * x
                        + coef[1]
                    )

                    # Cálculo de R² manual
                    ss_res = np.sum(
                        (s.values - yhat) ** 2
                    )

                    ss_tot_calc = np.sum(
                        (s.values - s.mean()) ** 2
                    )

                    ss_tot = (
                        ss_tot_calc
                        if ss_tot_calc != 0
                        else 0
                    )

                    r2 = (
                        0.0
                        if ss_tot == 0
                        else 1 - ss_res / ss_tot
                    )

                    change_pct = (
                        (
                            s.iloc[-1]
                            / s.iloc[0]
                            - 1
                        )
                        * 100
                        if s.iloc[0] != 0
                        else np.nan
                    )

                    dir_txt = (
                        "al alza 📈"
                        if change_pct > 0
                        else (
                            "a la baja 📉"
                            if change_pct < 0
                            else "estable ➖"
                        )
                    )

                    # Clasificación verbal
                    if r2 >= 0.7:
                        fuerza = "fuerte"

                    elif r2 >= 0.4:
                        fuerza = "moderada"

                    else:
                        fuerza = "débil"

                    return (
                        f"Tendencia {dir_txt} "
                        f"en el periodo "
                        f"{freq_label.lower()} "
                        f"({change_pct:+.2f}%). "
                        f"Señal {fuerza} "
                        f"(R²={r2:.2f})."
                    )


                def dist_text(s):

                    """
                    Resumen de distribución:
                    media, mediana, desviación,
                    rango y sesgo.
                    """

                    s = s.dropna()

                    if s.empty:
                        return (
                            "Sin datos para distribución."
                        )

                    rango = (
                        s.min(),
                        s.max()
                    )

                    skew = s.skew()

                    if abs(skew) < 0.3:
                        sesgo = "simétrica"

                    elif skew > 0:
                        sesgo = (
                            "con cola a la derecha "
                            "(picos altos poco frecuentes)"
                        )

                    else:
                        sesgo = (
                            "con cola a la izquierda "
                            "(picos bajos poco frecuentes)"
                        )

                    return (
                        f"Media {s.mean():.2f}, "
                        f"mediana {s.median():.2f}, "
                        f"desviación {s.std():.2f}. "
                        f"Rango "
                        f"[{rango[0]:.2f}, "
                        f"{rango[1]:.2f}]. "
                        f"Distribución {sesgo}."
                    )


                def box_text(df_box):

                    """
                    Comentario dinámico
                    para boxplot mensual.
                    """

                    if df_box.empty:
                        return (
                            "Sin datos mensuales suficientes."
                        )

                    med = (
                        df_box
                        .groupby("Mes")["Valor"]
                        .median()
                        .sort_values(
                            ascending=False
                        )
                    )

                    iqr = (
                        df_box
                        .groupby("Mes")["Valor"]
                        .apply(
                            lambda x:
                            x.quantile(0.75)
                            - x.quantile(0.25)
                        )
                        .sort_values(
                            ascending=False
                        )
                    )

                    top_mes = med.index[0]

                    bot_mes = med.index[-1]

                    var_mes = iqr.index[0]

                    return (
                        "Mes con mediana más alta: "
                        f"**{top_mes}**; "
                        "más baja: "
                        f"**{bot_mes}**. "
                        "Mayor variabilidad (IQR) en "
                        f"**{var_mes}**."
                    )


                def heat_text(piv):

                    """
                    Lee máximos y mínimos
                    del mapa de calor Año-Mes.
                    """

                    if piv.isna().all().all():
                        return (
                            "Sin datos suficientes "
                            "para mapa de calor."
                        )

                    max_val = np.nanmax(
                        piv.values
                    )

                    min_val = np.nanmin(
                        piv.values
                    )

                    max_pos = np.where(
                        piv.values == max_val
                    )

                    min_pos = np.where(
                        piv.values == min_val
                    )

                    y_max = piv.index[
                        max_pos[0][0]
                    ]

                    m_max = piv.columns[
                        max_pos[1][0]
                    ]

                    y_min = piv.index[
                        min_pos[0][0]
                    ]

                    m_min = piv.columns[
                        min_pos[1][0]
                    ]

                    return (
                        f"Máximo promedio: "
                        f"**{max_val:.2f}** "
                        f"en **{m_max} {y_max}**. "
                        f"Mínimo promedio: "
                        f"**{min_val:.2f}** "
                        f"en **{m_min} {y_min}**."
                    )


                def pers_text(corr):

                    """
                    Traduce la correlación lag-1
                    a una interpretación cualitativa.
                    """

                    if np.isnan(corr):

                        return (
                            "No se puede calcular "
                            "persistencia "
                            "(datos insuficientes)."
                        )

                    if corr >= 0.8:
                        lvl = "muy alta"

                    elif corr >= 0.6:
                        lvl = "alta"

                    elif corr >= 0.4:
                        lvl = "moderada"

                    elif corr >= 0.2:
                        lvl = "baja"

                    else:
                        lvl = "muy baja"

                    dirr = (
                        "positiva"
                        if corr >= 0
                        else "negativa"
                    )

                    return (
                        f"Persistencia {lvl} "
                        f"({dirr}), "
                        f"correlación lag-1 = "
                        f"{corr:.2f}."
                    )

                # =========================
                # Preparar datos base
                # =========================
                df_vis = df.copy()

                df_vis["Fecha"] = pd.to_datetime(
                    df_vis["Fecha"]
                )

                df_vis["Valor"] = pd.to_numeric(
                    df_vis["Valor"],
                    errors="coerce"
                )

                df_vis = (
                    df_vis
                    .dropna(
                        subset=["Valor"]
                    )
                    .sort_values("Fecha")
                )

                # Selector de frecuencia
                freq = st.radio(
                    "Frecuencia de agregación",
                    [
                        "Diaria",
                        "Semanal",
                        "Mensual"
                    ],
                    index=0,
                    horizontal=True
                )

                freq_map = {
                    "Diaria": "D",
                    "Semanal": "W",
                    "Mensual": "MS"
                }

                res = (
                    df_vis
                    .set_index("Fecha")
                    .resample(
                        freq_map[freq]
                    )["Valor"]
                    .mean()
                    .reset_index()
                    .rename(
                        columns={
                            "Valor": "Precio"
                        }
                    )
                )

                # =========================
                # 1) Serie temporal
                # =========================
                st.markdown(
                    "#### 1) Serie temporal "
                    "con media móvil"
                )

                win = (
                    7
                    if freq == "Diaria"
                    else (
                        4
                        if freq == "Semanal"
                        else 3
                    )
                )

                fig1, ax1 = plt.subplots(
                    figsize=(12, 5)
                )

                sns.lineplot(
                    data=res,
                    x="Fecha",
                    y="Precio",
                    linewidth=2,
                    ax=ax1,
                    label="Serie"
                )

                ax1.plot(
                    res["Fecha"],
                    res["Precio"]
                    .rolling(
                        win,
                        min_periods=1
                    )
                    .mean(),
                    linestyle="--",
                    linewidth=2,
                    label=f"Media móvil ({win})"
                )

                ax1.set_xlabel("Fecha")

                ax1.set_ylabel(
                    "Precio (COP/kWh)"
                )

                ax1.set_title(
                    f"Evolución {freq.lower()} "
                    "y media móvil"
                )

                ax1.legend(
                    loc="upper left"
                )

                ax1.grid(
                    True,
                    alpha=0.3
                )

                plt.tight_layout()

                st.pyplot(fig1)

                st.markdown(
                    f"**Explicación:** "
                    f"La línea azul es el precio promedio "
                    f"{freq.lower()} y la discontinua "
                    f"suaviza con una ventana de "
                    f"{win} periodos.\n\n"
                    f"**Análisis:** "
                    f"{trend_text(res['Precio'], freq)}"
                )

                # =========================
                # 2) Distribución
                # =========================
                st.markdown(
                    "#### 2) Distribución de precios"
                )

                fig2, ax2 = plt.subplots(
                    figsize=(12, 5)
                )

                sns.histplot(
                    res["Precio"],
                    bins=30,
                    kde=True,
                    ax=ax2
                )

                ax2.set_title(
                    "Distribución de precios"
                )

                ax2.set_xlabel(
                    "Precio (COP/kWh)"
                )

                ax2.set_ylabel(
                    "Frecuencia"
                )

                ax2.grid(
                    True,
                    alpha=0.3
                )

                plt.tight_layout()

                st.pyplot(fig2)

                st.markdown(
                    "**Explicación:** "
                    "Histograma con densidad (KDE) "
                    "para conocer rangos típicos.\n\n"
                    f"**Análisis:** "
                    f"{dist_text(res['Precio'])}"
                )

                # =========================
                # 3) Boxplot por mes
                # =========================
                st.markdown(
                    "#### 3) Estacionalidad "
                    "por mes (boxplot)"
                )

                df_box = df_vis.copy()

                df_box["MesN"] = (
                    df_box["Fecha"].dt.month
                )

                df_box["Mes"] = (
                    df_box["MesN"]
                    .apply(
                        lambda m:
                        calendar.month_name[m]
                    )
                )

                order_months = list(
                    calendar.month_name
                )[1:]

                fig3, ax3 = plt.subplots(
                    figsize=(14, 5)
                )

                sns.boxplot(
                    data=df_box,
                    x="Mes",
                    y="Valor",
                    order=order_months,
                    ax=ax3
                )

                ax3.set_xlabel("Mes")

                ax3.set_ylabel(
                    "Precio (COP/kWh)"
                )

                ax3.set_title(
                    "Distribución de precios por mes"
                )

                ax3.tick_params(
                    axis="x",
                    rotation=30
                )

                ax3.grid(
                    True,
                    axis="y",
                    alpha=0.3
                )

                plt.tight_layout()

                st.pyplot(fig3)

                st.markdown(
                    "**Explicación:** "
                    "Cada caja resume la variación mensual "
                    "(mediana, cuartiles y atípicos).\n\n"
                    f"**Análisis:** "
                    f"{box_text(df_box)}"
                )

                # =========================
                # 4) Mapa de calor
                # =========================
                st.markdown(
                    "#### 4) Mapa de calor "
                    "Año vs Mes (promedio)"
                )

                df_hm = df_vis.copy()

                df_hm["Año"] = (
                    df_hm["Fecha"].dt.year
                )

                df_hm["MesN"] = (
                    df_hm["Fecha"].dt.month
                )

                piv = (
                    df_hm
                    .pivot_table(
                        index="Año",
                        columns="MesN",
                        values="Valor",
                        aggfunc="mean"
                    )
                    .reindex(
                        columns=range(1, 13)
                    )
                )

                piv.columns = [
                    calendar.month_abbr[c]
                    for c in piv.columns
                ]

                fig4, ax4 = plt.subplots(
                    figsize=(12, 6)
                )

                sns.heatmap(
                    piv,
                    annot=False,
                    fmt=".1f",
                    linewidths=0.3,
                    ax=ax4
                )

                ax4.set_title(
                    "Promedio de precios "
                    "por Año y Mes"
                )

                plt.tight_layout()

                st.pyplot(fig4)

                st.markdown(
                    "**Explicación:** "
                    "Colores más intensos indican "
                    "promedios más altos.\n\n"
                    f"**Análisis:** "
                    f"{heat_text(piv)}"
                )

                # =========================
                # 5) Persistencia
                # =========================
                st.markdown(
                    "#### 5) Persistencia "
                    "(Valor vs. Valor anterior)"
                )

                df_lag = df_vis.copy()

                df_lag["Valor_lag1"] = (
                    df_lag["Valor"]
                    .shift(1)
                )

                df_lag = (
                    df_lag.dropna()
                )

                fig5, ax5 = plt.subplots(
                    figsize=(12, 5)
                )

                sns.regplot(
                    data=df_lag,
                    x="Valor_lag1",
                    y="Valor",
                    color="purple",
                    ax=ax5,
                    scatter_kws={
                        "s": 25,
                        "alpha": 0.6
                    }
                )

                ax5.set_xlabel(
                    "Precio periodo anterior "
                    "(COP/kWh)"
                )

                ax5.set_ylabel(
                    "Precio actual (COP/kWh)"
                )

                ax5.set_title(
                    "Relación precio vs. "
                    "rezago (lag-1)"
                )

                ax5.grid(
                    True,
                    alpha=0.3
                )

                plt.tight_layout()

                st.pyplot(fig5)

                corr = (
                    df_lag["Valor_lag1"]
                    .corr(
                        df_lag["Valor"]
                    )
                    if not df_lag.empty
                    else np.nan
                )

                st.markdown(
                    "**Explicación:** "
                    "Compara el precio actual "
                    "con el del periodo previo "
                    "para medir inercia.\n\n"
                    f"**Análisis:** "
                    f"{pers_text(float(corr) if pd.notna(corr) else np.nan)}"
                )

                # =========================
                # 6) Top picos y valles
                # =========================
                st.markdown(
                    "#### 6) Top 10 picos y valles "
                    "(últimos 12 meses)"
                )

                ult_12m = df_vis[
                    df_vis["Fecha"]
                    >= (
                        df_vis["Fecha"].max()
                        - pd.Timedelta(
                            days=365
                        )
                    )
                ]

                if ult_12m.empty:

                    st.info(
                        "No hay suficientes datos "
                        "en los últimos 12 meses "
                        "para este resumen."
                    )

                else:

                    top_max = (
                        ult_12m
                        .nlargest(
                            10,
                            "Valor"
                        )[
                            [
                                "Fecha",
                                "Valor"
                            ]
                        ]
                        .rename(
                            columns={
                                "Valor": "Precio"
                            }
                        )
                    )

                    top_min = (
                        ult_12m
                        .nsmallest(
                            10,
                            "Valor"
                        )[
                            [
                                "Fecha",
                                "Valor"
                            ]
                        ]
                        .rename(
                            columns={
                                "Valor": "Precio"
                            }
                        )
                    )

                    colm1, colm2 = (
                        st.columns(2)
                    )

                    with colm1:

                        st.write(
                            "**Máximos (Top 10)**"
                        )

                        st.dataframe(
                            top_max.reset_index(
                                drop=True
                            )
                        )

                    with colm2:

                        st.write(
                            "**Mínimos (Top 10)**"
                        )

                        st.dataframe(
                            top_min.reset_index(
                                drop=True
                            )
                        )

                    # Análisis dinámico
                    r = (
                        ult_12m["Valor"].max()
                        - ult_12m["Valor"].min()
                    )

                    st.markdown(
                        "**Explicación:** "
                        "Listado de los picos más altos "
                        "y más bajos del último año.\n\n"
                        f"**Análisis:** "
                        f"Amplitud anual ≈ "
                        f"**{r:.2f}** COP/kWh. "
                        f"Último valor real: "
                        f"**{df_vis['Valor'].iloc[-1]:.2f}** "
                        "COP/kWh."
                    )

    else:

        st.info(
            "Activa **Conectar a API** "
            "para visualizar las gráficas."
        )


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
