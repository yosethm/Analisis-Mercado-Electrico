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

st.set_page_config(
    page_title="Precios XM",
    layout="wide"
)

st.title(
    "⚡ Análisis del Precio del Mercado Eléctrico Colombiano 📈"
)

st.caption(
    "Estudio histórico del precio de la energía en Colombia, "
    "estadísticas descriptivas, análisis y visualizaciones."
)


# Tema visual por defecto para seaborn
sns.set_theme(style="whitegrid")


# =========================
# Sidebar
# =========================

st.sidebar.header("Parámetros de consulta")

fecha_inicio = st.sidebar.date_input(
    "Fecha inicial"
)

fecha_fin = st.sidebar.date_input(
    "Fecha final"
)

usar_api = st.sidebar.checkbox(
    "Conectar a API",
    value=False
)


# =========================
# CSS ORIGINAL
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


    .stSelectbox,
    .stSlider,
    .stNumberInput,
    .stTextInput {

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

            box-shadow:
                0 0 0 0
                rgba(255,110,64,0.7);
        }

        70% {

            box-shadow:
                0 0 0 10px
                rgba(255,110,64,0);
        }

        100% {

            box-shadow:
                0 0 0 0
                rgba(255,110,64,0);
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

        transition:
            transform 0.3s ease-in-out,
            opacity 0.3s ease-in-out;
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

    <a
        href="https://www.udea.edu.co"
        target="_blank"
    >

        <img
            src="https://raw.githubusercontent.com/Emma-Ok/BootcampTalentoTech/main/Escudo-UdeA.svg.png"
            alt="Escudo UdeA"
        >

    </a>

</div>

""", unsafe_allow_html=True)


# =========================
# Validación de fechas
# =========================

if fecha_inicio > fecha_fin:

    st.sidebar.error(
        "La fecha inicial no puede ser mayor a la final."
    )

    st.stop()


# =========================
# Función para obtener datos
# =========================

@st.cache_data(show_spinner=True)
def obtener_datos_por_rango(
    f_ini,
    f_fin
):

    dataset_id = "96D56E"

    # Ajustar al primer día del mes
    f_ini = (
        pd.to_datetime(f_ini)
        .replace(day=1)
    )

    f_fin = (
        pd.to_datetime(f_fin)
        .replace(day=1)
    )

    meses = pd.date_range(
        f_ini,
        f_fin,
        freq="MS"
    )

    dfs = []


    for fecha in meses:

        f_inicio_mes = fecha.date()

        f_fin_mes = (
            fecha
            + pd.offsets.MonthEnd(0)
        ).date()


        url = (

            "https://www.simem.co/backend-files/api/PublicData"

            f"?startDate={f_inicio_mes}"

            f"&enddate={f_fin_mes}"

            f"&datasetId={dataset_id}"
        )


        try:

            r = requests.get(
                url,
                timeout=30
            )


            if r.status_code == 200:

                payload = r.json()

                datos = (
                    payload
                    .get(
                        "result",
                        {}
                    )
                    .get(
                        "records",
                        []
                    )
                )


                if datos:

                    df_mes = pd.DataFrame(
                        datos
                    )


                    df_mes["Fecha"] = (
                        pd.to_datetime(
                            df_mes["Fecha"]
                        )
                    )


                    df_mes["Valor"] = (
                        pd.to_numeric(
                            df_mes["Valor"],
                            errors="coerce"
                        )
                    )


                    df_mes = (
                        df_mes
                        .dropna(
                            subset=["Valor"]
                        )
                    )


                    dfs.append(
                        df_mes
                    )


            else:

                st.error(

                    f"Error en "
                    f"{fecha.strftime('%B %Y')}: "
                    f"Código {r.status_code}"
                )


        except Exception as e:

            st.error(

                f"Error en "
                f"{fecha.strftime('%B %Y')}: "
                f"{e}"
            )


    if dfs:

        out = (

            pd.concat(dfs)

            .sort_values("Fecha")

            .reset_index(drop=True)
        )

        return out


    return pd.DataFrame()


# =========================
# Generar GIF mensual
# =========================

def generar_gif(df):

    """

    Genera un GIF animado por mes con:

    - Serie diaria
    - Promedio
    - Máximo
    - Mínimo
    - Media móvil

    """

    df = df.copy()


    df["Mes"] = (

        df["Fecha"]
        .dt
        .to_period("M")
    )


    imgs = []


    fixed_size = (
        1200,
        600
    )


    for mes in df["Mes"].unique():

        data = (
            df[
                df["Mes"] == mes
            ]
        )


        mes_txt = (

            datetime
            .strptime(
                str(mes),
                "%Y-%m"
            )
            .strftime(
                "%B %Y"
            )
        )


        fig, ax = plt.subplots(
            figsize=(12, 6)
        )


        # Serie principal
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


        # Promedio
        ax.axhline(

            data["Valor"].mean(),

            color="purple",

            linestyle="-",

            linewidth=1,

            label="Promedio"
        )


        # Máximo
        ax.axhline(

            data["Valor"].max(),

            color="red",

            linestyle="--",

            linewidth=1,

            label="Máximo"
        )


        # Mínimo
        ax.axhline(

            data["Valor"].min(),

            color="blue",

            linestyle="--",

            linewidth=1,

            label="Mínimo"
        )


        # Media móvil
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


        ax.set_title(

            f"Precio Energía - "
            f"{mes_txt}"
        )


        ax.set_xlabel(
            "Fecha"
        )


        ax.set_ylabel(
            "Precio (COP/kWh)"
        )


        ax.legend()


        ax.grid(
            True
        )


        ax.xaxis.set_major_formatter(

            mdates.DateFormatter(
                "%b %Y"
            )
        )


        fig.tight_layout()


        # Convertir gráfica en imagen
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

            Image
            .open(buf)
            .convert("RGB")
        )


        img_pil = (

            img_pil
            .resize(
                fixed_size
            )
        )


        imgs.append(
            img_pil
        )


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


# =========================================================
# TABS
# =========================================================

tab, tab2 = st.tabs(

    [
        "Consulta & Análisis",
        "Graficas"
    ]
)


# =========================================================
# TAB 1 — CONSULTA Y ANÁLISIS
# =========================================================

with tab:


    if usar_api:


        df = obtener_datos_por_rango(

            fecha_inicio,

            fecha_fin
        )


        if not df.empty:


            st.subheader(
                "Datos obtenidos"
            )


            st.dataframe(
                df
            )


            # =========================
            # Descargar CSV
            # =========================

            csv = (

                df
                .to_csv(
                    index=False
                )
                .encode(
                    "utf-8"
                )
            )


            st.download_button(

                "Descargar CSV",

                csv,

                file_name="precios_xm.csv",

                mime="text/csv"
            )


            st.success(

                f"Datos obtenidos: "
                f"{len(df)} registros"
            )


            # =========================
            # Estadísticas descriptivas
            # =========================

            st.subheader(
                "Estadísticas descriptivas"
            )


            col1, col2, col3, col4, col5 = (
                st.columns(5)
            )


            col1.metric(

                "Promedio",

                f"{df['Valor'].mean():.2f} COP"
            )


            col2.metric(

                "Máximo",

                f"{df['Valor'].max():.2f} COP"
            )


            col3.metric(

                "Mínimo",

                f"{df['Valor'].min():.2f} COP"
            )


            col4.metric(

                "Desviación",

                f"{df['Valor'].std():.2f} COP"
            )


            col5.metric(

                "Mediana",

                f"{df['Valor'].median():.2f} COP"
            )


            # =========================
            # GIF
            # =========================

            st.markdown(
                "---"
            )


            st.subheader(
                "GIF de Precios Mensuales"
            )


            gif_img = generar_gif(
                df
            )


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
            "para consultar y visualizar "
            "los datos."
        )


# =========================================================
# TAB 2 — GRÁFICAS
# =========================================================

with tab2:


    st.subheader(

        "📊 Visualizaciones clave (sin modelo)"
    )


    if usar_api:


        try:

            df


        except NameError:


            st.warning(

                "Primero ve a "
                "**Consulta & Análisis**, "
                "activa **Conectar a API** "
                "y carga los datos."
            )


        else:


            if df.empty:


                st.info(

                    "No hay datos para "
                    "graficar todavía."
                )


            else:


                import calendar

                import numpy as np


                # =========================================
                # Función análisis de tendencia
                # =========================================

                def trend_text(
                    series_vals,
                    freq_label
                ):

                    """

                    Describe:

                    - tendencia
                    - cambio %
                    - R²

                    """

                    s = (

                        pd.Series(
                            series_vals
                        )
                        .dropna()
                    )


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


                    ss_res = np.sum(

                        (
                            s.values
                            - yhat
                        ) ** 2
                    )


                    suma_total = np.sum(

                        (
                            s.values
                            - s.mean()
                        ) ** 2
                    )


                    ss_tot = (

                        suma_total

                        if suma_total != 0

                        else 0
                    )


                    r2 = (

                        0.0

                        if ss_tot == 0

                        else

                        1
                        -
                        ss_res
                        /
                        ss_tot
                    )


                    change_pct = (

                        (
                            s.iloc[-1]
                            /
                            s.iloc[0]
                            -
                            1
                        )
                        *
                        100

                        if s.iloc[0] != 0

                        else np.nan
                    )


                    if change_pct > 0:

                        dir_txt = (
                            "al alza 📈"
                        )

                    elif change_pct < 0:

                        dir_txt = (
                            "a la baja 📉"
                        )

                    else:

                        dir_txt = (
                            "estable ➖"
                        )


                    if r2 >= 0.7:

                        fuerza = (
                            "fuerte"
                        )


                    elif r2 >= 0.4:

                        fuerza = (
                            "moderada"
                        )


                    else:

                        fuerza = (
                            "débil"
                        )


                    return (

                        f"Tendencia "
                        f"{dir_txt} "
                        f"en el periodo "
                        f"{freq_label.lower()} "
                        f"({change_pct:+.2f}%). "

                        f"Señal "
                        f"{fuerza} "
                        f"(R²={r2:.2f})."
                    )


                # =========================================
                # Distribución
                # =========================================

                def dist_text(s):

                    s = (
                        s
                        .dropna()
                    )


                    if s.empty:

                        return (
                            "Sin datos "
                            "para distribución."
                        )


                    rango = (

                        s.min(),

                        s.max()
                    )


                    skew = (
                        s.skew()
                    )


                    if abs(skew) < 0.3:

                        sesgo = (
                            "simétrica"
                        )


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

                        f"mediana "
                        f"{s.median():.2f}, "

                        f"desviación "
                        f"{s.std():.2f}. "

                        f"Rango "
                        f"[{rango[0]:.2f}, "
                        f"{rango[1]:.2f}]. "

                        f"Distribución "
                        f"{sesgo}."
                    )


                # =========================================
                # Boxplot
                # =========================================

                def box_text(
                    df_box
                ):


                    if df_box.empty:

                        return (

                            "Sin datos mensuales "
                            "suficientes."
                        )


                    med = (

                        df_box

                        .groupby(
                            "Mes"
                        )["Valor"]

                        .median()

                        .sort_values(
                            ascending=False
                        )
                    )


                    iqr = (

                        df_box

                        .groupby(
                            "Mes"
                        )["Valor"]

                        .apply(

                            lambda x:

                            x.quantile(0.75)

                            -

                            x.quantile(0.25)
                        )

                        .sort_values(
                            ascending=False
                        )
                    )


                    top_mes = (
                        med.index[0]
                    )


                    bot_mes = (
                        med.index[-1]
                    )


                    var_mes = (
                        iqr.index[0]
                    )


                    return (

                        f"Mes con mediana "
                        f"más alta: "
                        f"**{top_mes}**; "

                        f"más baja: "
                        f"**{bot_mes}**. "

                        f"Mayor variabilidad "
                        f"(IQR) en "
                        f"**{var_mes}**."
                    )


                # =========================================
                # Heatmap
                # =========================================

                def heat_text(
                    piv
                ):


                    if (
                        piv
                        .isna()
                        .all()
                        .all()
                    ):

                        return (

                            "Sin datos suficientes "
                            "para mapa de calor."
                        )


                    max_val = (
                        np.nanmax(
                            piv.values
                        )
                    )


                    min_val = (
                        np.nanmin(
                            piv.values
                        )
                    )


                    max_pos = (
                        np.where(
                            piv.values
                            ==
                            max_val
                        )
                    )


                    min_pos = (
                        np.where(
                            piv.values
                            ==
                            min_val
                        )
                    )


                    y_max = (
                        piv.index[
                            max_pos[0][0]
                        ]
                    )


                    m_max = (
                        piv.columns[
                            max_pos[1][0]
                        ]
                    )


                    y_min = (
                        piv.index[
                            min_pos[0][0]
                        ]
                    )


                    m_min = (
                        piv.columns[
                            min_pos[1][0]
                        ]
                    )


                    return (

                        f"Máximo promedio: "
                        f"**{max_val:.2f}** "

                        f"en "
                        f"**{m_max} "
                        f"{y_max}**. "

                        f"Mínimo promedio: "
                        f"**{min_val:.2f}** "

                        f"en "
                        f"**{m_min} "
                        f"{y_min}**."
                    )


                # =========================================
                # Persistencia
                # =========================================

                def pers_text(
                    corr
                ):


                    if np.isnan(corr):

                        return (

                            "No se puede calcular "
                            "persistencia "
                            "(datos insuficientes)."
                        )


                    if corr >= 0.8:

                        lvl = (
                            "muy alta"
                        )


                    elif corr >= 0.6:

                        lvl = (
                            "alta"
                        )


                    elif corr >= 0.4:

                        lvl = (
                            "moderada"
                        )


                    elif corr >= 0.2:

                        lvl = (
                            "baja"
                        )


                    else:

                        lvl = (
                            "muy baja"
                        )


                    dirr = (

                        "positiva"

                        if corr >= 0

                        else "negativa"
                    )


                    return (

                        f"Persistencia "
                        f"{lvl} "
                        f"({dirr}), "

                        f"correlación "
                        f"lag-1 = "
                        f"{corr:.2f}."
                    )


                # =========================================
                # Preparar datos
                # =========================================

                df_vis = (
                    df.copy()
                )


                df_vis["Fecha"] = (
                    pd.to_datetime(
                        df_vis["Fecha"]
                    )
                )


                df_vis["Valor"] = (
                    pd.to_numeric(

                        df_vis["Valor"],

                        errors="coerce"
                    )
                )


                df_vis = (

                    df_vis

                    .dropna(
                        subset=["Valor"]
                    )

                    .sort_values(
                        "Fecha"
                    )
                )


                # =========================================
                # Selector frecuencia
                # =========================================

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

                    .set_index(
                        "Fecha"
                    )

                    .resample(
                        freq_map[
                            freq
                        ]
                    )["Valor"]

                    .mean()

                    .reset_index()

                    .rename(

                        columns={

                            "Valor":
                            "Precio"
                        }
                    )
                )


                # =========================================
                # 1. SERIE TEMPORAL
                # =========================================

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


                ax1.set_xlabel(
                    "Fecha"
                )


                ax1.set_ylabel(
                    "Precio (COP/kWh)"
                )


                ax1.set_title(

                    f"Evolución "
                    f"{freq.lower()} "
                    f"y media móvil"
                )


                ax1.legend(
                    loc="upper left"
                )


                ax1.grid(

                    True,

                    alpha=0.3
                )


                plt.tight_layout()


                st.pyplot(
                    fig1
                )


                st.markdown(

                    f"**Explicación:** "
                    f"La línea azul es el "
                    f"precio promedio "
                    f"{freq.lower()} y la "
                    f"discontinua suaviza "
                    f"con una ventana de "
                    f"{win} periodos.\n\n"

                    f"**Análisis:** "
                    f"{trend_text(res['Precio'], freq)}"
                )


                # =========================================
                # 2. DISTRIBUCIÓN
                # =========================================

                st.markdown(

                    "#### 2) Distribución "
                    "de precios"
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


                st.pyplot(
                    fig2
                )


                st.markdown(

                    f"**Explicación:** "
                    f"Histograma con densidad "
                    f"(KDE) para conocer "
                    f"rangos típicos.\n\n"

                    f"**Análisis:** "
                    f"{dist_text(res['Precio'])}"
                )


                # =========================================
                # 3. BOXPLOT POR MES
                # =========================================

                st.markdown(

                    "#### 3) Estacionalidad "
                    "por mes (boxplot)"
                )


                df_box = (
                    df_vis.copy()
                )


                df_box["MesN"] = (

                    df_box["Fecha"]
                    .dt
                    .month
                )


                df_box["Mes"] = (

                    df_box["MesN"]

                    .apply(

                        lambda m:

                        calendar.month_name[
                            m
                        ]
                    )
                )


                order_months = (

                    list(
                        calendar.month_name
                    )[1:]
                )


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


                ax3.set_xlabel(
                    "Mes"
                )


                ax3.set_ylabel(

                    "Precio (COP/kWh)"
                )


                ax3.set_title(

                    "Distribución de "
                    "precios por mes"
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


                st.pyplot(
                    fig3
                )


                st.markdown(

                    f"**Explicación:** "
                    f"Cada caja resume la "
                    f"variación mensual "
                    f"(mediana, cuartiles "
                    f"y atípicos).\n\n"

                    f"**Análisis:** "
                    f"{box_text(df_box)}"
                )


                # =========================================
                # 4. MAPA DE CALOR
                # =========================================

                st.markdown(

                    "#### 4) Mapa de calor "
                    "Año vs Mes (promedio)"
                )


                df_hm = (
                    df_vis.copy()
                )


                df_hm["Año"] = (

                    df_hm["Fecha"]
                    .dt
                    .year
                )


                df_hm["MesN"] = (

                    df_hm["Fecha"]
                    .dt
                    .month
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

                        columns=range(
                            1,
                            13
                        )
                    )
                )


                piv.columns = [

                    calendar.month_abbr[c]

                    for c

                    in piv.columns
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


                st.pyplot(
                    fig4
                )


                st.markdown(

                    f"**Explicación:** "
                    f"Colores más intensos "
                    f"indican promedios "
                    f"más altos.\n\n"

                    f"**Análisis:** "
                    f"{heat_text(piv)}"
                )


                # =========================================
                # 5. PERSISTENCIA
                # =========================================

                st.markdown(

                    "#### 5) Persistencia "
                    "(Valor vs. Valor anterior)"
                )


                df_lag = (
                    df_vis.copy()
                )


                df_lag["Valor_lag1"] = (

                    df_lag["Valor"]
                    .shift(1)
                )


                df_lag = (
                    df_lag
                    .dropna()
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

                    "Precio actual "
                    "(COP/kWh)"
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


                st.pyplot(
                    fig5
                )


                corr = (

                    df_lag[
                        "Valor_lag1"
                    ]

                    .corr(
                        df_lag[
                            "Valor"
                        ]
                    )

                    if not df_lag.empty

                    else np.nan
                )


                st.markdown(

                    f"**Explicación:** "
                    f"Compara el precio actual "
                    f"con el del periodo previo "
                    f"para medir inercia.\n\n"

                    f"**Análisis:** "

                    f"{pers_text(float(corr) if pd.notna(corr) else np.nan)}"
                )


                # =========================================
                # 6. PICOS Y VALLES
                # =========================================

                st.markdown(

                    "#### 6) Top 10 picos y "
                    "valles (últimos 12 meses)"
                )


                ult_12m = (

                    df_vis[

                        df_vis["Fecha"]

                        >=

                        (
                            df_vis["Fecha"]
                            .max()

                            -

                            pd.Timedelta(
                                days=365
                            )
                        )
                    ]
                )


                if ult_12m.empty:


                    st.info(

                        "No hay suficientes "
                        "datos en los últimos "
                        "12 meses para este "
                        "resumen."
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

                                "Valor":
                                "Precio"
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

                                "Valor":
                                "Precio"
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

                            top_max

                            .reset_index(
                                drop=True
                            )
                        )


                    with colm2:


                        st.write(

                            "**Mínimos (Top 10)**"
                        )


                        st.dataframe(

                            top_min

                            .reset_index(
                                drop=True
                            )
                        )


                    r = (

                        ult_12m[
                            "Valor"
                        ]
                        .max()

                        -

                        ult_12m[
                            "Valor"
                        ]
                        .min()
                    )


                    st.markdown(

                        f"**Explicación:** "
                        f"Listado de los picos "
                        f"más altos y más bajos "
                        f"del último año.\n\n"

                        f"**Análisis:** "
                        f"Amplitud anual ≈ "
                        f"**{r:.2f}** COP/kWh. "

                        f"Último valor real: "
                        f"**{df_vis['Valor'].iloc[-1]:.2f}** "
                        f"COP/kWh."
                    )


    else:


        st.info(

            "Activa **Conectar a API** "
            "para visualizar las gráficas."
        )


# =========================================================
# FOOTER ORIGINAL
# =========================================================

st.markdown("""
<style>

.footer {

    position: relative;

    bottom: 0;

    width: 100%;

    background:
        linear-gradient(
            90deg,
            #4e89ae,
            #43658b
        );

    color: white;

    text-align: center;

    padding: 15px 10px;

    border-radius: 8px;

    font-size: 0.9rem;

    box-shadow:
        0 4px 12px
        rgba(0,0,0,0.2);
}


.footer p {

    margin: 4px 0;
}

</style>


<div class="footer">

    <p>
        ⚡ Autor:
        <b>Yoseth Mosquera</b>
    </p>

    <p>
        🎓 Universidad:
        <b>Universidad de Antioquia</b>
    </p>

    <p>
        📊 Fuente:
        <b>Datos obtenidos de SIMEM</b>
    </p>

    <p>
        © 2024
    </p>

</div>

""", unsafe_allow_html=True)
