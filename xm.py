import io
import calendar
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image


# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================
st.set_page_config(
    page_title="Precios XM | Mercado Eléctrico Colombiano",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = "https://www.simem.co/backend-files/api/PublicData"
DATASET_ID = "96D56E"
REQUEST_TIMEOUT = 20
MAX_WORKERS = 6


# =========================================================
# CSS — DISEÑO MODERNO, LIMPIO Y LIGERO
# =========================================================
st.markdown(
    """
    <style>
        :root {
            --bg: #f6f8fc;
            --surface: #ffffff;
            --surface-soft: #eef4fb;
            --primary: #123b63;
            --primary-2: #1f5f93;
            --accent: #f59e0b;
            --text: #172033;
            --muted: #667085;
            --border: #dce5ef;
            --success: #157347;
            --radius: 16px;
            --shadow: 0 8px 24px rgba(18, 59, 99, 0.08);
        }

        html {
            scroll-behavior: smooth;
        }

        .stApp {
            background:
                radial-gradient(
                    circle at 100% 0%,
                    rgba(31,95,147,.08),
                    transparent 28%
                ),
                radial-gradient(
                    circle at 0% 20%,
                    rgba(245,158,11,.05),
                    transparent 24%
                ),
                var(--bg);

            color: var(--text);
        }

        [data-testid="stAppViewContainer"] > .main .block-container {
            max-width: 1500px;
            padding-top: 1.4rem;
            padding-bottom: 2.5rem;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(
                180deg,
                #0f2f4d 0%,
                #123b63 100%
            );

            border-right: 1px solid rgba(255,255,255,.08);
        }

        [data-testid="stSidebar"] * {
            color: #f8fbff;
        }

        [data-testid="stSidebar"] input,
        [data-testid="stSidebar"] [data-baseweb="input"],
        [data-testid="stSidebar"] [data-baseweb="select"] {
            color: var(--text) !important;
        }

        .hero {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 24px;

            padding: 24px 26px;
            margin-bottom: 18px;

            border: 1px solid var(--border);
            border-radius: 22px;

            background: linear-gradient(
                135deg,
                rgba(255,255,255,.98),
                rgba(238,244,251,.96)
            );

            box-shadow: var(--shadow);
        }

        .hero-copy h1 {
            margin: 0 0 8px 0;
            color: var(--primary);

            font-size: clamp(
                1.8rem,
                3vw,
                2.75rem
            );

            line-height: 1.08;
            letter-spacing: -0.035em;
        }

        .hero-copy p {
            margin: 0;
            color: var(--muted);

            font-size: 1rem;
            max-width: 850px;
        }

        .hero-logo {
            flex: 0 0 auto;

            background: #fff;

            padding: 10px 14px;

            border: 1px solid var(--border);
            border-radius: 14px;
        }

        .hero-logo img {
            width: 76px;
            height: 76px;

            object-fit: contain;
            display: block;
        }

        h2,
        h3 {
            color: var(--primary);
            letter-spacing: -0.02em;
        }

        [data-testid="stMetric"] {
            background: rgba(255,255,255,.94);

            border: 1px solid var(--border);
            border-radius: var(--radius);

            padding: 14px 16px;

            box-shadow:
                0 4px 16px
                rgba(18,59,99,.06);
        }

        [data-testid="stMetricLabel"] {
            color: var(--muted);
            font-weight: 650;
        }

        [data-testid="stMetricValue"] {
            color: var(--primary);
            font-weight: 800;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            padding: 6px;

            background:
                rgba(255,255,255,.86);

            border:
                1px solid
                var(--border);

            border-radius: 14px;

            box-shadow:
                0 3px 14px
                rgba(18,59,99,.05);
        }

        .stTabs [data-baseweb="tab"] {
            height: 44px;
            border-radius: 10px;

            padding: 0 18px;

            color: var(--muted);
            font-weight: 700;
        }

        .stTabs [aria-selected="true"] {
            background:
                var(--primary)
                !important;

            color:
                #fff
                !important;
        }

        .stButton > button,
        .stDownloadButton > button {
            min-height: 42px;

            border:
                1px solid
                var(--primary-2);

            border-radius: 11px;

            background:
                var(--primary);

            color: #fff;

            font-weight: 750;

            transition:
                transform .12s ease,
                box-shadow .12s ease,
                background .12s ease;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
            transform:
                translateY(-1px);

            background:
                var(--primary-2);

            color: #fff;

            box-shadow:
                0 7px 18px
                rgba(18,59,99,.16);
        }

        [data-testid="stDataFrame"] {
            border:
                1px solid
                var(--border);

            border-radius:
                var(--radius);

            overflow:
                hidden;

            box-shadow:
                0 4px 18px
                rgba(18,59,99,.05);
        }

        [data-testid="stAlert"] {
            border-radius:
                12px;

            border:
                1px solid
                var(--border);
        }

        hr {
            border: none;

            border-top:
                1px solid
                var(--border);

            margin:
                1.6rem 0;
        }

        .analysis-card {
            background:
                rgba(255,255,255,.95);

            border:
                1px solid
                var(--border);

            border-radius:
                var(--radius);

            padding:
                16px 18px;

            box-shadow:
                0 4px 16px
                rgba(18,59,99,.05);

            margin-top:
                10px;
        }

        .analysis-card b {
            color:
                var(--primary);
        }

        .footer {
            margin-top:
                34px;

            padding:
                20px;

            border-radius:
                18px;

            background:
                linear-gradient(
                    135deg,
                    #0f2f4d,
                    #1f5f93
                );

            color:
                #fff;

            text-align:
                center;

            box-shadow:
                var(--shadow);
        }

        .footer p {
            margin:
                4px 0;
        }

        .footer .muted {
            color:
                rgba(
                    255,
                    255,
                    255,
                    .74
                );

            font-size:
                .86rem;
        }

        @media (max-width: 800px) {

            .hero {
                align-items:
                    flex-start;

                padding:
                    18px;
            }

            .hero-logo {
                display:
                    none;
            }

            [data-testid="stAppViewContainer"] > .main .block-container {
                padding-left:
                    1rem;

                padding-right:
                    1rem;
            }
        }

        @media (prefers-reduced-motion: reduce) {

            *,
            *::before,
            *::after {

                animation-duration:
                    .01ms !important;

                transition-duration:
                    .01ms !important;

                scroll-behavior:
                    auto !important;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# CABECERA
# =========================================================
st.markdown(
    """
    <div class="hero">

        <div class="hero-copy">

            <h1>
                ⚡ Análisis del Mercado Eléctrico Colombiano
            </h1>

            <p>
                Consulta histórica de precios de energía de SIMEM,
                estadísticas descriptivas y visualizaciones interactivas.
            </p>

        </div>

        <div class="hero-logo">

            <a
                href="https://www.udea.edu.co"
                target="_blank"
                rel="noopener noreferrer"
            >

                <img
                    src="https://raw.githubusercontent.com/Emma-Ok/BootcampTalentoTech/main/Escudo-UdeA.svg.png"
                    alt="Universidad de Antioquia"
                >

            </a>

        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header(
    "⚙️ Parámetros de consulta"
)

hoy = date.today()

fecha_inicio = st.sidebar.date_input(
    "Fecha inicial",
    value=hoy - timedelta(days=30),
    max_value=hoy,
)

fecha_fin = st.sidebar.date_input(
    "Fecha final",
    value=hoy,
    max_value=hoy,
)

usar_api = st.sidebar.checkbox(
    "Conectar a API",
    value=False,
)

if fecha_inicio > fecha_fin:

    st.sidebar.error(
        "La fecha inicial no puede ser mayor a la fecha final."
    )

    st.stop()


st.sidebar.caption(
    """
    Los datos se consultan desde SIMEM
    y se almacenan temporalmente en caché
    para evitar solicitudes repetidas.
    """
)


# =========================================================
# CONSULTA DE UN MES
# =========================================================
def _consultar_mes(
    fecha_mes: pd.Timestamp
):

    inicio_mes = fecha_mes.date()

    fin_mes = (
        fecha_mes
        + pd.offsets.MonthEnd(0)
    ).date()

    params = {
        "startDate": inicio_mes,
        "enddate": fin_mes,
        "datasetId": DATASET_ID,
    }

    try:

        respuesta = requests.get(
            API_URL,
            params=params,
            timeout=REQUEST_TIMEOUT,
        )

        respuesta.raise_for_status()

        payload = respuesta.json()

        registros = (
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

        return (
            fecha_mes,
            registros,
            None,
        )

    except (
        requests.RequestException,
        ValueError,
    ) as exc:

        return (
            fecha_mes,
            [],
            str(exc),
        )


# =========================================================
# CONSULTA COMPLETA
# =========================================================
@st.cache_data(
    ttl=3600,
    show_spinner=False,
)
def obtener_datos_por_rango(
    f_ini,
    f_fin,
):

    inicio = (
        pd.Timestamp(f_ini)
        .normalize()
    )

    fin = (
        pd.Timestamp(f_fin)
        .normalize()
    )

    meses = pd.date_range(
        inicio.replace(
            day=1
        ),
        fin.replace(
            day=1
        ),
        freq="MS",
    )

    registros_totales = []

    errores = []

    workers = min(
        MAX_WORKERS,
        max(
            1,
            len(meses)
        ),
    )

    with ThreadPoolExecutor(
        max_workers=workers
    ) as executor:

        futuros = {
            executor.submit(
                _consultar_mes,
                mes,
            ): mes

            for mes in meses
        }

        for futuro in as_completed(
            futuros
        ):

            mes, registros, error = (
                futuro.result()
            )

            if error:

                errores.append(
                    f"{mes.strftime('%Y-%m')}: {error}"
                )

            elif registros:

                registros_totales.extend(
                    registros
                )

    if not registros_totales:

        return (
            pd.DataFrame(),
            errores,
        )

    df = pd.DataFrame.from_records(
        registros_totales
    )

    if (
        "Fecha" not in df.columns
        or
        "Valor" not in df.columns
    ):

        errores.append(
            """
            La respuesta de la API
            no contiene las columnas
            'Fecha' y 'Valor'.
            """
        )

        return (
            pd.DataFrame(),
            errores,
        )

    df["Fecha"] = pd.to_datetime(
        df["Fecha"],
        errors="coerce",
    )

    df["Valor"] = pd.to_numeric(
        df["Valor"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "Fecha",
            "Valor",
        ]
    )

    # Filtrar exactamente el rango seleccionado
    df = df[
        (
            df["Fecha"]
            >= inicio
        )
        &
        (
            df["Fecha"]
            < fin
            + pd.Timedelta(days=1)
        )
    ]

    df = (
        df
        .drop_duplicates()
        .sort_values(
            "Fecha",
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )

    return (
        df,
        errores,
    )


# =========================================================
# PREPARAR DATOS DE VISUALIZACIÓN
# =========================================================
@st.cache_data(
    show_spinner=False
)
def preparar_visualizaciones(
    df: pd.DataFrame
) -> pd.DataFrame:

    out = df[
        [
            "Fecha",
            "Valor",
        ]
    ].copy()

    out["Fecha"] = pd.to_datetime(
        out["Fecha"],
        errors="coerce",
    )

    out["Valor"] = pd.to_numeric(
        out["Valor"],
        errors="coerce",
    )

    out = (
        out
        .dropna()
        .sort_values("Fecha")
        .reset_index(drop=True)
    )

    return out


# =========================================================
# GENERAR GIF
# =========================================================
@st.cache_data(
    show_spinner=False
)
def generar_gif_bytes(
    df: pd.DataFrame
) -> bytes:

    datos = preparar_visualizaciones(
        df
    )

    datos["Mes"] = (
        datos["Fecha"]
        .dt
        .to_period("M")
    )

    imagenes = []

    for mes, data in datos.groupby(
        "Mes",
        sort=True,
    ):

        fig, ax = plt.subplots(
            figsize=(
                11,
                5.2,
            )
        )

        ax.plot(
            data["Fecha"],
            data["Valor"],
            linewidth=1.7,
            marker="o",
            markersize=3.5,
            label="Datos",
        )

        ax.axhline(
            data["Valor"].mean(),
            linestyle="-",
            linewidth=1.2,
            label="Promedio",
        )

        ax.axhline(
            data["Valor"].max(),
            linestyle="--",
            linewidth=1,
            label="Máximo",
        )

        ax.axhline(
            data["Valor"].min(),
            linestyle="--",
            linewidth=1,
            label="Mínimo",
        )

        ax.plot(
            data["Fecha"],
            data["Valor"]
            .rolling(
                5,
                min_periods=1,
            )
            .mean(),

            linestyle="--",
            linewidth=2,

            label="Media móvil",
        )

        ax.set_title(
            f"Precio de energía — {mes.strftime('%B %Y')}"
        )

        ax.set_xlabel(
            "Fecha"
        )

        ax.set_ylabel(
            "Precio (COP/kWh)"
        )

        ax.xaxis.set_major_formatter(
            mdates.DateFormatter(
                "%d %b"
            )
        )

        ax.grid(
            alpha=0.22
        )

        ax.legend(
            loc="best"
        )

        fig.tight_layout()

        buffer = io.BytesIO()

        fig.savefig(
            buffer,
            format="png",
            dpi=105,
            bbox_inches="tight",
        )

        plt.close(fig)

        buffer.seek(0)

        with Image.open(
            buffer
        ) as imagen:

            imagenes.append(
                imagen
                .convert("RGB")
                .resize(
                    (
                        1000,
                        470,
                    )
                )
            )

    if not imagenes:

        return b""

    gif_buffer = io.BytesIO()

    imagenes[0].save(
        gif_buffer,
        format="GIF",
        save_all=True,
        append_images=imagenes[1:],
        duration=900,
        loop=0,
        optimize=True,
    )

    return gif_buffer.getvalue()


# =========================================================
# ANÁLISIS DE TENDENCIA
# =========================================================
def trend_text(
    series_vals,
    freq_label,
):

    s = pd.Series(
        series_vals
    ).dropna()

    if len(s) < 3:

        return (
            "Serie muy corta para evaluar tendencia."
        )

    x = np.arange(
        len(s),
        dtype=float,
    )

    pendiente, intercepto = (
        np.polyfit(
            x,
            s.to_numpy(),
            1,
        )
    )

    yhat = (
        pendiente * x
        + intercepto
    )

    ss_res = np.sum(
        (
            s.to_numpy()
            - yhat
        ) ** 2
    )

    ss_tot = np.sum(
        (
            s.to_numpy()
            - s.mean()
        ) ** 2
    )

    r2 = (
        0.0
        if ss_tot == 0
        else
        1 - (
            ss_res
            /
            ss_tot
        )
    )

    cambio = (
        np.nan
        if s.iloc[0] == 0
        else
        (
            s.iloc[-1]
            /
            s.iloc[0]
            - 1
        ) * 100
    )

    if np.isnan(
        cambio
    ):

        direccion = (
            "sin cambio porcentual calculable"
        )

    elif cambio > 0:

        direccion = (
            "al alza 📈"
        )

    elif cambio < 0:

        direccion = (
            "a la baja 📉"
        )

    else:

        direccion = (
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

    cambio_txt = (
        "N/D"
        if np.isnan(cambio)
        else
        f"{cambio:+.2f}%"
    )

    return (
        f"Tendencia {direccion} "
        f"en el periodo {freq_label.lower()} "
        f"({cambio_txt}). "
        f"Señal {fuerza} "
        f"(R²={r2:.2f})."
    )


# =========================================================
# ANÁLISIS DISTRIBUCIÓN
# =========================================================
def dist_text(
    s
):

    s = pd.Series(
        s
    ).dropna()

    if s.empty:

        return (
            "Sin datos para distribución."
        )

    skew = s.skew()

    if abs(skew) < 0.3:

        sesgo = (
            "aproximadamente simétrica"
        )

    elif skew > 0:

        sesgo = (
            "con cola hacia valores altos"
        )

    else:

        sesgo = (
            "con cola hacia valores bajos"
        )

    return (
        f"Media {s.mean():.2f}, "
        f"mediana {s.median():.2f}, "
        f"desviación {s.std():.2f}. "
        f"Rango [{s.min():.2f}, {s.max():.2f}]. "
        f"Distribución {sesgo}."
    )


# =========================================================
# ANÁLISIS BOXPLOT
# =========================================================
def box_text(
    df_box
):

    if df_box.empty:

        return (
            "Sin datos mensuales suficientes."
        )

    med = (
        df_box
        .groupby(
            "Mes",
            observed=True,
        )["Valor"]
        .median()
        .sort_values(
            ascending=False
        )
    )

    iqr = (
        df_box
        .groupby(
            "Mes",
            observed=True,
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

    if (
        med.empty
        or
        iqr.empty
    ):

        return (
            "Sin datos mensuales suficientes."
        )

    return (
        f"Mes con mediana más alta: "
        f"**{med.index[0]}**; "
        f"más baja: "
        f"**{med.index[-1]}**. "
        f"Mayor variabilidad intercuartílica: "
        f"**{iqr.index[0]}**."
    )


# =========================================================
# ANÁLISIS HEATMAP
# =========================================================
def heat_text(
    piv
):

    if (
        piv.empty
        or
        piv.isna()
        .all()
        .all()
    ):

        return (
            "Sin datos suficientes para el mapa de calor."
        )

    valores = piv.to_numpy(
        dtype=float
    )

    max_idx = np.unravel_index(
        np.nanargmax(
            valores
        ),
        valores.shape,
    )

    min_idx = np.unravel_index(
        np.nanargmin(
            valores
        ),
        valores.shape,
    )

    max_val = valores[
        max_idx
    ]

    min_val = valores[
        min_idx
    ]

    y_max = piv.index[
        max_idx[0]
    ]

    m_max = piv.columns[
        max_idx[1]
    ]

    y_min = piv.index[
        min_idx[0]
    ]

    m_min = piv.columns[
        min_idx[1]
    ]

    return (
        f"Máximo promedio: "
        f"**{max_val:.2f}** "
        f"en **{m_max} {y_max}**. "
        f"Mínimo promedio: "
        f"**{min_val:.2f}** "
        f"en **{m_min} {y_min}**."
    )


# =========================================================
# ANÁLISIS PERSISTENCIA
# =========================================================
def pers_text(
    corr
):

    if pd.isna(
        corr
    ):

        return (
            "No se puede calcular persistencia "
            "con los datos disponibles."
        )

    abs_corr = abs(
        corr
    )

    if abs_corr >= 0.8:

        nivel = (
            "muy alta"
        )

    elif abs_corr >= 0.6:

        nivel = (
            "alta"
        )

    elif abs_corr >= 0.4:

        nivel = (
            "moderada"
        )

    elif abs_corr >= 0.2:

        nivel = (
            "baja"
        )

    else:

        nivel = (
            "muy baja"
        )

    direccion = (
        "positiva"
        if corr >= 0
        else
        "negativa"
    )

    return (
        f"Persistencia {nivel} "
        f"({direccion}), "
        f"correlación lag-1 = "
        f"{corr:.2f}."
    )


# =========================================================
# TARJETA DE ANÁLISIS
# =========================================================
def mostrar_analisis(
    texto
):

    st.markdown(
        f"""
        <div class="analysis-card">
            <b>Análisis:</b>
            {texto}
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# CARGA ÚNICA DE DATOS
# =========================================================
df = pd.DataFrame()

errores_api = []

if usar_api:

    with st.spinner(
        "Consultando SIMEM..."
    ):

        df, errores_api = (
            obtener_datos_por_rango(
                fecha_inicio,
                fecha_fin,
            )
        )

    if errores_api:

        with st.expander(
            f"⚠️ Detalles de conexión ({len(errores_api)})"
        ):

            for error in errores_api:

                st.caption(
                    error
                )


# =========================================================
# TABS PRINCIPALES
# =========================================================
tab_consulta, tab_graficas = st.tabs(
    [
        "📋 Consulta & Análisis",
        "📊 Gráficas",
    ]
)


# =========================================================
# TAB 1
# =========================================================
with tab_consulta:

    if not usar_api:

        st.info(
            """
            Activa **Conectar a API**
            en el panel lateral
            para consultar datos.
            """
        )

    elif df.empty:

        st.warning(
            """
            No se encontraron datos válidos
            para el rango seleccionado.
            """
        )

    else:

        st.subheader(
            "Resumen del periodo"
        )

        col1, col2, col3, col4, col5 = (
            st.columns(5)
        )

        col1.metric(
            "Promedio",
            f"{df['Valor'].mean():,.2f} COP",
        )

        col2.metric(
            "Máximo",
            f"{df['Valor'].max():,.2f} COP",
        )

        col3.metric(
            "Mínimo",
            f"{df['Valor'].min():,.2f} COP",
        )

        col4.metric(
            "Desviación",
            f"{df['Valor'].std():,.2f} COP",
        )

        col5.metric(
            "Mediana",
            f"{df['Valor'].median():,.2f} COP",
        )

        st.markdown(
            "---"
        )

        st.subheader(
            "Datos obtenidos"
        )

        c_info, c_download = (
            st.columns(
                [
                    3,
                    1,
                ]
            )
        )

        with c_info:

            st.caption(
                f"{len(df):,} registros · "
                f"{df['Fecha'].min().date()} "
                f"a "
                f"{df['Fecha'].max().date()}"
            )

        with c_download:

            csv = (
                df
                .to_csv(
                    index=False
                )
                .encode(
                    "utf-8-sig"
                )
            )

            st.download_button(
                "⬇️ Descargar CSV",
                data=csv,
                file_name=(
                    f"precios_xm_"
                    f"{fecha_inicio}_"
                    f"{fecha_fin}.csv"
                ),
                mime="text/csv",
                use_container_width=True,
            )

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            height=430,
        )

        st.markdown(
            "---"
        )

        st.subheader(
            "GIF mensual"
        )

        st.caption(
            """
            Se genera solo cuando lo solicitas
            para evitar recalcular imágenes
            en cada interacción.
            """
        )

        if st.button(
            "🎞️ Generar GIF",
            use_container_width=False,
        ):

            with st.spinner(
                "Generando animación..."
            ):

                st.session_state[
                    "gif_xm"
                ] = generar_gif_bytes(
                    df
                )

                st.session_state[
                    "gif_xm_key"
                ] = (
                    str(fecha_inicio),
                    str(fecha_fin),
                    len(df),
                )

        clave_actual = (
            str(fecha_inicio),
            str(fecha_fin),
            len(df),
        )

        if (
            st.session_state.get(
                "gif_xm_key"
            )
            ==
            clave_actual

            and

            st.session_state.get(
                "gif_xm"
            )
        ):

            gif_bytes = (
                st.session_state[
                    "gif_xm"
                ]
            )

            st.image(
                gif_bytes,
                caption=(
                    "Evolución mensual "
                    "del precio de energía"
                ),
                use_container_width=True,
            )

            st.download_button(
                "⬇️ Descargar GIF",
                data=gif_bytes,
                file_name="precios_mes.gif",
                mime="image/gif",
            )


# =========================================================
# TAB 2 — GRÁFICAS
# =========================================================
with tab_graficas:

    if not usar_api:

        st.info(
            """
            Activa **Conectar a API**
            para visualizar las gráficas.
            """
        )

    elif df.empty:

        st.warning(
            """
            No hay datos para graficar
            en el rango seleccionado.
            """
        )

    else:

        df_vis = (
            preparar_visualizaciones(
                df
            )
        )

        st.subheader(
            "Visualizaciones clave"
        )

        st.caption(
            """
            Para mejorar el rendimiento,
            la aplicación dibuja únicamente
            la visualización seleccionada.
            """
        )

        control1, control2 = (
            st.columns(
                [
                    1,
                    2,
                ]
            )
        )

        with control1:

            freq = st.radio(
                "Frecuencia",
                [
                    "Diaria",
                    "Semanal",
                    "Mensual",
                ],
                index=0,
                horizontal=True,
            )

        with control2:

            grafica = st.selectbox(
                "Visualización",
                [
                    "1. Serie temporal con media móvil",
                    "2. Distribución de precios",
                    "3. Estacionalidad mensual (boxplot)",
                    "4. Mapa de calor Año vs Mes",
                    "5. Persistencia (lag-1)",
                    "6. Top 10 picos y valles",
                ],
            )

        freq_map = {
            "Diaria": "D",
            "Semanal": "W",
            "Mensual": "MS",
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
            .dropna()
            .rename(
                "Precio"
            )
            .reset_index()
        )


        # =================================================
        # 1. SERIE TEMPORAL
        # =================================================
        if grafica.startswith(
            "1."
        ):

            if freq == "Diaria":

                win = 7

            elif freq == "Semanal":

                win = 4

            else:

                win = 3


            fig, ax = plt.subplots(
                figsize=(
                    12,
                    5,
                )
            )

            ax.plot(
                res["Fecha"],
                res["Precio"],
                linewidth=2,
                label="Serie",
            )

            ax.plot(
                res["Fecha"],
                res["Precio"]
                .rolling(
                    win,
                    min_periods=1,
                )
                .mean(),

                linestyle="--",

                linewidth=2,

                label=f"Media móvil ({win})",
            )

            ax.set_title(
                f"Evolución {freq.lower()} "
                f"y media móvil"
            )

            ax.set_xlabel(
                "Fecha"
            )

            ax.set_ylabel(
                "Precio (COP/kWh)"
            )

            ax.grid(
                alpha=0.25
            )

            ax.legend(
                loc="upper left"
            )

            fig.tight_layout()

            st.pyplot(
                fig,
                clear_figure=True,
            )

            mostrar_analisis(
                trend_text(
                    res["Precio"],
                    freq,
                )
            )


        # =================================================
        # 2. DISTRIBUCIÓN
        # =================================================
        elif grafica.startswith(
            "2."
        ):

            fig, ax = plt.subplots(
                figsize=(
                    12,
                    5,
                )
            )

            numero_bins = min(
                30,
                max(
                    8,
                    int(
                        np.sqrt(
                            len(res)
                        )
                    ),
                ),
            )

            ax.hist(
                res["Precio"]
                .dropna(),

                bins=numero_bins,

                alpha=0.85,
            )

            ax.axvline(
                res["Precio"].mean(),
                linestyle="--",
                linewidth=1.5,
                label="Media",
            )

            ax.axvline(
                res["Precio"].median(),
                linestyle=":",
                linewidth=1.8,
                label="Mediana",
            )

            ax.set_title(
                "Distribución de precios"
            )

            ax.set_xlabel(
                "Precio (COP/kWh)"
            )

            ax.set_ylabel(
                "Frecuencia"
            )

            ax.grid(
                axis="y",
                alpha=0.22,
            )

            ax.legend()

            fig.tight_layout()

            st.pyplot(
                fig,
                clear_figure=True,
            )

            mostrar_analisis(
                dist_text(
                    res["Precio"]
                )
            )


        # =================================================
        # 3. BOXPLOT
        # =================================================
        elif grafica.startswith(
            "3."
        ):

            df_box = (
                df_vis.copy()
            )

            df_box[
                "MesN"
            ] = (
                df_box[
                    "Fecha"
                ]
                .dt
                .month
            )

            meses_presentes = (
                sorted(
                    df_box[
                        "MesN"
                    ]
                    .dropna()
                    .unique()
                    .tolist()
                )
            )

            etiquetas = [
                calendar.month_name[
                    m
                ]

                for m
                in meses_presentes
            ]

            series = [
                df_box.loc[
                    df_box[
                        "MesN"
                    ]
                    == m,
                    "Valor",
                ]
                .dropna()
                .to_numpy()

                for m
                in meses_presentes
            ]

            fig, ax = plt.subplots(
                figsize=(
                    14,
                    5,
                )
            )

            if series:

                ax.boxplot(
                    series,
                    labels=etiquetas,
                    showfliers=True,
                )

            ax.set_title(
                "Distribución de precios por mes"
            )

            ax.set_xlabel(
                "Mes"
            )

            ax.set_ylabel(
                "Precio (COP/kWh)"
            )

            ax.tick_params(
                axis="x",
                rotation=28,
            )

            ax.grid(
                axis="y",
                alpha=0.22,
            )

            fig.tight_layout()

            st.pyplot(
                fig,
                clear_figure=True,
            )

            df_box[
                "Mes"
            ] = (
                df_box[
                    "MesN"
                ]
                .map(
                    lambda m:
                    calendar.month_name[
                        m
                    ]
                )
            )

            mostrar_analisis(
                box_text(
                    df_box
                )
            )


        # =================================================
        # 4. MAPA DE CALOR
        # =================================================
        elif grafica.startswith(
            "4."
        ):

            df_hm = (
                df_vis.copy()
            )

            df_hm[
                "Año"
            ] = (
                df_hm[
                    "Fecha"
                ]
                .dt
                .year
            )

            df_hm[
                "MesN"
            ] = (
                df_hm[
                    "Fecha"
                ]
                .dt
                .month
            )

            piv = (
                df_hm
                .pivot_table(
                    index="Año",
                    columns="MesN",
                    values="Valor",
                    aggfunc="mean",
                )
                .reindex(
                    columns=range(
                        1,
                        13,
                    )
                )
            )

            piv.columns = [
                calendar.month_abbr[
                    m
                ]

                for m
                in piv.columns
            ]

            fig, ax = plt.subplots(
                figsize=(
                    12,
                    5.5,
                )
            )

            masked = np.ma.masked_invalid(
                piv.to_numpy(
                    dtype=float
                )
            )

            im = ax.imshow(
                masked,
                aspect="auto",
                interpolation="nearest",
            )

            ax.set_title(
                "Promedio de precios por Año y Mes"
            )

            ax.set_xlabel(
                "Mes"
            )

            ax.set_ylabel(
                "Año"
            )

            ax.set_xticks(
                np.arange(
                    len(
                        piv.columns
                    )
                ),

                labels=piv.columns,
            )

            ax.set_yticks(
                np.arange(
                    len(
                        piv.index
                    )
                ),

                labels=piv.index,
            )

            fig.colorbar(
                im,
                ax=ax,
                label="COP/kWh",
            )

            fig.tight_layout()

            st.pyplot(
                fig,
                clear_figure=True,
            )

            mostrar_analisis(
                heat_text(
                    piv
                )
            )


        # =================================================
        # 5. PERSISTENCIA
        # =================================================
        elif grafica.startswith(
            "5."
        ):

            df_lag = (
                df_vis[
                    [
                        "Valor"
                    ]
                ]
                .copy()
            )

            df_lag[
                "Valor_lag1"
            ] = (
                df_lag[
                    "Valor"
                ]
                .shift(1)
            )

            df_lag = (
                df_lag
                .dropna()
            )

            if len(
                df_lag
            ) > 1:

                corr = (
                    df_lag[
                        "Valor_lag1"
                    ]
                    .corr(
                        df_lag[
                            "Valor"
                        ]
                    )
                )

            else:

                corr = np.nan


            fig, ax = plt.subplots(
                figsize=(
                    12,
                    5,
                )
            )

            ax.scatter(
                df_lag[
                    "Valor_lag1"
                ],

                df_lag[
                    "Valor"
                ],

                s=24,

                alpha=0.55,
            )

            if len(
                df_lag
            ) >= 2:

                coef = np.polyfit(
                    df_lag[
                        "Valor_lag1"
                    ],

                    df_lag[
                        "Valor"
                    ],

                    1,
                )

                x_line = np.linspace(
                    df_lag[
                        "Valor_lag1"
                    ]
                    .min(),

                    df_lag[
                        "Valor_lag1"
                    ]
                    .max(),

                    100,
                )

                ax.plot(
                    x_line,

                    (
                        coef[0]
                        * x_line
                        + coef[1]
                    ),

                    linewidth=2,
                )

            ax.set_title(
                """
                Relación precio actual
                vs. periodo anterior
                """
            )

            ax.set_xlabel(
                """
                Precio periodo anterior
                (COP/kWh)
                """
            )

            ax.set_ylabel(
                """
                Precio actual
                (COP/kWh)
                """
            )

            ax.grid(
                alpha=0.22
            )

            fig.tight_layout()

            st.pyplot(
                fig,
                clear_figure=True,
            )

            mostrar_analisis(
                pers_text(
                    corr
                )
            )


        # =================================================
        # 6. TOP Picos y Valles
        # =================================================
        else:

            ult_12m = df_vis[
                df_vis[
                    "Fecha"
                ]
                >=
                (
                    df_vis[
                        "Fecha"
                    ]
                    .max()
                    -
                    pd.Timedelta(
                        days=365
                    )
                )
            ]

            if ult_12m.empty:

                st.info(
                    """
                    No hay datos suficientes
                    en los últimos 12 meses.
                    """
                )

            else:

                top_max = (
                    ult_12m
                    .nlargest(
                        10,
                        "Valor",
                    )[
                        [
                            "Fecha",
                            "Valor",
                        ]
                    ]
                    .rename(
                        columns={
                            "Valor":
                            "Precio"
                        }
                    )
                    .reset_index(
                        drop=True
                    )
                )

                top_min = (
                    ult_12m
                    .nsmallest(
                        10,
                        "Valor",
                    )[
                        [
                            "Fecha",
                            "Valor",
                        ]
                    ]
                    .rename(
                        columns={
                            "Valor":
                            "Precio"
                        }
                    )
                    .reset_index(
                        drop=True
                    )
                )

                c1, c2 = (
                    st.columns(2)
                )

                with c1:

                    st.markdown(
                        "#### 🔺 Máximos"
                    )

                    st.dataframe(
                        top_max,
                        use_container_width=True,
                        hide_index=True,
                    )

                with c2:

                    st.markdown(
                        "#### 🔻 Mínimos"
                    )

                    st.dataframe(
                        top_min,
                        use_container_width=True,
                        hide_index=True,
                    )

                amplitud = (
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

                mostrar_analisis(
                    f"Amplitud del último año: "
                    f"**{amplitud:.2f} COP/kWh**. "
                    f"Último valor disponible: "
                    f"**{df_vis['Valor'].iloc[-1]:.2f} COP/kWh**."
                )


# =========================================================
# FOOTER
# =========================================================
st.markdown(
    f"""
    <div class="footer">

        <p>
            ⚡ <b>Yoseth Mosquera</b>
            · Universidad de Antioquia
        </p>

        <p>
            📊 Fuente de datos:
            <b>SIMEM</b>
        </p>

        <p class="muted">
            © {datetime.now().year}
            · Aplicación optimizada para
            consulta y visualización
        </p>

    </div>
    """,
    unsafe_allow_html=True,
)
