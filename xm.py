import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import io
import imageio
from datetime import datetime

# Configuración inicial
st.set_page_config(page_title="Precios XM", layout="wide")
st.title("Consulta de Precios XM")

# Panel lateral
st.sidebar.header("Parámetros de consulta")
fecha_inicio = st.sidebar.date_input("Fecha inicial")
fecha_fin = st.sidebar.date_input("Fecha final")
usar_api = st.sidebar.checkbox("Conectar a API", value=False)

# Validación
if fecha_inicio > fecha_fin:
    st.sidebar.error("La fecha inicial no puede ser mayor a la final.")
    st.stop()

# Función para obtener datos por rango de meses
def obtener_datos_por_rango(f_ini, f_fin):
    dataset_id = '96D56E'
    f_ini = pd.to_datetime(f_ini).replace(day=1)
    f_fin = pd.to_datetime(f_fin).replace(day=1)
    meses = pd.date_range(f_ini, f_fin, freq='MS')

    dfs = []
    for fecha in meses:
        f_inicio_mes = fecha.date()
        f_fin_mes = (fecha + pd.offsets.MonthEnd(0)).date()
        url = f"https://www.simem.co/backend-files/api/PublicData?startDate={f_inicio_mes}&enddate={f_fin_mes}&datasetId={dataset_id}"

        try:
            r = requests.get(url)
            if r.status_code == 200:
                datos = r.json()["result"]["records"]
                if datos:
                    df_mes = pd.DataFrame(datos)
                    df_mes["Fecha"] = pd.to_datetime(df_mes["Fecha"])
                    dfs.append(df_mes)
            else:
                st.error(f"Error en {fecha.strftime('%B %Y')}: Código {r.status_code}")
        except Exception as e:
            st.error(f"Error en {fecha.strftime('%B %Y')}: {e}")

    if dfs:
        return pd.concat(dfs).sort_values("Fecha").reset_index(drop=True)
    else:
        return pd.DataFrame()

# Función para generar el GIF
def gif(df):
    df["Mes"] = df["Fecha"].dt.to_period("M")
    imgs = []

    for mes in df["Mes"].unique():
        data = df[df["Mes"] == mes]
        mes_txt = datetime.strptime(str(mes), "%Y-%m").strftime("%B %Y")

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(data["Fecha"], data["Valor"], color="black", marker="o", markersize=4,
                markerfacecolor="blue", linewidth=1.5, label="Datos")

        
        ax.axhline(data["Valor"].mean(), color="purple", linestyle='-', linewidth=1, label="Promedio")
        ax.axhline(data["Valor"].max(), color="red", linestyle='--', linewidth=1, label="Máximo")
        ax.axhline(data["Valor"].min(), color="blue", linestyle='--', linewidth=1, label="Mínimo")
        
        ax.plot(data["Fecha"], data["Valor"].rolling(5, min_periods=1).mean(),
                linestyle="--", color="black", linewidth=2, label="Tendencia")



        ax.set_title(f"Precio Energía - {mes_txt}")
        ax.set_xlabel("Fecha")
        ax.set_ylabel("Precio (COP/kWh)")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        plt.close(fig)
        imgs.append(imageio.imread(buf))


    gif_buf = io.BytesIO()
    imageio.mimsave(gif_buf, imgs, format="GIF", duration=500, loop=0) 
    gif_buf.seek(0)
    gif_buf.name = "grafico_precios_mes.gif"
    return gif_buf

# Lógica principal
if usar_api:
    df = obtener_datos_por_rango(fecha_inicio, fecha_fin)

    if not df.empty:
        st.subheader("Datos obtenidos")
        st.dataframe(df)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Descargar CSV", csv, "precios_xm.csv", "text/csv")

        st.success(f"Datos obtenidos: {len(df)} registros")

        st.subheader("Estadísticas descriptivas")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Promedio", f"{df['Valor'].mean():.2f} COP")
        col2.metric("Máximo", f"{df['Valor'].max():.2f} COP")
        col3.metric("Mínimo", f"{df['Valor'].min():.2f} COP")
        col4.metric("Desviación", f"{df['Valor'].std():.2f} COP")
        col5.metric("Mediana", f"{df['Valor'].median():.2f} COP")

        st.subheader("Visualización")
        tipo = st.selectbox("Tipo de gráfico", ["Línea", "Barras", "Boxplot"])

        df.set_index("Fecha", inplace=True)

        if tipo == "Línea":
            st.line_chart(df["Valor"])

        elif tipo == "Barras":
            plt.figure(figsize=(10, 6))
            plt.bar(df.index, df["Valor"])
            plt.xticks(rotation=45)
            plt.title("Precios XM - Barras")
            plt.ylim(0, df["Valor"].max() * 1.2)
            st.pyplot(plt)
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            st.download_button("Descargar gráfico (PNG)", buf, "barras.png", "image/png")

        else:
            flierprops = dict(marker='o', markerfacecolor='red', markersize=8, linestyle='none')
            sns.boxplot(x=df["Valor"], flierprops=flierprops)
            plt.title("Boxplot de Precios")
            plt.xlabel("Valor")
            st.pyplot(plt.gcf())
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            st.download_button("Descargar boxplot (PNG)", buf, "boxplot.png", "image/png")

        # GIF
        st.markdown("---")
        st.subheader("GIF de Precios Mensuales")
        gif_img = gif(df.reset_index())
        st.image(gif_img, caption="Evolución mensual del precio de energía", use_container_width=True)
        st.download_button("Descargar GIF", gif_img, "precios_mes.gif", "image/gif")

st.markdown("---")
st.markdown("**Autor:** Yoseth Mosquera")
st.markdown("**Universidad:** Universidad de Antioquia")
st.markdown("**Fuente:** Datos obtenidos de SIMEM")

