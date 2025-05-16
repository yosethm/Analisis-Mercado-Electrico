import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Configuración inicial
st.set_page_config(page_title="Precios XM", layout="wide", page_icon="⚡")
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
        col1.metric("Promedio", f"{df['Valor'].mean():.2f}")
        col2.metric("Máximo", f"{df['Valor'].max():.2f}")
        col3.metric("Mínimo", f"{df['Valor'].min():.2f}")
        col4.metric("Desviación", f"{df['Valor'].std():.2f}")
        col5.metric("Mediana", f"{df['Valor'].median():.2f}")

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
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=df["Valor"])
            plt.title("Boxplot de Precios")
            st.pyplot(plt)
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            st.download_button("Descargar boxplot (PNG)", buf, "boxplot.png", "image/png")

st.markdown("---")
st.markdown("**Autor:** Yoseth Mosquera")
st.markdown("**Universidad:** Universidad de Antioquia")
st.markdown("**Datos:** Datos obtenidos de SIMEM")



