import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
matplotlib.use('Agg') 
import seaborn as sns
import io


# Configuración inicial de la aplicación
st.set_page_config(page_title="Precios XM", layout="wide", page_icon="⚡")
st.title("Consulta de Precios XM", )

# Panel lateral para filtrado por fechas
st.sidebar.header("Filtro de Fechas")
fecha_inicio = st.sidebar.date_input("Fecha inicial")
fecha_fin = st.sidebar.date_input("Fecha final")

st.sidebar.info("El rango de fechas no puede ser mayor a un mes.")

# Validación de las fechas ingresadas
if fecha_inicio > fecha_fin:
    st.sidebar.error("La fecha inicial no puede ser posterior a la fecha final.")
    st.stop()

# Opción para conectar a la API
usar_api = st.sidebar.checkbox("Conectar a API XM", value=False)

# Función para obtener datos de la API XM
def obtener_datos_desde_api(start_date, end_date):
    dataset_id = '96D56E'
    url = f"https://www.simem.co/backend-files/api/PublicData?startDate={start_date}&enddate={end_date}&datasetId={dataset_id}"

    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            records = data.get("result", {}).get("records", [])
            if records:
                df = pd.DataFrame(records)
                df.columns = df.columns.str.lower()
                df['fecha'] = pd.to_datetime(df['fecha'])
                return df
            else:
                st.warning("No se encontraron registros en la respuesta.")
        else:
            st.error(f"Error al consultar la API: Código {response.status_code}")
    except Exception as e:
        st.error(f"Error de conexión: {e}")
    return pd.DataFrame()

# Lógica principal cuando se conecta a la API
if usar_api:
    start_str = fecha_inicio.strftime("%Y-%m-%d")
    end_str = fecha_fin.strftime("%Y-%m-%d")

    df = obtener_datos_desde_api(start_str, end_str)

    if not df.empty:
        # Muestra los datos obtenidos
        st.subheader("Datos obtenidos")
        st.dataframe(df)

        # Opción para descargar los datos
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Descargar datos como CSV",
            data=csv,
            file_name='precios_xm.csv',
            mime='text/csv')

        st.success(f"Datos obtenidos correctamente: {len(df)} registros")

        # Muestra estadísticas descriptivas
        st.subheader("Estadísticas descriptivas")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Promedio", f"{df['valor'].mean():.2f}")
        col2.metric("Valor máximo", f"{df['valor'].max():.2f}")
        col3.metric("Valor mínimo", f"{df['valor'].min():.2f}")
        col4.metric("Desviación estándar", f"{df['valor'].std():.2f}")
        col5.metric("Mediana", f"{df['valor'].median():.2f}")

        # Visualización de gráficos
        st.subheader("Visualización de series temporales")
        tipo_grafico = st.selectbox(
            "Tipo de gráfico", 
            ["Línea", "Barras", "Boxplot"],
            key="tipo_grafico")

        df.set_index("fecha", inplace=True)

        if tipo_grafico == "Línea":
            st.line_chart(df["valor"])
                   
        elif tipo_grafico == "Barras":
            plt.figure(figsize=(10, 6))
            plt.bar(df.index, df["valor"])
            plt.title("Precios XM - Gráfico de Barras")
            plt.ylim(0, df["valor"].max() * 1.2)
            plt.xticks(rotation=45)
            st.pyplot(plt)
            
            # Botón para descargar gráfico de barras
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            st.download_button(
                label="Descargar (PNG)",
                data=buf,
                file_name="grafico_barras.png",
                mime="image/png")
            
        else:  # Gráfico de cajas
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=df["valor"])
            plt.title("Distribución de Precios")
            st.pyplot(plt)
            
            # Botón para descargar boxplot
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            st.download_button(
            label="Descargar (PNG)",
            data=buf,
            file_name="boxplot_precios.png",
            mime="image/png")
        
# Pie de página con información del autor
st.markdown("---")
st.markdown("**Autor:** Yoseth Mosquera")
st.markdown("**Universidad:** Universidad de Antioquia")

