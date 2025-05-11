import streamlit as st
import pandas as pd
import requests
from datetime import datetime
from io import StringIO

# Configuración de la página
st.set_page_config(page_title="Precios XM", layout="wide")
st.title("Consulta de Precios XM")

# Panel lateral
st.sidebar.header("Configuración")
usar_api = st.sidebar.checkbox("Conectar a API XM", value=True)

st.sidebar.header("Filtro de Fechas")
fecha_inicio = st.sidebar.date_input("Fecha inicial", value=datetime(2025, 5, 1))
fecha_fin = st.sidebar.date_input("Fecha final", value=datetime(2025, 5, 10))

# Validación de fechas
if fecha_inicio > fecha_fin:
    st.sidebar.error("La fecha inicial no puede ser posterior a la fecha final.")

# Función para obtener datos
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

# Lógica principal
if usar_api:
    start_str = fecha_inicio.strftime("%Y-%m-%d")
    end_str = fecha_fin.strftime("%Y-%m-%d")

    df = obtener_datos_desde_api(start_str, end_str)

    if not df.empty:
        st.subheader("Datos obtenidos")
        st.dataframe(df)

        # Opción para descargar CSV
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Descargar datos como CSV",
            data=csv,
            file_name='precios_xm.csv',
            mime='text/csv'
        )

        st.success(f"Datos obtenidos correctamente: {len(df)} registros")

        st.subheader("Estadísticas descriptivas")
        col1, col2, col3 = st.columns(3)
        col1.metric("Promedio", f"{df['valor'].mean():.2f}")
        col2.metric("Valor máximo", f"{df['valor'].max():.2f}")
        col3.metric("Valor mínimo", f"{df['valor'].min():.2f}")

        st.subheader("Visualización de series temporales")
        tipo_grafico = st.selectbox("Tipo de gráfico", ["Línea", "Barras"])
        if tipo_grafico == "Línea":
            st.line_chart(df.set_index("fecha")["valor"])
        else:
            st.bar_chart(df.set_index("fecha")["valor"])
    else:
        st.error("No se encontraron datos para el período seleccionado")


st.markdown("---")
st.markdown("**Autor:** Yoseth Mosquera")
st.markdown("**Universidad:** Universidad de Antioquia")
st.markdown("**Asignatura:** Computacion Numerica")