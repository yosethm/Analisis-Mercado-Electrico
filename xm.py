import streamlit as st
import pandas as pd
import requests
from datetime import datetime

st.set_page_config(page_title="Precios XM", layout="wide")
st.title("💡 Obtener Datos de XM")

st.sidebar.header("🔌 Conectarse directamente a XM")
usar_api = st.sidebar.checkbox("Conectarse directamente a XM", value=True)

st.sidebar.header("📅 Filtro de Fechas")
fecha_inicio = st.sidebar.date_input("Desde", value=datetime(2025, 5, 1))
fecha_fin = st.sidebar.date_input("Hasta", value=datetime(2025, 5, 10))

if fecha_inicio > fecha_fin:
    st.sidebar.error("La fecha de inicio no puede ser posterior a la fecha de fin.")

def obtener_datos_desde_api(start_date, end_date):
    dataset_id = '96D56E'
    column_destiny = 'null'
    values = 'null'

    url = f"https://www.simem.co/backend-files/api/PublicData?startDate={start_date}&enddate={end_date}&datasetId={dataset_id}&columnDestinyName={column_destiny}&values={values}"
    
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
                st.warning("⚠️ No se encontraron registros en la respuesta.")
        else:
            st.error(f"Error {response.status_code} al consultar la API.")
    except Exception as e:
        st.error(f"Error al conectar: {e}")
    return pd.DataFrame()

if usar_api:
    start_str = fecha_inicio.strftime("%Y-%m-%d")
    end_str = fecha_fin.strftime("%Y-%m-%d")

    df = obtener_datos_desde_api(start_str, end_str)

    if not df.empty:
        st.subheader("📄 Datos Filtrados")
        st.dataframe(df)

        st.success(f"✅ Se cargaron {len(df)} registros desde la API.")

        st.subheader("📈 Estadísticas Básicas")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Promedio", f"{df['valor'].mean():.2f}")
        col2.metric("Mediana", f"{df['valor'].median():.2f}")
        col3.metric("Máximo", f"{df['valor'].max():.2f}")
        col4.metric("Mínimo", f"{df['valor'].min():.2f}")
        col5.metric("Desviación estándar", f"{df['valor'].std():.2f}")

        st.subheader("📊 Visualización de Datos")
        tipo_grafico = st.selectbox("Selecciona tipo de gráfico", ["Línea", "Barras"])
        if tipo_grafico == "Línea":
            st.line_chart(df.set_index("fecha")["valor"])
        else:
            st.bar_chart(df.set_index("fecha")["valor"])
    else:
        st.error("❌ No se pudieron obtener datos para el rango seleccionado.")