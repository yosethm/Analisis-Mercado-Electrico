# Migración de Streamlit a Vercel

## Principio de la migración

La versión Streamlit no se modificó. La aplicación Next.js vive junto a ella y comparte solamente el CSV de respaldo. Esto permite comparar ambas versiones, mantener el enlace histórico y volver atrás sin reconstruir el proyecto Python.

## Equivalencias técnicas

- `st.sidebar.date_input` → campos de fecha HTML y estado React.
- `requests.get` → `fetch` dentro de una Route Handler de Next.js.
- `pandas.resample` → agrupación tipada en `statistics.ts`.
- `pandas.describe` → funciones puras de media, mediana, desviación y cuantiles.
- Matplotlib/Seaborn → componentes SVG responsive.
- `st.download_button` → Blob API del navegador.
- Pillow + Matplotlib GIF → encoder GIF JavaScript en una función de servidor.
- `st.cache_data` → caché y revalidación de Next.js.

## Ventajas de Vercel

1. **Interfaz más flexible:** navegación, animaciones suaves, diseño móvil y estados de carga controlados con precisión.
2. **Despliegues por Git:** cada pull request obtiene una vista previa y `main` publica producción.
3. **Mejor separación de responsabilidades:** UI, datos, estadísticas y API viven en módulos distintos.
4. **Escalabilidad:** la página estática se distribuye por CDN y las rutas dinámicas se ejecutan bajo demanda.
5. **Integración con Supabase:** variables sincronizadas y datos persistentes sin incluir credenciales en GitHub.
6. **TypeScript:** los contratos reducen errores por columnas faltantes, valores no numéricos o respuestas inesperadas.
7. **Experiencia del usuario:** los filtros y cálculos no recargan toda la aplicación como ocurre en cada interacción de Streamlit.

## Compromisos

- Python y pandas permiten análisis exploratorio más rápido; en Next.js los cálculos deben implementarse explícitamente.
- Las funciones muy pesadas de machine learning deberían ejecutarse en un servicio Python separado y consumirse por API.
- Vercel tiene límites de tiempo y memoria en funciones; por eso la consulta se divide por meses y el GIF limita el número de filas.
