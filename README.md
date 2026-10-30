# Pulso XM

Aplicación de análisis del mercado eléctrico colombiano. El repositorio contiene dos implementaciones independientes:

- **Streamlit original:** `xm.py`, `requirements.txt` y `.streamlit/` permanecen intactos.
- **Aplicación Vercel:** Next.js, React y TypeScript dentro de `src/`.

La versión web conserva la consulta a SIMEM, estadísticas, tabla, exportación CSV, animación/GIF mensual y las seis visualizaciones analíticas del proyecto original.

## Funcionalidad migrada

| Streamlit | Next.js / Vercel |
| --- | --- |
| Consulta por rango a SIMEM | `GET /api/market` con validación y solicitudes mensuales desde el servidor |
| Tabla `st.dataframe` | Tabla responsive paginada |
| Descargar CSV | Exportación nativa desde el navegador |
| Promedio, máximo, mínimo, desviación y mediana | Tarjetas calculadas en TypeScript |
| GIF mensual con Matplotlib/Pillow | Historia mensual interactiva y GIF generado por `POST /api/gif` |
| Agregación diaria, semanal y mensual | Selector inmediato en cliente |
| Serie y media móvil | SVG responsive sin librería de gráficos pesada |
| Histograma y sesgo | Histograma SVG y explicación automática |
| Boxplot mensual | Cuartiles, mediana, IQR y bigotes en SVG |
| Heatmap Año × Mes | Cuadrícula responsive con escala de color |
| Persistencia lag‑1 | Dispersión, correlación e interpretación |
| Top 10 máximos y mínimos | Tablas comparativas de los últimos 12 meses |

## Estructura

```text
.
├── .streamlit/                     # configuración original
├── xm.py                           # aplicación Streamlit original
├── requirements.txt                # dependencias Python originales
├── precios_xm.csv                  # respaldo de datos
├── src/
│   ├── app/
│   │   ├── api/market/route.ts     # proxy seguro hacia SIMEM
│   │   ├── api/gif/route.ts        # generación de GIF en Vercel
│   │   ├── globals.css             # sistema visual responsive
│   │   ├── layout.tsx
│   │   └── page.tsx
│   ├── components/
│   │   ├── charts.tsx              # visualizaciones SVG
│   │   └── market-workspace.tsx    # interfaz e interacciones
│   ├── lib/
│   │   ├── csv.ts                  # importación y exportación CSV
│   │   ├── data-source.ts          # Supabase, SIMEM y fallback local
│   │   └── statistics.ts           # cálculos estadísticos puros
│   └── types/                      # contratos TypeScript
├── supabase/                       # migración y semilla de Postgres
├── scripts/seed-supabase.mjs
└── docs/                           # arquitectura y guía de migración
```

## Fuentes de datos

Al abrir la aplicación se usa este orden:

1. **Supabase**, si las variables están configuradas y la tabla contiene registros.
2. **CSV local**, si Supabase no está disponible.
3. **SIMEM**, cuando el usuario ejecuta una consulta por fechas.

La página principal se regenera cada 15 minutos. Las consultas manuales a SIMEM no se almacenan automáticamente en Supabase.

## Desarrollo

```bash
npm install
npm run dev
```

Validación de producción:

```bash
npm run typecheck
npm run build
```

La aplicación estará en `http://localhost:3000`.

## Variables de entorno

```text
NEXT_PUBLIC_SUPABASE_URL=
NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY=
```

También se admite `NEXT_PUBLIC_SUPABASE_ANON_KEY` para proyectos antiguos. `SUPABASE_SERVICE_ROLE_KEY` se utiliza únicamente al ejecutar la semilla local y nunca debe exponerse en el navegador.

## Despliegue en Vercel

1. Importa este repositorio en Vercel.
2. Usa **Next.js** como framework y `./` como directorio raíz.
3. Conecta Supabase desde el Marketplace de Vercel o agrega las variables manualmente.
4. Ejecuta `supabase/migrations/001_create_electricity_prices.sql`.
5. Importa `supabase/seed/electricity_prices.csv` o ejecuta `npm run seed:supabase`.
6. Haz un nuevo despliegue para aplicar variables recién creadas.

## Documentación

- [Arquitectura y flujo de datos](docs/ARCHITECTURE.md)
- [Decisiones y equivalencias de migración](docs/MIGRATION.md)

Fuente: SIMEM · Autor: Yoseth Mosquera · Universidad de Antioquia
