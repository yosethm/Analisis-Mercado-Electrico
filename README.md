# Pulso XM — Inteligencia del mercado eléctrico colombiano

Pulso XM convierte datos de precios de SIMEM/XM en un centro de mando interactivo. La nueva interfaz está construida con Next.js para desplegarse directamente en Vercel y conserva `xm.py` como versión histórica en Streamlit.

## Qué incluye

- Lectura de precio actual, promedio, volatilidad y máximo del periodo.
- Filtros por producto (`PPBO`, `PPBOGReal`) y versión (`TXR`, `TX1`, `TX2`).
- Serie de tiempo SVG sin dependencias pesadas de gráficos.
- Pronóstico exploratorio de siete días con banda de incertidumbre.
- Simulador de sensibilidad a demanda e hidrología.
- Ranking de movimientos atípicos mediante puntuación z.
- Diseño responsive para escritorio y móvil.
- Supabase como fuente principal y CSV como respaldo automático.

## Arquitectura

```text
Supabase (electricity_prices) ─┐
                              ├─> lib/market-data.ts ─> Server Component ─> Dashboard interactivo
precios_xm.csv (respaldo) ────┘
```

`lib/market-data.ts` intenta consultar Supabase. Si faltan variables de entorno, la tabla no existe, está vacía o la petición falla, lee `precios_xm.csv`. La página se regenera cada 15 minutos para evitar solicitudes innecesarias.

## Ejecutar localmente

```bash
npm install
npm run dev
```

Abre `http://localhost:3000`.

## Activar Supabase

1. Ejecuta `supabase/migrations/001_create_electricity_prices.sql` en el SQL Editor de Supabase.
2. Copia `.env.example` como `.env.local` y agrega las credenciales de tu proyecto.
3. Agrega temporalmente `SUPABASE_SERVICE_ROLE_KEY` solo en tu entorno local.
4. Ejecuta `npm run seed:supabase` para importar el CSV.
5. Elimina la clave de servicio cuando termines. La aplicación web solo necesita la clave pública `anon`.

La migración activa Row Level Security y permite únicamente lectura pública. La escritura queda reservada al servidor o a la clave de servicio.

## Desplegar en Vercel

Vercel detecta Next.js automáticamente. Configura estas variables en **Project Settings → Environment Variables**:

- `NEXT_PUBLIC_SUPABASE_URL`
- `NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY` (recomendado actualmente)
- `NEXT_PUBLIC_SUPABASE_ANON_KEY` también funciona en proyectos antiguos

Si la integración de Supabase con Vercel ya las creó, no es necesario duplicarlas. El despliegue también funciona sin ellas gracias al CSV de respaldo.

## Cómo funciona el pronóstico

Se toman las últimas 14 observaciones filtradas y se ajusta una regresión lineal simple. La línea discontinua extiende esa tendencia siete días; la banda usa la desviación de los residuos como estimación de incertidumbre al 80%. Es una herramienta exploratoria, no una recomendación de compra o cobertura.

## Stack

- Next.js 16 + React 19 + TypeScript
- Supabase Postgres + Row Level Security
- CSS y SVG nativos
- Vercel para CI/CD y hosting

Fuente de datos: [SIMEM](https://www.simem.co/) · Autor: Yoseth Mosquera

