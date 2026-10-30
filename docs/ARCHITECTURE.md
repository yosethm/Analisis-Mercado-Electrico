# Arquitectura

## Flujo principal

```text
Supabase ───────────────┐
                       ├─ getInitialMarketData ─ Server Component ─ MarketWorkspace
precios_xm.csv ────────┘                                  │
                                                          ├─ estadísticas TypeScript
Navegador ─ /api/market ─ SIMEM                            ├─ gráficos SVG
Navegador ─ /api/gif ─── gifenc                            └─ exportaciones
```

## Capas

### Presentación

`src/components/market-workspace.tsx` controla navegación, filtros, consulta por fechas, paginación y exportaciones. `src/components/charts.tsx` contiene componentes SVG independientes y accesibles.

### Dominio

`src/lib/statistics.ts` contiene funciones puras: agregación, media móvil, regresión, R², cuantiles, sesgo, histograma, boxplot, heatmap y correlación lag‑1. No depende de React ni del servidor, por lo que se puede probar y reutilizar.

### Datos

`src/lib/data-source.ts` normaliza todas las fuentes al contrato `MarketRow`. El navegador nunca llama directamente a SIMEM ni utiliza claves privadas.

### API

- `GET /api/market`: valida el rango, divide la consulta por meses y consolida la respuesta de SIMEM.
- `POST /api/gif`: recibe filas filtradas, dibuja un frame por mes y devuelve un GIF animado.

## Seguridad

- Solo la clave publicable de Supabase llega a la aplicación.
- Row Level Security permite lectura pública y mantiene escritura cerrada.
- La clave de servicio se usa exclusivamente en el script local de semilla.
- La API de SIMEM se consume desde el servidor para evitar CORS y ocultar detalles de integración.

## Rendimiento

- La primera vista se prerenderiza y se revalida cada 15 minutos.
- Los cálculos interactivos se realizan localmente sin nuevas solicitudes.
- Los gráficos son SVG y no incorporan una biblioteca pesada de visualización.
- El GIF se genera bajo demanda para no aumentar el JavaScript inicial.
