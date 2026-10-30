import { promises as fs } from "node:fs";
import path from "node:path";
import { parseMarketCsv } from "./csv";
import type { MarketPayload, MarketRow } from "@/types/market";

type SupabaseRow = {
  fecha: string;
  codigo_variable: string;
  codigo_duracion: string;
  unidad_medida: string;
  version: string;
  valor: number | string;
};

const normalize = (rows: SupabaseRow[]): MarketRow[] => rows
  .map((row) => ({
    date: row.fecha,
    variable: row.codigo_variable,
    duration: row.codigo_duracion,
    unit: row.unidad_medida,
    version: row.version,
    value: Number(row.valor),
  }))
  .filter((row) => row.date && Number.isFinite(row.value));

const simemMemoryCache = new Map<string, { expiresAt: number; rows: MarketRow[] }>();

async function readSupabase(): Promise<MarketRow[]> {
  const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const key = process.env.NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY ?? process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;
  if (!url || !key) return [];
  const response = await fetch(
    `${url}/rest/v1/electricity_prices?select=fecha,codigo_variable,codigo_duracion,unidad_medida,version,valor&order=fecha.asc&limit=10000`,
    {
      headers: { apikey: key, Authorization: `Bearer ${key}` },
      next: { revalidate: 900 },
      // Una conexión incompleta a Supabase no debe retrasar el primer render.
      signal: AbortSignal.timeout(4_500),
    },
  );
  if (!response.ok) return [];
  return normalize((await response.json()) as SupabaseRow[]);
}

export async function getInitialMarketData(): Promise<MarketPayload> {
  try {
    const rows = await readSupabase();
    if (rows.length) return { rows, source: "Supabase", syncedAt: new Date().toISOString() };
  } catch {
    // El CSV mantiene la aplicación disponible durante una caída o configuración incompleta.
  }
  const csv = await fs.readFile(path.join(process.cwd(), "precios_xm.csv"), "utf8");
  return { rows: parseMarketCsv(csv), source: "CSV local", syncedAt: new Date().toISOString() };
}

export async function fetchSimemRange(start: string, end: string): Promise<MarketRow[]> {
  const first = new Date(`${start}T12:00:00Z`);
  const last = new Date(`${end}T12:00:00Z`);
  if (!Number.isFinite(first.getTime()) || !Number.isFinite(last.getTime()) || first > last) {
    throw new Error("El rango de fechas no es válido.");
  }
  const cacheKey = `${start}:${end}`;
  const cached = simemMemoryCache.get(cacheKey);
  if (cached && cached.expiresAt > Date.now()) return cached.rows;
  const months: Date[] = [];
  const cursor = new Date(Date.UTC(first.getUTCFullYear(), first.getUTCMonth(), 1));
  while (cursor <= last && months.length < 120) {
    months.push(new Date(cursor));
    cursor.setUTCMonth(cursor.getUTCMonth() + 1);
  }
  const fetchMonth = async (month: Date) => {
    const from = new Date(Date.UTC(month.getUTCFullYear(), month.getUTCMonth(), 1));
    const to = new Date(Date.UTC(month.getUTCFullYear(), month.getUTCMonth() + 1, 0));
    const url = new URL("https://www.simem.co/backend-files/api/PublicData");
    url.searchParams.set("startDate", from.toISOString().slice(0, 10));
    url.searchParams.set("enddate", to.toISOString().slice(0, 10));
    url.searchParams.set("datasetId", "96D56E");
    let response: Response;
    try {
      response = await fetch(url, {
        // La consulta frecuente del mismo rango se responde desde caché durante 10 min.
        next: { revalidate: 600 },
        signal: AbortSignal.timeout(12_000),
      });
    } catch {
      throw new Error("SIMEM no respondió a tiempo. Intenta de nuevo en unos segundos.");
    }
    if (!response.ok) throw new Error(`SIMEM respondió ${response.status}.`);
    const payload = await response.json() as { result?: { records?: Record<string, unknown>[] } };
    return (payload.result?.records ?? []).map((record) => ({
      date: String(record.Fecha ?? "").slice(0, 10),
      variable: String(record.CodigoVariable ?? ""),
      duration: String(record.CodigoDuracion ?? "P1D"),
      unit: String(record.UnidadMedida ?? "COP/kWh"),
      version: String(record.Version ?? ""),
      value: Number(record.Valor),
    })).filter((row) => row.date >= start && row.date <= end && Number.isFinite(row.value));
  };
  // Evita saturar SIMEM cuando se solicita un histórico extenso.
  const batches: MarketRow[][] = [];
  for (let index = 0; index < months.length; index += 4) {
    batches.push(...await Promise.all(months.slice(index, index + 4).map(fetchMonth)));
  }
  const rows = batches.flat().sort((a, b) => a.date.localeCompare(b.date));
  simemMemoryCache.set(cacheKey, { rows, expiresAt: Date.now() + 600_000 });
  return rows;
}
