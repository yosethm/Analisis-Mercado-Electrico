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

async function readSupabase(): Promise<MarketRow[]> {
  const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const key = process.env.NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY ?? process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;
  if (!url || !key) return [];
  const response = await fetch(
    `${url}/rest/v1/electricity_prices?select=fecha,codigo_variable,codigo_duracion,unidad_medida,version,valor&order=fecha.asc&limit=10000`,
    { headers: { apikey: key, Authorization: `Bearer ${key}` }, next: { revalidate: 900 } },
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
  const months: Date[] = [];
  const cursor = new Date(Date.UTC(first.getUTCFullYear(), first.getUTCMonth(), 1));
  while (cursor <= last && months.length < 120) {
    months.push(new Date(cursor));
    cursor.setUTCMonth(cursor.getUTCMonth() + 1);
  }
  const batches = await Promise.all(months.map(async (month) => {
    const from = new Date(Date.UTC(month.getUTCFullYear(), month.getUTCMonth(), 1));
    const to = new Date(Date.UTC(month.getUTCFullYear(), month.getUTCMonth() + 1, 0));
    const url = new URL("https://www.simem.co/backend-files/api/PublicData");
    url.searchParams.set("startDate", from.toISOString().slice(0, 10));
    url.searchParams.set("enddate", to.toISOString().slice(0, 10));
    url.searchParams.set("datasetId", "96D56E");
    const response = await fetch(url, { signal: AbortSignal.timeout(30_000), cache: "no-store" });
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
  }));
  return batches.flat().sort((a, b) => a.date.localeCompare(b.date));
}
