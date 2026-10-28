import { promises as fs } from "node:fs";
import path from "node:path";

export type PricePoint = {
  date: string;
  variable: string;
  duration: string;
  unit: string;
  version: string;
  value: number;
};

export type MarketPayload = {
  points: PricePoint[];
  source: "Supabase" | "CSV local";
  syncedAt: string;
};

type SupabaseRow = {
  fecha: string;
  codigo_variable: string;
  codigo_duracion: string;
  unidad_medida: string;
  version: string;
  valor: number | string;
};

function parseCsv(text: string): PricePoint[] {
  return text
    .trim()
    .split(/\r?\n/)
    .slice(1)
    .map((line) => line.split(","))
    .filter((row) => row.length >= 6 && Number.isFinite(Number(row[5])))
    .map(([variable, date, duration, unit, version, value]) => ({
      date,
      variable,
      duration,
      unit,
      version,
      value: Number(value),
    }));
}

function normalizeRows(rows: SupabaseRow[]): PricePoint[] {
  return rows
    .map((row) => ({
      date: row.fecha,
      variable: row.codigo_variable,
      duration: row.codigo_duracion,
      unit: row.unidad_medida,
      version: row.version,
      value: Number(row.valor),
    }))
    .filter((row) => row.date && Number.isFinite(row.value));
}

async function readSupabase(): Promise<PricePoint[]> {
  const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const key =
    process.env.NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY ??
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;
  if (!url || !key) return [];

  const response = await fetch(
    `${url}/rest/v1/electricity_prices?select=fecha,codigo_variable,codigo_duracion,unidad_medida,version,valor&order=fecha.asc&limit=5000`,
    {
      headers: { apikey: key, Authorization: `Bearer ${key}` },
      next: { revalidate: 900 },
    },
  );
  if (!response.ok) return [];
  return normalizeRows((await response.json()) as SupabaseRow[]);
}

export async function getMarketData(): Promise<MarketPayload> {
  try {
    const points = await readSupabase();
    if (points.length) return { points, source: "Supabase", syncedAt: new Date().toISOString() };
  } catch {
    // El respaldo local mantiene el dashboard disponible si Supabase no responde.
  }

  const csvPath = path.join(process.cwd(), "precios_xm.csv");
  const csv = await fs.readFile(csvPath, "utf8");
  return { points: parseCsv(csv), source: "CSV local", syncedAt: new Date().toISOString() };
}

