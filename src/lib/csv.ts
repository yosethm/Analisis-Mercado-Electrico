import type { MarketRow } from "@/types/market";

export function parseMarketCsv(text: string): MarketRow[] {
  return text
    .trim()
    .split(/\r?\n/)
    .slice(1)
    .map((line) => line.split(","))
    .filter((columns) => columns.length >= 6 && Number.isFinite(Number(columns[5])))
    .map(([variable, date, duration, unit, version, value]) => ({
      variable,
      date,
      duration,
      unit,
      version,
      value: Number(value),
    }));
}

export function marketRowsToCsv(rows: MarketRow[]): string {
  const header = "CodigoVariable,Fecha,CodigoDuracion,UnidadMedida,Version,Valor";
  const body = rows.map((row) =>
    [row.variable, row.date, row.duration, row.unit, row.version, row.value].join(","),
  );
  return [header, ...body].join("\n");
}
