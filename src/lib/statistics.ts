import type { Frequency, MarketRow, SeriesPoint } from "@/types/market";

export const mean = (values: number[]) => values.reduce((sum, value) => sum + value, 0) / Math.max(1, values.length);

export function median(values: number[]): number {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

export function quantile(values: number[], q: number): number {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const index = (sorted.length - 1) * q;
  const lower = Math.floor(index);
  const weight = index - lower;
  return sorted[lower + 1] === undefined ? sorted[lower] : sorted[lower] * (1 - weight) + sorted[lower + 1] * weight;
}

export const deviation = (values: number[]) => {
  if (values.length < 2) return 0;
  const avg = mean(values);
  return Math.sqrt(values.reduce((sum, value) => sum + (value - avg) ** 2, 0) / (values.length - 1));
};

function bucketKey(dateText: string, frequency: Frequency): string {
  const date = new Date(`${dateText}T12:00:00Z`);
  if (frequency === "monthly") return `${date.getUTCFullYear()}-${String(date.getUTCMonth() + 1).padStart(2, "0")}-01`;
  if (frequency === "weekly") {
    const monday = new Date(date);
    const day = (monday.getUTCDay() + 6) % 7;
    monday.setUTCDate(monday.getUTCDate() - day);
    return monday.toISOString().slice(0, 10);
  }
  return dateText;
}

export function aggregate(rows: MarketRow[], frequency: Frequency = "daily"): SeriesPoint[] {
  const buckets = new Map<string, number[]>();
  rows.forEach((row) => {
    const key = bucketKey(row.date, frequency);
    buckets.set(key, [...(buckets.get(key) ?? []), row.value]);
  });
  return [...buckets.entries()]
    .map(([date, values]) => ({ date, value: mean(values) }))
    .sort((a, b) => a.date.localeCompare(b.date));
}

export function movingAverage(series: SeriesPoint[], window: number): SeriesPoint[] {
  return series.map((point, index) => ({
    date: point.date,
    value: mean(series.slice(Math.max(0, index - window + 1), index + 1).map((item) => item.value)),
  }));
}

export function linearRegression(values: number[]) {
  const n = values.length;
  if (n < 2) return { slope: 0, intercept: values[0] ?? 0, r2: 0 };
  const sx = n * (n - 1) / 2;
  const sy = values.reduce((sum, value) => sum + value, 0);
  const sxy = values.reduce((sum, value, index) => sum + index * value, 0);
  const sx2 = values.reduce((sum, _, index) => sum + index ** 2, 0);
  const slope = (n * sxy - sx * sy) / (n * sx2 - sx ** 2 || 1);
  const intercept = (sy - slope * sx) / n;
  const average = mean(values);
  const residual = values.reduce((sum, value, index) => sum + (value - (intercept + slope * index)) ** 2, 0);
  const total = values.reduce((sum, value) => sum + (value - average) ** 2, 0);
  return { slope, intercept, r2: total ? 1 - residual / total : 0 };
}

export function trendSummary(series: SeriesPoint[], frequencyLabel: string): string {
  if (series.length < 3) return "La serie es demasiado corta para evaluar una tendencia.";
  const values = series.map((point) => point.value);
  const { r2 } = linearRegression(values);
  const change = values[0] ? (values.at(-1)! / values[0] - 1) * 100 : 0;
  const direction = change > .05 ? "alcista" : change < -.05 ? "bajista" : "estable";
  const strength = r2 >= .7 ? "fuerte" : r2 >= .4 ? "moderada" : "débil";
  return `Tendencia ${direction} en la agregación ${frequencyLabel.toLowerCase()} (${change >= 0 ? "+" : ""}${change.toFixed(2)}%). Señal ${strength}, con R²=${r2.toFixed(2)}.`;
}

export function skewness(values: number[]): number {
  if (values.length < 3) return 0;
  const avg = mean(values);
  const sd = deviation(values);
  if (!sd) return 0;
  return values.reduce((sum, value) => sum + ((value - avg) / sd) ** 3, 0) / values.length;
}

export function distributionSummary(values: number[]): string {
  if (!values.length) return "No hay datos para analizar la distribución.";
  const skew = skewness(values);
  const shape = Math.abs(skew) < .3 ? "aproximadamente simétrica" : skew > 0 ? "sesgada a la derecha, con picos altos poco frecuentes" : "sesgada a la izquierda, con picos bajos poco frecuentes";
  return `Media ${mean(values).toFixed(2)}, mediana ${median(values).toFixed(2)} y desviación ${deviation(values).toFixed(2)}. Rango [${Math.min(...values).toFixed(2)}, ${Math.max(...values).toFixed(2)}]. Distribución ${shape}.`;
}

export function histogram(values: number[], count = 12) {
  if (!values.length) return [];
  const min = Math.min(...values), max = Math.max(...values);
  const size = (max - min || 1) / count;
  return Array.from({ length: count }, (_, index) => {
    const from = min + size * index;
    const to = index === count - 1 ? max + Number.EPSILON : from + size;
    return { from, to, count: values.filter((value) => value >= from && value < to).length };
  });
}

export function monthlyBoxes(rows: MarketRow[]) {
  const groups = new Map<number, number[]>();
  rows.forEach((row) => {
    const month = new Date(`${row.date}T12:00:00Z`).getUTCMonth();
    groups.set(month, [...(groups.get(month) ?? []), row.value]);
  });
  return [...groups.entries()].sort(([a], [b]) => a - b).map(([month, values]) => ({
    month, min: Math.min(...values), q1: quantile(values, .25), median: median(values), q3: quantile(values, .75), max: Math.max(...values), count: values.length,
  }));
}

export function monthlyBoxSummary(rows: MarketRow[]): string {
  const boxes = monthlyBoxes(rows);
  if (!boxes.length) return "No hay datos mensuales suficientes.";
  const names = ["enero","febrero","marzo","abril","mayo","junio","julio","agosto","septiembre","octubre","noviembre","diciembre"];
  const high = [...boxes].sort((a,b) => b.median-a.median)[0];
  const low = [...boxes].sort((a,b) => a.median-b.median)[0];
  const variable = [...boxes].sort((a,b) => (b.q3-b.q1)-(a.q3-a.q1))[0];
  return `La mediana más alta aparece en ${names[high.month]} y la más baja en ${names[low.month]}. ${names[variable.month]} presenta la mayor variabilidad intercuartílica.`;
}

export function heatmap(rows: MarketRow[]) {
  const groups = new Map<string, number[]>();
  rows.forEach((row) => {
    const date = new Date(`${row.date}T12:00:00Z`);
    const key = `${date.getUTCFullYear()}-${date.getUTCMonth()}`;
    groups.set(key, [...(groups.get(key) ?? []), row.value]);
  });
  return [...groups.entries()].map(([key, values]) => {
    const [year, month] = key.split("-").map(Number);
    return { year, month, value: mean(values) };
  }).sort((a,b) => a.year-b.year || a.month-b.month);
}

export function persistence(series: SeriesPoint[]) {
  const pairs = series.slice(1).map((point, index) => ({ previous: series[index].value, current: point.value, date: point.date }));
  if (!pairs.length) return { pairs, correlation: 0, regression: { slope: 0, intercept: 0, r2: 0 } };
  const xs = pairs.map((pair) => pair.previous), ys = pairs.map((pair) => pair.current);
  const mx = mean(xs), my = mean(ys);
  const numerator = xs.reduce((sum, x, i) => sum + (x-mx)*(ys[i]-my), 0);
  const denominator = Math.sqrt(xs.reduce((sum,x)=>sum+(x-mx)**2,0)*ys.reduce((sum,y)=>sum+(y-my)**2,0));
  return { pairs, correlation: denominator ? numerator/denominator : 0, regression: linearRegression(xs.map((_, i) => ys[i])) };
}

export function persistenceSummary(correlation: number): string {
  const absolute = Math.abs(correlation);
  const level = absolute >= .8 ? "muy alta" : absolute >= .6 ? "alta" : absolute >= .4 ? "moderada" : absolute >= .2 ? "baja" : "muy baja";
  return `Persistencia ${level} y ${correlation >= 0 ? "positiva" : "negativa"}, con correlación lag‑1 de ${correlation.toFixed(2)}.`;
}
