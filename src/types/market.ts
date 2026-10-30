export type MarketRow = {
  date: string;
  variable: string;
  duration: string;
  unit: string;
  version: string;
  value: number;
};

export type DataSource = "Supabase" | "CSV local" | "SIMEM";

export type MarketPayload = {
  rows: MarketRow[];
  source: DataSource;
  syncedAt: string;
};

export type Frequency = "daily" | "weekly" | "monthly";
export type SeriesPoint = { date: string; value: number };
