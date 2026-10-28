import { readFile } from "node:fs/promises";

const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
const key = process.env.SUPABASE_SERVICE_ROLE_KEY;
if (!url || !key) {
  console.error("Faltan NEXT_PUBLIC_SUPABASE_URL y SUPABASE_SERVICE_ROLE_KEY.");
  process.exit(1);
}

const csv = await readFile(new URL("../precios_xm.csv", import.meta.url), "utf8");
const rows = csv.trim().split(/\r?\n/).slice(1).map((line) => {
  const [codigo_variable, fecha, codigo_duracion, unidad_medida, version, valor] = line.split(",");
  return { codigo_variable, fecha, codigo_duracion, unidad_medida, version, valor: Number(valor) };
});

const response = await fetch(`${url}/rest/v1/electricity_prices?on_conflict=codigo_variable,fecha,version`, {
  method: "POST",
  headers: {
    apikey: key,
    Authorization: `Bearer ${key}`,
    "Content-Type": "application/json",
    Prefer: "resolution=merge-duplicates,return=minimal",
  },
  body: JSON.stringify(rows),
});

if (!response.ok) {
  console.error(`Supabase respondió ${response.status}: ${await response.text()}`);
  process.exit(1);
}
console.log(`Listo: ${rows.length} registros sincronizados con Supabase.`);

