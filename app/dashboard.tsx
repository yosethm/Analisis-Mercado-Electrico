"use client";

import { useMemo, useState } from "react";
import type { MarketPayload, PricePoint } from "@/lib/market-data";

type DailyPoint = { date: string; value: number };
type ForecastPoint = DailyPoint & { low: number; high: number };

const money = new Intl.NumberFormat("es-CO", { maximumFractionDigits: 1 });
const shortDate = new Intl.DateTimeFormat("es-CO", { day: "2-digit", month: "short" });
const longDate = new Intl.DateTimeFormat("es-CO", { day: "numeric", month: "long", year: "numeric" });

function Icon({ name }: { name: "bolt" | "trend" | "wave" | "alert" | "drop" | "github" }) {
  const paths = {
    bolt: <path d="M13 2 4.8 13H11l-1 9L19.2 11H13l0-9Z" />,
    trend: <><path d="m3 17 6-6 4 4 8-9"/><path d="M15 6h6v6"/></>,
    wave: <path d="M3 12h3l2-7 4 14 2-7h7" />,
    alert: <><path d="M12 3 2.8 20h18.4L12 3Z"/><path d="M12 9v5M12 17.5v.1"/></>,
    drop: <path d="M12 2S6 9 6 14a6 6 0 0 0 12 0c0-5-6-12-6-12Z" />,
    github: <path d="M12 2a10 10 0 0 0-3.2 19.5v-2.3c-2.6.6-3.2-1.1-3.2-1.1-.4-1.1-1.1-1.4-1.1-1.4-.9-.6.1-.6.1-.6 1 0 1.5 1 1.5 1 .9 1.5 2.3 1.1 2.8.8.1-.6.3-1.1.6-1.3-2.1-.2-4.3-1-4.3-4.6 0-1 .4-1.9 1-2.6-.1-.3-.4-1.2.1-2.6 0 0 .8-.3 2.7 1a9.4 9.4 0 0 1 5 0c1.9-1.3 2.7-1 2.7-1 .5 1.4.2 2.3.1 2.6.6.7 1 1.6 1 2.6 0 3.6-2.2 4.4-4.3 4.6.4.3.7 1 .7 2v2.9A10 10 0 0 0 12 2Z" />,
  };
  return <svg viewBox="0 0 24 24" aria-hidden="true">{paths[name]}</svg>;
}

function aggregate(points: PricePoint[], variable: string, version: string): DailyPoint[] {
  const filtered = points.filter((p) => p.variable === variable && p.version === version);
  const chosen = filtered.length ? filtered : points.filter((p) => p.variable === variable);
  const buckets = new Map<string, number[]>();
  chosen.forEach((p) => buckets.set(p.date, [...(buckets.get(p.date) ?? []), p.value]));
  return [...buckets.entries()]
    .map(([date, values]) => ({ date, value: values.reduce((a, b) => a + b, 0) / values.length }))
    .sort((a, b) => a.date.localeCompare(b.date));
}

function forecast(data: DailyPoint[]): ForecastPoint[] {
  const sample = data.slice(-14);
  if (sample.length < 2) return [];
  const n = sample.length;
  const sx = (n * (n - 1)) / 2;
  const sy = sample.reduce((sum, p) => sum + p.value, 0);
  const sxy = sample.reduce((sum, p, i) => sum + i * p.value, 0);
  const sx2 = sample.reduce((sum, _, i) => sum + i * i, 0);
  const slope = (n * sxy - sx * sy) / (n * sx2 - sx * sx || 1);
  const intercept = (sy - slope * sx) / n;
  const residuals = sample.map((p, i) => p.value - (intercept + slope * i));
  const sigma = Math.sqrt(residuals.reduce((sum, r) => sum + r * r, 0) / n);
  const last = new Date(`${sample[n - 1].date}T12:00:00`);
  return Array.from({ length: 7 }, (_, index) => {
    const date = new Date(last);
    date.setDate(last.getDate() + index + 1);
    const value = Math.max(0, intercept + slope * (n + index));
    return {
      date: date.toISOString().slice(0, 10),
      value,
      low: Math.max(0, value - sigma * 1.28),
      high: value + sigma * 1.28,
    };
  });
}

function LineChart({ actual, projected }: { actual: DailyPoint[]; projected: ForecastPoint[] }) {
  const width = 900, height = 310, padX = 48, padY = 30;
  const visible = actual.slice(-40);
  const all = [...visible, ...projected];
  if (!all.length) return <div className="empty">No hay datos para este filtro.</div>;
  const min = Math.min(...visible.map((p) => p.value), ...projected.map((p) => p.low)) * .92;
  const max = Math.max(...visible.map((p) => p.value), ...projected.map((p) => p.high)) * 1.05;
  const x = (i: number) => padX + i * ((width - padX * 2) / Math.max(1, all.length - 1));
  const y = (v: number) => height - padY - ((v - min) / Math.max(1, max - min)) * (height - padY * 2);
  const actualPath = visible.map((p, i) => `${i ? "L" : "M"}${x(i)},${y(p.value)}`).join(" ");
  const start = Math.max(0, visible.length - 1);
  const projectedWithAnchor = visible.length ? [visible[visible.length - 1], ...projected] : projected;
  const forecastPath = projectedWithAnchor.map((p, i) => `${i ? "L" : "M"}${x(start + i)},${y(p.value)}`).join(" ");
  const band = projected.length ? [
    ...projected.map((p, i) => `${i ? "L" : "M"}${x(visible.length + i)},${y(p.high)}`),
    ...[...projected].reverse().map((p, i) => `L${x(all.length - 1 - i)},${y(p.low)}`), "Z",
  ].join(" ") : "";
  const ticks = [min, min + (max-min)/2, max];
  return (
    <div className="chart-wrap">
      <svg className="line-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Evolución del precio y pronóstico">
        <defs>
          <linearGradient id="area" x1="0" y1="0" x2="0" y2="1"><stop stopColor="#78F7C5" stopOpacity=".24"/><stop offset="1" stopColor="#78F7C5" stopOpacity="0"/></linearGradient>
          <linearGradient id="band" x1="0" x2="1"><stop stopColor="#F3C969" stopOpacity=".08"/><stop offset="1" stopColor="#F3C969" stopOpacity=".28"/></linearGradient>
        </defs>
        {ticks.map((tick) => <g key={tick}><line x1={padX} x2={width-padX} y1={y(tick)} y2={y(tick)} className="grid-line"/><text x={padX-10} y={y(tick)+4} textAnchor="end" className="axis-label">{money.format(tick)}</text></g>)}
        {visible.length > 1 && <path d={`${actualPath} L${x(visible.length-1)},${height-padY} L${x(0)},${height-padY} Z`} fill="url(#area)"/>}
        <path d={band} fill="url(#band)"/>
        <path d={actualPath} className="actual-line"/>
        <path d={forecastPath} className="forecast-line"/>
        {all.map((p, i) => i % Math.max(1, Math.floor(all.length/5)) === 0 && <text key={p.date} x={x(i)} y={height-7} textAnchor="middle" className="axis-label">{shortDate.format(new Date(`${p.date}T12:00:00`))}</text>)}
        <line x1={x(start)} x2={x(start)} y1={padY} y2={height-padY} className="today-line"/>
        <text x={x(start)+8} y={padY+10} className="today-label">HOY</text>
      </svg>
    </div>
  );
}

function Metric({ icon, label, value, detail, tone = "green" }: { icon: "bolt" | "trend" | "wave" | "alert"; label: string; value: string; detail: string; tone?: string }) {
  return <article className="metric-card"><div className={`metric-icon ${tone}`}><Icon name={icon}/></div><div><span>{label}</span><strong>{value}</strong><small>{detail}</small></div></article>;
}

export default function Dashboard({ initialData }: { initialData: MarketPayload }) {
  const variables = useMemo(() => [...new Set(initialData.points.map((p) => p.variable))], [initialData.points]);
  const versions = useMemo(() => [...new Set(initialData.points.map((p) => p.version))], [initialData.points]);
  const [variable, setVariable] = useState(variables.includes("PPBOGReal") ? "PPBOGReal" : variables[0]);
  const [version, setVersion] = useState(versions.includes("TXR") ? "TXR" : versions[0]);
  const [demand, setDemand] = useState(0);
  const [hydrology, setHydrology] = useState(0);
  const daily = useMemo(() => aggregate(initialData.points, variable, version), [initialData.points, variable, version]);
  const projection = useMemo(() => forecast(daily), [daily]);
  const values = daily.map((p) => p.value);
  const latest = values.at(-1) ?? 0;
  const previous = values.at(-2) ?? latest;
  const average = values.reduce((a,b) => a+b, 0) / Math.max(1, values.length);
  const max = Math.max(...values, 0);
  const std = Math.sqrt(values.reduce((sum, v) => sum + (v-average)**2, 0) / Math.max(1, values.length));
  const volatility = average ? std / average * 100 : 0;
  const delta = previous ? (latest-previous)/previous*100 : 0;
  const scenario = latest * (1 + demand * .012 - hydrology * .009);
  const scenarioDelta = latest ? (scenario-latest)/latest*100 : 0;
  const anomalies = daily.map((p) => ({...p, score: std ? Math.abs((p.value-average)/std) : 0})).sort((a,b) => b.score-a.score).slice(0,4);
  const latestDate = daily.at(-1)?.date;

  return (
    <main>
      <header className="topbar">
        <a className="brand" href="#inicio"><span className="brand-mark"><Icon name="bolt"/></span><span><b>PULSO</b> XM<small>INTELIGENCIA ENERGÉTICA</small></span></a>
        <nav><a className="active" href="#mercado">Mercado</a><a href="#pronostico">Pronóstico</a><a href="#escenarios">Escenarios</a></nav>
        <a className="github-link" href="https://github.com/yosethm/Analisis-Mercado-Electrico" target="_blank" rel="noreferrer"><Icon name="github"/> GitHub</a>
      </header>

      <section className="hero" id="inicio">
        <div className="eyebrow"><span/> MERCADO ELÉCTRICO COLOMBIANO</div>
        <div className="hero-grid"><div><h1>El mercado no espera.<br/><em>Anticípalo.</em></h1><p>Datos, señales y escenarios para convertir la complejidad energética en decisiones claras.</p></div>
          <div className="live-card"><div><span className="pulse"/> SEÑAL ACTUAL</div><strong>{money.format(latest)} <small>COP/kWh</small></strong><p className={delta >= 0 ? "up" : "down"}>{delta >= 0 ? "↗" : "↘"} {Math.abs(delta).toFixed(1)}% vs. dato anterior</p><footer>{latestDate ? longDate.format(new Date(`${latestDate}T12:00:00`)) : "Sin fecha"}</footer></div>
        </div>
        <div className="source-line"><span>FUENTE DE DATOS</span><b>{initialData.source}</b><i>•</i><span>{initialData.points.length} REGISTROS</span></div>
      </section>

      <section className="dashboard" id="mercado">
        <div className="section-head"><div><span>01 / PANORAMA</span><h2>Lectura del mercado</h2></div><div className="filters"><label>Producto<select value={variable} onChange={(e)=>setVariable(e.target.value)}>{variables.map(v=><option key={v}>{v}</option>)}</select></label><label>Versión<select value={version} onChange={(e)=>setVersion(e.target.value)}>{versions.map(v=><option key={v}>{v}</option>)}</select></label></div></div>
        <div className="metrics"><Metric icon="bolt" label="Último precio" value={`${money.format(latest)}`} detail="COP/kWh"/><Metric icon="trend" label="Promedio del periodo" value={money.format(average)} detail={`${daily.length} observaciones`} tone="gold"/><Metric icon="wave" label="Volatilidad" value={`${volatility.toFixed(1)}%`} detail="desviación relativa" tone="blue"/><Metric icon="alert" label="Pico del periodo" value={money.format(max)} detail="COP/kWh" tone="red"/></div>
      </section>

      <section className="analysis-grid" id="pronostico">
        <article className="panel chart-panel"><div className="panel-head"><div><span>02 / SEÑAL + MODELO</span><h2>Evolución y proyección</h2></div><div className="legend"><i className="actual"/>Real <i className="projected"/>Proyección 7 días</div></div><LineChart actual={daily} projected={projection}/><p className="method-note">Pronóstico exploratorio por tendencia lineal sobre 14 observaciones; la franja representa incertidumbre aproximada del 80%. No constituye una recomendación comercial.</p></article>
        <aside className="panel insight-panel"><span>SEÑAL DEL MODELO</span><h3>{projection.at(-1)?.value && projection.at(-1)!.value > latest ? "Presión alcista" : "Moderación esperada"}</h3><p>El modelo proyecta un precio de <b>{money.format(projection.at(-1)?.value ?? latest)} COP/kWh</b> al cierre de los próximos 7 días.</p><div className="confidence"><div><span>Confianza indicativa</span><b>{Math.max(42, Math.min(88, 92-volatility)).toFixed(0)}%</b></div><progress value={Math.max(42, 92-volatility)} max="100"/></div><div className="insight-callout"><Icon name="wave"/><p><b>Qué vigilar</b>La volatilidad del periodo es {volatility > 20 ? "elevada" : "moderada"}; contrasta la señal con hidrología y demanda antes de decidir.</p></div></aside>
      </section>

      <section className="scenario-section" id="escenarios">
        <div className="section-head light"><div><span>03 / LABORATORIO</span><h2>Simula el próximo movimiento</h2><p>Mueve las variables y observa el impacto indicativo sobre la última señal.</p></div></div>
        <div className="scenario-grid"><div className="sliders"><label><div><span>Demanda del sistema</span><b>{demand > 0 ? "+" : ""}{demand}%</b></div><input type="range" min="-10" max="10" value={demand} onChange={(e)=>setDemand(Number(e.target.value))}/><small>Menor demanda <i/> Mayor demanda</small></label><label><div><span>Nivel hidrológico</span><b>{hydrology > 0 ? "+" : ""}{hydrology}%</b></div><input type="range" min="-10" max="10" value={hydrology} onChange={(e)=>setHydrology(Number(e.target.value))}/><small>Sequía <i/> Aportes altos</small></label></div>
          <div className="scenario-output"><span>PRECIO ESTIMADO DEL ESCENARIO</span><strong>{money.format(scenario)} <small>COP/kWh</small></strong><p className={scenarioDelta >= 0 ? "up" : "down"}>{scenarioDelta >= 0 ? "▲" : "▼"} {Math.abs(scenarioDelta).toFixed(1)}% frente a la señal base</p><div className="impact-scale"><i/><i/><i/><i/><i className="marker" style={{left:`${Math.max(4, Math.min(96, 50+scenarioDelta*3))}%`}}/></div><small>Modelo pedagógico de sensibilidad, no predicción causal.</small></div>
        </div>
      </section>

      <section className="anomalies"><div className="section-head"><div><span>04 / EVENTOS</span><h2>Movimientos fuera de lo común</h2></div></div><div className="anomaly-list">{anomalies.map((item, index)=><article key={item.date}><b>{String(index+1).padStart(2,"0")}</b><div><span>{longDate.format(new Date(`${item.date}T12:00:00`))}</span><strong>{money.format(item.value)} COP/kWh</strong></div><em>{item.score.toFixed(1)}σ</em></article>)}</div></section>

      <footer className="footer"><div><span className="brand-mark"><Icon name="bolt"/></span><div><b>PULSO XM</b><small>Proyecto de análisis del mercado eléctrico colombiano</small></div></div><p>Datos de SIMEM · Construido por Yoseth Mosquera</p><p>Supabase + Next.js + Vercel</p></footer>
    </main>
  );
}

