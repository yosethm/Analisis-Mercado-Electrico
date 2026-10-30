"use client";

import { useEffect, useMemo, useState } from "react";
import type { Frequency, MarketPayload, MarketRow } from "@/types/market";
import { marketRowsToCsv } from "@/lib/csv";
import {
  aggregate, deviation, distributionSummary, heatmap, histogram, mean, median,
  monthlyBoxes, monthlyBoxSummary, movingAverage, persistence, persistenceSummary,
  trendSummary,
} from "@/lib/statistics";
import { BoxPlot, Heatmap, Histogram, LineChart, ScatterPlot } from "./charts";

const money = new Intl.NumberFormat("es-CO", { maximumFractionDigits: 2 });
const fullDate = new Intl.DateTimeFormat("es-CO", { day: "numeric", month: "long", year: "numeric" });
const frequencyLabels: Record<Frequency,string> = { daily: "Diaria", weekly: "Semanal", monthly: "Mensual" };
type Tab = "overview" | "data" | "analytics";

function inputDate(value: Date) {
  return `${value.getFullYear()}-${String(value.getMonth()+1).padStart(2,"0")}-${String(value.getDate()).padStart(2,"0")}`;
}

function defaultDateRange() {
  const end = new Date();
  const start = new Date(end);
  start.setMonth(start.getMonth() - 1);
  return { start: inputDate(start), end: inputDate(end) };
}

function Icon({ name }: { name: "bolt"|"calendar"|"download"|"database"|"chart"|"play"|"pause"|"refresh" }) {
  const paths={
    bolt:<path d="M13 2 5 13h6l-1 9 9-12h-6V2Z"/>, calendar:<><path d="M5 3v3m14-3v3M3 9h18M5 5h14a2 2 0 0 1 2 2v13H3V7a2 2 0 0 1 2-2Z"/></>,
    download:<><path d="M12 3v12m-5-5 5 5 5-5"/><path d="M5 20h14"/></>,database:<><ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v6c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 11v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6"/></>,
    chart:<><path d="M4 19V9m6 10V5m6 14v-7m5 7H2"/></>,play:<path d="m8 5 11 7-11 7V5Z"/>,pause:<><path d="M8 5v14M16 5v14"/></>,refresh:<><path d="M20 7h-5V2"/><path d="M20 7a8 8 0 1 0 1 7"/></>,
  };
  return <svg viewBox="0 0 24 24" aria-hidden="true">{paths[name]}</svg>;
}

function downloadText(content:string,filename:string,type:string){
  const url=URL.createObjectURL(new Blob([content],{type}));
  const anchor=document.createElement("a");anchor.href=url;anchor.download=filename;anchor.click();URL.revokeObjectURL(url);
}

export function MarketWorkspace({ initialPayload }: { initialPayload: MarketPayload }) {
  const [payload,setPayload]=useState(initialPayload);
  const [activeTab,setActiveTab]=useState<Tab>("overview");
  const [frequency,setFrequency]=useState<Frequency>("daily");
  const [loading,setLoading]=useState(false);
  const [error,setError]=useState("");
  const [dateRange]=useState(defaultDateRange);
  const [start,setStart]=useState(dateRange.start);
  const [end,setEnd]=useState(dateRange.end);
  const variables=useMemo(()=>[...new Set(payload.rows.map(row=>row.variable))].filter(Boolean),[payload.rows]);
  const [variable,setVariable]=useState(variables.includes("PPBOGReal")?"PPBOGReal":variables[0]??"");
  const versions=useMemo(()=>[...new Set(payload.rows.filter(row=>row.variable===variable).map(row=>row.version))].filter(Boolean),[payload.rows,variable]);
  const [version,setVersion]=useState(versions.includes("TXR")?"TXR":versions[0]??"");

  useEffect(()=>{if(!versions.includes(version))setVersion(versions.includes("TXR")?"TXR":versions[0]??"")},[versions,version]);

  const selected=useMemo(()=>payload.rows.filter(row=>row.variable===variable&&row.version===version),[payload.rows,variable,version]);
  const series=useMemo(()=>aggregate(selected,frequency),[selected,frequency]);
  const smooth=useMemo(()=>movingAverage(series,frequency==="daily"?7:frequency==="weekly"?4:3),[series,frequency]);
  const values=selected.map(row=>row.value);
  const latest=series.at(-1)?.value??0;
  const previous=series.at(-2)?.value??latest;
  const delta=previous?(latest/previous-1)*100:0;
  const hist=useMemo(()=>histogram(series.map(point=>point.value),14),[series]);
  const boxes=useMemo(()=>monthlyBoxes(selected),[selected]);
  const heat=useMemo(()=>heatmap(selected),[selected]);
  const persistenceData=useMemo(()=>persistence(series),[series]);
  const cutoff=series.length?new Date(`${series.at(-1)!.date}T12:00:00Z`):new Date();cutoff.setUTCFullYear(cutoff.getUTCFullYear()-1);
  const lastYear=series.filter(point=>new Date(`${point.date}T12:00:00Z`)>=cutoff);
  const peaks=[...lastYear].sort((a,b)=>b.value-a.value).slice(0,10);
  const valleys=[...lastYear].sort((a,b)=>a.value-b.value).slice(0,10);

  async function querySimem(){
    if(!start||!end||start>end){setError("La fecha inicial debe ser anterior a la fecha final.");return}
    setLoading(true);setError("");
    try{
      const response=await fetch(`/api/market?start=${start}&end=${end}`);
      const result=await response.json();
      if(!response.ok)throw new Error(result.error??"No se pudo consultar SIMEM.");
      setPayload(result);
      const newVariables=[...new Set((result.rows as MarketRow[]).map(row=>row.variable))];
      const preferred=newVariables.includes("PPBOGReal")?"PPBOGReal":newVariables[0];
      setVariable(preferred??"");
    }catch(cause){setError(cause instanceof Error?cause.message:"No se pudo consultar SIMEM.")}
    finally{setLoading(false)}
  }

  async function downloadGif(){
    setLoading(true);setError("");
    try{
      const response=await fetch("/api/gif",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({rows:selected})});
      if(!response.ok)throw new Error("No se pudo generar el GIF.");
      const url=URL.createObjectURL(await response.blob());const anchor=document.createElement("a");anchor.href=url;anchor.download="precios-xm-mensual.gif";anchor.click();URL.revokeObjectURL(url);
    }catch(cause){setError(cause instanceof Error?cause.message:"No se pudo generar el GIF.")}
    finally{setLoading(false)}
  }

  return <main>
    <header className="topbar"><a className="brand" href="#top"><span><Icon name="bolt"/></span><div><b>Pulso XM</b><small>Mercado eléctrico colombiano</small></div></a><nav aria-label="Navegación principal"><button className={activeTab==="overview"?"active":""} onClick={()=>setActiveTab("overview")}>Resumen</button><button className={activeTab==="data"?"active":""} onClick={()=>setActiveTab("data")}>Datos</button><button className={activeTab==="analytics"?"active":""} onClick={()=>setActiveTab("analytics")}>Analítica</button></nav><a className="source-link" href="https://www.simem.co/" target="_blank" rel="noreferrer">Fuente SIMEM ↗</a></header>

    <section className="hero" id="top"><div><p className="kicker">INTELIGENCIA ENERGÉTICA · COLOMBIA</p><h1>Entiende el precio.<br/><em>Decide con contexto.</em></h1><p className="hero-copy">Consulta el mercado eléctrico, identifica patrones y exporta análisis claros desde una sola herramienta.</p></div><div className="live-price"><span><i/> ÚLTIMA SEÑAL</span><strong>{money.format(latest)}</strong><small>COP/kWh</small><p className={delta>=0?"positive":"negative"}>{delta>=0?"↑":"↓"} {Math.abs(delta).toFixed(2)}% frente al periodo anterior</p><footer>{series.at(-1)?.date?fullDate.format(new Date(`${series.at(-1)!.date}T12:00:00Z`)):"Sin fecha"}</footer></div></section>

    <section className="query-shell"><div className="query-heading"><span className="query-icon"><Icon name="calendar"/></span><div><b>Consulta de mercado</b><small>Rango inicial: últimos 30 días · consulta directa a SIMEM</small></div></div><div className="query-fields"><label>Desde<input type="date" value={start} onChange={event=>setStart(event.target.value)}/></label><label>Hasta<input type="date" value={end} onChange={event=>setEnd(event.target.value)}/></label><button className="primary-button" disabled={loading} onClick={querySimem}><Icon name="refresh"/>{loading?"Consultando…":"Consultar SIMEM"}</button></div>{error&&<p className="error" role="alert">{error}</p>}</section>

    <section className="workspace">
      <div className="section-heading"><div><span>Fuente: {payload.source}</span><h2>{activeTab==="overview"?"Panorama del mercado":activeTab==="data"?"Explorador de datos":"Analítica descriptiva"}</h2></div><div className="filters"><label>Producto<select value={variable} onChange={event=>setVariable(event.target.value)}>{variables.map(item=><option key={item}>{item}</option>)}</select></label><label>Versión<select value={version} onChange={event=>setVersion(event.target.value)}>{versions.map(item=><option key={item}>{item}</option>)}</select></label></div></div>

      {activeTab==="overview"&&<Overview values={values} series={series} smooth={smooth} frequency={frequency} setFrequency={setFrequency}/>} 
      {activeTab==="data"&&<DataExplorer rows={selected} series={aggregate(selected,"daily")} loading={loading} onCsv={()=>downloadText(marketRowsToCsv(selected),"precios_xm.csv","text/csv;charset=utf-8")} onGif={downloadGif}/>} 
      {activeTab==="analytics"&&<div className="analytics-grid">
        <AnalysisCard index="01" title="Tendencia y media móvil" text={trendSummary(series,frequencyLabels[frequency])} wide><div className="frequency-control">{(["daily","weekly","monthly"] as Frequency[]).map(item=><button key={item} className={frequency===item?"active":""} onClick={()=>setFrequency(item)}>{frequencyLabels[item]}</button>)}</div><LineChart series={series} comparison={smooth}/></AnalysisCard>
        <AnalysisCard index="02" title="Distribución de precios" text={distributionSummary(series.map(point=>point.value))}><Histogram bins={hist}/></AnalysisCard>
        <AnalysisCard index="03" title="Estacionalidad mensual" text={monthlyBoxSummary(selected)}><BoxPlot boxes={boxes}/></AnalysisCard>
        <AnalysisCard index="04" title="Mapa Año × Mes" text={heat.length?`El promedio más alto fue ${money.format(Math.max(...heat.map(cell=>cell.value)))} COP/kWh y el más bajo ${money.format(Math.min(...heat.map(cell=>cell.value)))} COP/kWh.`:"Sin datos suficientes."} wide><Heatmap cells={heat}/></AnalysisCard>
        <AnalysisCard index="05" title="Persistencia lag‑1" text={persistenceSummary(persistenceData.correlation)} wide><ScatterPlot data={persistenceData}/></AnalysisCard>
        <AnalysisCard index="06" title="Picos y valles · 12 meses" text={`Amplitud del periodo: ${money.format((peaks[0]?.value??0)-(valleys[0]?.value??0))} COP/kWh.`} wide><Extremes peaks={peaks} valleys={valleys}/></AnalysisCard>
      </div>}
    </section>
    <footer className="footer"><div className="brand footer-brand"><span><Icon name="bolt"/></span><div><b>Pulso XM</b><small>Proyecto de Yoseth Mosquera · Universidad de Antioquia</small></div></div><p>Datos de SIMEM · Next.js + TypeScript + Supabase + Vercel</p></footer>
  </main>;
}

function Overview({values,series,smooth,frequency,setFrequency}:{values:number[];series:{date:string;value:number}[];smooth:{date:string;value:number}[];frequency:Frequency;setFrequency:(value:Frequency)=>void}){
  const cards=[
    ["Promedio",mean(values),"Media del periodo","average"],["Máximo",values.length?Math.max(...values):0,"Mayor valor observado","maximum"],["Mínimo",values.length?Math.min(...values):0,"Menor valor observado","minimum"],["Desviación",deviation(values),"Variabilidad del periodo","deviation"],["Mediana",median(values),"Punto medio de los datos","median"],
  ];
  return <><div className="metrics">{cards.map(([label,value,note,tone])=><article className={`metric ${tone}`} key={String(label)}><span>{label}</span><strong>{money.format(Number(value))}</strong><small>COP/kWh · {note}</small></article>)}</div><article className="card main-chart"><div className="card-head"><div><span>EVOLUCIÓN DEL PRECIO</span><h3>Serie temporal</h3></div><div className="frequency-control">{(["daily","weekly","monthly"] as Frequency[]).map(item=><button key={item} className={frequency===item?"active":""} onClick={()=>setFrequency(item)}>{frequencyLabels[item]}</button>)}</div></div><LineChart series={series} comparison={smooth}/><div className="legend"><span><i className="real"/>Precio promedio</span><span><i className="smooth"/>Media móvil</span></div><p className="analysis-note"><b>Lectura automática.</b> {trendSummary(series,frequencyLabels[frequency])}</p></article></>;
}

function DataExplorer({rows,series,loading,onCsv,onGif}:{rows:MarketRow[];series:{date:string;value:number}[];loading:boolean;onCsv:()=>void;onGif:()=>void}){
  const [page,setPage]=useState(0);const [playing,setPlaying]=useState(true);
  const months=[...new Set(series.map(point=>point.date.slice(0,7)))];const [frame,setFrame]=useState(Math.max(0,months.length-1));
  useEffect(()=>{setFrame(Math.max(0,months.length-1))},[months.length]);
  useEffect(()=>{if(!playing||months.length<2)return;const timer=setInterval(()=>setFrame(value=>(value+1)%months.length),1200);return()=>clearInterval(timer)},[playing,months.length]);
  const current=months[frame];const monthSeries=series.filter(point=>point.date.startsWith(current??""));const pageRows=rows.slice(page*25,page*25+25);const pages=Math.max(1,Math.ceil(rows.length/25));
  return <div className="data-layout"><article className="card table-card"><div className="card-head"><div><span>REGISTROS</span><h3>{rows.length.toLocaleString("es-CO")} observaciones</h3></div><button className="secondary-button" onClick={onCsv}><Icon name="download"/>Descargar CSV</button></div><div className="table-scroll"><table><thead><tr><th>Fecha</th><th>Variable</th><th>Versión</th><th>Duración</th><th>Unidad</th><th>Valor</th></tr></thead><tbody>{pageRows.map((row,index)=><tr key={`${row.date}-${row.variable}-${row.version}-${index}`}><td>{row.date}</td><td><b>{row.variable}</b></td><td><span className="tag">{row.version}</span></td><td>{row.duration}</td><td>{row.unit}</td><td className="numeric">{money.format(row.value)}</td></tr>)}</tbody></table></div><div className="pagination"><button disabled={page===0} onClick={()=>setPage(value=>value-1)}>Anterior</button><span>Página {page+1} de {pages}</span><button disabled={page>=pages-1} onClick={()=>setPage(value=>value+1)}>Siguiente</button></div></article><article className="card story-card"><div className="card-head"><div><span>HISTORIA MENSUAL</span><h3>{current??"Sin datos"}</h3></div><button className="icon-button" aria-label={playing?"Pausar":"Reproducir"} onClick={()=>setPlaying(value=>!value)}><Icon name={playing?"pause":"play"}/></button></div><LineChart series={monthSeries}/><div className="story-actions"><div className="dots">{months.map((month,index)=><button key={month} aria-label={`Ver ${month}`} className={index===frame?"active":""} onClick={()=>setFrame(index)}/>)}</div><button className="secondary-button" disabled={loading} onClick={onGif}><Icon name="download"/>{loading?"Generando…":"Descargar GIF"}</button></div></article></div>;
}

function AnalysisCard({index,title,text,wide=false,children}:{index:string;title:string;text:string;wide?:boolean;children:React.ReactNode}){return <article className={`card analysis-card ${wide?"wide":""}`}><div className="analysis-title"><span>{index}</span><h3>{title}</h3></div>{children}<p className="analysis-note"><b>Análisis.</b> {text}</p></article>}

function Extremes({peaks,valleys}:{peaks:{date:string;value:number}[];valleys:{date:string;value:number}[]}){return <div className="extremes"><div><h4>Máximos</h4>{peaks.map((point,index)=><div className="extreme-row" key={point.date}><span>{String(index+1).padStart(2,"0")}</span><b>{point.date}</b><strong>{money.format(point.value)}</strong></div>)}</div><div><h4>Mínimos</h4>{valleys.map((point,index)=><div className="extreme-row" key={point.date}><span>{String(index+1).padStart(2,"0")}</span><b>{point.date}</b><strong>{money.format(point.value)}</strong></div>)}</div></div>}
