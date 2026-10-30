"use client";

import type { SeriesPoint } from "@/types/market";
import type { monthlyBoxes, heatmap, persistence } from "@/lib/statistics";

const number = new Intl.NumberFormat("es-CO", { maximumFractionDigits: 1 });
const date = new Intl.DateTimeFormat("es-CO", { day: "2-digit", month: "short" });
const monthNames = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"];

type Box = ReturnType<typeof monthlyBoxes>[number];
type HeatCell = ReturnType<typeof heatmap>[number];
type PersistenceData = ReturnType<typeof persistence>;

const frame = { width: 920, height: 330, left: 58, right: 24, top: 24, bottom: 42 };

function scale(min: number, max: number, start: number, end: number) {
  return (value: number) => start + ((value - min) / (max - min || 1)) * (end - start);
}

export function LineChart({ series, comparison, label = "Precio" }: { series: SeriesPoint[]; comparison?: SeriesPoint[]; label?: string }) {
  if (!series.length) return <EmptyChart/>;
  const values = [...series, ...(comparison ?? [])].map((point) => point.value);
  const min = Math.min(...values) * .94, max = Math.max(...values) * 1.04;
  const x = scale(0, Math.max(1, series.length - 1), frame.left, frame.width - frame.right);
  const y = scale(min, max, frame.height - frame.bottom, frame.top);
  const path = series.map((point, index) => `${index ? "L" : "M"}${x(index)},${y(point.value)}`).join(" ");
  const comparisonPath = comparison?.map((point, index) => `${index ? "L" : "M"}${x(index)},${y(point.value)}`).join(" ");
  const ticks = [min, min + (max-min)/2, max];
  return <div className="chart-scroll"><svg className="chart" viewBox={`0 0 ${frame.width} ${frame.height}`} role="img" aria-label={`${label} en el tiempo`}>
    <defs><linearGradient id="lineArea" x1="0" y1="0" x2="0" y2="1"><stop stopColor="#20a875" stopOpacity=".22"/><stop offset="1" stopColor="#20a875" stopOpacity="0"/></linearGradient></defs>
    {ticks.map((tick) => <g key={tick}><line className="grid" x1={frame.left} x2={frame.width-frame.right} y1={y(tick)} y2={y(tick)}/><text className="axis" x={frame.left-10} y={y(tick)+4} textAnchor="end">{number.format(tick)}</text></g>)}
    <path d={`${path} L${x(series.length-1)},${frame.height-frame.bottom} L${x(0)},${frame.height-frame.bottom} Z`} fill="url(#lineArea)"/>
    <path d={path} className="line primary"/>
    {comparisonPath && <path d={comparisonPath} className="line comparison"/>}
    {series.map((point,index) => index % Math.max(1,Math.floor(series.length/5)) === 0 && <text key={point.date} className="axis" x={x(index)} y={frame.height-13} textAnchor="middle">{date.format(new Date(`${point.date}T12:00:00Z`))}</text>)}
    {series.map((point,index)=><circle key={`${point.date}-${index}`} className="hit" cx={x(index)} cy={y(point.value)} r="9"><title>{`${point.date}: ${number.format(point.value)} COP/kWh`}</title></circle>)}
  </svg></div>;
}

export function Histogram({ bins }: { bins: { from: number; to: number; count: number }[] }) {
  if (!bins.length) return <EmptyChart/>;
  const max = Math.max(...bins.map((bin) => bin.count), 1);
  const width=920,height=300,left=50,bottom=38,top=20;
  const barWidth=(width-left-20)/bins.length;
  const y=scale(0,max,height-bottom,top);
  return <div className="chart-scroll"><svg className="chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Distribución de precios">
    {[0,Math.ceil(max/2),max].map(tick=><g key={tick}><line className="grid" x1={left} x2={width-20} y1={y(tick)} y2={y(tick)}/><text className="axis" x={left-10} y={y(tick)+4} textAnchor="end">{tick}</text></g>)}
    {bins.map((bin,index)=><g key={bin.from}><rect className="hist-bar" x={left+index*barWidth+2} y={y(bin.count)} width={Math.max(1,barWidth-4)} height={height-bottom-y(bin.count)} rx="3"><title>{`${number.format(bin.from)}–${number.format(bin.to)}: ${bin.count} observaciones`}</title></rect>{index%3===0&&<text className="axis" x={left+index*barWidth+barWidth/2} y={height-13} textAnchor="middle">{number.format(bin.from)}</text>}</g>)}
  </svg></div>;
}

export function BoxPlot({ boxes }: { boxes: Box[] }) {
  if (!boxes.length) return <EmptyChart/>;
  const all=boxes.flatMap(box=>[box.min,box.max]);
  const min=Math.min(...all)*.96,max=Math.max(...all)*1.04;
  const width=920,height=310,left=56,bottom=42,top=20;
  const slot=(width-left-20)/boxes.length;
  const y=scale(min,max,height-bottom,top);
  return <div className="chart-scroll"><svg className="chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Boxplot por mes">
    {[min,(min+max)/2,max].map(tick=><g key={tick}><line className="grid" x1={left} x2={width-20} y1={y(tick)} y2={y(tick)}/><text className="axis" x={left-10} y={y(tick)+4} textAnchor="end">{number.format(tick)}</text></g>)}
    {boxes.map((box,index)=>{const cx=left+slot*index+slot/2;const bw=Math.min(46,slot*.55);return <g key={box.month}><line className="whisker" x1={cx} x2={cx} y1={y(box.min)} y2={y(box.max)}/><line className="whisker" x1={cx-bw/3} x2={cx+bw/3} y1={y(box.min)} y2={y(box.min)}/><line className="whisker" x1={cx-bw/3} x2={cx+bw/3} y1={y(box.max)} y2={y(box.max)}/><rect className="box" x={cx-bw/2} y={y(box.q3)} width={bw} height={Math.max(2,y(box.q1)-y(box.q3))} rx="3"><title>{`${monthNames[box.month]} · mediana ${number.format(box.median)} · IQR ${number.format(box.q3-box.q1)}`}</title></rect><line className="median" x1={cx-bw/2} x2={cx+bw/2} y1={y(box.median)} y2={y(box.median)}/><text className="axis" x={cx} y={height-14} textAnchor="middle">{monthNames[box.month]}</text></g>})}
  </svg></div>;
}

export function Heatmap({ cells }: { cells: HeatCell[] }) {
  if (!cells.length) return <EmptyChart/>;
  const years=[...new Set(cells.map(cell=>cell.year))];
  const values=cells.map(cell=>cell.value),min=Math.min(...values),max=Math.max(...values);
  const getColor=(value:number)=>{const t=(value-min)/(max-min||1);return `color-mix(in srgb, #e7f5ee ${100-t*90}%, #087653)`};
  return <div className="heatmap" role="img" aria-label="Mapa de calor de precios por año y mes"><div className="heat-head"><span/>{monthNames.map(name=><b key={name}>{name}</b>)}</div>{years.map(year=><div className="heat-row" key={year}><strong>{year}</strong>{Array.from({length:12},(_,month)=>{const cell=cells.find(item=>item.year===year&&item.month===month);const label=cell?`${monthNames[month]} ${year}: ${number.format(cell.value)} COP/kWh`:"Sin datos";return <i key={month} aria-label={label} title={label} className={!cell?"empty-cell":""} style={cell?{background:getColor(cell.value)}:undefined}/>})}</div>)}</div>;
}

export function ScatterPlot({ data }: { data: PersistenceData }) {
  if (!data.pairs.length) return <EmptyChart/>;
  const values=data.pairs.flatMap(pair=>[pair.previous,pair.current]);
  const min=Math.min(...values)*.95,max=Math.max(...values)*1.05;
  const width=920,height=310,left=58,bottom=46,top=22;
  const x=scale(min,max,left,width-24),y=scale(min,max,height-bottom,top);
  return <div className="chart-scroll"><svg className="chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Persistencia del precio">
    {[min,(min+max)/2,max].map(tick=><g key={tick}><line className="grid" x1={left} x2={width-24} y1={y(tick)} y2={y(tick)}/><text className="axis" x={left-10} y={y(tick)+4} textAnchor="end">{number.format(tick)}</text><text className="axis" x={x(tick)} y={height-15} textAnchor="middle">{number.format(tick)}</text></g>)}
    <line className="identity" x1={x(min)} y1={y(min)} x2={x(max)} y2={y(max)}/>
    {data.pairs.map(pair=><circle className="dot" key={pair.date} cx={x(pair.previous)} cy={y(pair.current)} r="5"><title>{`${pair.date}: anterior ${number.format(pair.previous)}, actual ${number.format(pair.current)}`}</title></circle>)}
    <text className="axis axis-title" x={(left+width-24)/2} y={height-2} textAnchor="middle">Precio anterior</text>
  </svg></div>;
}

function EmptyChart(){return <div className="chart-empty">No hay suficientes datos para esta visualización.</div>}
