import { NextRequest, NextResponse } from "next/server";
import { GIFEncoder, applyPalette, quantize } from "gifenc";
import type { MarketRow } from "@/types/market";
import { mean, movingAverage } from "@/lib/statistics";

export const runtime = "nodejs";

// Compact frames keep the export comfortably inside Vercel's serverless limits.
const WIDTH = 560, HEIGHT = 300;
type Color = [number, number, number, number];

function frameBuffer(color: Color): Uint8ClampedArray {
  const pixels = new Uint8ClampedArray(WIDTH * HEIGHT * 4);
  for (let index = 0; index < pixels.length; index += 4) pixels.set(color, index);
  return pixels;
}

function pixel(buffer: Uint8ClampedArray, x: number, y: number, color: Color) {
  if (x < 0 || y < 0 || x >= WIDTH || y >= HEIGHT) return;
  buffer.set(color, (Math.floor(y) * WIDTH + Math.floor(x)) * 4);
}

function rect(buffer: Uint8ClampedArray, x: number, y: number, width: number, height: number, color: Color) {
  for (let py = Math.max(0, Math.floor(y)); py < Math.min(HEIGHT, Math.ceil(y + height)); py++) {
    for (let px = Math.max(0, Math.floor(x)); px < Math.min(WIDTH, Math.ceil(x + width)); px++) pixel(buffer, px, py, color);
  }
}

function line(buffer: Uint8ClampedArray, x0: number, y0: number, x1: number, y1: number, color: Color, thickness = 2) {
  const dx = Math.abs(x1-x0), sx=x0<x1?1:-1, dy=-Math.abs(y1-y0), sy=y0<y1?1:-1;
  let error=dx+dy;
  while (true) {
    rect(buffer, x0-thickness/2, y0-thickness/2, thickness, thickness, color);
    if (Math.round(x0)===Math.round(x1)&&Math.round(y0)===Math.round(y1)) break;
    const e2=2*error;
    if (e2>=dy){error+=dy;x0+=sx}
    if (e2<=dx){error+=dx;y0+=sy}
  }
}

function drawFrame(rows: MarketRow[]): Uint8ClampedArray {
  const buffer=frameBuffer([247,248,245,255]);
  const grouped=new Map<string,number[]>();
  rows.forEach(row=>grouped.set(row.date,[...(grouped.get(row.date)??[]),row.value]));
  const series=[...grouped.entries()].sort(([a],[b])=>a.localeCompare(b)).map(([date,values])=>({date,value:mean(values)}));
  const average=movingAverage(series,5);
  const values=series.map(point=>point.value);
  const min=Math.min(...values),max=Math.max(...values);
  const left=64,right=30,top=52,bottom=52;
  const x=(index:number)=>left+index*((WIDTH-left-right)/Math.max(1,series.length-1));
  const y=(value:number)=>HEIGHT-bottom-((value-min)/(max-min||1))*(HEIGHT-top-bottom);
  rect(buffer,0,0,WIDTH,10,[24,171,119,255]);
  for(let i=0;i<4;i++){const py=top+i*(HEIGHT-top-bottom)/3;line(buffer,left,py,WIDTH-right,py,[216,223,218,255],1)}
  [mean(values),Math.max(...values),Math.min(...values)].forEach((value,index)=>line(buffer,left,y(value),WIDTH-right,y(value),index===0?[109,79,190,255]:index===1?[214,78,78,255]:[48,113,190,255],1));
  series.slice(1).forEach((point,index)=>line(buffer,x(index),y(series[index].value),x(index+1),y(point.value),[17,49,39,255],3));
  average.slice(1).forEach((point,index)=>line(buffer,x(index),y(average[index].value),x(index+1),y(point.value),[24,171,119,255],2));
  series.forEach((point,index)=>rect(buffer,x(index)-3,y(point.value)-3,6,6,[24,171,119,255]));
  return buffer;
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json() as { rows?: MarketRow[] };
    const rows=(body.rows??[]).filter(row=>row.date&&Number.isFinite(row.value)).slice(0,5000);
    if (!rows.length) return NextResponse.json({ error: "No hay datos para animar." }, { status: 400 });
    const months=new Map<string,MarketRow[]>();
    rows.forEach(row=>{const key=row.date.slice(0,7);months.set(key,[...(months.get(key)??[]),row])});
    const gif=GIFEncoder();
    [...months.entries()].sort(([a],[b])=>a.localeCompare(b)).forEach(([,monthRows])=>{
      const rgba=drawFrame(monthRows);
      const palette=quantize(rgba,32);
      gif.writeFrame(applyPalette(rgba,palette),WIDTH,HEIGHT,{palette,delay:1000,repeat:0});
    });
    gif.finish();
    return new NextResponse(gif.bytes() as BodyInit,{headers:{"Content-Type":"image/gif","Content-Disposition":"attachment; filename=precios-xm-mensual.gif","Cache-Control":"no-store"}});
  } catch {
    return NextResponse.json({ error: "No se pudo generar el GIF." }, { status: 500 });
  }
}
