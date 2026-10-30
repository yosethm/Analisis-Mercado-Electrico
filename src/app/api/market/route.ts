import { NextRequest, NextResponse } from "next/server";
import { fetchSimemRange } from "@/lib/data-source";

export const runtime = "nodejs";

export async function GET(request: NextRequest) {
  const start = request.nextUrl.searchParams.get("start") ?? "";
  const end = request.nextUrl.searchParams.get("end") ?? "";
  try {
    const rows = await fetchSimemRange(start, end);
    return NextResponse.json(
      { rows, source: "SIMEM", syncedAt: new Date().toISOString() },
      { headers: { "Cache-Control": "public, s-maxage=600, stale-while-revalidate=86400" } },
    );
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "No se pudo consultar SIMEM." },
      { status: 502 },
    );
  }
}
