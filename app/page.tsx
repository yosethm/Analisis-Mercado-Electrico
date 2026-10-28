import Dashboard from "./dashboard";
import { getMarketData } from "@/lib/market-data";

export const revalidate = 900;

export default async function Home() {
  const payload = await getMarketData();
  return <Dashboard initialData={payload} />;
}

