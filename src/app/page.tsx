import { getInitialMarketData } from "@/lib/data-source";
import { MarketWorkspace } from "@/components/market-workspace";

export const revalidate = 900;

export default async function Home() {
  return <MarketWorkspace initialPayload={await getInitialMarketData()}/>;
}
