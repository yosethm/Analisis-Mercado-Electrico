import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Pulso XM | Mercado eléctrico colombiano",
  description: "Consulta, analiza y exporta indicadores del mercado eléctrico colombiano.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="es"><body>{children}</body></html>;
}
