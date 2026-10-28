import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Pulso XM | Inteligencia del mercado eléctrico",
  description: "Centro de inteligencia para explorar y anticipar el precio de la energía en Colombia.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="es">
      <body>{children}</body>
    </html>
  );
}

