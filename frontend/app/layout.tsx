import type React from "react"
import type { Metadata } from "next"
import { Inter } from "next/font/google"
import "./globals.css"

const inter = Inter({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-inter",
})

export const metadata: Metadata = {
  title: "SPJIMR PGPM Assistant",
  description: "Get answers about the SPJIMR Post Graduate Programme in Management (PGPM)",
  icons: [
    { rel: "icon", url: "/favicon.png", type: "image/png", sizes: "192x192" },
    { rel: "apple-touch-icon", url: "/favicon.png" },
  ],
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en">
      <body className={`font-sans ${inter.variable} antialiased`}>{children}</body>
    </html>
  )
}
