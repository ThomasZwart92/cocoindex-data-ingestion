import type { Metadata } from "next";
import "./globals.css";
import { Providers } from "@/lib/providers";
import Navigation from "@/components/Navigation";

export const metadata: Metadata = {
  title: "Data Ingestion Portal",
  description: "Document processing and review system",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">
        <Providers>
          <div className="min-h-screen flex flex-col">
            {/* Header */}
            <header style={{ borderBottom: '1px solid #E5E7EB', paddingTop: '24px', paddingBottom: '24px' }}>
              <div style={{ maxWidth: '85%', margin: '0 auto' }}>
                <div className="flex justify-between items-start">
                  <div>
                    <h1 className="font-bold" style={{ fontSize: '24px', color: '#2C3E50' }}>
                      DATA INGESTION PORTAL
                    </h1>
                    <p style={{ fontSize: '14px', color: '#7F8C8D', marginTop: '4px' }}>
                      Document processing and review system
                    </p>
                  </div>
                  <div style={{ fontSize: '14px', color: '#7F8C8D' }}>
                    user@email
                  </div>
                </div>
              </div>
            </header>
            
            {/* Navigation */}
            <Navigation />
            
            {/* Main Content */}
            <main className="flex-1">
              {children}
            </main>
          </div>
        </Providers>
      </body>
    </html>
  );
}