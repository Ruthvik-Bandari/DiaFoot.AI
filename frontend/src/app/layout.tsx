import type { Metadata } from "next";
import type { ReactNode } from "react";
import { AppRouterCacheProvider } from "@mui/material-nextjs/v16-appRouter";
import { Box } from "@/lib/mui";
import Providers from "@/components/Providers";
import Sidebar, { DRAWER_WIDTH } from "@/components/layout/Sidebar";

export const metadata: Metadata = {
  title: "DiaFoot.AI — Wound Analysis Dashboard",
  description:
    "Academic DFU detection and segmentation system. Not for clinical use.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link
          href="https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;600;700&display=swap"
          rel="stylesheet"
        />
      </head>
      <body>
        <AppRouterCacheProvider>
          <Providers>
            <Box sx={{ display: "flex", minHeight: "100vh" }}>
              <Sidebar />
              <Box
                component="main"
                sx={{
                  flexGrow: 1,
                  ml: `${DRAWER_WIDTH}px`,
                  p: { xs: 2, md: 4 },
                  backgroundColor: "background.default",
                  minHeight: "100vh",
                }}
              >
                {children}
              </Box>
            </Box>
          </Providers>
        </AppRouterCacheProvider>
      </body>
    </html>
  );
}
