"use client";

import { useCallback, useState } from "react";
import type { ReactNode } from "react";
import { Box, Button } from "@/lib/mui";
import { AmbientBackground } from "@/components/ui";
import Sidebar from "./Sidebar";

export default function AppShell({ children }: { children: ReactNode }) {
  const [mobileOpen, setMobileOpen] = useState(false);
  const handleToggle = useCallback(() => setMobileOpen((prev) => !prev), []);
  const handleClose = useCallback(() => setMobileOpen(false), []);

  return (
    <Box sx={{ display: "flex", minHeight: "100dvh", position: "relative" }}>
      <AmbientBackground />
      <Sidebar mobileOpen={mobileOpen} onMobileClose={handleClose} />

      <Box
        component="main"
        sx={{
          position: "relative",
          zIndex: 1,
          flexGrow: 1,
          minWidth: 0,
          p: { xs: 2, sm: 2.5, lg: 4 },
          minHeight: "100dvh",
        }}
      >
        {/* Mobile top bar with menu toggle — only below lg */}
        <Box
          sx={{
            display: { xs: "flex", lg: "none" },
            alignItems: "center",
            gap: 1.5,
            mb: 2,
          }}
        >
          <Button
            variant="outlined"
            size="small"
            onClick={handleToggle}
            aria-label="Open navigation menu"
            sx={{ minWidth: 0, px: 1.5, py: 1 }}
          >
            <Box component="span" sx={{ display: "block", width: 18, height: 2, bgcolor: "currentColor", boxShadow: "0 6px 0 currentColor, 0 -6px 0 currentColor" }} />
          </Button>
          <Box
            component="span"
            sx={{
              fontFamily: "var(--font-display)",
              fontWeight: 700,
              fontSize: "1.05rem",
              backgroundImage: "var(--grad-brand)",
              WebkitBackgroundClip: "text",
              backgroundClip: "text",
              color: "transparent",
            }}
          >
            DiaFoot.AI
          </Box>
        </Box>

        <Box sx={{ maxWidth: 1320, mx: "auto" }}>{children}</Box>
      </Box>
    </Box>
  );
}
