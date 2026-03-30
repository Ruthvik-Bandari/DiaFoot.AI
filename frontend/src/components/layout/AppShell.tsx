"use client";

import { useCallback, useState } from "react";
import type { ReactNode } from "react";
import { Box, Button } from "@/lib/mui";
import Sidebar, { SIDEBAR_WIDTH } from "./Sidebar";

export default function AppShell({ children }: { children: ReactNode }) {
  const [mobileOpen, setMobileOpen] = useState(false);
  const handleToggle = useCallback(() => setMobileOpen((prev) => !prev), []);
  const handleClose = useCallback(() => setMobileOpen(false), []);

  return (
    <Box sx={{ display: "flex", minHeight: "100vh" }}>
      <Sidebar mobileOpen={mobileOpen} onMobileClose={handleClose} />

      <Box
        component="main"
        sx={{
          flexGrow: 1,
          minWidth: 0,
          p: { xs: 2, sm: 2.5, lg: 3 },
          backgroundColor: "background.default",
          minHeight: "100vh",
        }}
      >
        {/* Mobile menu button — only visible below lg */}
        <Box sx={{ display: { xs: "block", lg: "none" }, mb: 2 }}>
          <Button
            variant="outlined"
            size="small"
            onClick={handleToggle}
            sx={{ minWidth: 0, px: 1.5 }}
          >
            &#9776;
          </Button>
        </Box>

        {children}
      </Box>
    </Box>
  );
}
