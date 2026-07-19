"use client";

import { usePathname } from "next/navigation";
import Link from "next/link";
import {
  Box,
  CameraAltIcon,
  Chip,
  DashboardIcon,
  Drawer,
  InfoIcon,
  List,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Typography,
} from "@/lib/mui";
import { useHealth, getIsDemoMode } from "@/lib/api";

/** Sidebar width in pixels — used by layout for main content offset. */
const SIDEBAR_WIDTH = 256;

const API_BASE = (process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");

interface NavItem {
  label: string;
  href: string;
  icon: React.ReactNode;
}

const NAV_ITEMS: NavItem[] = [
  { label: "Dashboard", href: "/", icon: <DashboardIcon /> },
  { label: "Analyze", href: "/predict", icon: <CameraAltIcon /> },
  { label: "About", href: "/about", icon: <InfoIcon /> },
];

interface SidebarProps {
  mobileOpen: boolean;
  onMobileClose: () => void;
}

function StatusDot({ color, pulse }: { color: string; pulse?: boolean }) {
  return (
    <Box
      sx={{
        width: 8,
        height: 8,
        borderRadius: "50%",
        backgroundColor: color,
        boxShadow: `0 0 10px ${color}`,
        flexShrink: 0,
        ...(pulse && {
          "@keyframes pulseDot": {
            "0%,100%": { opacity: 1 },
            "50%": { opacity: 0.35 },
          },
          animation: "pulseDot 2s ease-in-out infinite",
        }),
      }}
    />
  );
}

function SidebarContent() {
  const pathname = usePathname();
  const { data: health, isError: healthError } = useHealth();
  const isDemoMode = getIsDemoMode();
  const isLiveMode = !isDemoMode && !!health && !healthError;
  const accent = isLiveMode ? "#34D399" : "#FBBF24";

  return (
    <Box sx={{ display: "flex", flexDirection: "column", height: "100%", px: 2, pt: 2.5, pb: 3 }}>
      {/* Logo */}
      <Box sx={{ display: "flex", alignItems: "center", gap: 1.5, mb: 3 }}>
        <Box
          sx={{
            width: 42,
            height: 42,
            borderRadius: 3,
            display: "grid",
            placeItems: "center",
            background: "var(--grad-brand)",
            boxShadow: "0 8px 24px -6px rgba(45,212,191,0.6)",
            flexShrink: 0,
            fontFamily: "var(--font-display)",
            fontWeight: 700,
            fontSize: "1.05rem",
            color: "#04110E",
          }}
        >
          DF
        </Box>
        <Box sx={{ minWidth: 0 }}>
          <Typography
            sx={{
              fontFamily: "var(--font-display)",
              fontWeight: 700,
              fontSize: "1.1rem",
              lineHeight: 1.1,
              color: "#F1F5F9",
            }}
            noWrap
          >
            DiaFoot<Box component="span" sx={{ color: "primary.main" }}>.AI</Box>
          </Typography>
          <Typography variant="caption" sx={{ color: "text.secondary", fontSize: "0.66rem", letterSpacing: "0.04em" }} noWrap>
            v2.1.0 · DFU Intelligence
          </Typography>
        </Box>
      </Box>

      {/* Navigation */}
      <Typography variant="overline" sx={{ color: "text.secondary", px: 1, mb: 0.5, display: "block" }}>
        Navigate
      </Typography>
      <List sx={{ p: 0 }}>
        {NAV_ITEMS.map((item) => {
          const isActive = pathname === item.href;
          return (
            <ListItemButton
              key={item.href}
              component={Link}
              href={item.href}
              selected={isActive}
              sx={{
                position: "relative",
                borderRadius: 3,
                mb: 0.5,
                px: 2,
                py: 1.15,
                overflow: "hidden",
                transition: "background-color .2s ease, color .2s ease",
                "&::before": {
                  content: '""',
                  position: "absolute",
                  left: 0,
                  top: "22%",
                  bottom: "22%",
                  width: 3,
                  borderRadius: 999,
                  background: "var(--grad-brand)",
                  opacity: isActive ? 1 : 0,
                  transition: "opacity .2s ease",
                },
                "&.Mui-selected, &.Mui-selected:hover": {
                  backgroundColor: "rgba(45,212,191,0.10)",
                  color: "#5EEAD4",
                  "& .MuiListItemIcon-root": { color: "#5EEAD4" },
                },
                "&:hover": { backgroundColor: "rgba(255,255,255,0.04)" },
              }}
            >
              <ListItemIcon sx={{ minWidth: 38, color: isActive ? "#5EEAD4" : "text.secondary" }}>
                {item.icon}
              </ListItemIcon>
              <ListItemText
                primary={item.label}
                slotProps={{ primary: { fontSize: "0.9rem", fontWeight: isActive ? 600 : 500 } }}
              />
            </ListItemButton>
          );
        })}
      </List>

      {/* Status panel */}
      <Box sx={{ mt: "auto", pt: 2 }}>
        <Box
          sx={{
            p: 1.75,
            borderRadius: 2.5,
            backgroundColor: "rgba(18,24,38,0.6)",
            border: "1px solid rgba(255,255,255,0.08)",
            backdropFilter: "blur(12px)",
          }}
        >
          <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 1.5 }}>
            <StatusDot color={accent} pulse={isLiveMode} />
            <Typography variant="subtitle2" sx={{ fontSize: "0.78rem", color: accent }}>
              {isLiveMode ? "Live Backend" : isDemoMode ? "Demo Mode" : "Backend Offline"}
            </Typography>
          </Box>

          <Box sx={{ display: "flex", gap: 0.75, flexWrap: "wrap", mb: 1.5 }}>
            <Chip
              size="small"
              label={health?.model_loaded ? "Model loaded" : "No model"}
              color={health?.model_loaded ? "success" : "default"}
              variant="outlined"
              sx={{ fontSize: "0.66rem", height: 22 }}
            />
            <Chip
              size="small"
              label={`v${health?.version ?? "—"}`}
              variant="outlined"
              sx={{ fontSize: "0.66rem", height: 22 }}
            />
          </Box>

          <Typography
            variant="caption"
            sx={{ color: "text.secondary", fontSize: "0.62rem", display: "block", wordBreak: "break-all", fontFamily: "var(--font-mono)" }}
          >
            {API_BASE}
          </Typography>
        </Box>

        <Typography
          variant="caption"
          sx={{ display: "block", mt: 1.5, fontSize: "0.6rem", textAlign: "center", color: "text.secondary", lineHeight: 1.5 }}
        >
          Academic project · not a medical device
        </Typography>
      </Box>
    </Box>
  );
}

const drawerPaperSx = {
  width: SIDEBAR_WIDTH,
  boxSizing: "border-box" as const,
  backgroundColor: "rgba(9,12,20,0.72)",
  backdropFilter: "blur(18px)",
  WebkitBackdropFilter: "blur(18px)",
  borderRight: "1px solid rgba(255,255,255,0.08)",
};

export default function Sidebar({ mobileOpen, onMobileClose }: SidebarProps) {
  return (
    <>
      {/* Mobile: temporary drawer */}
      <Drawer
        variant="temporary"
        open={mobileOpen}
        onClose={onMobileClose}
        ModalProps={{ keepMounted: true }}
        sx={{
          display: { xs: "block", lg: "none" },
          "& .MuiDrawer-paper": drawerPaperSx,
        }}
      >
        <SidebarContent />
      </Drawer>

      {/* Desktop: permanent drawer */}
      <Drawer
        variant="permanent"
        sx={{
          display: { xs: "none", lg: "block" },
          width: SIDEBAR_WIDTH,
          flexShrink: 0,
          "& .MuiDrawer-paper": drawerPaperSx,
        }}
      >
        <SidebarContent />
      </Drawer>
    </>
  );
}

export { SIDEBAR_WIDTH };
