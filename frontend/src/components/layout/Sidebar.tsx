"use client";

import { usePathname } from "next/navigation";
import Link from "next/link";
import {
  Box,
  CameraAltIcon,
  Chip,
  DashboardIcon,
  Divider,
  Drawer,
  FavoriteIcon,
  InfoIcon,
  List,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Typography,
} from "@/lib/mui";
import { useHealth, getIsDemoMode } from "@/lib/api";

/** Sidebar width in pixels — used by layout for main content offset. */
const SIDEBAR_WIDTH = 240;

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

function SidebarContent() {
  const pathname = usePathname();
  const { data: health, isError: healthError } = useHealth();
  const isDemoMode = getIsDemoMode();
  const isLiveMode = !isDemoMode && !!health && !healthError;

  return (
    <Box sx={{ display: "flex", flexDirection: "column", height: "100%" }}>
      {/* Logo */}
      <Box sx={{ px: 2, py: 2.5, display: "flex", alignItems: "center", gap: 1.5 }}>
        <Box
          sx={{
            width: 36,
            height: 36,
            borderRadius: 2,
            background: "linear-gradient(135deg, #065A82 0%, #00A896 100%)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            flexShrink: 0,
          }}
        >
          <FavoriteIcon sx={{ color: "#fff", fontSize: 20 }} />
        </Box>
        <Box sx={{ minWidth: 0 }}>
          <Typography variant="h6" sx={{ fontSize: "1rem", lineHeight: 1.2, color: "#065A82" }} noWrap>
            DiaFoot.AI
          </Typography>
          <Typography variant="caption" sx={{ color: "text.secondary", fontSize: "0.65rem" }} noWrap>
            v2.0.0 — Wound Analysis
          </Typography>
        </Box>
      </Box>

      <Divider sx={{ mx: 1.5 }} />

      {/* Navigation */}
      <List sx={{ px: 1, py: 1.5 }}>
        {NAV_ITEMS.map((item) => {
          const isActive = pathname === item.href;
          return (
            <ListItemButton
              key={item.href}
              component={Link}
              href={item.href}
              selected={isActive}
              sx={{
                borderRadius: 2,
                mb: 0.5,
                px: 2,
                py: 1,
                "&.Mui-selected": {
                  backgroundColor: "rgba(6,90,130,0.08)",
                  color: "#065A82",
                  "& .MuiListItemIcon-root": { color: "#065A82" },
                },
                "&:hover": {
                  backgroundColor: "rgba(6,90,130,0.04)",
                },
              }}
            >
              <ListItemIcon sx={{ minWidth: 36, color: isActive ? "#065A82" : "text.secondary" }}>
                {item.icon}
              </ListItemIcon>
              <ListItemText
                primary={item.label}
                primaryTypographyProps={{
                  fontSize: "0.85rem",
                  fontWeight: isActive ? 600 : 400,
                }}
              />
            </ListItemButton>
          );
        })}
      </List>

      {/* Status indicator */}
      <Box sx={{ mt: "auto", px: 2, pb: 2 }}>
        <Box sx={{ display: "flex", gap: 0.75, mb: 1, flexWrap: "wrap" }}>
          <Chip
            size="small"
            label={isLiveMode ? "Live Backend" : "Demo / Offline"}
            color={isLiveMode ? "success" : "warning"}
            variant={isLiveMode ? "filled" : "outlined"}
            sx={{ fontSize: "0.7rem" }}
          />
          <Chip
            size="small"
            label={health?.model_loaded ? "Model Loaded" : "Model Unknown"}
            color={health?.model_loaded ? "success" : "default"}
            variant="outlined"
            sx={{ fontSize: "0.7rem" }}
          />
        </Box>

        <Box
          sx={{
            p: 1.5,
            borderRadius: 2,
            backgroundColor: isDemoMode ? "rgba(243,156,18,0.08)" : "rgba(39,174,96,0.08)",
            border: `1px solid ${isDemoMode ? "rgba(243,156,18,0.2)" : "rgba(39,174,96,0.2)"}`,
          }}
        >
          <Box sx={{ display: "flex", alignItems: "center", gap: 0.75, mb: 0.25 }}>
            <Box
              sx={{
                width: 7,
                height: 7,
                borderRadius: "50%",
                backgroundColor: isDemoMode ? "#F39C12" : "#27AE60",
                flexShrink: 0,
              }}
            />
            <Typography variant="caption" fontWeight={600} sx={{ fontSize: "0.7rem" }} color={isDemoMode ? "warning.dark" : "success.dark"}>
              {isDemoMode ? "Demo Mode" : "API Connected"}
            </Typography>
          </Box>
          <Typography variant="caption" color="text.secondary" sx={{ fontSize: "0.65rem" }}>
            {isDemoMode ? "Using simulated results" : `Model: ${health?.version ?? "..."}`}
          </Typography>
          <Typography
            variant="caption"
            color="text.secondary"
            sx={{ fontSize: "0.6rem", display: "block", mt: 0.5, wordBreak: "break-all" }}
          >
            API: {API_BASE}
          </Typography>
        </Box>

        <Typography
          variant="caption"
          color="text.secondary"
          sx={{ display: "block", mt: 1.5, fontSize: "0.6rem", textAlign: "center" }}
        >
          Academic project only — not for clinical use
        </Typography>
      </Box>
    </Box>
  );
}

const drawerPaperSx = {
  width: SIDEBAR_WIDTH,
  boxSizing: "border-box" as const,
  background: "linear-gradient(180deg, #FFFFFF 0%, #F0F7FA 100%)",
};

export default function Sidebar({ mobileOpen, onMobileClose }: SidebarProps) {
  return (
    <>
      {/* Mobile / tablet: temporary drawer */}
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
