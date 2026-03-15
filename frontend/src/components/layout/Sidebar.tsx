"use client";

import { usePathname } from "next/navigation";
import Link from "next/link";
import Box from "@mui/material/Box";
import Drawer from "@mui/material/Drawer";
import List from "@mui/material/List";
import ListItemButton from "@mui/material/ListItemButton";
import ListItemIcon from "@mui/material/ListItemIcon";
import ListItemText from "@mui/material/ListItemText";
import Typography from "@mui/material/Typography";
import Chip from "@mui/material/Chip";
import Divider from "@mui/material/Divider";
import DashboardIcon from "@mui/icons-material/Dashboard";
import CameraAltIcon from "@mui/icons-material/CameraAlt";
import InfoIcon from "@mui/icons-material/Info";
import FavoriteIcon from "@mui/icons-material/Favorite";
import { useHealth, getIsDemoMode } from "@/lib/api";

const DRAWER_WIDTH = 260;
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

export default function Sidebar() {
  const pathname = usePathname();
  const { data: health, isError: healthError } = useHealth();
  const isDemoMode = getIsDemoMode();
  const isLiveMode = !isDemoMode && !!health && !healthError;

  return (
    <Drawer
      variant="permanent"
      sx={{
        width: DRAWER_WIDTH,
        flexShrink: 0,
        "& .MuiDrawer-paper": {
          width: DRAWER_WIDTH,
          boxSizing: "border-box",
          background: "linear-gradient(180deg, #FFFFFF 0%, #F0F7FA 100%)",
        },
      }}
    >
      {/* Logo */}
      <Box sx={{ px: 2.5, py: 3, display: "flex", alignItems: "center", gap: 1.5 }}>
        <Box
          sx={{
            width: 40,
            height: 40,
            borderRadius: 2,
            background: "linear-gradient(135deg, #065A82 0%, #00A896 100%)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          <FavoriteIcon sx={{ color: "#fff", fontSize: 22 }} />
        </Box>
        <Box>
          <Typography variant="h6" sx={{ fontSize: "1.1rem", lineHeight: 1.2, color: "#065A82" }}>
            DiaFoot.AI
          </Typography>
          <Typography variant="caption" sx={{ color: "text.secondary", fontSize: "0.7rem" }}>
            v2.0.0 — Wound Analysis
          </Typography>
        </Box>
      </Box>

      <Divider sx={{ mx: 2 }} />

      {/* Navigation */}
      <List sx={{ px: 1.5, py: 2 }}>
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
                py: 1.2,
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
              <ListItemIcon sx={{ minWidth: 40, color: isActive ? "#065A82" : "text.secondary" }}>
                {item.icon}
              </ListItemIcon>
              <ListItemText
                primary={item.label}
                primaryTypographyProps={{
                  fontSize: "0.9rem",
                  fontWeight: isActive ? 600 : 400,
                }}
              />
            </ListItemButton>
          );
        })}
      </List>

      {/* Status indicator */}
      <Box sx={{ mt: "auto", px: 2.5, pb: 3 }}>
        <Box sx={{ display: "flex", gap: 1, mb: 1.5, flexWrap: "wrap" }}>
          <Chip
            size="small"
            label={isLiveMode ? "Live Backend" : "Demo / Offline"}
            color={isLiveMode ? "success" : "warning"}
            variant={isLiveMode ? "filled" : "outlined"}
          />
          <Chip
            size="small"
            label={health?.model_loaded ? "Model Loaded" : "Model Unknown"}
            color={health?.model_loaded ? "success" : "default"}
            variant="outlined"
          />
        </Box>

        <Box
          sx={{
            p: 2,
            borderRadius: 2,
            backgroundColor: isDemoMode ? "rgba(243,156,18,0.08)" : "rgba(39,174,96,0.08)",
            border: `1px solid ${isDemoMode ? "rgba(243,156,18,0.2)" : "rgba(39,174,96,0.2)"}`,
          }}
        >
          <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 0.5 }}>
            <Box
              sx={{
                width: 8,
                height: 8,
                borderRadius: "50%",
                backgroundColor: isDemoMode ? "#F39C12" : "#27AE60",
              }}
            />
            <Typography variant="caption" fontWeight={600} color={isDemoMode ? "warning.dark" : "success.dark"}>
              {isDemoMode ? "Demo Mode" : "API Connected"}
            </Typography>
          </Box>
          <Typography variant="caption" color="text.secondary" sx={{ fontSize: "0.7rem" }}>
            {isDemoMode
              ? "Using simulated results"
              : `Model: ${health?.version ?? "..."}`}
          </Typography>
          <Typography
            variant="caption"
            color="text.secondary"
            sx={{
              fontSize: "0.66rem",
              display: "block",
              mt: 0.75,
              wordBreak: "break-all",
            }}
          >
            API: {API_BASE}
          </Typography>
        </Box>

        <Typography
          variant="caption"
          color="text.secondary"
          sx={{ display: "block", mt: 2, fontSize: "0.65rem", textAlign: "center" }}
        >
          Academic project only — not for clinical use
        </Typography>
      </Box>
    </Drawer>
  );
}

export { DRAWER_WIDTH };
