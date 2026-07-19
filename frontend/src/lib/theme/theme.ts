"use client";

import { createTheme } from "@mui/material/styles";

declare module "@mui/material/styles" {
  interface Palette {
    teal: Palette["primary"];
    seafoam: Palette["primary"];
  }
  interface PaletteOptions {
    teal?: PaletteOptions["primary"];
    seafoam?: PaletteOptions["primary"];
  }
}

const DISPLAY = "'Space Grotesk', 'Inter', system-ui, sans-serif";
const SANS = "'Inter', system-ui, -apple-system, sans-serif";

const theme = createTheme({
  palette: {
    mode: "dark",
    primary: {
      main: "#2DD4BF",
      light: "#5EEAD4",
      dark: "#0D9488",
      contrastText: "#04110E",
    },
    secondary: {
      main: "#38BDF8",
      light: "#7DD3FC",
      dark: "#0284C7",
      contrastText: "#03121C",
    },
    error: {
      main: "#FB7185",
      light: "#FDA4AF",
      dark: "#E11D48",
    },
    warning: {
      main: "#FBBF24",
      light: "#FCD34D",
      dark: "#D97706",
    },
    success: {
      main: "#34D399",
      light: "#6EE7B7",
      dark: "#059669",
    },
    info: {
      main: "#818CF8",
      light: "#A5B4FC",
      dark: "#4F46E5",
    },
    background: {
      default: "#060910",
      paper: "#0C111D",
    },
    text: {
      primary: "#E6EDF5",
      secondary: "#94A3B8",
    },
    divider: "rgba(255,255,255,0.08)",
    teal: {
      main: "#2DD4BF",
      light: "#5EEAD4",
      dark: "#0D9488",
      contrastText: "#04110E",
    },
    seafoam: {
      main: "#38BDF8",
      light: "#7DD3FC",
      dark: "#0284C7",
      contrastText: "#03121C",
    },
  },
  typography: {
    fontFamily: SANS,
    h1: { fontFamily: DISPLAY, fontWeight: 700, letterSpacing: "-0.03em" },
    h2: { fontFamily: DISPLAY, fontWeight: 700, letterSpacing: "-0.03em" },
    h3: { fontFamily: DISPLAY, fontWeight: 700, letterSpacing: "-0.025em" },
    h4: { fontFamily: DISPLAY, fontWeight: 600, letterSpacing: "-0.02em" },
    h5: { fontFamily: DISPLAY, fontWeight: 600, letterSpacing: "-0.015em" },
    h6: { fontFamily: DISPLAY, fontWeight: 600, letterSpacing: "-0.01em" },
    subtitle1: { fontWeight: 500, color: "#94A3B8" },
    subtitle2: { fontWeight: 600, letterSpacing: "0.01em" },
    overline: { fontWeight: 600, letterSpacing: "0.14em", fontSize: "0.68rem" },
    button: { textTransform: "none", fontWeight: 600, letterSpacing: "0.01em" },
  },
  shape: { borderRadius: 14 },
  components: {
    MuiCssBaseline: {
      styleOverrides: {
        body: { backgroundColor: "#060910" },
      },
    },
    MuiCard: {
      defaultProps: { elevation: 0 },
      styleOverrides: {
        root: {
          position: "relative",
          borderRadius: 20,
          backgroundColor: "rgba(18,24,38,0.62)",
          backgroundImage:
            "linear-gradient(160deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.01) 40%, rgba(255,255,255,0) 100%)",
          backdropFilter: "blur(18px)",
          WebkitBackdropFilter: "blur(18px)",
          border: "1px solid rgba(255,255,255,0.08)",
          boxShadow:
            "0 1px 0 rgba(255,255,255,0.05) inset, 0 20px 48px -24px rgba(0,0,0,0.9)",
          transition:
            "transform 0.25s cubic-bezier(0.16,1,0.3,1), border-color 0.25s ease, box-shadow 0.25s ease",
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          padding: "10px 22px",
          fontSize: "0.9rem",
          transition: "transform 0.18s cubic-bezier(0.16,1,0.3,1), box-shadow 0.2s ease, filter 0.2s ease",
          "&:active": { transform: "scale(0.97)" },
        },
        contained: {
          color: "#04110E",
          backgroundImage: "linear-gradient(120deg, #2dd4bf 0%, #38bdf8 55%, #6366f1 130%)",
          boxShadow: "0 8px 28px -10px rgba(45,212,191,0.65)",
          "&:hover": {
            backgroundImage: "linear-gradient(120deg, #34e0cb 0%, #4cc6ff 55%, #7a7dff 130%)",
            boxShadow: "0 12px 34px -8px rgba(56,189,248,0.7)",
          },
        },
        outlined: {
          borderColor: "rgba(255,255,255,0.18)",
          color: "#E6EDF5",
          "&:hover": {
            borderColor: "rgba(45,212,191,0.6)",
            backgroundColor: "rgba(45,212,191,0.08)",
          },
        },
        text: { color: "#CBD5E1" },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          borderRadius: 9,
          fontWeight: 600,
          fontSize: "0.72rem",
          letterSpacing: "0.01em",
          backdropFilter: "blur(6px)",
        },
        outlined: { borderColor: "rgba(255,255,255,0.16)" },
      },
    },
    MuiDivider: {
      styleOverrides: { root: { borderColor: "rgba(255,255,255,0.08)" } },
    },
    MuiDrawer: {
      styleOverrides: {
        paper: {
          border: "none",
          backgroundColor: "transparent",
          backgroundImage: "none",
        },
      },
    },
    MuiAppBar: {
      styleOverrides: {
        root: {
          backgroundColor: "rgba(8,11,18,0.6)",
          backdropFilter: "blur(14px)",
          color: "#E6EDF5",
          boxShadow: "none",
          borderBottom: "1px solid rgba(255,255,255,0.08)",
        },
      },
    },
    MuiTableCell: {
      styleOverrides: {
        root: { borderColor: "rgba(255,255,255,0.07)" },
        head: {
          color: "#94A3B8",
          fontWeight: 700,
          fontSize: "0.7rem",
          letterSpacing: "0.08em",
          textTransform: "uppercase",
        },
      },
    },
    MuiLinearProgress: {
      styleOverrides: {
        root: {
          height: 8,
          borderRadius: 999,
          backgroundColor: "rgba(255,255,255,0.08)",
        },
        bar: { borderRadius: 999 },
      },
    },
    MuiAlert: {
      styleOverrides: {
        root: {
          borderRadius: 16,
          border: "1px solid rgba(255,255,255,0.08)",
          backdropFilter: "blur(12px)",
          alignItems: "center",
        },
        standardError: { backgroundColor: "rgba(251,113,133,0.1)", color: "#FECDD3" },
        standardWarning: { backgroundColor: "rgba(251,191,36,0.1)", color: "#FDE68A" },
        standardSuccess: { backgroundColor: "rgba(52,211,153,0.1)", color: "#A7F3D0" },
        standardInfo: { backgroundColor: "rgba(129,140,248,0.1)", color: "#C7D2FE" },
      },
    },
    MuiTooltip: {
      styleOverrides: {
        tooltip: {
          backgroundColor: "rgba(20,27,43,0.96)",
          border: "1px solid rgba(255,255,255,0.1)",
          borderRadius: 10,
          fontSize: "0.72rem",
          backdropFilter: "blur(8px)",
          padding: "8px 10px",
        },
      },
    },
    MuiSkeleton: {
      styleOverrides: {
        root: { borderRadius: 12, backgroundColor: "rgba(255,255,255,0.06)" },
      },
    },
    MuiLink: {
      styleOverrides: { root: { color: "#5EEAD4", textUnderlineOffset: 3 } },
    },
  },
});

export default theme;
