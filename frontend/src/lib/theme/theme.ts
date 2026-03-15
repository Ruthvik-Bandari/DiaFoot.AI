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

const theme = createTheme({
  palette: {
    mode: "light",
    primary: {
      main: "#065A82",
      light: "#1C7293",
      dark: "#0B1D26",
      contrastText: "#FFFFFF",
    },
    secondary: {
      main: "#00A896",
      light: "#33B9AB",
      dark: "#007568",
      contrastText: "#FFFFFF",
    },
    error: {
      main: "#E74C3C",
      light: "#EC7063",
      dark: "#C0392B",
    },
    warning: {
      main: "#F39C12",
      light: "#F5B041",
      dark: "#D68910",
    },
    success: {
      main: "#27AE60",
      light: "#52BE80",
      dark: "#1E8449",
    },
    background: {
      default: "#F0F7FA",
      paper: "#FFFFFF",
    },
    text: {
      primary: "#0B1D26",
      secondary: "#6B7B8D",
    },
    divider: "#B0C4D8",
    teal: {
      main: "#1C7293",
      light: "#2E94B9",
      dark: "#065A82",
      contrastText: "#FFFFFF",
    },
    seafoam: {
      main: "#00A896",
      light: "#33B9AB",
      dark: "#007568",
      contrastText: "#FFFFFF",
    },
  },
  typography: {
    fontFamily: "'Google Sans', 'Roboto', 'Helvetica Neue', Arial, sans-serif",
    h4: {
      fontWeight: 600,
      letterSpacing: "-0.02em",
    },
    h5: {
      fontWeight: 600,
      letterSpacing: "-0.01em",
    },
    h6: {
      fontWeight: 600,
    },
    subtitle1: {
      fontWeight: 500,
      color: "#6B7B8D",
    },
    button: {
      textTransform: "none",
      fontWeight: 600,
    },
  },
  shape: {
    borderRadius: 12,
  },
  components: {
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 16,
          boxShadow: "0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.06)",
          border: "1px solid rgba(176,196,216,0.3)",
          transition: "box-shadow 0.2s ease",
          "&:hover": {
            boxShadow:
              "0 4px 12px rgba(6,90,130,0.1), 0 2px 4px rgba(6,90,130,0.06)",
          },
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 10,
          padding: "10px 24px",
          fontSize: "0.9rem",
        },
        contained: {
          boxShadow: "none",
          "&:hover": {
            boxShadow: "0 2px 8px rgba(6,90,130,0.25)",
          },
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          fontWeight: 500,
        },
      },
    },
    MuiDrawer: {
      styleOverrides: {
        paper: {
          borderRight: "1px solid rgba(176,196,216,0.3)",
          backgroundColor: "#FFFFFF",
        },
      },
    },
    MuiAppBar: {
      styleOverrides: {
        root: {
          backgroundColor: "#FFFFFF",
          color: "#0B1D26",
          boxShadow: "0 1px 3px rgba(0,0,0,0.06)",
        },
      },
    },
    MuiSkeleton: {
      styleOverrides: {
        root: {
          borderRadius: 8,
        },
      },
    },
  },
});

export default theme;
