"use client";

import { useEffect, useRef, useState, type ReactNode } from "react";
import type { SxProps, Theme } from "@mui/material/styles";
import { Box, Card, CardContent, Typography } from "@/lib/mui";

/* ─────────────────────────────────────────────────────────────
   AmbientBackground — slow-drifting glow blobs behind the app.
   Purely decorative, non-interactive, respects reduced-motion.
   ───────────────────────────────────────────────────────────── */
export function AmbientBackground() {
  return (
    <Box
      aria-hidden
      sx={{
        position: "fixed",
        inset: 0,
        zIndex: 0,
        pointerEvents: "none",
        overflow: "hidden",
      }}
    >
      <Box
        sx={{
          position: "absolute",
          top: "-18%",
          left: "-10%",
          width: "48vw",
          height: "48vw",
          borderRadius: "50%",
          background: "radial-gradient(circle, rgba(45,212,191,0.22), transparent 62%)",
          filter: "blur(24px)",
          animation: "drift-a 22s ease-in-out infinite",
        }}
      />
      <Box
        sx={{
          position: "absolute",
          top: "-6%",
          right: "-14%",
          width: "44vw",
          height: "44vw",
          borderRadius: "50%",
          background: "radial-gradient(circle, rgba(99,102,241,0.22), transparent 60%)",
          filter: "blur(24px)",
          animation: "drift-b 26s ease-in-out infinite",
        }}
      />
      <Box
        sx={{
          position: "absolute",
          bottom: "-24%",
          left: "36%",
          width: "40vw",
          height: "40vw",
          borderRadius: "50%",
          background: "radial-gradient(circle, rgba(56,189,248,0.14), transparent 60%)",
          filter: "blur(26px)",
          animation: "drift-a 30s ease-in-out infinite",
        }}
      />
    </Box>
  );
}

/* ─────────────────────────────────────────────────────────────
   GradientText — brand gradient clipped to text.
   ───────────────────────────────────────────────────────────── */
export function GradientText({ children, sx }: { children: ReactNode; sx?: SxProps<Theme> }) {
  return (
    <Box
      component="span"
      sx={{
        backgroundImage: "var(--grad-brand)",
        WebkitBackgroundClip: "text",
        backgroundClip: "text",
        color: "transparent",
        WebkitTextFillColor: "transparent",
        ...sx,
      }}
    >
      {children}
    </Box>
  );
}

/* ─────────────────────────────────────────────────────────────
   SectionHeader — icon chip + title (+ optional subtitle/action).
   ───────────────────────────────────────────────────────────── */
export function SectionHeader({
  icon,
  title,
  subtitle,
  action,
}: {
  icon?: ReactNode;
  title: string;
  subtitle?: string;
  action?: ReactNode;
}) {
  return (
    <Box sx={{ display: "flex", alignItems: "center", gap: 1.5, mb: 2.5 }}>
      {icon && (
        <Box
          sx={{
            width: 38,
            height: 38,
            borderRadius: 2.5,
            display: "grid",
            placeItems: "center",
            color: "primary.light",
            background: "var(--grad-brand-soft)",
            border: "1px solid rgba(255,255,255,0.08)",
            flexShrink: 0,
          }}
        >
          {icon}
        </Box>
      )}
      <Box sx={{ minWidth: 0, flexGrow: 1 }}>
        <Typography variant="h6" sx={{ lineHeight: 1.2 }}>
          {title}
        </Typography>
        {subtitle && (
          <Typography variant="caption" color="text.secondary">
            {subtitle}
          </Typography>
        )}
      </Box>
      {action}
    </Box>
  );
}

/* ─────────────────────────────────────────────────────────────
   useCountUp — animate a number 0 → value with ease-out.
   Honors prefers-reduced-motion (snaps to final value).
   ───────────────────────────────────────────────────────────── */
export function useCountUp(value: number, durationMs = 1100, startDelayMs = 0) {
  const [display, setDisplay] = useState(0);
  const ref = useRef<number>(0);

  useEffect(() => {
    const reduce =
      typeof window !== "undefined" &&
      window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    if (reduce) {
      setDisplay(value);
      return;
    }
    let raf = 0;
    let start = 0;
    const from = ref.current;
    const tick = (t: number) => {
      if (!start) start = t + startDelayMs;
      const elapsed = t - start;
      if (elapsed < 0) {
        raf = requestAnimationFrame(tick);
        return;
      }
      const p = Math.min(1, elapsed / durationMs);
      const eased = 1 - Math.pow(1 - p, 3);
      const current = from + (value - from) * eased;
      ref.current = current;
      setDisplay(current);
      if (p < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [value, durationMs, startDelayMs]);

  return display;
}

function formatNumber(n: number, decimals: number, thousands: boolean) {
  const fixed = n.toFixed(decimals);
  if (!thousands) return fixed;
  const [int, frac] = fixed.split(".");
  const withSep = int.replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  return frac ? `${withSep}.${frac}` : withSep;
}

/* ─────────────────────────────────────────────────────────────
   StatTile — hero metric card: animated figure, icon, accent
   glow, and a thin meter bar for a data-rich feel.
   ───────────────────────────────────────────────────────────── */
export function StatTile({
  label,
  value,
  decimals = 0,
  suffix = "",
  thousands = false,
  sublabel,
  icon,
  accent = "#2DD4BF",
  meter,
  delayMs = 0,
}: {
  label: string;
  value: number;
  decimals?: number;
  suffix?: string;
  thousands?: boolean;
  sublabel?: string;
  icon?: ReactNode;
  accent?: string;
  meter?: number; // 0..1 fills the bottom accent bar
  delayMs?: number;
}) {
  const animated = useCountUp(value, 1100, delayMs);
  return (
    <Card
      sx={{
        height: "100%",
        overflow: "hidden",
        "&:hover": {
          transform: "translateY(-4px)",
          borderColor: "rgba(45,212,191,0.32)",
          boxShadow: `0 1px 0 rgba(255,255,255,0.05) inset, 0 26px 56px -26px ${accent}80`,
        },
      }}
    >
      <CardContent sx={{ p: 3 }}>
        <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 1 }}>
          <Typography variant="overline" color="text.secondary" sx={{ display: "block" }}>
            {label}
          </Typography>
          {icon && (
            <Box
              sx={{
                width: 42,
                height: 42,
                borderRadius: 2.5,
                display: "grid",
                placeItems: "center",
                color: accent,
                backgroundColor: `${accent}1f`,
                border: `1px solid ${accent}33`,
                flexShrink: 0,
              }}
            >
              {icon}
            </Box>
          )}
        </Box>
        <Typography
          className="metric-figure"
          sx={{ mt: 1, fontSize: "2.2rem", fontWeight: 700, lineHeight: 1.1, color: "#F1F5F9" }}
        >
          {formatNumber(animated, decimals, thousands)}
          {suffix && (
            <Box component="span" sx={{ fontSize: "1.2rem", color: accent, ml: 0.25 }}>
              {suffix}
            </Box>
          )}
        </Typography>
        {sublabel && (
          <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, display: "block" }}>
            {sublabel}
          </Typography>
        )}
        <Box sx={{ mt: 2, height: 4, borderRadius: 999, backgroundColor: "rgba(255,255,255,0.07)", overflow: "hidden" }}>
          <Box
            sx={{
              height: "100%",
              width: `${Math.round((meter ?? 0.72) * 100)}%`,
              borderRadius: 999,
              background: `linear-gradient(90deg, ${accent}, ${accent}55)`,
              boxShadow: `0 0 12px ${accent}80`,
            }}
          />
        </Box>
      </CardContent>
    </Card>
  );
}

/* ─────────────────────────────────────────────────────────────
   MetricRow — label/value row with mono value; used in info cards.
   ───────────────────────────────────────────────────────────── */
export function MetricRow({
  label,
  value,
  mono = false,
  valueColor,
}: {
  label: string;
  value: ReactNode;
  mono?: boolean;
  valueColor?: string;
}) {
  return (
    <Box
      sx={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        gap: 2,
        py: 1.1,
        borderBottom: "1px solid rgba(255,255,255,0.06)",
        "&:last-of-type": { borderBottom: "none" },
      }}
    >
      <Typography variant="body2" color="text.secondary">
        {label}
      </Typography>
      <Typography
        variant="body2"
        component="div"
        className={mono ? "metric-figure" : undefined}
        sx={{ fontWeight: 600, textAlign: "right", color: valueColor ?? "#E6EDF5" }}
      >
        {value}
      </Typography>
    </Box>
  );
}
