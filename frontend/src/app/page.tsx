"use client";

import { useRef } from "react";
import { useGSAP } from "@gsap/react";
import gsap from "gsap";
import {
  Alert,
  AlertTitle,
  BalanceIcon,
  Box,
  Button,
  CameraAltIcon,
  Card,
  CardContent,
  Chip,
  DatasetIcon,
  Grid,
  MemoryIcon,
  ScienceIcon,
  Skeleton,
  SpeedIcon,
  Typography,
  VerifiedIcon,
} from "@/lib/mui";
import Link from "next/link";
import { getIsDemoMode, useModelInfo, useHealth } from "@/lib/api";
import { GradientText, MetricRow, SectionHeader, StatTile } from "@/components/ui";

export default function DashboardPage() {
  const containerRef = useRef<HTMLDivElement>(null);
  const {
    data: modelInfo,
    isLoading: modelLoading,
    isError: modelInfoError,
    error: modelInfoErrorValue,
  } = useModelInfo();
  const { data: health, isError: healthError, error: healthErrorValue } = useHealth();
  const isDemoMode = getIsDemoMode();
  const isLive = !isDemoMode && !!health && !healthError;

  useGSAP(
    () => {
      gsap.from(".hero-reveal", { y: 24, opacity: 0, duration: 0.7, ease: "power3.out" });
      gsap.from(".stat-card", { y: 28, opacity: 0, duration: 0.6, stagger: 0.08, ease: "power3.out", delay: 0.15 });
      gsap.from(".info-reveal", { y: 22, opacity: 0, duration: 0.55, stagger: 0.1, ease: "power2.out", delay: 0.45 });
    },
    { scope: containerRef }
  );

  return (
    <Box ref={containerRef}>
      {/* Header */}
      <Box className="hero-reveal" sx={{ mb: 3 }}>
        <Typography variant="overline" sx={{ color: "primary.light" }}>
          DiaFoot.AI · v2.1.0
        </Typography>
        <Typography variant="h3" sx={{ mt: 0.5, lineHeight: 1.05 }}>
          Diabetic Foot Ulcer <GradientText>Intelligence</GradientText>
        </Typography>
        <Typography variant="subtitle1" sx={{ mt: 1, maxWidth: 620 }}>
          A cascaded DINOv2 pipeline for foot-image triage and wound-boundary segmentation —
          evaluated on leakage-audited splits with honest, reproducible metrics.
        </Typography>
      </Box>

      {isDemoMode && (
        <Alert severity="warning" sx={{ mb: 3 }} className="hero-reveal">
          <AlertTitle>Demo mode</AlertTitle>
          Backend responses are simulated. Disable <code>NEXT_PUBLIC_ENABLE_DEMO_MODE</code> for real inference.
        </Alert>
      )}
      {(healthError || modelInfoError) && !isDemoMode && (
        <Alert severity="error" sx={{ mb: 3 }} className="hero-reveal">
          <AlertTitle>Backend connection issue</AlertTitle>
          {(healthErrorValue as Error | undefined)?.message ||
            (modelInfoErrorValue as Error | undefined)?.message ||
            "Unable to reach the backend. Check NEXT_PUBLIC_API_URL and that the API is running."}
        </Alert>
      )}

      {/* Hero CTA band */}
      <Card
        className="hero-reveal"
        sx={{
          mb: 4,
          overflow: "hidden",
          border: "1px solid rgba(45,212,191,0.22)",
          background:
            "radial-gradient(120% 140% at 0% 0%, rgba(45,212,191,0.16), transparent 45%), radial-gradient(120% 140% at 100% 100%, rgba(99,102,241,0.18), transparent 45%), rgba(14,19,31,0.7)",
        }}
      >
        <CardContent
          sx={{
            p: { xs: 3, md: 4 },
            display: "flex",
            flexDirection: { xs: "column", md: "row" },
            alignItems: { xs: "flex-start", md: "center" },
            justifyContent: "space-between",
            gap: 3,
          }}
        >
          <Box sx={{ minWidth: 0 }}>
            <Box sx={{ display: "flex", gap: 1, mb: 1.5, flexWrap: "wrap" }}>
              <Chip
                size="small"
                label={isLive ? "Live backend" : isDemoMode ? "Demo mode" : "Backend offline"}
                color={isLive ? "success" : "warning"}
                variant="outlined"
              />
              <Chip
                size="small"
                label={health?.model_loaded ? "Model loaded" : "Model not loaded"}
                color={health?.model_loaded ? "success" : "default"}
                variant="outlined"
              />
            </Box>
            <Typography variant="h5">Ready to analyze</Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mt: 0.75, maxWidth: 520 }}>
              Upload a foot image for instant triage classification, wound segmentation, and calibrated
              clinical-style metrics through the cascaded classifier → segmenter pipeline.
            </Typography>
          </Box>
          <Button
            variant="contained"
            size="large"
            component={Link}
            href="/predict"
            startIcon={<CameraAltIcon />}
            sx={{ flexShrink: 0, px: 3.5, py: 1.5 }}
          >
            Start analysis
          </Button>
        </CardContent>
      </Card>

      {/* Key metrics */}
      <Grid container spacing={2.5} sx={{ mb: 4 }}>
        <Grid size={{ xs: 12, sm: 6, lg: 3 }}>
          <Box className="stat-card">
            <StatTile
              label="Classification Accuracy"
              value={98.36}
              decimals={2}
              suffix="%"
              sublabel="DINOv2 ViT-B/14 · 3-class triage"
              icon={<VerifiedIcon />}
              accent="#2DD4BF"
              meter={0.9836}
              delayMs={150}
            />
          </Box>
        </Grid>
        <Grid size={{ xs: 12, sm: 6, lg: 3 }}>
          <Box className="stat-card">
            <StatTile
              label="DFU Segmentation Dice"
              value={89.12}
              decimals={2}
              suffix="%"
              sublabel="DFU-only slice (n=263) · +UPerNet"
              icon={<SpeedIcon />}
              accent="#38BDF8"
              meter={0.8912}
              delayMs={230}
            />
          </Box>
        </Grid>
        <Grid size={{ xs: 12, sm: 6, lg: 3 }}>
          <Box className="stat-card">
            <StatTile
              label="DFU Sensitivity"
              value={96.58}
              decimals={2}
              suffix="%"
              sublabel="defer @0.95 → 99.7% accuracy on kept"
              icon={<BalanceIcon />}
              accent="#34D399"
              meter={0.9658}
              delayMs={310}
            />
          </Box>
        </Grid>
        <Grid size={{ xs: 12, sm: 6, lg: 3 }}>
          <Box className="stat-card">
            <StatTile
              label="Dataset Size"
              value={8105}
              thousands
              sublabel="train+val+test · leakage-audited"
              icon={<DatasetIcon />}
              accent="#818CF8"
              meter={0.86}
              delayMs={390}
            />
          </Box>
        </Grid>
      </Grid>

      {/* Info cards */}
      <Grid container spacing={2.5}>
        <Grid size={{ xs: 12, lg: 4 }}>
          <Box className="info-reveal" sx={{ height: "100%" }}>
            <Card sx={{ height: "100%" }}>
              <CardContent sx={{ p: 3 }}>
                <SectionHeader icon={<MemoryIcon />} title="Model Architecture" subtitle="Live from /model/info" />
                {modelLoading ? (
                  <Box>
                    <Skeleton variant="text" height={30} />
                    <Skeleton variant="text" width="70%" height={30} />
                    <Skeleton variant="text" width="55%" height={30} />
                  </Box>
                ) : (
                  <>
                    <MetricRow label="Classifier" value={modelInfo?.classifier ?? "—"} />
                    <MetricRow label="Segmenter" value={modelInfo?.segmenter ?? "—"} />
                    <MetricRow
                      label="Input size"
                      mono
                      value={modelInfo ? `${modelInfo.input_size[0]}×${modelInfo.input_size[1]}` : "—"}
                    />
                    <MetricRow
                      label="Status"
                      value={
                        <Chip
                          size="small"
                          label={health?.model_loaded ? "Loaded" : "Not loaded"}
                          color={health?.model_loaded ? "success" : "error"}
                          variant="outlined"
                        />
                      }
                    />
                  </>
                )}
              </CardContent>
            </Card>
          </Box>
        </Grid>

        <Grid size={{ xs: 12, lg: 4 }}>
          <Box className="info-reveal" sx={{ height: "100%" }}>
            <Card sx={{ height: "100%" }}>
              <CardContent sx={{ p: 3 }}>
                <SectionHeader icon={<ScienceIcon />} title="Key Experiments" subtitle="What the study established" />
                <MetricRow label="Transfer learning" value="DINOv2 ViT-B/14 (frozen)" />
                <MetricRow label="Data composition" value="Composition > size" valueColor="#5EEAD4" />
                <MetricRow label="Composition Δ Dice" mono value="87.9% vs 79.5%" />
                <MetricRow label="Calibration (ECE)" mono value="0.039 → 0.007" />
                <MetricRow label="Fairness gap (DFU)" mono value="0.00" valueColor="#6EE7B7" />
              </CardContent>
            </Card>
          </Box>
        </Grid>

        <Grid size={{ xs: 12, lg: 4 }}>
          <Box className="info-reveal" sx={{ height: "100%" }}>
            <Card sx={{ height: "100%" }}>
              <CardContent sx={{ p: 3 }}>
                <SectionHeader icon={<SpeedIcon />} title="Segmentation — test set" subtitle="Dice by slice" />
                <MetricRow label="DFU wounds only (n=263)" mono value="0.891" valueColor="#5EEAD4" />
                <MetricRow label="Full mixed (n=1,161) mean" mono value="0.72" />
                <MetricRow label="Full mixed — median" mono value="0.93" />
                <MetricRow label="5-fold CV (DFU)" mono value="0.853 ± 0.009" />
                <Typography variant="caption" color="text.secondary" sx={{ mt: 1.5, display: "block", lineHeight: 1.6 }}>
                  Mixed-set mean is pulled down by empty-mask healthy/non-DFU images; on real wounds the model is strong.
                </Typography>
              </CardContent>
            </Card>
          </Box>
        </Grid>
      </Grid>
    </Box>
  );
}
