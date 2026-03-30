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

interface StatCardProps {
  title: string;
  value: string;
  subtitle: string;
  icon: React.ReactNode;
  color: string;
}

function StatCard({ title, value, subtitle, icon, color }: StatCardProps) {
  return (
    <Card sx={{ height: "100%" }}>
      <CardContent sx={{ p: 3 }}>
        <Box sx={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between" }}>
          <Box>
            <Typography variant="caption" color="text.secondary" fontWeight={500}>
              {title}
            </Typography>
            <Typography variant="h4" sx={{ mt: 0.5, color, fontWeight: 700 }}>
              {value}
            </Typography>
            <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, display: "block" }}>
              {subtitle}
            </Typography>
          </Box>
          <Box
            sx={{
              width: 48,
              height: 48,
              borderRadius: 2,
              backgroundColor: `${color}14`,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            <Box sx={{ color }}>{icon}</Box>
          </Box>
        </Box>
      </CardContent>
    </Card>
  );
}

const KEY_STATS: StatCardProps[] = [
  {
    title: "Classification Accuracy",
    value: "98.36%",
    subtitle: "DINOv2 ViT-B/14 (frozen backbone)",
    icon: <VerifiedIcon />,
    color: "#065A82",
  },
  {
    title: "DFU Segmentation Dice",
    value: "82.73%",
    subtitle: "DINOv2 + UPerNet (10 epochs, frozen)",
    icon: <SpeedIcon />,
    color: "#00A896",
  },
  {
    title: "DFU Sensitivity",
    value: "96.58%",
    subtitle: "calibrated defer at 99.72% accuracy",
    icon: <BalanceIcon />,
    color: "#27AE60",
  },
  {
    title: "Dataset Size",
    value: "8,105",
    subtitle: "DFU + Healthy + Non-DFU",
    icon: <DatasetIcon />,
    color: "#1C7293",
  },
];

export default function DashboardPage() {
  const containerRef = useRef<HTMLDivElement>(null);
  const {
    data: modelInfo,
    isLoading: modelLoading,
    isError: modelInfoError,
    error: modelInfoErrorValue,
  } = useModelInfo();
  const {
    data: health,
    isError: healthError,
    error: healthErrorValue,
  } = useHealth();
  const isDemoMode = getIsDemoMode();

  useGSAP(
    () => {
      gsap.from(".stat-card", {
        y: 30,
        opacity: 0,
        duration: 0.6,
        stagger: 0.1,
        ease: "power3.out",
      });
      gsap.from(".info-section", {
        y: 20,
        opacity: 0,
        duration: 0.5,
        delay: 0.5,
        ease: "power2.out",
      });
    },
    { scope: containerRef }
  );

  return (
    <Box ref={containerRef}>
      {/* Header */}
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" color="primary.dark">
          Dashboard
        </Typography>
        <Typography variant="subtitle1" sx={{ mt: 0.5 }}>
          DiaFoot.AI v2 — Diabetic Foot Ulcer Detection & Segmentation
        </Typography>
      </Box>

      {isDemoMode && (
        <Alert severity="warning" sx={{ mb: 3 }}>
          <AlertTitle>Demo Mode Enabled</AlertTitle>
          Backend responses are mocked. Disable `NEXT_PUBLIC_ENABLE_DEMO_MODE` for real inference.
        </Alert>
      )}

      {(healthError || modelInfoError) && (
        <Alert severity="error" sx={{ mb: 3 }}>
          <AlertTitle>Backend Connection Issue</AlertTitle>
          {(healthErrorValue as Error | undefined)?.message ||
            (modelInfoErrorValue as Error | undefined)?.message ||
            "Unable to fetch backend status. Check NEXT_PUBLIC_API_URL and backend availability."}
        </Alert>
      )}

      {/* CTA */}
      <Card
        sx={{
          mb: 4,
          background: "linear-gradient(135deg, #065A82 0%, #1C7293 50%, #00A896 100%)",
          color: "#fff",
          border: "none",
        }}
      >
        <CardContent sx={{ p: 4, display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <Box>
            <Typography variant="h5" fontWeight={700} color="inherit">
              Ready to Analyze
            </Typography>
            <Typography variant="body2" sx={{ mt: 1, opacity: 0.85, maxWidth: 480 }}>
              Upload a foot image to get instant classification, wound segmentation, and clinical metrics.
              The model uses a cascaded pipeline: triage classifier → wound segmenter.
            </Typography>
          </Box>
          <Button
            variant="contained"
            size="large"
            component={Link}
            href="/predict"
            startIcon={<CameraAltIcon />}
            sx={{
              backgroundColor: "#fff",
              color: "#065A82",
              fontWeight: 700,
              px: 4,
              py: 1.5,
              "&:hover": { backgroundColor: "rgba(255,255,255,0.9)" },
            }}
          >
            Start Analysis
          </Button>
        </CardContent>
      </Card>

      {/* Key Metrics */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        {KEY_STATS.map((stat) => (
          <Grid size={{ xs: 12, sm: 6, lg: 3 }} key={stat.title}>
            <Box className="stat-card">
              <StatCard {...stat} />
            </Box>
          </Grid>
        ))}
      </Grid>

      {/* Model Info */}
      <Box className="info-section">
        <Grid container spacing={3}>
          <Grid size={{ xs: 12, lg: 6 }}>
            <Card>
              <CardContent sx={{ p: 3 }}>
                <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 2 }}>
                  <MemoryIcon color="primary" />
                  <Typography variant="h6" color="primary.dark">
                    Model Architecture
                  </Typography>
                </Box>
                {modelLoading ? (
                  <Box>
                    <Skeleton variant="text" width="80%" height={28} />
                    <Skeleton variant="text" width="60%" height={28} />
                    <Skeleton variant="text" width="70%" height={28} />
                  </Box>
                ) : (
                  <Box sx={{ display: "flex", flexDirection: "column", gap: 1.5 }}>
                    <Box sx={{ display: "flex", justifyContent: "space-between" }}>
                      <Typography variant="body2" color="text.secondary">Classifier</Typography>
                      <Typography variant="body2" fontWeight={600}>{modelInfo?.classifier ?? "—"}</Typography>
                    </Box>
                    <Box sx={{ display: "flex", justifyContent: "space-between" }}>
                      <Typography variant="body2" color="text.secondary">Segmenter</Typography>
                      <Typography variant="body2" fontWeight={600}>{modelInfo?.segmenter ?? "—"}</Typography>
                    </Box>
                    <Box sx={{ display: "flex", justifyContent: "space-between" }}>
                      <Typography variant="body2" color="text.secondary">Input Size</Typography>
                      <Typography variant="body2" fontWeight={600}>{modelInfo ? `${modelInfo.input_size[0]}×${modelInfo.input_size[1]}` : "—"}</Typography>
                    </Box>
                    <Box sx={{ display: "flex", justifyContent: "space-between" }}>
                      <Typography variant="body2" color="text.secondary">Status</Typography>
                      <Chip
                        label={health?.model_loaded ? "Loaded" : "Not Loaded"}
                        size="small"
                        color={health?.model_loaded ? "success" : "error"}
                      />
                    </Box>
                  </Box>
                )}
              </CardContent>
            </Card>
          </Grid>

          <Grid size={{ xs: 12, lg: 6 }}>
            <Card>
              <CardContent sx={{ p: 3 }}>
                <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 2 }}>
                  <ScienceIcon color="primary" />
                  <Typography variant="h6" color="primary.dark">
                    Key Experiments
                  </Typography>
                </Box>
                <Box sx={{ display: "flex", flexDirection: "column", gap: 1.5 }}>
                  <Box sx={{ display: "flex", justifyContent: "space-between" }}>
                    <Typography variant="body2" color="text.secondary">Transfer Learning</Typography>
                    <Typography variant="body2" fontWeight={600}>DINOv2 ViT-B/14 (Meta, self-supervised)</Typography>
                  </Box>
                  <Box sx={{ display: "flex", justifyContent: "space-between" }}>
                    <Typography variant="body2" color="text.secondary">Data Ablation</Typography>
                    <Typography variant="body2" fontWeight={600}>DFU-only wins (87.4% vs 68.7%)</Typography>
                  </Box>
                  <Box sx={{ display: "flex", justifyContent: "space-between" }}>
                    <Typography variant="body2" color="text.secondary">Calibration</Typography>
                    <Typography variant="body2" fontWeight={600}>ECE 0.0075 after temperature scaling</Typography>
                  </Box>
                  <Box sx={{ display: "flex", justifyContent: "space-between" }}>
                    <Typography variant="body2" color="text.secondary">Fairness Gap</Typography>
                    <Typography variant="body2" fontWeight={600}>0.00% on DFU images</Typography>
                  </Box>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </Box>
    </Box>
  );
}
