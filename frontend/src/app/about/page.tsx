"use client";

import { useRef } from "react";
import { useGSAP } from "@gsap/react";
import gsap from "gsap";
import {
  Alert,
  AlertTitle,
  ArchitectureIcon,
  Box,
  Card,
  CardContent,
  Chip,
  Divider,
  GavelIcon,
  GitHubIcon,
  Grid,
  GroupIcon,
  SchoolIcon,
  ScienceIcon,
  StorageIcon,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Typography,
  WarningAmberIcon,
} from "@/lib/mui";
import { GradientText, MetricRow, SectionHeader } from "@/components/ui";

const ARCHITECTURE_STAGES = [
  {
    stage: "Stage 1",
    title: "Triage Classifier",
    desc: "DINOv2 ViT-B/14 (frozen) + linear head → Healthy · Non-DFU · DFU",
    color: "#2DD4BF",
  },
  {
    stage: "Stage 2",
    title: "Wound Segmenter",
    desc: "DINOv2 ViT-B/14 (frozen) + UPerNet decoder → pixel-wise wound mask",
    color: "#38BDF8",
  },
];

const CV_DATA = [
  { fold: "0", dice: "84.68%", iou: "77.70%" },
  { fold: "1", dice: "85.94%", iou: "79.30%" },
  { fold: "2", dice: "86.63%", iou: "79.71%" },
  { fold: "3", dice: "84.83%", iou: "78.01%" },
  { fold: "4", dice: "84.56%", iou: "77.69%" },
];

const ABLATION_DATA = [
  { variant: "U-Net++ (DFU-only)", dice: "85.13%", iou: "77.51%", note: "Best" },
  { variant: "U-Net++ (All classes)", dice: "82.35%", iou: "73.67%", note: "" },
  { variant: "FUSegNet (DFU+nonDFU)", dice: "81.75%", iou: "73.00%", note: "" },
  { variant: "U-Net++ v2 (DFU+nonDFU)", dice: "80.39%", iou: "70.72%", note: "" },
  { variant: "U-Net++ (DFU+nonDFU)", dice: "79.03%", iou: "69.03%", note: "Worst" },
];

const TECH_STACK = [
  { component: "Deep Learning", tool: "PyTorch 2.13" },
  { component: "Medical Imaging", tool: "MONAI 1.5" },
  { component: "Segmentation", tool: "SMP 0.5" },
  { component: "Augmentation", tool: "Albumentations 1.4" },
  { component: "API", tool: "FastAPI 0.139" },
  { component: "Inference", tool: "ONNX Runtime 1.21" },
  { component: "Compute", tool: "Northeastern HPC (H200/A100)" },
];

const SCOPE_NOTES = [
  "Shortcut-learning risk: an EfficientNet baseline reached 100% internal accuracy but only 21% external (0% DFU sensitivity). The deployed DINOv2 classifier (98.4%) is far more robust, but external classification still does not generalize — re-validate on your own image source before any use.",
  "Data leakage — found and fixed: an audit detected near-duplicate train↔test pairs; they were removed and the splits rebuilt. Re-audit confirms zero overlap across path, canonical-ID, content-hash, and perceptual-hash.",
  "Segmentation mean vs. median: the mixed-set mean Dice (0.65–0.72) is dragged down by empty-mask healthy/non-DFU images; the median (0.93) and DFU-only mean (0.89) reflect real-wound performance.",
  "Limited skin-tone diversity: fairness was validated primarily on one ITA group (Brown, n=285 for DFU). Broader-spectrum validation is still needed.",
  "Wound-area agreement was evaluated on only 3 images — statistically insufficient for any clinical claim.",
  "Research and education only. This is NOT a medical device and must not be used for diagnosis or treatment.",
];

export default function AboutPage() {
  const containerRef = useRef<HTMLDivElement>(null);

  useGSAP(
    () => {
      gsap.from(".about-hero", { y: 24, opacity: 0, duration: 0.7, ease: "power3.out" });
      gsap.from(".about-section", {
        y: 25,
        opacity: 0,
        duration: 0.5,
        stagger: 0.12,
        ease: "power3.out",
        delay: 0.15,
      });
    },
    { scope: containerRef }
  );

  return (
    <Box ref={containerRef}>
      {/* Header */}
      <Box className="about-hero" sx={{ mb: 3 }}>
        <Typography variant="overline" sx={{ color: "primary.light" }}>
          DiaFoot.AI · Documentation
        </Typography>
        <Typography variant="h3" sx={{ mt: 0.5, lineHeight: 1.05 }}>
          About <GradientText>DiaFoot.AI</GradientText>
        </Typography>
        <Typography variant="subtitle1" sx={{ mt: 1, maxWidth: 640 }}>
          Project details, methodology, results, and limitations — the full story behind the
          cascaded triage-and-segmentation pipeline.
        </Typography>
      </Box>

      {/* Disclaimer */}
      <Alert severity="error" icon={<GavelIcon />} className="about-section" sx={{ mb: 4 }}>
        <AlertTitle sx={{ fontWeight: 700 }}>Regulatory & Ethical Notice</AlertTitle>
        <Typography variant="body2">
          This is an academic project developed for educational purposes only (AAI6630,
          Northeastern University). This software is <strong>NOT</strong> a medical device, is{" "}
          <strong>NOT</strong> FDA-cleared, and is <strong>NOT</strong> intended for clinical use,
          diagnosis, treatment, or any medical decision-making. Any use for clinical purposes is
          strictly prohibited.
        </Typography>
      </Alert>

      <Grid container spacing={2.5}>
        {/* Project Overview */}
        <Grid size={{ xs: 12, lg: 6 }} className="about-section">
          <Card sx={{ height: "100%" }}>
            <CardContent sx={{ p: 3 }}>
              <SectionHeader icon={<SchoolIcon />} title="Project Overview" subtitle="Who built this, and why" />
              <Typography variant="body2" color="text.secondary" sx={{ mb: 1, lineHeight: 1.7 }}>
                DiaFoot.AI v2 is a production-grade, multi-task pipeline for diabetic foot ulcer
                (DFU) triage and wound-boundary segmentation. It is a ground-up rebuild of v1,
                which posted strong-looking Dice but had near-zero clinical specificity because it
                was trained only on ulcer images.
              </Typography>
              <Divider sx={{ my: 1.5 }} />
              <MetricRow label="Author" value="Ruthvik Bandari" />
              <MetricRow label="Course" value="AAI6630 Computer Vision" />
              <MetricRow label="University" value="Northeastern University" />
              <MetricRow label="Date" value="July 2026" />
              <MetricRow label="Version" mono value="2.1.0" />
            </CardContent>
          </Card>
        </Grid>

        {/* Architecture */}
        <Grid size={{ xs: 12, lg: 6 }} className="about-section">
          <Card sx={{ height: "100%" }}>
            <CardContent sx={{ p: 3 }}>
              <SectionHeader icon={<ArchitectureIcon />} title="Architecture" subtitle="Cascaded two-stage pipeline" />
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2, lineHeight: 1.7 }}>
                Cascaded two-stage pipeline, validated by a data-composition ablation:
              </Typography>
              <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
                {ARCHITECTURE_STAGES.map((s) => (
                  <Box
                    key={s.stage}
                    sx={{
                      p: 2,
                      borderRadius: 2.5,
                      backgroundColor: `${s.color}14`,
                      border: `1px solid ${s.color}33`,
                      borderLeft: `3px solid ${s.color}`,
                    }}
                  >
                    <Chip
                      label={s.stage}
                      size="small"
                      sx={{
                        mb: 1,
                        backgroundColor: `${s.color}26`,
                        color: s.color,
                        border: `1px solid ${s.color}55`,
                      }}
                    />
                    <Typography variant="body2" fontWeight={600} sx={{ color: "#E6EDF5" }}>
                      {s.title}
                    </Typography>
                    <Typography variant="caption" color="text.secondary" sx={{ display: "block", mt: 0.5 }}>
                      {s.desc}
                    </Typography>
                  </Box>
                ))}
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* 5-Fold Cross-Validation */}
        <Grid size={{ xs: 12, lg: 6 }} className="about-section">
          <Card sx={{ height: "100%" }}>
            <CardContent sx={{ p: 3 }}>
              <SectionHeader
                icon={<ScienceIcon />}
                title="5-Fold Cross-Validation"
                subtitle="DFU wound segmentation (U-Net++)"
              />
              <Box
                sx={{
                  textAlign: "center",
                  mb: 2.5,
                  p: 2.5,
                  borderRadius: 3,
                  background: "var(--grad-brand-soft)",
                  border: "1px solid rgba(45,212,191,0.22)",
                }}
              >
                <Typography className="metric-figure" sx={{ fontSize: "2.3rem", fontWeight: 700, lineHeight: 1.1 }}>
                  <GradientText>85.33 ± 0.91%</GradientText>
                </Typography>
                <Typography variant="caption" color="text.secondary" sx={{ display: "block", mt: 0.5 }}>
                  Dice score (mean ± std across folds)
                </Typography>
              </Box>
              <TableContainer>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Fold</TableCell>
                      <TableCell align="right">Dice</TableCell>
                      <TableCell align="right">IoU</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {CV_DATA.map((row) => (
                      <TableRow key={row.fold}>
                        <TableCell className="metric-figure">{row.fold}</TableCell>
                        <TableCell align="right" className="metric-figure">{row.dice}</TableCell>
                        <TableCell align="right" className="metric-figure">{row.iou}</TableCell>
                      </TableRow>
                    ))}
                    <TableRow sx={{ backgroundColor: "rgba(45,212,191,0.08)" }}>
                      <TableCell sx={{ fontWeight: 700 }}>Mean ± Std</TableCell>
                      <TableCell
                        align="right"
                        className="metric-figure"
                        sx={{ fontWeight: 700, color: "primary.light" }}
                      >
                        85.33 ± 0.91%
                      </TableCell>
                      <TableCell
                        align="right"
                        className="metric-figure"
                        sx={{ fontWeight: 700, color: "primary.light" }}
                      >
                        78.48 ± 0.95%
                      </TableCell>
                    </TableRow>
                  </TableBody>
                </Table>
              </TableContainer>
            </CardContent>
          </Card>
        </Grid>

        {/* Data Composition Ablation */}
        <Grid size={{ xs: 12, lg: 6 }} className="about-section">
          <Card sx={{ height: "100%" }}>
            <CardContent sx={{ p: 3 }}>
              <SectionHeader
                icon={<StorageIcon />}
                title="Data Composition Ablation"
                subtitle="Core finding: DFU-only training beats every mixed-data variant"
              />
              <TableContainer>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Model / Data</TableCell>
                      <TableCell align="right">Dice</TableCell>
                      <TableCell align="right">IoU</TableCell>
                      <TableCell>Note</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {ABLATION_DATA.map((row) => (
                      <TableRow
                        key={row.variant}
                        sx={
                          row.note === "Best"
                            ? { backgroundColor: "rgba(52,211,153,0.08)" }
                            : row.note === "Worst"
                              ? { backgroundColor: "rgba(251,113,133,0.08)" }
                              : {}
                        }
                      >
                        <TableCell>{row.variant}</TableCell>
                        <TableCell
                          align="right"
                          className="metric-figure"
                          sx={{ fontWeight: row.note === "Best" ? 700 : 400 }}
                        >
                          {row.dice}
                        </TableCell>
                        <TableCell align="right" className="metric-figure">
                          {row.iou}
                        </TableCell>
                        <TableCell>
                          {row.note && (
                            <Chip
                              label={row.note}
                              size="small"
                              color={row.note === "Best" ? "success" : "error"}
                              variant="outlined"
                            />
                          )}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </CardContent>
          </Card>
        </Grid>

        {/* Tech Stack */}
        <Grid size={{ xs: 12, lg: 6 }} className="about-section">
          <Card sx={{ height: "100%" }}>
            <CardContent sx={{ p: 3 }}>
              <SectionHeader icon={<StorageIcon />} title="Tech Stack" subtitle="What runs under the hood" />
              {TECH_STACK.map((item) => (
                <MetricRow key={item.component} label={item.component} mono value={item.tool} />
              ))}
              <Divider sx={{ my: 2 }} />
              <Box sx={{ display: "flex", gap: 1, flexWrap: "wrap" }}>
                <Chip
                  icon={<GitHubIcon />}
                  label="GitHub"
                  size="small"
                  component="a"
                  href="https://github.com/Ruthvik-Bandari/DiaFoot.AI"
                  target="_blank"
                  clickable
                />
                <Chip
                  label="HuggingFace"
                  size="small"
                  component="a"
                  href="https://huggingface.co/RuthvikBandari/DiaFoot.AI-v2"
                  target="_blank"
                  clickable
                />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Scope Notes */}
        <Grid size={{ xs: 12, lg: 6 }} className="about-section">
          <Card sx={{ height: "100%" }}>
            <CardContent sx={{ p: 3 }}>
              <SectionHeader
                icon={<WarningAmberIcon />}
                title="Scope Notes"
                subtitle="What we found, fixed, and still can't claim"
              />
              {SCOPE_NOTES.map((note, i) => (
                <Box
                  key={note.slice(0, 24)}
                  sx={{
                    p: 1.5,
                    mb: 1,
                    borderRadius: 2,
                    backgroundColor: i % 2 === 0 ? "rgba(251,191,36,0.06)" : "transparent",
                  }}
                >
                  <Typography variant="body2" color="text.secondary" sx={{ lineHeight: 1.6 }}>
                    <Typography component="span" fontWeight={700} sx={{ color: "warning.light" }}>
                      {i + 1}.{" "}
                    </Typography>
                    {note}
                  </Typography>
                </Box>
              ))}
            </CardContent>
          </Card>
        </Grid>

        {/* Dataset */}
        <Grid size={{ xs: 12 }} className="about-section">
          <Card>
            <CardContent sx={{ p: 3 }}>
              <SectionHeader icon={<GroupIcon />} title="Dataset" subtitle="Leakage-audited splits" />
              <Grid container spacing={{ xs: 0, sm: 5 }}>
                <Grid size={{ xs: 12, sm: 6 }}>
                  <MetricRow label="Total" mono value="8,105 images" />
                  <MetricRow label="DFU" mono value="2,119 (FUSeg + AZH)" />
                  <MetricRow label="Healthy" mono value="3,300" />
                </Grid>
                <Grid size={{ xs: 12, sm: 6 }}>
                  <MetricRow label="Non-DFU" mono value="2,686" />
                  <MetricRow label="Splits (train/val/test)" mono value="5,782 / 1,162 / 1,161" />
                  <MetricRow label="Basis" value="Leakage-audited" valueColor="#5EEAD4" />
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
}
