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

interface InfoRowProps {
  label: string;
  value: string;
}

function InfoRow({ label, value }: InfoRowProps) {
  return (
    <Box sx={{ display: "flex", justifyContent: "space-between", py: 1 }}>
      <Typography variant="body2" color="text.secondary">{label}</Typography>
      <Typography variant="body2" fontWeight={600}>{value}</Typography>
    </Box>
  );
}

const ABLATION_DATA = [
  { variant: "U-Net++ (DFU-only)", dice: "85.13%", iou: "77.51%", note: "Best" },
  { variant: "U-Net++ (All classes)", dice: "82.35%", iou: "73.67%", note: "" },
  { variant: "FUSegNet (DFU+nonDFU)", dice: "81.75%", iou: "73.00%", note: "" },
  { variant: "U-Net++ v2 (DFU+nonDFU)", dice: "80.39%", iou: "70.72%", note: "" },
  { variant: "U-Net++ (DFU+nonDFU)", dice: "79.03%", iou: "69.03%", note: "Worst" },
];

const CV_DATA = [
  { fold: "0", dice: "84.68%", iou: "77.70%" },
  { fold: "1", dice: "85.94%", iou: "79.30%" },
  { fold: "2", dice: "86.63%", iou: "79.71%" },
  { fold: "3", dice: "84.83%", iou: "78.01%" },
  { fold: "4", dice: "84.56%", iou: "77.69%" },
];

const LIMITATIONS = [
  "Classifier learns dataset shortcuts: EfficientNet achieves 100% internal but only 21% external accuracy (0% DFU sensitivity). DINOv2 (98.36%) is less extreme but the same risk applies.",
  "Data leakage detected: 20,774 near-duplicate healthy foot pairs across train-val splits. Content overlap: 87 train-val, 9 train-test pairs.",
  "Limited skin tone diversity: 929/1,057 test images labeled Unknown ITA. Fairness validated on single group only (Brown, n=285).",
  "Wound area agreement evaluated on only 3 images. Statistically insufficient for clinical claims.",
  "The software is intended for research and education, not for clinical diagnosis or treatment.",
];

const TECH_STACK = [
  { component: "Deep Learning", tool: "PyTorch 2.10.0" },
  { component: "Medical Imaging", tool: "MONAI 1.5.2" },
  { component: "Segmentation", tool: "SMP 0.5.0" },
  { component: "Augmentation", tool: "Albumentations 1.4" },
  { component: "API", tool: "FastAPI 0.133.0" },
  { component: "Inference", tool: "ONNX Runtime 1.21" },
  { component: "Compute", tool: "Northeastern HPC (H200/A100)" },
];

export default function AboutPage() {
  const containerRef = useRef<HTMLDivElement>(null);

  useGSAP(
    () => {
      gsap.from(".about-section", {
        y: 25,
        opacity: 0,
        duration: 0.5,
        stagger: 0.12,
        ease: "power3.out",
      });
    },
    { scope: containerRef }
  );

  return (
    <Box ref={containerRef}>
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" color="primary.dark">About DiaFoot.AI</Typography>
        <Typography variant="subtitle1" sx={{ mt: 0.5 }}>
          Project details, methodology, results, and limitations
        </Typography>
      </Box>

      {/* Disclaimer */}
      <Alert
        severity="error"
        icon={<GavelIcon />}
        className="about-section"
        sx={{ mb: 3, borderRadius: 3 }}
      >
        <AlertTitle sx={{ fontWeight: 700 }}>Regulatory & Ethical Notice</AlertTitle>
        <Typography variant="body2">
          This is an academic project developed for educational purposes only (AAI6630, Northeastern University).
          This software is <strong>NOT</strong> a medical device, is <strong>NOT</strong> FDA-cleared, and is{" "}
          <strong>NOT</strong> intended for clinical use, diagnosis, treatment, or any medical decision-making.
          Any use for clinical purposes is strictly prohibited.
        </Typography>
      </Alert>

      <Grid container spacing={3}>
        {/* Project Overview */}
        <Grid size={{ xs: 12, lg: 6 }} className="about-section">
          <Card sx={{ height: "100%" }}>
            <CardContent sx={{ p: 3 }}>
              <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 2 }}>
                <SchoolIcon color="primary" />
                <Typography variant="h6" color="primary.dark">Project Overview</Typography>
              </Box>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2, lineHeight: 1.7 }}>
                DiaFoot.AI v2 is a production-grade multi-task pipeline for automated diabetic foot ulcer
                (DFU) detection and wound boundary segmentation. It was built as a complete ground-up rebuild
                of v1, which achieved seemingly strong metrics (91.73% Dice) but had zero clinical specificity
                because it was trained exclusively on ulcer images.
              </Typography>
              <Divider sx={{ my: 2 }} />
              <InfoRow label="Author" value="Ruthvik Bandari" />
              <InfoRow label="Course" value="AAI6630 Computer Vision" />
              <InfoRow label="University" value="Northeastern University" />
              <InfoRow label="Date" value="April 2026" />
              <InfoRow label="Version" value="2.0.0" />
            </CardContent>
          </Card>
        </Grid>

        {/* Architecture */}
        <Grid size={{ xs: 12, lg: 6 }} className="about-section">
          <Card sx={{ height: "100%" }}>
            <CardContent sx={{ p: 3 }}>
              <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 2 }}>
                <ArchitectureIcon color="primary" />
                <Typography variant="h6" color="primary.dark">Architecture</Typography>
              </Box>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2, lineHeight: 1.7 }}>
                Cascaded two-stage pipeline validated by data composition ablation:
              </Typography>
              <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
                {[
                  { stage: "Stage 1", title: "Triage Classifier", desc: "DINOv2 ViT-B/14 (frozen) + linear head → Healthy | Non-DFU | DFU", color: "#1C7293" },
                  { stage: "Stage 2", title: "Wound Segmenter", desc: "DINOv2 ViT-B/14 (frozen) + UPerNet decoder → pixel-wise wound mask", color: "#065A82" },
                ].map((s) => (
                  <Box
                    key={s.stage}
                    sx={{
                      p: 2,
                      borderRadius: 2,
                      backgroundColor: `${s.color}08`,
                      borderLeft: `4px solid ${s.color}`,
                    }}
                  >
                    <Chip label={s.stage} size="small" sx={{ mb: 0.5, backgroundColor: s.color, color: "#fff" }} />
                    <Typography variant="body2" fontWeight={600}>{s.title}</Typography>
                    <Typography variant="caption" color="text.secondary">{s.desc}</Typography>
                  </Box>
                ))}
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Cross-Validation Results */}
        <Grid size={{ xs: 12, lg: 6 }} className="about-section">
          <Card sx={{ height: "100%" }}>
            <CardContent sx={{ p: 3 }}>
              <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 2 }}>
                <ScienceIcon color="primary" />
                <Typography variant="h6" color="primary.dark">5-Fold Cross-Validation</Typography>
              </Box>
              <Box sx={{ textAlign: "center", mb: 2, p: 2, borderRadius: 2, backgroundColor: "rgba(6,90,130,0.04)" }}>
                <Typography variant="h4" fontWeight={700} color="primary.main">
                  85.33 ± 0.91%
                </Typography>
                <Typography variant="caption" color="text.secondary">Dice Score (mean ± std)</Typography>
              </Box>
              <TableContainer>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell sx={{ fontWeight: 700 }}>Fold</TableCell>
                      <TableCell align="right" sx={{ fontWeight: 700 }}>Dice</TableCell>
                      <TableCell align="right" sx={{ fontWeight: 700 }}>IoU</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {CV_DATA.map((row) => (
                      <TableRow key={row.fold}>
                        <TableCell>{row.fold}</TableCell>
                        <TableCell align="right">{row.dice}</TableCell>
                        <TableCell align="right">{row.iou}</TableCell>
                      </TableRow>
                    ))}
                    <TableRow sx={{ backgroundColor: "rgba(6,90,130,0.04)" }}>
                      <TableCell sx={{ fontWeight: 700 }}>Mean ± Std</TableCell>
                      <TableCell align="right" sx={{ fontWeight: 700 }}>85.33 ± 0.91%</TableCell>
                      <TableCell align="right" sx={{ fontWeight: 700 }}>78.48 ± 0.95%</TableCell>
                    </TableRow>
                  </TableBody>
                </Table>
              </TableContainer>
            </CardContent>
          </Card>
        </Grid>

        {/* Data Ablation */}
        <Grid size={{ xs: 12, lg: 6 }} className="about-section">
          <Card sx={{ height: "100%" }}>
            <CardContent sx={{ p: 3 }}>
              <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 2 }}>
                <StorageIcon color="primary" />
                <Typography variant="h6" color="primary.dark">Data Composition Ablation</Typography>
              </Box>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                Core finding: DFU-only training outperforms all mixed-data variants.
              </Typography>
              <TableContainer>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell sx={{ fontWeight: 700 }}>Model / Data</TableCell>
                      <TableCell align="right" sx={{ fontWeight: 700 }}>Dice</TableCell>
                      <TableCell align="right" sx={{ fontWeight: 700 }}>IoU</TableCell>
                      <TableCell sx={{ fontWeight: 700 }}>Note</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {ABLATION_DATA.map((row) => (
                      <TableRow
                        key={row.variant}
                        sx={row.note === "Best" ? { backgroundColor: "rgba(39,174,96,0.06)" } : {}}
                      >
                        <TableCell>{row.variant}</TableCell>
                        <TableCell align="right" sx={{ fontWeight: row.note === "Best" ? 700 : 400 }}>
                          {row.dice}
                        </TableCell>
                        <TableCell align="right">{row.iou}</TableCell>
                        <TableCell>
                          {row.note && (
                            <Chip
                              label={row.note}
                              size="small"
                              color={row.note === "Best" ? "success" : row.note === "Worst" ? "error" : "default"}
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
              <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 2 }}>
                <StorageIcon color="primary" />
                <Typography variant="h6" color="primary.dark">Tech Stack</Typography>
              </Box>
              {TECH_STACK.map((item) => (
                <Box key={item.component}>
                  <InfoRow label={item.component} value={item.tool} />
                </Box>
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

        {/* Honest Limitations */}
        <Grid size={{ xs: 12, lg: 6 }} className="about-section">
          <Card sx={{ height: "100%" }}>
            <CardContent sx={{ p: 3 }}>
              <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 2 }}>
                <WarningAmberIcon sx={{ color: "#F39C12" }} />
                <Typography variant="h6" color="primary.dark">Scope Notes</Typography>
              </Box>
              {LIMITATIONS.map((lim, i) => (
                <Box
                  key={i}
                  sx={{
                    p: 1.5,
                    mb: 1,
                    borderRadius: 1.5,
                    backgroundColor: i % 2 === 0 ? "rgba(243,156,18,0.04)" : "transparent",
                  }}
                >
                  <Typography variant="body2" color="text.secondary" sx={{ lineHeight: 1.6 }}>
                    <Typography component="span" fontWeight={700} color="warning.dark">
                      {i + 1}.{" "}
                    </Typography>
                    {lim}
                  </Typography>
                </Box>
              ))}
            </CardContent>
          </Card>
        </Grid>

        {/* Dataset + Peer Feedback */}
        <Grid size={{ xs: 12 }} className="about-section">
          <Card>
            <CardContent sx={{ p: 3 }}>
              <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 2 }}>
                <GroupIcon color="primary" />
                <Typography variant="h6" color="primary.dark">Dataset & Peer Feedback</Typography>
              </Box>
              <Grid container spacing={3}>
                <Grid size={{ xs: 12, lg: 4 }}>
                  <Typography variant="subtitle2" color="primary.main" gutterBottom>Dataset</Typography>
                  <InfoRow label="Processed" value="8,105 images" />
                  <InfoRow label="In splits" value="6,996 images" />
                  <InfoRow label="DFU" value="1,010 (FUSeg + AZH)" />
                  <InfoRow label="Healthy" value="3,300" />
                  <InfoRow label="Non-DFU" value="2,686" />
                  <InfoRow label="Splits" value="4,894 / 1,045 / 1,057" />
                </Grid>
                <Grid size={{ xs: 12, lg: 8 }}>
                  <Typography variant="subtitle2" color="primary.main" gutterBottom>
                    Peer Feedback → Implementation
                  </Typography>
                  {[
                    { from: "Sudeep K.S.", feedback: "Handle skin tone diversity → ITA-stratified fairness audit" },
                    { from: "Shivam Dubey", feedback: "Add attention mechanisms → scSE in decoder" },
                    { from: "Yash Jain", feedback: "Benchmark stronger segmentation baselines → integrated MedSAM2 and nnU-Net with matched-split evaluation" },
                    { from: "Yucheng Yan", feedback: "Prioritize ablation → Data composition as core experiment" },
                    { from: "Om Patel", feedback: "Implement TTA → 16-augmentation TTA (+3.88% Dice overall)" },
                    { from: "Ching-Yi Mao", feedback: "Address algorithmic bias → ITA audit + honest disclosure" },
                  ].map((fb) => (
                    <Box key={fb.from} sx={{ display: "flex", gap: 1, mb: 0.8 }}>
                      <Chip label={fb.from} size="small" sx={{ minWidth: 100 }} />
                      <Typography variant="caption" color="text.secondary" sx={{ lineHeight: 1.8 }}>
                        {fb.feedback}
                      </Typography>
                    </Box>
                  ))}
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
}
