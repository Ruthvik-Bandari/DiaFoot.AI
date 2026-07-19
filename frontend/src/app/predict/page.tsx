"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useGSAP } from "@gsap/react";
import gsap from "gsap";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import {
  Alert,
  AlertTitle,
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  Grid,
  LinearProgress,
  Skeleton,
  Typography,
} from "@/lib/mui";
import CameraAltIcon from "@mui/icons-material/CameraAlt";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import CloudUploadIcon from "@mui/icons-material/CloudUpload";
import CropFreeIcon from "@mui/icons-material/CropFree";
import ErrorIcon from "@mui/icons-material/Error";
import ImageSearchIcon from "@mui/icons-material/ImageSearch";
import InfoIcon from "@mui/icons-material/Info";
import ReplayIcon from "@mui/icons-material/Replay";
import StraightenIcon from "@mui/icons-material/Straighten";
import TimerIcon from "@mui/icons-material/Timer";
import WarningIcon from "@mui/icons-material/Warning";
import { usePrediction, getIsDemoMode } from "@/lib/api";
import { uploadFormSchema, type UploadFormData, type PredictionResponse } from "@/lib/api/schemas";
import { GradientText, MetricRow, SectionHeader, StatTile } from "@/components/ui";

/* ─────────────────────────────────────────────────────────────
   Class → semantic color/icon mapping, shared by the badge and
   the per-class confidence bars so everything stays in sync.
   ───────────────────────────────────────────────────────────── */
const CLASS_COLORS = {
  success: "#34D399",
  warning: "#FBBF24",
  error: "#FB7185",
  info: "#818CF8",
} as const;

type SemanticColor = keyof typeof CLASS_COLORS;

const CLASS_META: Record<string, { color: SemanticColor; icon: React.ReactElement }> = {
  Healthy: { color: "success", icon: <CheckCircleIcon sx={{ fontSize: 16 }} /> },
  "Non-DFU": { color: "info", icon: <WarningIcon sx={{ fontSize: 16 }} /> },
  DFU: { color: "error", icon: <ErrorIcon sx={{ fontSize: 16 }} /> },
  "Manual Review Required": { color: "warning", icon: <WarningIcon sx={{ fontSize: 16 }} /> },
};

function ClassificationBadge({ classification }: { classification: string }) {
  const c = CLASS_META[classification];

  return (
    <Chip
      label={classification}
      color={c?.color ?? "info"}
      icon={c?.icon}
      sx={{ fontSize: "0.9rem", fontWeight: 700, py: 2.5, px: 1 }}
    />
  );
}

function ConfidenceBar({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <Box sx={{ mb: 1.5 }}>
      <Box sx={{ display: "flex", justifyContent: "space-between", mb: 0.5 }}>
        <Typography variant="caption" fontWeight={500}>{label}</Typography>
        <Typography variant="caption" fontWeight={700} sx={{ color }}>
          {(value * 100).toFixed(1)}%
        </Typography>
      </Box>
      <LinearProgress
        variant="determinate"
        value={value * 100}
        sx={{
          height: 8,
          borderRadius: 999,
          backgroundColor: "rgba(255,255,255,0.08)",
          "& .MuiLinearProgress-bar": { backgroundColor: color, borderRadius: 999 },
        }}
      />
    </Box>
  );
}

/**
 * Compact stat tile for a single wound metric. Delegates rendering to the
 * shared `StatTile` primitive (animated `metric-figure`, icon accent, meter)
 * so wound metrics visually match the rest of the design system.
 */
function MetricCard({
  icon,
  label,
  value,
  decimals = 0,
  unit,
  accent = "#2DD4BF",
  meter,
}: {
  icon: React.ReactNode;
  label: string;
  value: number;
  decimals?: number;
  unit?: string;
  accent?: string;
  meter?: number;
}) {
  return (
    <StatTile
      label={label}
      value={value}
      decimals={decimals}
      suffix={unit ?? ""}
      icon={icon}
      accent={accent}
      meter={meter}
    />
  );
}

function ResultsSkeleton() {
  return (
    <Box>
      <Skeleton variant="rounded" height={72} sx={{ mb: 2.5, borderRadius: 3 }} />
      <Skeleton variant="rounded" height={190} sx={{ mb: 2.5, borderRadius: 3 }} />
      <Box sx={{ display: "flex", gap: 2, mb: 2.5 }}>
        <Skeleton variant="rounded" height={140} sx={{ flex: 1, borderRadius: 3 }} />
        <Skeleton variant="rounded" height={140} sx={{ flex: 1, borderRadius: 3 }} />
      </Box>
      <Skeleton variant="rounded" height={320} sx={{ borderRadius: 3 }} />
    </Box>
  );
}

function DeferBanner({ result }: { result: PredictionResponse }) {
  if (!result.defer_to_clinician) return null;
  if (result.classification_confidence >= 0.80) return null;

  const message =
    result.defer_reason === "low_confidence" ||
    result.defer_reason === "low_classification_confidence" ||
    result.defer_reason === "below_confidence_threshold"
      ? "The model is not confident enough in its prediction. Please consult a healthcare professional."
      : result.defer_reason === "low_image_quality"
        ? `Image quality issues detected: ${result.quality_flags.join(", ")}. Please retake the image.`
        : result.defer_reason === "segmentation_classifier_disagreement"
          ? "Classifier and segmenter disagree on this case. Please review manually."
          : "This case requires clinical review.";

  return (
    <Alert severity="warning" icon={<WarningIcon />} sx={{ mb: 2.5 }} className="result-card">
      <AlertTitle sx={{ fontWeight: 700 }}>Manual Review Required</AlertTitle>
      {message}
    </Alert>
  );
}

function ClassificationCard({ result }: { result: PredictionResponse }) {
  const hasProbs = Object.keys(result.classification_probs ?? {}).length > 0;
  const meta = CLASS_META[result.classification];
  const accent = CLASS_COLORS[meta?.color ?? "info"];

  return (
    <Card className="result-card" sx={{ mb: 2.5 }}>
      <CardContent sx={{ p: 3 }}>
        <SectionHeader
          icon={meta?.icon ?? <WarningIcon />}
          title="Classification Result"
          subtitle="Stage 1 · DINOv2 triage classifier"
          action={
            <Chip
              size="small"
              variant="outlined"
              icon={<TimerIcon sx={{ fontSize: 14 }} />}
              label={`${Math.round(result.inference_time_ms)} ms`}
            />
          }
        />

        <Box
          sx={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            flexWrap: "wrap",
            gap: 2,
          }}
        >
          <Box sx={{ display: "flex", gap: 1, flexWrap: "wrap" }}>
            <ClassificationBadge classification={result.classification} />
            {result.defer_to_clinician &&
              result.classification_confidence < 0.80 &&
              result.classification !== "Manual Review Required" && (
                <ClassificationBadge classification="Manual Review Required" />
              )}
          </Box>
          <Typography
            className="metric-figure"
            sx={{ fontSize: "2.4rem", fontWeight: 700, lineHeight: 1, color: accent }}
          >
            {hasProbs ? `${(result.classification_confidence * 100).toFixed(1)}%` : "N/A"}
          </Typography>
        </Box>

        <Box sx={{ mt: 3 }}>
          {hasProbs ? (
            Object.entries(result.classification_probs).map(([cls, prob]) => (
              <ConfidenceBar
                key={cls}
                label={cls}
                value={prob}
                color={CLASS_COLORS[CLASS_META[cls]?.color ?? "info"]}
              />
            ))
          ) : (
            <Typography variant="body2" color="text.secondary">
              Classification was skipped due to image-quality checks.
            </Typography>
          )}
        </Box>
      </CardContent>
    </Card>
  );
}

function WoundMetrics({ result }: { result: PredictionResponse }) {
  if (!result.has_wound) return null;

  return (
    <Grid container spacing={2.5} sx={{ mb: 2.5 }}>
      <Grid size={{ xs: 12, sm: 6 }} className="result-card">
        <MetricCard
          icon={<CropFreeIcon />}
          label="Wound Coverage"
          value={result.wound_coverage_pct}
          decimals={2}
          unit="%"
          accent="#34D399"
          meter={Math.min(1, Math.max(0, result.wound_coverage_pct / 100))}
        />
      </Grid>
      <Grid size={{ xs: 12, sm: 6 }} className="result-card">
        <MetricCard
          icon={<StraightenIcon />}
          label="Wound Area"
          value={result.wound_area_mm2}
          decimals={1}
          unit="mm²"
          accent="#38BDF8"
        />
      </Grid>
    </Grid>
  );
}

/**
 * Renders the segmentation mask as a colored overlay on top of the original
 * image using a canvas. The mask (grayscale: 255=wound, 0=background) is
 * painted as semi-transparent red so it's clearly visible on any photo.
 */
function WoundOverlay({ result, preview }: { result: PredictionResponse; preview: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!result.has_wound || !result.segmentation_mask_base64) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const photo = new Image();
    const mask = new Image();

    let loaded = 0;
    const onBothLoaded = () => {
      loaded++;
      if (loaded < 2) return;

      canvas.width = photo.naturalWidth;
      canvas.height = photo.naturalHeight;

      // Draw the original photo
      ctx.drawImage(photo, 0, 0, canvas.width, canvas.height);

      // Draw mask to an offscreen canvas to read pixel data
      const offscreen = document.createElement("canvas");
      offscreen.width = canvas.width;
      offscreen.height = canvas.height;
      const offCtx = offscreen.getContext("2d")!;
      offCtx.drawImage(mask, 0, 0, canvas.width, canvas.height);
      const maskData = offCtx.getImageData(0, 0, canvas.width, canvas.height);

      // Create a colored overlay: red (rgba 220, 40, 40) where mask is white
      const overlay = ctx.createImageData(canvas.width, canvas.height);
      for (let i = 0; i < maskData.data.length; i += 4) {
        const isWound = maskData.data[i] > 128;
        if (isWound) {
          overlay.data[i] = 220;     // R
          overlay.data[i + 1] = 40;  // G
          overlay.data[i + 2] = 40;  // B
          overlay.data[i + 3] = 140; // A — semi-transparent
        }
      }

      ctx.putImageData(overlay, 0, 0);

      // Re-draw the photo underneath by compositing
      // We need photo first, then overlay, so let's redo:
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(photo, 0, 0, canvas.width, canvas.height);
      // Now draw overlay on top
      const overlayCanvas = document.createElement("canvas");
      overlayCanvas.width = canvas.width;
      overlayCanvas.height = canvas.height;
      const overlayCtx = overlayCanvas.getContext("2d")!;
      overlayCtx.putImageData(overlay, 0, 0);
      ctx.drawImage(overlayCanvas, 0, 0);
    };

    photo.onload = onBothLoaded;
    mask.onload = onBothLoaded;
    photo.src = preview;
    mask.src = `data:image/png;base64,${result.segmentation_mask_base64}`;
  }, [result.has_wound, result.segmentation_mask_base64, preview]);

  if (!result.has_wound || !result.segmentation_mask_base64) return null;

  return (
    <Card className="result-card" sx={{ mb: 2.5, overflow: "hidden" }}>
      <CardContent sx={{ p: 3 }}>
        <SectionHeader
          icon={<CropFreeIcon />}
          title="Segmentation Overlay"
          subtitle="Stage 2 · DINOv2 + UPerNet wound mask"
        />
        <Box
          sx={{
            borderRadius: 3,
            overflow: "hidden",
            border: "1px solid rgba(255,255,255,0.08)",
            backgroundColor: "rgba(0,0,0,0.2)",
          }}
        >
          <canvas
            ref={canvasRef}
            style={{ width: "100%", height: "auto", display: "block" }}
          />
        </Box>
        <Typography variant="caption" color="text.secondary" sx={{ mt: 1.5, display: "block" }}>
          Red overlay highlights the detected wound region.
        </Typography>
      </CardContent>
    </Card>
  );
}

function WoundMissingNotice({ result, isDemoMode }: { result: PredictionResponse; isDemoMode: boolean }) {
  if (!result.has_wound || result.segmentation_mask_base64) return null;

  return (
    <Card className="result-card" sx={{ mb: 2.5 }}>
      <CardContent sx={{ p: 3, textAlign: "center" }}>
        <Typography variant="body2" color="text.secondary">
          Wound detected but segmentation mask not available in response.
          {isDemoMode && " (Demo mode — no real segmentation)"}
        </Typography>
      </CardContent>
    </Card>
  );
}

function DiagnosticsPanel({ diagnostics }: { diagnostics: Record<string, unknown> & { segmentation_ran?: boolean; segmentation_skip_reason?: string; seg_prob_min?: number; seg_prob_max?: number; seg_prob_mean?: number; pixels_above_threshold?: number; total_pixels?: number; seg_threshold_used?: number; confidence_threshold_used?: number; defer_threshold_used?: number } }) {
  return (
    <Card className="result-card">
      <CardContent sx={{ p: 3 }}>
        <SectionHeader icon={<InfoIcon />} title="Pipeline Diagnostics" subtitle="Internal thresholds & signal ranges" />
        <MetricRow label="Segmentation ran" value={diagnostics.segmentation_ran ? "Yes" : "No"} />
        {diagnostics.segmentation_skip_reason && (
          <MetricRow label="Skip reason" value={String(diagnostics.segmentation_skip_reason)} />
        )}
        {diagnostics.seg_prob_max != null && (
          <>
            <MetricRow
              label="Seg prob range"
              mono
              value={`${Number(diagnostics.seg_prob_min).toFixed(4)} – ${Number(diagnostics.seg_prob_max).toFixed(4)} (mean ${Number(diagnostics.seg_prob_mean).toFixed(4)})`}
            />
            <MetricRow
              label="Pixels above threshold"
              mono
              value={`${String(diagnostics.pixels_above_threshold)} / ${String(diagnostics.total_pixels)} (≥ ${Number(diagnostics.seg_threshold_used).toFixed(2)})`}
            />
          </>
        )}
        <MetricRow label="Confidence threshold" mono value={Number(diagnostics.confidence_threshold_used).toFixed(2)} />
        <MetricRow label="Defer threshold" mono value={Number(diagnostics.defer_threshold_used).toFixed(2)} />
      </CardContent>
    </Card>
  );
}

export default function PredictPage() {
  const containerRef = useRef<HTMLDivElement>(null);
  const resultsRef = useRef<HTMLDivElement>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const mutation = usePrediction();

  const {
    register,
    handleSubmit,
    formState: { errors },
    reset: resetForm,
    setValue,
    clearErrors,
    setError,
  } = useForm<UploadFormData>({
    resolver: zodResolver(uploadFormSchema),
  });

  useGSAP(
    () => {
      gsap.from(".hero-reveal", { y: 20, opacity: 0, duration: 0.6, ease: "power3.out" });
      gsap.from(".upload-area", { y: 24, opacity: 0, duration: 0.6, delay: 0.12, ease: "power3.out" });
    },
    { scope: containerRef }
  );

  useGSAP(
    () => {
      if (result) {
        gsap.from(".result-card", {
          y: 30,
          opacity: 0,
          duration: 0.5,
          stagger: 0.08,
          ease: "power3.out",
        });
      }
    },
    { scope: resultsRef, dependencies: [result] }
  );

  const onSubmit = useCallback(
    async (data: UploadFormData) => {
      const file = data.image instanceof File ? data.image : data.image?.[0] ?? selectedFile;
      if (!file) {
        setError("image", { type: "manual", message: "Please select an image" });
        return;
      }
      setResult(null);
      const response = await mutation.mutateAsync(file);
      setResult(response);
    },
    [mutation, selectedFile, setError]
  );

  const handleFileChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const files = e.target.files;
      if (files && files[0]) {
        const url = URL.createObjectURL(files[0]);
        setPreview(url);
        setSelectedFile(files[0]);
        setValue("image", files[0], { shouldValidate: true, shouldDirty: true, shouldTouch: true });
        clearErrors("image");
      }
    },
    [setValue, clearErrors]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const files = e.dataTransfer.files;
      if (files && files[0]) {
        const url = URL.createObjectURL(files[0]);
        setPreview(url);
        setSelectedFile(files[0]);
        setValue("image", files[0], { shouldValidate: true, shouldDirty: true, shouldTouch: true });
        clearErrors("image");
      }
    },
    [setValue, clearErrors]
  );

  const handleReset = useCallback(() => {
    setPreview(null);
    setSelectedFile(null);
    setResult(null);
    resetForm();
    mutation.reset();
  }, [resetForm, mutation]);

  const isDemoMode = getIsDemoMode();

  return (
    <Box ref={containerRef}>
      <Box className="hero-reveal" sx={{ mb: 4 }}>
        <Typography variant="overline" sx={{ color: "primary.light" }}>
          DiaFoot.AI · Inference
        </Typography>
        <Typography variant="h3" sx={{ mt: 0.5, lineHeight: 1.1 }}>
          Analyze a <GradientText>Foot Image</GradientText>
        </Typography>
        <Typography variant="subtitle1" sx={{ mt: 1, maxWidth: 620 }}>
          Upload a photo for automated triage classification and wound-boundary segmentation
          through the cascaded classifier → segmenter pipeline.
        </Typography>
      </Box>

      {isDemoMode && (
        <Alert severity="warning" className="hero-reveal" sx={{ mb: 3 }}>
          <AlertTitle>Demo Mode</AlertTitle>
          API is not available. Showing simulated results for demonstration purposes.
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* Upload panel */}
        <Grid size={{ xs: 12, lg: 5 }}>
          <Card className="upload-area" sx={{ height: "100%" }}>
            <CardContent sx={{ p: 3 }}>
              <SectionHeader
                icon={<CloudUploadIcon />}
                title="Upload Image"
                subtitle="JPEG, PNG, or WebP · Max 20MB"
              />

              <form onSubmit={handleSubmit(onSubmit)}>
                <Box
                  onDrop={handleDrop}
                  onDragOver={(e) => e.preventDefault()}
                  sx={{
                    border: "2px dashed",
                    borderColor: errors.image ? "error.main" : "rgba(45,212,191,0.35)",
                    borderRadius: 4,
                    p: 4,
                    textAlign: "center",
                    cursor: "pointer",
                    transition: "border-color 0.25s ease, box-shadow 0.25s ease",
                    background: preview ? "transparent" : "var(--grad-brand-soft)",
                    "&:hover": {
                      borderColor: "rgba(45,212,191,0.65)",
                      boxShadow: "var(--glow-teal)",
                    },
                    position: "relative",
                    minHeight: 280,
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "center",
                    justifyContent: "center",
                  }}
                  onClick={() => document.getElementById("file-input")?.click()}
                >
                  {preview ? (
                    <Box
                      component="img"
                      src={preview}
                      alt="Preview"
                      sx={{ maxWidth: "100%", maxHeight: 260, borderRadius: 3, objectFit: "contain" }}
                    />
                  ) : (
                    <>
                      <Box
                        sx={{
                          width: 64,
                          height: 64,
                          borderRadius: "50%",
                          display: "grid",
                          placeItems: "center",
                          mb: 2,
                          color: "primary.light",
                          backgroundColor: "rgba(45,212,191,0.1)",
                          border: "1px solid rgba(45,212,191,0.3)",
                        }}
                      >
                        <CloudUploadIcon sx={{ fontSize: 30 }} />
                      </Box>
                      <Typography variant="body1" fontWeight={600}>
                        Drop image here or click to browse
                      </Typography>
                      <Typography variant="caption" color="text.secondary" sx={{ mt: 1 }}>
                        JPEG, PNG, or WebP — Max 20MB
                      </Typography>
                    </>
                  )}
                  <input
                    id="file-input"
                    type="file"
                    accept="image/jpeg,image/png,image/webp"
                    hidden
                    {...register("image", { onChange: handleFileChange })}
                  />
                </Box>

                {errors.image && (
                  <Typography variant="caption" color="error" sx={{ mt: 1, display: "block" }}>
                    {errors.image.message as string}
                  </Typography>
                )}

                <Box sx={{ display: "flex", gap: 2, mt: 3 }}>
                  <Button
                    type="submit"
                    variant="contained"
                    fullWidth
                    disabled={!preview || mutation.isPending}
                    startIcon={<CameraAltIcon />}
                    sx={{ py: 1.5 }}
                  >
                    {mutation.isPending ? "Analyzing…" : "Analyze"}
                  </Button>
                  <Button
                    variant="outlined"
                    onClick={handleReset}
                    startIcon={<ReplayIcon />}
                    sx={{ minWidth: 120 }}
                  >
                    Reset
                  </Button>
                </Box>
              </form>

              {mutation.isPending && <LinearProgress sx={{ mt: 2 }} />}

              <Typography
                variant="caption"
                color="text.secondary"
                sx={{ mt: 3, display: "block", lineHeight: 1.6 }}
              >
                Research prototype — not a medical device, not for clinical use.
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        {/* Results */}
        <Grid size={{ xs: 12, lg: 7 }}>
          <Box ref={resultsRef}>
            {mutation.isPending && <ResultsSkeleton />}

            {mutation.isError && (
              <Alert severity="error" sx={{ mb: 2.5 }}>
                <AlertTitle>Analysis Failed</AlertTitle>
                {mutation.error instanceof Error
                  ? mutation.error.message
                  : "An error occurred during analysis. Please try again."}
              </Alert>
            )}

            {result && (
              <Box>
                <DeferBanner result={result} />
                <ClassificationCard result={result} />
                <WoundMetrics result={result} />
                {preview && <WoundOverlay result={result} preview={preview} />}
                <WoundMissingNotice result={result} isDemoMode={isDemoMode} />
                {result.diagnostics && <DiagnosticsPanel diagnostics={result.diagnostics} />}
              </Box>
            )}

            {!mutation.isPending && !result && !mutation.isError && (
              <Card
                sx={{
                  height: "100%",
                  minHeight: 400,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                }}
              >
                <CardContent sx={{ textAlign: "center" }}>
                  <ImageSearchIcon sx={{ fontSize: 56, color: "text.secondary", mb: 2, opacity: 0.5 }} />
                  <Typography variant="h6">Upload an image to see results</Typography>
                  <Typography variant="body2" color="text.secondary" sx={{ mt: 1, maxWidth: 320, mx: "auto" }}>
                    The model will classify the image and segment any detected wounds.
                  </Typography>
                </CardContent>
              </Card>
            )}
          </Box>
        </Grid>
      </Grid>
    </Box>
  );
}
