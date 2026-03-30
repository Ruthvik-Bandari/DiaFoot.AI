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
  CameraAltIcon,
  Card,
  CardContent,
  CheckCircleIcon,
  Chip,
  CloudUploadIcon,
  CropFreeIcon,
  ErrorIcon,
  LinearProgress,
  ReplayIcon,
  Skeleton,
  StraightenIcon,
  Typography,
  WarningIcon,
} from "@/lib/mui";
import { usePrediction, getIsDemoMode } from "@/lib/api";
import { uploadFormSchema, type UploadFormData, type PredictionResponse } from "@/lib/api/schemas";

function ClassificationBadge({ classification }: { classification: string }) {
  const config: Record<string, { color: "success" | "warning" | "error" | "info"; icon: React.ReactElement }> = {
    Healthy: { color: "success", icon: <CheckCircleIcon sx={{ fontSize: 16 }} /> },
    "Non-DFU": { color: "info", icon: <WarningIcon sx={{ fontSize: 16 }} /> },
    DFU: { color: "error", icon: <ErrorIcon sx={{ fontSize: 16 }} /> },
    "Manual Review Required": { color: "warning", icon: <WarningIcon sx={{ fontSize: 16 }} /> },
  };
  const c = config[classification];

  return (
    <Chip
      label={classification}
      color={c?.color ?? "info"}
      icon={c?.icon}
      sx={{ fontSize: "1rem", fontWeight: 700, py: 2.5, px: 1 }}
    />
  );
}

function ConfidenceBar({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <Box sx={{ mb: 1.5 }}>
      <Box sx={{ display: "flex", justifyContent: "space-between", mb: 0.5 }}>
        <Typography variant="caption" fontWeight={500}>{label}</Typography>
        <Typography variant="caption" fontWeight={700} color={color}>
          {(value * 100).toFixed(1)}%
        </Typography>
      </Box>
      <LinearProgress
        variant="determinate"
        value={value * 100}
        sx={{
          height: 8,
          borderRadius: 4,
          backgroundColor: "rgba(176,196,216,0.2)",
          "& .MuiLinearProgress-bar": { backgroundColor: color, borderRadius: 4 },
        }}
      />
    </Box>
  );
}

function MetricCard({ icon, label, value, unit }: { icon: React.ReactNode; label: string; value: string; unit?: string }) {
  return (
    <Card sx={{ height: "100%" }}>
      <CardContent sx={{ p: 2.5, textAlign: "center" }}>
        <Box sx={{ color: "primary.main", mb: 1 }}>{icon}</Box>
        <Typography variant="h5" fontWeight={700} color="primary.dark">
          {value}
          {unit && <Typography component="span" variant="body2" color="text.secondary"> {unit}</Typography>}
        </Typography>
        <Typography variant="caption" color="text.secondary">{label}</Typography>
      </CardContent>
    </Card>
  );
}

function ResultsSkeleton() {
  return (
    <Box>
      <Skeleton variant="rounded" height={60} sx={{ mb: 2 }} />
      <Skeleton variant="rounded" height={160} sx={{ mb: 2 }} />
      <Skeleton variant="rounded" height={200} />
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
    <Alert severity="warning" sx={{ mb: 2 }} className="result-card">
      <AlertTitle>Manual Review Required</AlertTitle>
      {message}
    </Alert>
  );
}

function ClassificationCard({ result }: { result: PredictionResponse }) {
  const hasProbs = Object.keys(result.classification_probs ?? {}).length > 0;

  return (
    <Card sx={{ mb: 2 }} className="result-card">
      <CardContent sx={{ p: 3 }}>
        <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <Box>
            <Typography variant="caption" color="text.secondary" fontWeight={500}>
              Classification Result
            </Typography>
            <Box sx={{ mt: 1, display: "flex", gap: 1, flexWrap: "wrap" }}>
              <ClassificationBadge classification={result.classification} />
              {result.defer_to_clinician && result.classification_confidence < 0.80 && result.classification !== "Manual Review Required" && (
                <ClassificationBadge classification="Manual Review Required" />
              )}
            </Box>
          </Box>
          <Typography variant="h3" fontWeight={700} color="primary.main">
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
                color={cls === "DFU" ? "#E74C3C" : cls === "Healthy" ? "#27AE60" : "#1C7293"}
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
    <Box sx={{ display: "flex", gap: 2, mb: 2 }}>
      <Box sx={{ flex: 1 }} className="result-card">
        <MetricCard
          icon={<CropFreeIcon />}
          label="Wound Coverage"
          value={result.wound_coverage_pct.toFixed(2)}
          unit="%"
        />
      </Box>
      <Box sx={{ flex: 1 }} className="result-card">
        <MetricCard
          icon={<StraightenIcon />}
          label="Wound Area"
          value={result.wound_area_mm2.toFixed(1)}
          unit="mm²"
        />
      </Box>
    </Box>
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
    <Card className="result-card" sx={{ mb: 2 }}>
      <CardContent sx={{ p: 3 }}>
        <Typography variant="h6" color="primary.dark" gutterBottom>
          Wound Segmentation Overlay
        </Typography>
        <canvas
          ref={canvasRef}
          style={{ width: "100%", height: "auto", borderRadius: 8, display: "block" }}
        />
        <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: "block" }}>
          Red overlay highlights the detected wound region.
        </Typography>
      </CardContent>
    </Card>
  );
}

function WoundMissingNotice({ result, isDemoMode }: { result: PredictionResponse; isDemoMode: boolean }) {
  if (!result.has_wound || result.segmentation_mask_base64) return null;

  return (
    <Card className="result-card">
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
    <Card className="result-card" sx={{ mt: 2 }}>
      <CardContent sx={{ p: 3 }}>
        <Typography variant="subtitle2" color="text.secondary" gutterBottom>
          Pipeline Diagnostics
        </Typography>
        <Box
          component="table"
          sx={{
            width: "100%",
            fontSize: "0.8rem",
            "& td": { py: 0.5, px: 1, verticalAlign: "top" },
            "& td:first-of-type": { fontWeight: 600, whiteSpace: "nowrap", color: "text.secondary" },
          }}
        >
          <tbody>
            <tr>
              <td>Segmentation ran</td>
              <td>{diagnostics.segmentation_ran ? "Yes" : "No"}</td>
            </tr>
            {diagnostics.segmentation_skip_reason && (
              <tr>
                <td>Skip reason</td>
                <td>{String(diagnostics.segmentation_skip_reason)}</td>
              </tr>
            )}
            {diagnostics.seg_prob_max != null && (
              <>
                <tr>
                  <td>Seg prob range</td>
                  <td>
                    {Number(diagnostics.seg_prob_min).toFixed(4)} &mdash; {Number(diagnostics.seg_prob_max).toFixed(4)} (mean: {Number(diagnostics.seg_prob_mean).toFixed(4)})
                  </td>
                </tr>
                <tr>
                  <td>Pixels above threshold</td>
                  <td>
                    {String(diagnostics.pixels_above_threshold)} / {String(diagnostics.total_pixels)} (threshold: {Number(diagnostics.seg_threshold_used).toFixed(2)})
                  </td>
                </tr>
              </>
            )}
            <tr>
              <td>Confidence threshold</td>
              <td>{Number(diagnostics.confidence_threshold_used).toFixed(2)}</td>
            </tr>
            <tr>
              <td>Defer threshold</td>
              <td>{Number(diagnostics.defer_threshold_used).toFixed(2)}</td>
            </tr>
          </tbody>
        </Box>
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
      gsap.from(".upload-area", { y: 20, opacity: 0, duration: 0.5, ease: "power2.out" });
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
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" color="primary.dark">
          Analyze Image
        </Typography>
        <Typography variant="subtitle1" sx={{ mt: 0.5 }}>
          Upload a foot image for automated classification and wound segmentation
        </Typography>
      </Box>

      {isDemoMode && (
        <Alert severity="warning" sx={{ mb: 3 }}>
          <AlertTitle>Demo Mode</AlertTitle>
          API is not available. Showing simulated results for demonstration purposes.
        </Alert>
      )}

      <Box sx={{ display: "flex", gap: 3, flexDirection: { xs: "column", lg: "row" }, maxWidth: 1200 }}>
        {/* Upload Area */}
        <Box sx={{ width: { xs: "100%", lg: 340 }, flexShrink: 0 }}>
          <Card className="upload-area">
            <CardContent sx={{ p: 3 }}>
              <Typography variant="h6" color="primary.dark" gutterBottom>
                Upload Image
              </Typography>

              <form onSubmit={handleSubmit(onSubmit)}>
                <Box
                  onDrop={handleDrop}
                  onDragOver={(e) => e.preventDefault()}
                  sx={{
                    border: "2px dashed",
                    borderColor: errors.image ? "error.main" : "divider",
                    borderRadius: 3,
                    p: 4,
                    textAlign: "center",
                    cursor: "pointer",
                    transition: "all 0.2s",
                    backgroundColor: preview ? "transparent" : "rgba(6,90,130,0.02)",
                    "&:hover": {
                      borderColor: "primary.main",
                      backgroundColor: "rgba(6,90,130,0.04)",
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
                      sx={{ maxWidth: "100%", maxHeight: 260, borderRadius: 2, objectFit: "contain" }}
                    />
                  ) : (
                    <>
                      <CloudUploadIcon sx={{ fontSize: 48, color: "primary.main", mb: 2 }} />
                      <Typography variant="body1" fontWeight={500} color="primary.dark">
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
                    {mutation.isPending ? "Analyzing..." : "Analyze"}
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

              {mutation.isPending && (
                <LinearProgress sx={{ mt: 2, borderRadius: 2 }} />
              )}
            </CardContent>
          </Card>
        </Box>

        {/* Results */}
        <Box sx={{ flex: 1, minWidth: 0 }} ref={resultsRef}>
          {mutation.isPending && <ResultsSkeleton />}

          {mutation.isError && (
            <Alert severity="error">
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
            <Card sx={{ height: "100%", display: "flex", alignItems: "center", justifyContent: "center", minHeight: 400 }}>
              <CardContent sx={{ textAlign: "center" }}>
                <CameraAltIcon sx={{ fontSize: 64, color: "divider", mb: 2 }} />
                <Typography variant="h6" color="text.secondary">
                  Upload an image to see results
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                  The model will classify the image and segment any detected wounds
                </Typography>
              </CardContent>
            </Card>
          )}
        </Box>
      </Box>
    </Box>
  );
}
