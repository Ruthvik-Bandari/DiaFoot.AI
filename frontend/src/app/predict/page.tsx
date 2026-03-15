"use client";

import { useCallback, useRef, useState } from "react";
import { useGSAP } from "@gsap/react";
import gsap from "gsap";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import Box from "@mui/material/Box";
import Card from "@mui/material/Card";
import CardContent from "@mui/material/CardContent";
import Grid from "@mui/material/Grid";
import Typography from "@mui/material/Typography";
import Button from "@mui/material/Button";
import Skeleton from "@mui/material/Skeleton";
import Chip from "@mui/material/Chip";
import Alert from "@mui/material/Alert";
import AlertTitle from "@mui/material/AlertTitle";
import LinearProgress from "@mui/material/LinearProgress";
import CloudUploadIcon from "@mui/icons-material/CloudUpload";
import CameraAltIcon from "@mui/icons-material/CameraAlt";
import ReplayIcon from "@mui/icons-material/Replay";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import WarningIcon from "@mui/icons-material/Warning";
import ErrorIcon from "@mui/icons-material/Error";
import TimerIcon from "@mui/icons-material/Timer";
import CropFreeIcon from "@mui/icons-material/CropFree";
import StraightenIcon from "@mui/icons-material/Straighten";
import { usePrediction, getIsDemoMode } from "@/lib/api";
import { uploadFormSchema, type UploadFormData, type PredictionResponse } from "@/lib/api/schemas";

function ClassificationBadge({ classification }: { classification: string }) {
  const config: Record<string, { color: "success" | "warning" | "error" | "info"; icon: React.ReactNode }> = {
    Healthy: { color: "success", icon: <CheckCircleIcon sx={{ fontSize: 16 }} /> },
    "Non-DFU": { color: "info", icon: <WarningIcon sx={{ fontSize: 16 }} /> },
    DFU: { color: "error", icon: <ErrorIcon sx={{ fontSize: 16 }} /> },
    "Manual Review Required": { color: "warning", icon: <WarningIcon sx={{ fontSize: 16 }} /> },
  };
  const c = config[classification] ?? { color: "info" as const, icon: null };

  return (
    <Chip
      label={classification}
      color={c.color}
      icon={<>{c.icon}</>}
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
      <Grid container spacing={2}>
        {[1, 2, 3].map((i) => (
          <Grid size={{ xs: 4 }} key={i}>
            <Skeleton variant="rounded" height={100} />
          </Grid>
        ))}
      </Grid>
      <Skeleton variant="rounded" height={200} sx={{ mt: 2 }} />
    </Box>
  );
}

export default function PredictPage() {
  const containerRef = useRef<HTMLDivElement>(null);
  const resultsRef = useRef<HTMLDivElement>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const mutation = usePrediction();

  const {
    register,
    handleSubmit,
    formState: { errors },
    reset: resetForm,
    setValue,
  } = useForm<UploadFormData>({
    resolver: zodResolver(uploadFormSchema),
  });

  useGSAP(
    () => {
      gsap.from(".upload-area", { y: 20, opacity: 0, duration: 0.5, ease: "power2.out" });
    },
    { scope: containerRef }
  );

  // Animate results when they appear
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
      const file = data.image[0];
      setResult(null);
      const response = await mutation.mutateAsync(file);
      setResult(response);
    },
    [mutation]
  );

  const handleFileChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const files = e.target.files;
      if (files && files[0]) {
        const url = URL.createObjectURL(files[0]);
        setPreview(url);
        setValue("image", files);
      }
    },
    [setValue]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const files = e.dataTransfer.files;
      if (files && files[0]) {
        const url = URL.createObjectURL(files[0]);
        setPreview(url);
        const dt = new DataTransfer();
        dt.items.add(files[0]);
        setValue("image", dt.files);
      }
    },
    [setValue]
  );

  const handleReset = useCallback(() => {
    setPreview(null);
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

      <Grid container spacing={3}>
        {/* Upload Area */}
        <Grid size={{ xs: 12, md: 5 }}>
          <Card className="upload-area" sx={{ height: "100%" }}>
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
                      sx={{
                        maxWidth: "100%",
                        maxHeight: 260,
                        borderRadius: 2,
                        objectFit: "contain",
                      }}
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
        </Grid>

        {/* Results */}
        <Grid size={{ xs: 12, md: 7 }}>
          <Box ref={resultsRef}>
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
                {/* Defer Banner */}
                {result.defer_to_clinician && (
                  <Alert severity="warning" sx={{ mb: 2 }} className="result-card">
                    <AlertTitle>Manual Review Required</AlertTitle>
                    {result.defer_reason === "low_confidence"
                      ? "The model is not confident enough in its prediction. Please consult a healthcare professional."
                      : result.defer_reason === "low_image_quality"
                        ? `Image quality issues detected: ${result.quality_flags.join(", ")}. Please retake the image.`
                        : "This case requires clinical review."}
                  </Alert>
                )}

                {/* Classification */}
                <Card sx={{ mb: 2 }} className="result-card">
                  <CardContent sx={{ p: 3 }}>
                    <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                      <Box>
                        <Typography variant="caption" color="text.secondary" fontWeight={500}>
                          Classification Result
                        </Typography>
                        <Box sx={{ mt: 1 }}>
                          <ClassificationBadge classification={result.classification} />
                        </Box>
                      </Box>
                      <Typography variant="h3" fontWeight={700} color="primary.main">
                        {(result.classification_confidence * 100).toFixed(1)}%
                      </Typography>
                    </Box>

                    {/* Probability bars */}
                    <Box sx={{ mt: 3 }}>
                      {Object.entries(result.classification_probs).map(([cls, prob]) => (
                        <ConfidenceBar
                          key={cls}
                          label={cls}
                          value={prob}
                          color={cls === "DFU" ? "#E74C3C" : cls === "Healthy" ? "#27AE60" : "#1C7293"}
                        />
                      ))}
                    </Box>
                  </CardContent>
                </Card>

                {/* Metrics */}
                <Grid container spacing={2} sx={{ mb: 2 }}>
                  <Grid size={{ xs: 4 }} className="result-card">
                    <MetricCard
                      icon={<TimerIcon />}
                      label="Inference Time"
                      value={result.inference_time_ms.toFixed(0)}
                      unit="ms"
                    />
                  </Grid>
                  <Grid size={{ xs: 4 }} className="result-card">
                    <MetricCard
                      icon={<CropFreeIcon />}
                      label="Wound Coverage"
                      value={result.wound_coverage_pct.toFixed(2)}
                      unit="%"
                    />
                  </Grid>
                  <Grid size={{ xs: 4 }} className="result-card">
                    <MetricCard
                      icon={<StraightenIcon />}
                      label="Wound Area"
                      value={result.wound_area_mm2.toFixed(1)}
                      unit="mm²"
                    />
                  </Grid>
                </Grid>

                {/* Wound Overlay */}
                {result.has_wound && result.segmentation_mask_base64 && preview && (
                  <Card className="result-card">
                    <CardContent sx={{ p: 3 }}>
                      <Typography variant="h6" color="primary.dark" gutterBottom>
                        Wound Segmentation Overlay
                      </Typography>
                      <Box sx={{ position: "relative", display: "inline-block", width: "100%" }}>
                        <Box
                          component="img"
                          src={preview}
                          alt="Original"
                          sx={{ width: "100%", borderRadius: 2, display: "block" }}
                        />
                        <Box
                          component="img"
                          src={`data:image/png;base64,${result.segmentation_mask_base64}`}
                          alt="Segmentation mask"
                          sx={{
                            position: "absolute",
                            top: 0,
                            left: 0,
                            width: "100%",
                            height: "100%",
                            borderRadius: 2,
                            opacity: 0.4,
                            mixBlendMode: "multiply",
                            filter: "hue-rotate(170deg) saturate(3)",
                          }}
                        />
                      </Box>
                      <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: "block" }}>
                        Teal overlay shows detected wound boundary. Opacity: 40%.
                      </Typography>
                    </CardContent>
                  </Card>
                )}

                {result.has_wound && !result.segmentation_mask_base64 && (
                  <Card className="result-card">
                    <CardContent sx={{ p: 3, textAlign: "center" }}>
                      <Typography variant="body2" color="text.secondary">
                        Wound detected but segmentation mask not available in response.
                        {isDemoMode && " (Demo mode — no real segmentation)"}
                      </Typography>
                    </CardContent>
                  </Card>
                )}
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
        </Grid>
      </Grid>
    </Box>
  );
}
