import { z } from "zod";

export const predictionResponseSchema = z.object({
  classification: z.string(),
  classification_confidence: z.number(),
  classification_probs: z.record(z.string(), z.number()),
  defer_to_clinician: z.boolean(),
  defer_reason: z.string(),
  quality_flags: z.array(z.string()),
  has_wound: z.boolean(),
  wound_area_mm2: z.number(),
  wound_coverage_pct: z.number(),
  inference_time_ms: z.number(),
  segmentation_mask_base64: z.string().nullable().optional(),
  diagnostics: z.record(z.string(), z.any()).nullable().optional(),
});

export type PredictionResponse = z.infer<typeof predictionResponseSchema>;

export const healthResponseSchema = z.object({
  status: z.string(),
  model_loaded: z.boolean(),
  version: z.string(),
});

export type HealthResponse = z.infer<typeof healthResponseSchema>;

export const modelInfoResponseSchema = z.object({
  classifier: z.string(),
  segmenter: z.string(),
  input_size: z.tuple([z.number(), z.number()]),
  num_classes: z.number(),
  confidence_threshold: z.number(),
  defer_threshold: z.number(),
  defer_threshold_source: z.string(),
  max_image_size_mb: z.number(),
  rate_limit_rpm: z.number(),
  version: z.string(),
});

export type ModelInfoResponse = z.infer<typeof modelInfoResponseSchema>;

export const uploadFormSchema = z.object({
  image: z
    .custom<File | FileList>()
    .refine(
      (value) => {
        if (value instanceof File) return true;
        if (typeof FileList !== "undefined" && value instanceof FileList) return value.length === 1;
        return false;
      },
      "Please select an image"
    )
    .refine(
      (value) => {
        const file = value instanceof File ? value : value?.[0];
        return !!file && ["image/jpeg", "image/png", "image/webp"].includes(file.type);
      },
      "Only JPEG, PNG, or WebP images are accepted"
    )
    .refine(
      (value) => {
        const file = value instanceof File ? value : value?.[0];
        return !!file && file.size <= 20 * 1024 * 1024;
      },
      "Image must be under 20MB"
    ),
});

export type UploadFormData = z.infer<typeof uploadFormSchema>;
