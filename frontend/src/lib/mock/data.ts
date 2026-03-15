import type { PredictionResponse, HealthResponse, ModelInfoResponse } from "@/lib/api/schemas";

export const MOCK_DFU_RESULT: PredictionResponse = {
  classification: "DFU",
  classification_confidence: 0.9234,
  classification_probs: {
    Healthy: 0.0312,
    "Non-DFU": 0.0454,
    DFU: 0.9234,
  },
  defer_to_clinician: false,
  defer_reason: "",
  quality_flags: [],
  has_wound: true,
  wound_area_mm2: 1342.6,
  wound_coverage_pct: 5.12,
  inference_time_ms: 133.2,
  segmentation_mask_base64: null,
};

export const MOCK_HEALTHY_RESULT: PredictionResponse = {
  classification: "Healthy",
  classification_confidence: 0.9876,
  classification_probs: {
    Healthy: 0.9876,
    "Non-DFU": 0.0089,
    DFU: 0.0035,
  },
  defer_to_clinician: false,
  defer_reason: "",
  quality_flags: [],
  has_wound: false,
  wound_area_mm2: 0,
  wound_coverage_pct: 0,
  inference_time_ms: 42.1,
  segmentation_mask_base64: null,
};

export const MOCK_DEFER_RESULT: PredictionResponse = {
  classification: "Manual Review Required",
  classification_confidence: 0.42,
  classification_probs: {
    Healthy: 0.28,
    "Non-DFU": 0.42,
    DFU: 0.30,
  },
  defer_to_clinician: true,
  defer_reason: "low_confidence",
  quality_flags: [],
  has_wound: false,
  wound_area_mm2: 0,
  wound_coverage_pct: 0,
  inference_time_ms: 89.5,
  segmentation_mask_base64: null,
};

export const MOCK_HEALTH: HealthResponse = {
  status: "healthy",
  model_loaded: true,
  version: "2.0.0",
};

export const MOCK_MODEL_INFO: ModelInfoResponse = {
  classifier: "EfficientNet-V2-M (3-class triage)",
  segmenter: "U-Net++ / EfficientNet-B4",
  input_size: [512, 512],
  num_classes: 3,
  confidence_threshold: 0.95,
  defer_threshold: 0.6,
  defer_threshold_source: "calibration",
  max_image_size_mb: 20.0,
  rate_limit_rpm: 100,
  version: "2.0.0",
};

const MOCK_RESULTS = [MOCK_DFU_RESULT, MOCK_HEALTHY_RESULT, MOCK_DEFER_RESULT];

export function getRandomMockResult(): PredictionResponse {
  const idx = Math.floor(Math.random() * MOCK_RESULTS.length);
  return {
    ...MOCK_RESULTS[idx],
    inference_time_ms: 80 + Math.random() * 120,
  };
}
