import { useMutation, useQuery } from "@tanstack/react-query";
import {
  predictionResponseSchema,
  healthResponseSchema,
  modelInfoResponseSchema,
  type PredictionResponse,
  type HealthResponse,
  type ModelInfoResponse,
} from "./schemas";
import {
  MOCK_HEALTH,
  MOCK_MODEL_INFO,
  getRandomMockResult,
} from "@/lib/mock/data";

const API_BASE = (process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");
const DEMO_MODE_ENABLED = (process.env.NEXT_PUBLIC_ENABLE_DEMO_MODE ?? "false") === "true";

interface ApiState {
  isDemoMode: boolean;
}

const apiState: ApiState = {
  isDemoMode: false,
};

export function getIsDemoMode(): boolean {
  return apiState.isDemoMode;
}

class ApiError extends Error {
  constructor(message: string, public readonly status?: number) {
    super(message);
    this.name = "ApiError";
  }
}

interface IApiTransport {
  get(path: string, timeoutMs?: number): Promise<unknown>;
  postForm(path: string, formData: FormData, timeoutMs?: number): Promise<unknown>;
}

class FetchApiTransport implements IApiTransport {
  constructor(private readonly baseUrl: string) {}

  async get(path: string, timeoutMs = 5000): Promise<unknown> {
    return this.request(path, { method: "GET" }, timeoutMs);
  }

  async postForm(path: string, formData: FormData, timeoutMs = 30000): Promise<unknown> {
    return this.request(
      path,
      {
        method: "POST",
        body: formData,
      },
      timeoutMs,
    );
  }

  private async request(path: string, init: RequestInit, timeoutMs: number): Promise<unknown> {
    let res: Response;
    try {
      res = await fetch(`${this.baseUrl}${path}`, {
        ...init,
        signal: AbortSignal.timeout(timeoutMs),
      });
    } catch {
      throw new ApiError("Backend unreachable. Check NEXT_PUBLIC_API_URL and backend status.");
    }

    if (!res.ok) {
      const body = await res.text().catch(() => "");
      throw new ApiError(body || `Request failed with status ${res.status}`, res.status);
    }

    return res.json();
  }
}

interface IInferenceGateway {
  checkHealth(): Promise<HealthResponse>;
  getModelInfo(): Promise<ModelInfoResponse>;
  predictImage(file: File): Promise<PredictionResponse>;
}

class BackendInferenceGateway implements IInferenceGateway {
  constructor(private readonly transport: IApiTransport) {}

  async checkHealth(): Promise<HealthResponse> {
    return healthResponseSchema.parse(await this.transport.get("/health"));
  }

  async getModelInfo(): Promise<ModelInfoResponse> {
    return modelInfoResponseSchema.parse(await this.transport.get("/model/info"));
  }

  async predictImage(file: File): Promise<PredictionResponse> {
    const formData = new FormData();
    formData.append("file", file);
    return predictionResponseSchema.parse(await this.transport.postForm("/predict", formData));
  }
}

const gateway: IInferenceGateway = new BackendInferenceGateway(new FetchApiTransport(API_BASE));

async function withOptionalDemoFallback<T>(
  action: () => Promise<T>,
  fallback: () => Promise<T> | T,
): Promise<T> {
  try {
    const result = await action();
    apiState.isDemoMode = false;
    return result;
  } catch (error) {
    if (!DEMO_MODE_ENABLED) {
      apiState.isDemoMode = false;
      throw error;
    }
    apiState.isDemoMode = true;
    return await fallback();
  }
}

export async function checkHealth(): Promise<HealthResponse> {
  return withOptionalDemoFallback(
    () => gateway.checkHealth(),
    () => MOCK_HEALTH,
  );
}

export async function getModelInfo(): Promise<ModelInfoResponse> {
  return withOptionalDemoFallback(
    () => gateway.getModelInfo(),
    () => MOCK_MODEL_INFO,
  );
}

export async function predictImage(
  file: File
): Promise<PredictionResponse> {
  return withOptionalDemoFallback(
    () => gateway.predictImage(file),
    async () => {
      // Simulate inference delay in demo mode
      await new Promise((r) => setTimeout(r, 1500 + Math.random() * 1000));
      return getRandomMockResult();
    },
  );
}

// ── TanStack Query Hooks ──

export function useHealth() {
  return useQuery({
    queryKey: ["health"],
    queryFn: checkHealth,
    refetchInterval: 10000,
    staleTime: 5000,
  });
}

export function useModelInfo() {
  return useQuery({
    queryKey: ["modelInfo"],
    queryFn: getModelInfo,
    staleTime: 60000,
  });
}

export function usePrediction() {
  return useMutation({
    mutationKey: ["predict"],
    mutationFn: predictImage,
  });
}
