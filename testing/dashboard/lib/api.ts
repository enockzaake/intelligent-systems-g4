// API Configuration and Utilities
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface StopData {
  route_id: number;
  driver_id: number;
  stop_id: number;
  address_id: number;
  week_id: number;
  country: number;
  day_of_week: string;
  indexp: number;
  indexa: number;
  arrived_time: string;
  earliest_time: string;
  latest_time: string;
  distancep: number;
  distancea: number;
  depot: number;
  delivery: number;
}

export interface PredictionResponse {
  route_id: number;
  stop_id: number;
  driver_id: number;
  delayed_flag_pred: number;
  delay_probability: number | null;
  delay_minutes_pred: number;
  risk_level?: string;
  contributing_factors?: Array<{
    feature: string;
    value: number;
    importance: number;
  }>;
}

export interface RouteAggregate {
  route_id: number;
  route_delay_rate: number;
  avg_delay_probability: number;
  total_delay_minutes: number;
  avg_delay_minutes: number;
  max_delay_minutes: number;
}

export interface OptimizationResult {
  solution: {
    routes: Array<{
      vehicle_id: number;
      stops: Array<{
        location: number;
        time: number;
        stop_id?: number;
      }>;
      distance: number;
      time: number;
    }>;
    total_distance: number;
    total_time: number;
    num_vehicles_used: number;
  };
  num_reassignments: number;
  reassignments: Array<{
    stop_id: number;
    original_route: number;
    original_driver: number;
    new_driver: number;
    reason: string;
    expected_improvement: number;
  }>;
}

export interface ReassignmentResult {
  num_reassignments: number;
  reassignments: Array<{
    stop_id: number;
    original_route: number;
    original_driver: number;
    new_driver: number;
    reason: string;
    expected_improvement: number;
  }>;
  message: string;
}

// API Functions
export async function checkHealth() {
  const response = await fetch(`${API_BASE_URL}/health`);
  if (!response.ok) {
    const error = await response.text();
    throw new Error(`API health check failed: ${error}`);
  }
  return response.json();
}

export async function predictDelays(
  stops: StopData[]
): Promise<PredictionResponse[]> {
  console.log("Sending prediction request with", stops.length, "stops");
  console.log("Sample stop:", stops[0]);

  const response = await fetch(`${API_BASE_URL}/predict/delays`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ stops, model_type: "random_forest" }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    console.error("Prediction API error:", errorText);
    throw new Error(`Delay prediction failed: ${errorText}`);
  }

  const result = await response.json();
  console.log("Received predictions:", result.length);
  return result;
}

export async function predictRouteAggregates(
  stops: StopData[]
): Promise<RouteAggregate[]> {
  const response = await fetch(`${API_BASE_URL}/predict/routes`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ stops, model_type: "random_forest" }),
  });
  if (!response.ok) throw new Error("Route prediction failed");
  return response.json();
}

export async function optimizeRoutes(
  stops: StopData[]
): Promise<OptimizationResult> {
  const response = await fetch(`${API_BASE_URL}/optimize/routes`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ stops, model_type: "random_forest" }),
  });
  if (!response.ok) throw new Error("Route optimization failed");
  return response.json();
}

export async function reassignStops(
  stops: StopData[],
  delayThreshold: number = 0.7
): Promise<ReassignmentResult> {
  const response = await fetch(
    `${API_BASE_URL}/reassign/stops?delay_threshold=${delayThreshold}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ stops, model_type: "random_forest" }),
    }
  );
  if (!response.ok) throw new Error("Stop reassignment failed");
  return response.json();
}

export async function getModels() {
  const response = await fetch(`${API_BASE_URL}/models`);
  if (!response.ok) throw new Error("Failed to fetch models");
  return response.json();
}

export async function getOptimizationStatus() {
  const response = await fetch(`${API_BASE_URL}/optimization/status`);
  if (!response.ok) throw new Error("Failed to fetch optimization status");
  return response.json();
}

// Real Data Endpoints
export async function getSampleData(limit: number = 100): Promise<{
  total_records: number;
  sample_size: number;
  data: StopData[];
}> {
  const response = await fetch(`${API_BASE_URL}/data/sample?limit=${limit}`);
  if (!response.ok) throw new Error("Failed to fetch sample data");
  return response.json();
}

export async function getRoutesData(limit: number = 50): Promise<{
  total_routes: number;
  sample_size: number;
  routes: Array<{
    route_id: number;
    driver_id: number;
    total_stops: number;
    total_distance: number;
    day_of_week: string;
  }>;
}> {
  const response = await fetch(`${API_BASE_URL}/data/routes?limit=${limit}`);
  if (!response.ok) throw new Error("Failed to fetch routes data");
  return response.json();
}

export async function getDriversData(): Promise<{
  total_drivers: number;
  drivers: Array<{
    driver_id: number;
    completed_stops: number;
    total_routes: number;
    total_distance: number;
  }>;
}> {
  const response = await fetch(`${API_BASE_URL}/data/drivers`);
  if (!response.ok) throw new Error("Failed to fetch drivers data");
  return response.json();
}

// Scenario Simulation Types
export interface ScenarioModifications {
  stop_modifications?: Record<string, any>;
  driver_reassignments?: Record<number, number>; // stop_id -> new_driver_id
  traffic_multiplier?: number;
  distance_multiplier?: number;
  time_shift_minutes?: Record<number, number>; // stop_id -> minutes
  stops_to_remove?: number[];
}

export interface ScenarioComparison {
  metric: string;
  original: number;
  modified: number;
  improvement_percent: number;
}

export interface ScenarioResult {
  original: {
    predictions: PredictionResponse[];
    total_stops: number;
    total_delay_minutes: number;
    avg_delay_probability: number;
    high_risk_count: number;
    optimization: any;
  };
  modified: {
    predictions: PredictionResponse[];
    total_stops: number;
    total_delay_minutes: number;
    avg_delay_probability: number;
    high_risk_count: number;
    optimization: any;
  };
  comparisons: ScenarioComparison[];
  modifications_applied: {
    driver_reassignments: number;
    traffic_multiplier: number;
    distance_multiplier: number;
    time_shifts: number;
    stops_removed: number;
  };
}

// Scenario Simulation API
export async function simulateScenario(
  stops: StopData[],
  modifications?: ScenarioModifications
): Promise<ScenarioResult> {
  const response = await fetch(`${API_BASE_URL}/simulate/scenario`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      stops,
      model_type: "random_forest",
      modifications: modifications || {},
    }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    console.error("Scenario simulation error:", errorText);
    throw new Error(`Scenario simulation failed: ${errorText}`);
  }

  return response.json();
}

// ============================================================================
// NEW SIMULATION ENDPOINTS - Enhanced with explanations and reasoning
// ============================================================================

export interface StopPredictionWithReason {
  stop_id: number;
  route_id: number;
  delay_minutes: number;
  delay_probability: number;
  risk_level: string;
  reason: string;
  contributing_factors?: Array<{
    feature: string;
    value: number;
    importance: number;
  }>;
}

export interface RouteSummary {
  total_expected_delay: number;
  delay_risk: string; // "low" | "medium" | "high"
  expected_total_distance: number;
  expected_total_duration: number;
  high_risk_stops: number;
  on_time_probability: number;
}

export interface SimulationPredictionResponse {
  route_id: string;
  predictions: StopPredictionWithReason[];
  route_summary: RouteSummary;
  scenario_applied: {
    scenario_type: string;
    applied_changes: Array<{
      type: string;
      description: string;
      reason?: string;
      [key: string]: any;
    }>;
  };
}

export interface OptimizationAction {
  action: string;
  reason: string;
  expected_benefit: string;
  details?: Record<string, any>;
}

export interface OptimizationImpact {
  delay_reduction_minutes: number;
  energy_saving_km: number;
  on_time_rate_improvement: number;
}

export interface SimulationOptimizationResponse {
  old_sequence: number[];
  new_sequence: number[];
  actions: OptimizationAction[];
  impact: OptimizationImpact;
}

export interface SimulationScenarioConfig {
  route_id?: string;
  stops: StopData[];
  scenario_type:
    | "normal"
    | "traffic_congestion"
    | "delay_at_stop"
    | "driver_slowdown"
    | "increased_workload"
    | "random_faults"
    | "custom";
  custom_params?: {
    traffic_factor?: number;
    stop_id?: number;
    delay_minutes?: number;
    slowdown_factor?: number;
    workload_increase?: number;
    num_faults?: number;
    artificial_delay_minutes?: number;
    target_stop_id?: number;
  };
}

export interface FullSimulationAnalysis {
  predictions: SimulationPredictionResponse;
  optimizations: SimulationOptimizationResponse;
  analysis_timestamp: string;
}

/**
 * Simulate a scenario and get predictions with detailed explanations
 */
export async function simulateAndPredict(
  config: SimulationScenarioConfig
): Promise<SimulationPredictionResponse> {
  console.log("Sending simulation/predict request:", config.scenario_type);

  const response = await fetch(`${API_BASE_URL}/simulation/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config),
  });

  if (!response.ok) {
    const errorText = await response.text();
    console.error("Simulation predict error:", errorText);
    throw new Error(`Simulation predict failed: ${errorText}`);
  }

  const result = await response.json();
  console.log("Received predictions:", result.predictions.length);
  return result;
}

/**
 * Get optimization recommendations with explanations
 */
export async function simulateAndOptimize(
  config: SimulationScenarioConfig
): Promise<SimulationOptimizationResponse> {
  console.log("Sending simulation/optimize request");

  const response = await fetch(`${API_BASE_URL}/simulation/optimize`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config),
  });

  if (!response.ok) {
    const errorText = await response.text();
    console.error("Simulation optimize error:", errorText);
    throw new Error(`Simulation optimize failed: ${errorText}`);
  }

  const result = await response.json();
  console.log("Received optimization recommendations:", result.actions.length);
  return result;
}

/**
 * Get full simulation analysis (predictions + optimizations) in one call
 */
export async function runFullSimulationAnalysis(
  config: SimulationScenarioConfig
): Promise<FullSimulationAnalysis> {
  console.log("Running full simulation analysis:", config.scenario_type);

  const response = await fetch(`${API_BASE_URL}/simulation/full-analysis`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config),
  });

  if (!response.ok) {
    const errorText = await response.text();
    console.error("Full simulation analysis error:", errorText);
    throw new Error(`Full simulation analysis failed: ${errorText}`);
  }

  const result = await response.json();
  console.log("Full simulation analysis complete");
  return result;
}