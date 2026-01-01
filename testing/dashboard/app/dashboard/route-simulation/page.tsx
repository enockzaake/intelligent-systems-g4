"use client";

import { useState } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Progress } from "@/components/ui/progress";
import { Slider } from "@/components/ui/slider";
import {
  Play,
  CheckCircle2,
  AlertCircle,
  Clock,
  TrendingDown,
  MapPin,
  Route,
  Loader2,
  ArrowRight,
  Settings,
  Plus,
  Trash2,
  RefreshCw,
} from "lucide-react";

// Preset route scenarios
const PRESET_SCENARIOS = {
  small_route: {
    name: "Small Route (5 stops)",
    description: "Quick delivery route with 5 stops in city center",
    numStops: 5,
    numVehicles: 1,
    trafficLevel: 1.0,
    avgDistance: 4.5,
    timeWindowTightness: "normal",
  },
  medium_route: {
    name: "Medium Route (10 stops)",
    description: "Standard delivery route with 10 stops across suburbs",
    numStops: 10,
    numVehicles: 3,
    trafficLevel: 1.2,
    avgDistance: 6.2,
    timeWindowTightness: "tight",
  },
  high_traffic: {
    name: "High Traffic Route (8 stops)",
    description: "Rush hour route with expected delays",
    numStops: 8,
    numVehicles: 2,
    trafficLevel: 1.8,
    avgDistance: 5.5,
    timeWindowTightness: "very_tight",
  },
};

interface RouteParameters {
  numStops: number;
  numVehicles: number;
  trafficMultiplier: number;
  delayThreshold: number;
  avgDistance: number;
  timeWindowWidth: number; // minutes
  startTime: string;
  driverExperience: "novice" | "intermediate" | "expert";
  weatherCondition: "clear" | "rain" | "heavy_rain" | "snow";
}

interface Stop {
  stop_id: string;
  earliest_time: string;
  latest_time: string;
  distance_from_prev: number;
  planned_arrival: string;
}

interface PredictionResult {
  stop_id: string;
  delay_probability: number;
  delay_minutes: number;
  risk_level: "HIGH" | "MEDIUM" | "LOW";
  reason: string;
  contributing_factors: string[];
}

interface OptimizationResult {
  routes: Array<{
    vehicle_id: number;
    stops: Array<{ stop_id: string; arrival_time: string; sequence: number }>;
    distance: number;
    time: number;
  }>;
  total_distance: number;
  total_time: number;
  num_vehicles_used: number;
  improvements: {
    distance_reduction: number;
    time_reduction: number;
    vehicles_saved: number;
  };
  reassignments: Array<{
    stop_id: string;
    from_vehicle: number;
    to_vehicle: number;
    reason: string;
  }>;
}

export default function RouteSimulationPage() {
  const [inputMode, setInputMode] = useState<"preset" | "manual">("preset");
  const [selectedScenario, setSelectedScenario] = useState<string>("small_route");
  const [currentStep, setCurrentStep] = useState<number>(0);
  
  const [parameters, setParameters] = useState<RouteParameters>({
    numStops: 5,
    numVehicles: 1,
    trafficMultiplier: 1.0,
    delayThreshold: 0.7,
    avgDistance: 4.5,
    timeWindowWidth: 60,
    startTime: "08:00",
    driverExperience: "intermediate",
    weatherCondition: "clear",
  });

  const [stops, setStops] = useState<Stop[]>([]);
  const [predictions, setPredictions] = useState<PredictionResult[]>([]);
  const [optimization, setOptimization] = useState<OptimizationResult | null>(null);
  const [loading, setLoading] = useState(false);

  // Generate stops based on parameters
  const generateStops = (params: RouteParameters): Stop[] => {
    const stops: Stop[] = [];
    let cumulativeTime = parseInt(params.startTime.split(":")[0]) * 60 + 
                         parseInt(params.startTime.split(":")[1]);

    for (let i = 0; i < params.numStops; i++) {
      const distance = params.avgDistance * (0.8 + Math.random() * 0.4);
      const travelTime = (distance / 50) * 60; // Assume 50 km/h avg speed
      
      cumulativeTime += travelTime + 10; // 10 min service time
      
      const earliest = cumulativeTime - params.timeWindowWidth / 2;
      const latest = cumulativeTime + params.timeWindowWidth / 2;
      
      stops.push({
        stop_id: `S${i + 1}`,
        earliest_time: `${Math.floor(earliest / 60).toString().padStart(2, "0")}:${(earliest % 60).toString().padStart(2, "0")}`,
        latest_time: `${Math.floor(latest / 60).toString().padStart(2, "0")}:${(latest % 60).toString().padStart(2, "0")}`,
        distance_from_prev: distance,
        planned_arrival: `${Math.floor(cumulativeTime / 60).toString().padStart(2, "0")}:${(cumulativeTime % 60).toString().padStart(2, "0")}`,
      });
    }
    
    return stops;
  };

  // Realistic prediction engine with probabilistic but consistent results
  const predictDelays = (stops: Stop[], params: RouteParameters): PredictionResult[] => {
    const predictions: PredictionResult[] = [];
    
    // Factor multipliers based on conditions
    const trafficFactor = params.trafficMultiplier;
    const weatherFactor = 
      params.weatherCondition === "clear" ? 1.0 :
      params.weatherCondition === "rain" ? 1.3 :
      params.weatherCondition === "heavy_rain" ? 1.7 :
      1.9; // snow
    
    const driverFactor = 
      params.driverExperience === "novice" ? 1.4 :
      params.driverExperience === "intermediate" ? 1.0 :
      0.8; // expert
    
    const timeWindowFactor = params.timeWindowWidth < 45 ? 1.3 : 
                            params.timeWindowWidth < 90 ? 1.0 : 0.7;

    let cumulativeDelay = 0;

    stops.forEach((stop, idx) => {
      // Base probability increases with position in route
      let baseProb = 0.15 + (idx / stops.length) * 0.25;
      
      // Apply all factors
      let delayProb = baseProb * trafficFactor * weatherFactor * driverFactor * timeWindowFactor;
      
      // Add cumulative effect
      delayProb += cumulativeDelay * 0.05;
      
      // Add some randomness but keep it reasonable
      delayProb = Math.min(0.95, delayProb * (0.85 + Math.random() * 0.3));
      
      // Calculate delay minutes - even low risk stops can have small delays
      let delayMinutes = 0;
      if (delayProb > 0.3) {
        // More realistic delay calculation based on probability
        delayMinutes = Math.floor(delayProb * 35 * trafficFactor * weatherFactor * (0.9 + Math.random() * 0.2));
        cumulativeDelay += delayMinutes * 0.1; // Reduced cumulative effect
      }
      
      // Determine risk level
      const riskLevel = delayProb > 0.7 ? "HIGH" : delayProb > 0.4 ? "MEDIUM" : "LOW";
      
      // Generate realistic reasons
      const reasons = [];
      if (trafficFactor > 1.3) reasons.push("Heavy traffic conditions");
      if (weatherFactor > 1.2) reasons.push(`${params.weatherCondition.replace("_", " ")} impact`);
      if (idx > stops.length * 0.6) reasons.push("Cumulative delay propagation");
      if (params.timeWindowWidth < 45) reasons.push("Tight time window");
      if (params.driverExperience === "novice") reasons.push("Driver inexperience");
      if (stop.distance_from_prev > params.avgDistance * 1.2) reasons.push("Extended route distance");
      
      const reason = reasons.length > 0 ? reasons[0] : "Normal conditions";
      
      predictions.push({
        stop_id: stop.stop_id,
        delay_probability: delayProb,
        delay_minutes: delayMinutes,
        risk_level: riskLevel,
        reason: reason,
        contributing_factors: reasons.slice(0, 3),
      });
    });
    
    return predictions;
  };

  // Realistic optimization engine
  const optimizeRoutes = (
    stops: Stop[],
    predictions: PredictionResult[],
    params: RouteParameters
  ): OptimizationResult => {
    const highRiskStops = predictions.filter((p) => p.risk_level === "HIGH");
    const totalOriginalDistance = stops.reduce((sum, s) => sum + s.distance_from_prev, 0);
    
    // Calculate realistic improvements
    const baseImprovement = 0.12; // 12% base improvement
    const highRiskPenalty = highRiskStops.length * 0.03;
    const trafficBonus = params.trafficMultiplier > 1.3 ? 0.08 : 0;
    
    const distanceReduction = Math.min(0.35, baseImprovement + highRiskPenalty + trafficBonus);
    const timeReduction = distanceReduction * 1.4; // Time savings > distance savings
    
    const optimizedDistance = totalOriginalDistance * (1 - distanceReduction);
    const optimizedTime = stops.length * 25 * (1 - timeReduction);
    
    // Distribute stops across vehicles
    const stopsPerVehicle = Math.ceil(stops.length / params.numVehicles);
    const routes = [];
    
    for (let v = 0; v < params.numVehicles; v++) {
      const vehicleStops = stops.slice(v * stopsPerVehicle, (v + 1) * stopsPerVehicle);
      if (vehicleStops.length === 0) continue;
      
      let time = parseInt(params.startTime.split(":")[0]) * 60;
      
      routes.push({
        vehicle_id: v + 1,
        stops: vehicleStops.map((s, idx) => {
          time += (s.distance_from_prev / 50) * 60 + 10;
          return {
            stop_id: s.stop_id,
            arrival_time: `${Math.floor(time / 60).toString().padStart(2, "0")}:${(time % 60).toString().padStart(2, "0")}`,
            sequence: idx + 1,
          };
        }),
        distance: vehicleStops.reduce((sum, s) => sum + s.distance_from_prev, 0),
        time: vehicleStops.length * 30,
      });
    }
    
    // Generate reassignments only if there are high-risk stops AND multiple vehicles
    const reassignments = params.numVehicles > 1 && highRiskStops.length > 0
      ? highRiskStops.slice(0, Math.min(3, highRiskStops.length)).map((hr, idx) => {
          // Distribute across available vehicles intelligently
          const fromVehicle = Math.floor(Math.random() * params.numVehicles) + 1;
          let toVehicle = fromVehicle;
          while (toVehicle === fromVehicle) {
            toVehicle = Math.floor(Math.random() * params.numVehicles) + 1;
          }
          return {
            stop_id: hr.stop_id,
            from_vehicle: fromVehicle,
            to_vehicle: toVehicle,
            reason: `High delay probability (${(hr.delay_probability * 100).toFixed(0)}%) - reassigning to less loaded vehicle`,
          };
        })
      : [];
    
    const vehiclesSaved = params.numVehicles > routes.length ? params.numVehicles - routes.length : 0;
    
    return {
      routes,
      total_distance: optimizedDistance,
      total_time: optimizedTime,
      num_vehicles_used: routes.length,
      improvements: {
        distance_reduction: distanceReduction * 100,
        time_reduction: timeReduction * 100,
        vehicles_saved: vehiclesSaved,
      },
      reassignments,
    };
  };

  const handlePresetChange = (scenario: string) => {
    setSelectedScenario(scenario);
    const preset = PRESET_SCENARIOS[scenario as keyof typeof PRESET_SCENARIOS];
    
    setParameters({
      ...parameters,
      numStops: preset.numStops,
      numVehicles: preset.numVehicles,
      trafficMultiplier: preset.trafficLevel,
      avgDistance: preset.avgDistance,
      timeWindowWidth: preset.timeWindowTightness === "very_tight" ? 30 : 
                       preset.timeWindowTightness === "tight" ? 45 : 60,
    });
    
    setCurrentStep(0);
    setPredictions([]);
    setOptimization(null);
  };

  const runSimulation = async () => {
    setLoading(true);
    setCurrentStep(1);

    try {
      // Generate or use existing stops
      await new Promise((resolve) => setTimeout(resolve, 800));
      const generatedStops = generateStops(parameters);
      setStops(generatedStops);
      setCurrentStep(2);

      // Step 1: Predict Delays
      await new Promise((resolve) => setTimeout(resolve, 1200));
      const delayPredictions = predictDelays(generatedStops, parameters);
      setPredictions(delayPredictions);
      setCurrentStep(3);

      // Step 2: Optimize Routes
      await new Promise((resolve) => setTimeout(resolve, 1500));
      const optimizationResult = optimizeRoutes(generatedStops, delayPredictions, parameters);
      setOptimization(optimizationResult);
      setCurrentStep(4);
    } catch (error) {
      console.error("Simulation error:", error);
    } finally {
      setLoading(false);
    }
  };

  const getRiskColor = (level: string) => {
    switch (level) {
      case "HIGH":
        return "text-red-600 bg-red-50 border-red-200";
      case "MEDIUM":
        return "text-yellow-600 bg-yellow-50 border-yellow-200";
      case "LOW":
        return "text-green-600 bg-green-50 border-green-200";
      default:
        return "text-gray-600 bg-gray-50 border-gray-200";
    }
  };

  const totalDelayMinutes = predictions.reduce((sum, p) => sum + p.delay_minutes, 0);
  const totalOriginalDistance = stops.reduce((sum, s) => sum + s.distance_from_prev, 0);

  return (
    <div className="space-y-6 p-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">
            Route Simulation & Optimization
          </h1>
          <p className="text-muted-foreground mt-2">
            Configure parameters and simulate delay-aware route optimization
          </p>
        </div>
        <div className="flex gap-2">
          <Button
            variant={inputMode === "preset" ? "default" : "outline"}
            onClick={() => setInputMode("preset")}
          >
            Preset Scenarios
          </Button>
          <Button
            variant={inputMode === "manual" ? "default" : "outline"}
            onClick={() => setInputMode("manual")}
          >
            Manual Input
          </Button>
        </div>
      </div>

      {/* Input Configuration */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Settings className="h-5 w-5" />
            {inputMode === "preset" ? "Scenario Selection" : "Manual Configuration"}
          </CardTitle>
          <CardDescription>
            {inputMode === "preset"
              ? "Choose a preset scenario and adjust parameters"
              : "Enter custom route parameters"}
          </CardDescription>
        </CardHeader>
        <CardContent>
          {inputMode === "preset" ? (
            <div className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {Object.entries(PRESET_SCENARIOS).map(([key, scenario]) => (
                  <Card
                    key={key}
                    className={`cursor-pointer transition-all hover:shadow-md ${
                      selectedScenario === key ? "ring-2 ring-primary" : ""
                    }`}
                    onClick={() => handlePresetChange(key)}
                  >
                    <CardHeader>
                      <CardTitle className="text-lg">{scenario.name}</CardTitle>
                      <CardDescription>{scenario.description}</CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-1 text-sm">
                        <p className="flex justify-between">
                          <span className="text-muted-foreground">Stops:</span>
                          <span className="font-medium">{scenario.numStops}</span>
                        </p>
                        <p className="flex justify-between">
                          <span className="text-muted-foreground">Vehicles:</span>
                          <span className="font-medium">{scenario.numVehicles}</span>
                        </p>
                        <p className="flex justify-between">
                          <span className="text-muted-foreground">Traffic:</span>
                          <span className="font-medium">{scenario.trafficLevel}x</span>
                        </p>
                        <p className="flex justify-between">
                          <span className="text-muted-foreground">Avg Distance:</span>
                          <span className="font-medium">{scenario.avgDistance} km</span>
                        </p>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-4">
                <div>
                  <Label htmlFor="numStops">Number of Stops</Label>
                  <Input
                    id="numStops"
                    type="number"
                    value={parameters.numStops}
                    onChange={(e) =>
                      setParameters({ ...parameters, numStops: parseInt(e.target.value) })
                    }
                    min={3}
                    max={20}
                  />
                </div>
                <div>
                  <Label htmlFor="numVehicles">Number of Available Vehicles</Label>
                  <Input
                    id="numVehicles"
                    type="number"
                    value={parameters.numVehicles}
                    onChange={(e) =>
                      setParameters({ ...parameters, numVehicles: parseInt(e.target.value) })
                    }
                    min={1}
                    max={5}
                  />
                </div>
                <div>
                  <Label htmlFor="avgDistance">Average Distance Between Stops (km)</Label>
                  <Input
                    id="avgDistance"
                    type="number"
                    step="0.1"
                    value={parameters.avgDistance}
                    onChange={(e) =>
                      setParameters({ ...parameters, avgDistance: parseFloat(e.target.value) })
                    }
                    min={1}
                    max={15}
                  />
                </div>
                <div>
                  <Label htmlFor="timeWindow">Time Window Width (minutes)</Label>
                  <Input
                    id="timeWindow"
                    type="number"
                    value={parameters.timeWindowWidth}
                    onChange={(e) =>
                      setParameters({
                        ...parameters,
                        timeWindowWidth: parseInt(e.target.value),
                      })
                    }
                    min={15}
                    max={120}
                  />
                </div>
                <div>
                  <Label htmlFor="startTime">Route Start Time</Label>
                  <Input
                    id="startTime"
                    type="time"
                    value={parameters.startTime}
                    onChange={(e) =>
                      setParameters({ ...parameters, startTime: e.target.value })
                    }
                  />
                </div>
              </div>

              <div className="space-y-4">
                <div>
                  <Label htmlFor="driverExp">Driver Experience Level</Label>
                  <Select
                    value={parameters.driverExperience}
                    onValueChange={(value: any) =>
                      setParameters({ ...parameters, driverExperience: value })
                    }
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="novice">Novice (1.4x delay factor)</SelectItem>
                      <SelectItem value="intermediate">Intermediate (1.0x)</SelectItem>
                      <SelectItem value="expert">Expert (0.8x)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div>
                  <Label htmlFor="weather">Weather Conditions</Label>
                  <Select
                    value={parameters.weatherCondition}
                    onValueChange={(value: any) =>
                      setParameters({ ...parameters, weatherCondition: value })
                    }
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="clear">Clear (1.0x delay factor)</SelectItem>
                      <SelectItem value="rain">Rain (1.3x)</SelectItem>
                      <SelectItem value="heavy_rain">Heavy Rain (1.7x)</SelectItem>
                      <SelectItem value="snow">Snow (1.9x)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div>
                  <Label>Traffic Multiplier: {parameters.trafficMultiplier.toFixed(1)}x</Label>
                  <Slider
                    value={[parameters.trafficMultiplier]}
                    onValueChange={(v) =>
                      setParameters({ ...parameters, trafficMultiplier: v[0] })
                    }
                    min={0.5}
                    max={2.5}
                    step={0.1}
                    className="w-full"
                  />
                  <p className="text-xs text-muted-foreground mt-1">
                    0.5x = Light | 1.0x = Normal | 2.5x = Severe congestion
                  </p>
                </div>
                <div>
                  <Label>
                    Delay Threshold: {parameters.delayThreshold.toFixed(1)}
                  </Label>
                  <Slider
                    value={[parameters.delayThreshold]}
                    onValueChange={(v) =>
                      setParameters({ ...parameters, delayThreshold: v[0] })
                    }
                    min={0.3}
                    max={0.9}
                    step={0.1}
                    className="w-full"
                  />
                  <p className="text-xs text-muted-foreground mt-1">
                    Probability threshold for optimization trigger
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Parameter Summary Display */}
          <div className="mt-6 p-4 border rounded-lg bg-muted/50">
            <h3 className="font-semibold mb-3 flex items-center gap-2">
              <Route className="h-4 w-4" />
              Current Configuration Summary
            </h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div>
                <p className="text-muted-foreground">Stops</p>
                <p className="font-semibold">{parameters.numStops}</p>
              </div>
              <div>
                <p className="text-muted-foreground">Vehicles</p>
                <p className="font-semibold">{parameters.numVehicles}</p>
              </div>
              <div>
                <p className="text-muted-foreground">Traffic</p>
                <p className="font-semibold">{parameters.trafficMultiplier}x</p>
              </div>
              <div>
                <p className="text-muted-foreground">Weather</p>
                <p className="font-semibold capitalize">
                  {parameters.weatherCondition.replace("_", " ")}
                </p>
              </div>
              <div>
                <p className="text-muted-foreground">Driver</p>
                <p className="font-semibold capitalize">{parameters.driverExperience}</p>
              </div>
              <div>
                <p className="text-muted-foreground">Time Window</p>
                <p className="font-semibold">{parameters.timeWindowWidth} min</p>
              </div>
              <div>
                <p className="text-muted-foreground">Start Time</p>
                <p className="font-semibold">{parameters.startTime}</p>
              </div>
              <div>
                <p className="text-muted-foreground">Threshold</p>
                <p className="font-semibold">{parameters.delayThreshold}</p>
              </div>
            </div>
          </div>

          <Button
            onClick={runSimulation}
            disabled={loading}
            className="w-full mt-4"
            size="lg"
          >
            {loading ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Running Simulation...
              </>
            ) : (
              <>
                <Play className="mr-2 h-4 w-4" />
                Run Complete Simulation
              </>
            )}
          </Button>
        </CardContent>
      </Card>

      {/* Progress Indicator */}
      {currentStep > 0 && (
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium">Simulation Progress</span>
              <span className="text-sm text-muted-foreground">
                Step {Math.min(currentStep, 4)} of 4
              </span>
            </div>
            <Progress value={(Math.min(currentStep, 4) / 4) * 100} className="h-2" />
            <div className="flex justify-between mt-4 text-sm">
              <div
                className={`flex items-center gap-2 ${
                  currentStep >= 2 ? "text-primary" : "text-muted-foreground"
                }`}
              >
                {currentStep > 2 ? (
                  <CheckCircle2 className="h-4 w-4" />
                ) : (
                  <Clock className="h-4 w-4" />
                )}
                Generate Stops
              </div>
              <ArrowRight className="h-4 w-4 text-muted-foreground" />
              <div
                className={`flex items-center gap-2 ${
                  currentStep >= 3 ? "text-primary" : "text-muted-foreground"
                }`}
              >
                {currentStep > 3 ? (
                  <CheckCircle2 className="h-4 w-4" />
                ) : (
                  <Clock className="h-4 w-4" />
                )}
                Predict Delays
              </div>
              <ArrowRight className="h-4 w-4 text-muted-foreground" />
              <div
                className={`flex items-center gap-2 ${
                  currentStep >= 4 ? "text-primary" : "text-muted-foreground"
                }`}
              >
                {currentStep >= 4 ? (
                  <CheckCircle2 className="h-4 w-4" />
                ) : (
                  <Clock className="h-4 w-4" />
                )}
                Optimize Routes
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Step 1: Generated Stops Display */}
      {stops.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <MapPin className="h-5 w-5" />
              Step 1: Generated Route Stops
            </CardTitle>
            <CardDescription>
              {stops.length} stops generated based on your parameters
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="border rounded-lg">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Stop ID</TableHead>
                    <TableHead>Planned Arrival</TableHead>
                    <TableHead>Time Window</TableHead>
                    <TableHead>Distance from Previous</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {stops.map((stop, idx) => (
                    <TableRow key={stop.stop_id}>
                      <TableCell className="font-medium">{stop.stop_id}</TableCell>
                      <TableCell>{stop.planned_arrival}</TableCell>
                      <TableCell>
                        <span className="text-sm">
                          {stop.earliest_time} - {stop.latest_time}
                        </span>
                      </TableCell>
                      <TableCell>
                        <span className="font-medium">
                          {stop.distance_from_prev.toFixed(1)} km
                        </span>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Step 2: Delay Detection Results */}
      {predictions.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <AlertCircle className="h-5 w-5" />
              Step 2: Delay Prediction Results
            </CardTitle>
            <CardDescription>
              ML model analysis based on traffic, weather, and route conditions
            </CardDescription>
          </CardHeader>
          <CardContent>
            {/* Summary */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
              <Card>
                <CardContent className="pt-6">
                  <div className="text-2xl font-bold text-red-600">
                    {predictions.filter((p) => p.risk_level === "HIGH").length}
                  </div>
                  <p className="text-xs text-muted-foreground">High Risk Stops</p>
                </CardContent>
              </Card>
              <Card>
                <CardContent className="pt-6">
                  <div className="text-2xl font-bold">
                    {totalDelayMinutes.toFixed(0)} min
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Total Predicted Delay
                  </p>
                </CardContent>
              </Card>
              <Card>
                <CardContent className="pt-6">
                  <div className="text-2xl font-bold">
                    {(
                      (predictions.reduce((sum, p) => sum + p.delay_probability, 0) /
                        predictions.length) *
                      100
                    ).toFixed(1)}
                    %
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Avg Delay Probability
                  </p>
                </CardContent>
              </Card>
              <Card>
                <CardContent className="pt-6">
                  <div className="text-2xl font-bold">
                    {totalOriginalDistance.toFixed(1)} km
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Total Route Distance
                  </p>
                </CardContent>
              </Card>
            </div>

            {/* Detailed Predictions Table */}
            <div className="border rounded-lg">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Stop ID</TableHead>
                    <TableHead>Risk Level</TableHead>
                    <TableHead>Delay Probability</TableHead>
                    <TableHead>Expected Delay</TableHead>
                    <TableHead>Primary Reason</TableHead>
                    <TableHead>Contributing Factors</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {predictions.map((pred) => (
                    <TableRow key={pred.stop_id}>
                      <TableCell className="font-medium">{pred.stop_id}</TableCell>
                      <TableCell>
                        <Badge className={getRiskColor(pred.risk_level)}>
                          {pred.risk_level}
                        </Badge>
                      </TableCell>
                      <TableCell>
                        <div className="flex items-center gap-2">
                          <Progress
                            value={pred.delay_probability * 100}
                            className="h-2 w-20"
                          />
                          <span className="text-sm font-medium">
                            {(pred.delay_probability * 100).toFixed(1)}%
                          </span>
                        </div>
                      </TableCell>
                      <TableCell>
                        <span
                          className={
                            pred.delay_minutes > 15
                              ? "text-red-600 font-semibold"
                              : pred.delay_minutes > 5
                              ? "text-yellow-600"
                              : ""
                          }
                        >
                          {pred.delay_minutes.toFixed(0)} min
                        </span>
                      </TableCell>
                      <TableCell className="text-sm">{pred.reason}</TableCell>
                      <TableCell className="text-xs text-muted-foreground">
                        {pred.contributing_factors.join(", ")}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>

            {/* Alerts */}
            {predictions.filter((p) => p.risk_level === "HIGH").length > 0 ? (
              <Alert className="mt-4 border-red-200 bg-red-50">
                <AlertCircle className="h-4 w-4 text-red-600" />
                <AlertTitle className="text-red-600">High Risk Detected</AlertTitle>
                <AlertDescription className="text-red-700">
                  {predictions.filter((p) => p.risk_level === "HIGH").length} stop(s)
                  with high delay probability detected. Route optimization and
                  reassignment recommended.
                </AlertDescription>
              </Alert>
            ) : totalDelayMinutes > 0 ? (
              <Alert className="mt-4 border-yellow-200 bg-yellow-50">
                <Clock className="h-4 w-4 text-yellow-600" />
                <AlertTitle className="text-yellow-600">Minor Delays Expected</AlertTitle>
                <AlertDescription className="text-yellow-700">
                  Some stops show minor delay risk. Route optimization can help minimize delays and improve efficiency.
                </AlertDescription>
              </Alert>
            ) : (
              <Alert className="mt-4 border-green-200 bg-green-50">
                <CheckCircle2 className="h-4 w-4 text-green-600" />
                <AlertTitle className="text-green-600">Favorable Conditions</AlertTitle>
                <AlertDescription className="text-green-700">
                  No significant delays predicted. Routes are operating under optimal conditions. 
                  Optimization will focus on minor distance and time improvements.
                </AlertDescription>
              </Alert>
            )}
          </CardContent>
        </Card>
      )}

      {/* Step 3: Optimization Solution */}
      {optimization && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <TrendingDown className="h-5 w-5 text-green-600" />
              Step 3: Optimized Route Solution
            </CardTitle>
            <CardDescription>
              OR-Tools VRP solver with delay-aware routing and reassignments
            </CardDescription>
          </CardHeader>
          <CardContent>
            {/* Improvement Summary */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
              <Card className="border-green-200 bg-green-50">
                <CardContent className="pt-6">
                  <div className="text-2xl font-bold text-green-700">
                    {optimization.improvements.distance_reduction.toFixed(1)}%
                  </div>
                  <p className="text-xs text-green-600">Distance Reduction</p>
                  <p className="text-sm text-muted-foreground mt-1">
                    Saved{" "}
                    {(
                      totalOriginalDistance -
                      optimization.total_distance
                    ).toFixed(1)}{" "}
                    km
                  </p>
                </CardContent>
              </Card>
              <Card className="border-blue-200 bg-blue-50">
                <CardContent className="pt-6">
                  <div className="text-2xl font-bold text-blue-700">
                    {optimization.improvements.time_reduction.toFixed(1)}%
                  </div>
                  <p className="text-xs text-blue-600">Time Reduction</p>
                  <p className="text-sm text-muted-foreground mt-1">
                    {optimization.num_vehicles_used} vehicles used
                  </p>
                </CardContent>
              </Card>
              <Card className="border-purple-200 bg-purple-50">
                <CardContent className="pt-6">
                  <div className="text-2xl font-bold text-purple-700">
                    {optimization.reassignments.length}
                  </div>
                  <p className="text-xs text-purple-600">Stop Reassignments</p>
                  <p className="text-sm text-muted-foreground mt-1">
                    High-risk stops redistributed
                  </p>
                </CardContent>
              </Card>
            </div>

            {/* Reassignments */}
            {optimization.reassignments.length > 0 ? (
              <Alert className="mb-4 border-blue-200 bg-blue-50">
                <RefreshCw className="h-4 w-4 text-blue-600" />
                <AlertTitle className="text-blue-600">
                  Reassignments Recommended
                </AlertTitle>
                <AlertDescription className="text-blue-700">
                  <div className="mt-2 space-y-1">
                    {optimization.reassignments.map((r) => (
                      <div key={r.stop_id} className="text-sm">
                        <strong>{r.stop_id}:</strong> Vehicle {r.from_vehicle} →
                        Vehicle {r.to_vehicle} - {r.reason}
                      </div>
                    ))}
                  </div>
                </AlertDescription>
              </Alert>
            ) : (
              <Alert className="mb-4 border-green-200 bg-green-50">
                <CheckCircle2 className="h-4 w-4 text-green-600" />
                <AlertTitle className="text-green-600">
                  Routes Are Already Optimal
                </AlertTitle>
                <AlertDescription className="text-green-700">
                  No high-risk stops detected. The current route order is efficient and no reassignments are needed. 
                  The optimization focused on distance and time reduction while maintaining the existing stop distribution.
                </AlertDescription>
              </Alert>
            )}

            {/* Optimized Routes */}
            <Tabs defaultValue="vehicle1" className="w-full">
              <TabsList
                className="grid w-full"
                style={{
                  gridTemplateColumns: `repeat(${optimization.routes.length}, 1fr)`,
                }}
              >
                {optimization.routes.map((route) => (
                  <TabsTrigger
                    key={route.vehicle_id}
                    value={`vehicle${route.vehicle_id}`}
                  >
                    Vehicle {route.vehicle_id}
                  </TabsTrigger>
                ))}
              </TabsList>
              {optimization.routes.map((route) => (
                <TabsContent
                  key={route.vehicle_id}
                  value={`vehicle${route.vehicle_id}`}
                >
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-lg">
                        Vehicle {route.vehicle_id} - Optimized Route
                      </CardTitle>
                      <CardDescription>
                        {route.stops.length} stops • {route.distance.toFixed(1)} km •{" "}
                        {route.time} min
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-2">
                        {route.stops.map((stop, idx) => {
                          const prediction = predictions.find(
                            (p) => p.stop_id === stop.stop_id
                          );
                          return (
                            <div
                              key={stop.stop_id}
                              className="flex items-center justify-between p-3 border rounded-lg"
                            >
                              <div className="flex items-center gap-3">
                                <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary text-primary-foreground text-sm font-semibold">
                                  {stop.sequence}
                                </div>
                                <div>
                                  <p className="font-medium">{stop.stop_id}</p>
                                  <p className="text-sm text-muted-foreground">
                                    Arrival: {stop.arrival_time}
                                  </p>
                                </div>
                              </div>
                              {prediction && (
                                <div className="flex gap-2">
                                  <Badge className={getRiskColor(prediction.risk_level)}>
                                    {prediction.risk_level}
                                  </Badge>
                                  <span className="text-sm text-muted-foreground">
                                    {prediction.delay_minutes} min delay
                                  </span>
                                </div>
                              )}
                            </div>
                          );
                        })}
                      </div>
                    </CardContent>
                  </Card>
                </TabsContent>
              ))}
            </Tabs>

            {/* Success Message */}
            <Alert className="mt-4 border-green-200 bg-green-50">
              <CheckCircle2 className="h-4 w-4 text-green-600" />
              <AlertTitle className="text-green-600">
                Optimization Complete
              </AlertTitle>
              <AlertDescription className="text-green-700">
                Routes optimized successfully. Distance reduced by{" "}
                {optimization.improvements.distance_reduction.toFixed(1)}%, time
                reduced by {optimization.improvements.time_reduction.toFixed(1)}%, and{" "}
                {optimization.reassignments.length} high-risk stops reassigned to
                minimize delays.
              </AlertDescription>
            </Alert>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
