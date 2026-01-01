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
import { Play, CheckCircle2, AlertCircle, Clock } from "lucide-react";

interface StopInput {
  route_id: number;
  driver_id: string;
  stop_id: string;
  address_id: number;
  week_id: number;
  country: string;
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

interface PredictionResult {
  stop_id: string;
  delay_probability: number;
  delay_minutes: number;
  risk_level: "HIGH" | "MEDIUM" | "LOW";
  on_time: boolean;
}

interface OptimizationResult {
  old_sequence: number[];
  new_sequence: number[];
  distance_saved: number;
  time_saved: number;
  delay_reduction: number;
}

export default function OptimizationPage() {
  const [stops, setStops] = useState<StopInput[]>([
    {
      route_id: 1,
      driver_id: "D0001",
      stop_id: "S1_0",
      address_id: 0,
      week_id: 1,
      country: "Netherlands",
      day_of_week: "Monday",
      indexp: 0,
      indexa: 0,
      arrived_time: "08:00:00",
      earliest_time: "07:30:00",
      latest_time: "08:30:00",
      distancep: 0,
      distancea: 0,
      depot: 1,
      delivery: 0,
    },
  ]);

  const [predictions, setPredictions] = useState<PredictionResult[]>([]);
  const [optimization, setOptimization] = useState<OptimizationResult | null>(
    null
  );
  const [loading, setLoading] = useState(false);

  const addStop = () => {
    const newStop: StopInput = {
      route_id: stops[0]?.route_id || 1,
      driver_id: stops[0]?.driver_id || "D0001",
      stop_id: `S${stops[0]?.route_id || 1}_${stops.length}`,
      address_id: stops.length,
      week_id: stops[0]?.week_id || 1,
      country: stops[0]?.country || "Netherlands",
      day_of_week: stops[0]?.day_of_week || "Monday",
      indexp: stops.length,
      indexa: stops.length,
      arrived_time: "09:00:00",
      earliest_time: "08:30:00",
      latest_time: "09:30:00",
      distancep: stops.length * 10,
      distancea: stops.length * 10,
      depot: 0,
      delivery: 1,
    };
    setStops([...stops, newStop]);
  };

  const removeStop = (index: number) => {
    setStops(stops.filter((_, i) => i !== index));
  };

  const updateStop = (index: number, field: keyof StopInput, value: any) => {
    const updated = [...stops];
    updated[index] = { ...updated[index], [field]: value };
    setStops(updated);
  };

  const handleRunOptimization = async () => {
    setLoading(true);
    try {
      // Convert stops to API format
      const stopsForAPI = stops.map((stop) => ({
        route_id: stop.route_id,
        driver_id: stop.driver_id,
        stop_id: stop.stop_id,
        address_id: stop.address_id,
        week_id: stop.week_id,
        country: stop.country,
        day_of_week: stop.day_of_week,
        indexp: stop.indexp,
        indexa: stop.indexa,
        arrived_time: stop.arrived_time,
        earliest_time: stop.earliest_time,
        latest_time: stop.latest_time,
        distancep: stop.distancep,
        distancea: stop.distancea,
        depot: stop.depot,
        delivery: stop.delivery,
      }));

      // Call API
      const response = await fetch(
        "http://localhost:8000/simulation/full-analysis",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            route_id: stops[0].route_id.toString(),
            stops: stopsForAPI,
            scenario_type: "normal",
            custom_params: {},
          }),
        }
      );

      if (!response.ok) {
        throw new Error(`API error: ${response.statusText}`);
      }

      const data = await response.json();

      // Transform predictions
      const preds: PredictionResult[] = data.predictions.predictions.map(
        (p: any) => ({
          stop_id: p.stop_id.toString(),
          delay_probability: p.delay_probability || 0,
          delay_minutes: Math.max(0, p.delay_minutes || 0),
          risk_level:
            p.risk_level === "HIGH"
              ? "HIGH"
              : p.risk_level === "MEDIUM"
              ? "MEDIUM"
              : "LOW",
          on_time: (p.delay_probability || 0) < 0.5,
        })
      );
      setPredictions(preds);

      // Transform optimization
      if (data.optimizations) {
        setOptimization({
          old_sequence: data.optimizations.old_sequence || [],
          new_sequence: data.optimizations.new_sequence || [],
          distance_saved: data.optimizations.impact?.energy_saving_km || 0,
          time_saved: data.optimizations.impact?.delay_reduction_minutes || 0,
          delay_reduction:
            data.optimizations.impact?.delay_reduction_minutes || 0,
        });
      }
    } catch (error) {
      console.error("Error:", error);
      alert(`Error: ${error instanceof Error ? error.message : "Failed"}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">
          Route Optimization
        </h1>
        <p className="text-muted-foreground">
          Enter route parameters and run optimization
        </p>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        {/* Input Form */}
        <Card>
          <CardHeader>
            <CardTitle>Route Parameters</CardTitle>
            <CardDescription>Enter stop details</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {stops.map((stop, index) => (
              <div key={index} className="border rounded-lg p-4 space-y-3">
                <div className="flex items-center justify-between">
                  <h4 className="font-medium">Stop {index + 1}</h4>
                  {stops.length > 1 && (
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => removeStop(index)}
                    >
                      Remove
                    </Button>
                  )}
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <Label>Route ID</Label>
                    <Input
                      type="number"
                      value={stop.route_id}
                      onChange={(e) =>
                        updateStop(index, "route_id", parseInt(e.target.value))
                      }
                    />
                  </div>
                  <div>
                    <Label>Driver ID</Label>
                    <Input
                      value={stop.driver_id}
                      onChange={(e) =>
                        updateStop(index, "driver_id", e.target.value)
                      }
                    />
                  </div>
                  <div>
                    <Label>Stop ID</Label>
                    <Input
                      value={stop.stop_id}
                      onChange={(e) =>
                        updateStop(index, "stop_id", e.target.value)
                      }
                    />
                  </div>
                  <div>
                    <Label>Address ID</Label>
                    <Input
                      type="number"
                      value={stop.address_id}
                      onChange={(e) =>
                        updateStop(
                          index,
                          "address_id",
                          parseInt(e.target.value)
                        )
                      }
                    />
                  </div>
                  <div>
                    <Label>Week ID</Label>
                    <Input
                      type="number"
                      value={stop.week_id}
                      onChange={(e) =>
                        updateStop(index, "week_id", parseInt(e.target.value))
                      }
                    />
                  </div>
                  <div>
                    <Label>Country</Label>
                    <Select
                      value={stop.country}
                      onValueChange={(value) =>
                        updateStop(index, "country", value)
                      }
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="Netherlands">Netherlands</SelectItem>
                        <SelectItem value="Spain">Spain</SelectItem>
                        <SelectItem value="Italy">Italy</SelectItem>
                        <SelectItem value="Germany">Germany</SelectItem>
                        <SelectItem value="UK">UK</SelectItem>
                        <SelectItem value="France">France</SelectItem>
                        <SelectItem value="USA">USA</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div>
                    <Label>Day of Week</Label>
                    <Select
                      value={stop.day_of_week}
                      onValueChange={(value) =>
                        updateStop(index, "day_of_week", value)
                      }
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="Monday">Monday</SelectItem>
                        <SelectItem value="Tuesday">Tuesday</SelectItem>
                        <SelectItem value="Wednesday">Wednesday</SelectItem>
                        <SelectItem value="Thursday">Thursday</SelectItem>
                        <SelectItem value="Friday">Friday</SelectItem>
                        <SelectItem value="Saturday">Saturday</SelectItem>
                        <SelectItem value="Sunday">Sunday</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div>
                    <Label>IndexP</Label>
                    <Input
                      type="number"
                      value={stop.indexp}
                      onChange={(e) =>
                        updateStop(index, "indexp", parseInt(e.target.value))
                      }
                    />
                  </div>
                  <div>
                    <Label>IndexA</Label>
                    <Input
                      type="number"
                      value={stop.indexa}
                      onChange={(e) =>
                        updateStop(index, "indexa", parseInt(e.target.value))
                      }
                    />
                  </div>
                  <div>
                    <Label>Arrived Time</Label>
                    <Input
                      type="time"
                      value={stop.arrived_time.substring(0, 5)}
                      onChange={(e) =>
                        updateStop(
                          index,
                          "arrived_time",
                          e.target.value + ":00"
                        )
                      }
                    />
                  </div>
                  <div>
                    <Label>Earliest Time</Label>
                    <Input
                      type="time"
                      value={stop.earliest_time.substring(0, 5)}
                      onChange={(e) =>
                        updateStop(
                          index,
                          "earliest_time",
                          e.target.value + ":00"
                        )
                      }
                    />
                  </div>
                  <div>
                    <Label>Latest Time</Label>
                    <Input
                      type="time"
                      value={stop.latest_time.substring(0, 5)}
                      onChange={(e) =>
                        updateStop(index, "latest_time", e.target.value + ":00")
                      }
                    />
                  </div>
                  <div>
                    <Label>DistanceP (km)</Label>
                    <Input
                      type="number"
                      step="0.1"
                      value={stop.distancep}
                      onChange={(e) =>
                        updateStop(
                          index,
                          "distancep",
                          parseFloat(e.target.value)
                        )
                      }
                    />
                  </div>
                  <div>
                    <Label>DistanceA (km)</Label>
                    <Input
                      type="number"
                      step="0.1"
                      value={stop.distancea}
                      onChange={(e) =>
                        updateStop(
                          index,
                          "distancea",
                          parseFloat(e.target.value)
                        )
                      }
                    />
                  </div>
                  <div>
                    <Label>Depot</Label>
                    <Select
                      value={stop.depot.toString()}
                      onValueChange={(value) =>
                        updateStop(index, "depot", parseInt(value))
                      }
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="0">No</SelectItem>
                        <SelectItem value="1">Yes</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div>
                    <Label>Delivery</Label>
                    <Select
                      value={stop.delivery.toString()}
                      onValueChange={(value) =>
                        updateStop(index, "delivery", parseInt(value))
                      }
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="0">No</SelectItem>
                        <SelectItem value="1">Yes</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
              </div>
            ))}

            <Button onClick={addStop} variant="outline" className="w-full">
              Add Stop
            </Button>

            <Button
              onClick={handleRunOptimization}
              disabled={loading || stops.length === 0}
              className="w-full"
              size="lg"
            >
              {loading ? (
                <>
                  <Clock className="mr-2 h-4 w-4 animate-spin" />
                  Running...
                </>
              ) : (
                <>
                  <Play className="mr-2 h-4 w-4" />
                  Run Optimization
                </>
              )}
            </Button>
          </CardContent>
        </Card>

        {/* Results */}
        <Card>
          <CardHeader>
            <CardTitle>Results</CardTitle>
            <CardDescription>
              Delay predictions and optimization results
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {predictions.length === 0 ? (
              <div className="text-center py-12 text-muted-foreground">
                <AlertCircle className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <p>Run optimization to see results</p>
              </div>
            ) : (
              <>
                {/* Predictions */}
                <div>
                  <h3 className="font-semibold mb-3">Delay Predictions</h3>
                  <div className="space-y-2">
                    {predictions.map((pred, idx) => (
                      <div
                        key={idx}
                        className="flex items-center justify-between p-3 border rounded-lg"
                      >
                        <div className="flex items-center gap-3">
                          <span className="font-medium">{pred.stop_id}</span>
                          <Badge
                            variant={
                              pred.risk_level === "HIGH"
                                ? "destructive"
                                : pred.risk_level === "MEDIUM"
                                ? "secondary"
                                : "default"
                            }
                          >
                            {pred.risk_level}
                          </Badge>
                          {pred.on_time ? (
                            <CheckCircle2 className="h-4 w-4 text-green-500" />
                          ) : (
                            <AlertCircle className="h-4 w-4 text-red-500" />
                          )}
                        </div>
                        <div className="text-right">
                          <p className="text-sm font-medium">
                            {pred.delay_minutes.toFixed(1)} min
                          </p>
                          <p className="text-xs text-muted-foreground">
                            {(pred.delay_probability * 100).toFixed(0)}% prob
                          </p>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Optimization */}
                {optimization && (
                  <div>
                    <h3 className="font-semibold mb-3">Optimization Results</h3>
                    <div className="space-y-2">
                      <div className="p-3 border rounded-lg">
                        <p className="text-sm text-muted-foreground">
                          Old Sequence
                        </p>
                        <p className="font-medium">
                          {optimization.old_sequence.join(" → ")}
                        </p>
                      </div>
                      <div className="p-3 border rounded-lg">
                        <p className="text-sm text-muted-foreground">
                          New Sequence
                        </p>
                        <p className="font-medium">
                          {optimization.new_sequence.join(" → ")}
                        </p>
                      </div>
                      <div className="grid grid-cols-3 gap-2">
                        <div className="p-3 border rounded-lg text-center">
                          <p className="text-xs text-muted-foreground">
                            Distance Saved
                          </p>
                          <p className="text-lg font-bold">
                            {optimization.distance_saved.toFixed(1)} km
                          </p>
                        </div>
                        <div className="p-3 border rounded-lg text-center">
                          <p className="text-xs text-muted-foreground">
                            Time Saved
                          </p>
                          <p className="text-lg font-bold">
                            {optimization.time_saved.toFixed(1)} min
                          </p>
                        </div>
                        <div className="p-3 border rounded-lg text-center">
                          <p className="text-xs text-muted-foreground">
                            Delay Reduction
                          </p>
                          <p className="text-lg font-bold">
                            {optimization.delay_reduction.toFixed(1)} min
                          </p>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
