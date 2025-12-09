"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { Slider } from "@/components/ui/slider"
import { 
  Play, 
  AlertTriangle, 
  CloudRain, 
  Clock, 
  TrendingDown, 
  Shuffle,
  Settings,
  Sparkles
} from "lucide-react"
import { StopData, SimulationScenarioConfig } from "@/lib/api"
import { toast } from "sonner"

interface EnhancedScenarioConfiguratorProps {
  availableRoutes: Array<{ route_id: number; stop_count: number }>
  onRunSimulation: (config: SimulationScenarioConfig) => void
  loading?: boolean
}

const SCENARIO_TEMPLATES = [
  {
    id: "normal",
    name: "Normal Conditions",
    description: "Baseline scenario with optimal conditions",
    icon: Sparkles,
    color: "text-green-500",
    defaultParams: {}
  },
  {
    id: "traffic_congestion",
    name: "Traffic Congestion",
    description: "Heavy traffic conditions increasing travel time",
    icon: AlertTriangle,
    color: "text-orange-500",
    defaultParams: { traffic_factor: 1.5 }
  },
  {
    id: "delay_at_stop",
    name: "Stop Delay",
    description: "Unexpected delay at a specific delivery stop",
    icon: Clock,
    color: "text-red-500",
    defaultParams: { delay_minutes: 20 }
  },
  {
    id: "driver_slowdown",
    name: "Driver Slowdown",
    description: "Less experienced or slower driver performance",
    icon: TrendingDown,
    color: "text-yellow-500",
    defaultParams: { slowdown_factor: 1.3 }
  },
  {
    id: "increased_workload",
    name: "Increased Stop Workload",
    description: "More time required at each delivery stop",
    icon: CloudRain,
    color: "text-blue-500",
    defaultParams: { workload_increase: 10 }
  },
  {
    id: "random_faults",
    name: "Random Faults",
    description: "Random disruptions and unexpected events",
    icon: Shuffle,
    color: "text-purple-500",
    defaultParams: { num_faults: 3 }
  },
  {
    id: "custom",
    name: "Custom Scenario",
    description: "Define your own custom parameters",
    icon: Settings,
    color: "text-gray-500",
    defaultParams: {}
  }
]

export function EnhancedScenarioConfigurator({ 
  availableRoutes, 
  onRunSimulation, 
  loading = false 
}: EnhancedScenarioConfiguratorProps) {
  const [selectedRoute, setSelectedRoute] = useState<number | null>(null)
  const [selectedScenario, setSelectedScenario] = useState("normal")
  const [customParams, setCustomParams] = useState<any>({})
  const [routeStops, setRouteStops] = useState<StopData[]>([])

  const currentScenario = SCENARIO_TEMPLATES.find(s => s.id === selectedScenario)

  useEffect(() => {
    // Reset custom params when scenario changes
    if (currentScenario) {
      setCustomParams({ ...currentScenario.defaultParams })
    }
  }, [selectedScenario])

  const handleRunSimulation = async () => {
    if (!selectedRoute) {
      toast.error("Please select a route first")
      return
    }

    if (routeStops.length === 0) {
      toast.error("No stops loaded for selected route")
      return
    }

    const config: SimulationScenarioConfig = {
      route_id: selectedRoute.toString(),
      stops: routeStops,
      scenario_type: selectedScenario as any,
      custom_params: customParams
    }

    onRunSimulation(config)
  }

  const handleLoadRoute = async (routeId: number) => {
    setSelectedRoute(routeId)
    
    try {
      // In a real implementation, fetch stops for this specific route
      // For now, we'll use the getSampleData approach
      const { getSampleData } = await import("@/lib/api")
      const data = await getSampleData(100)
      
      // Filter stops for this route
      const stops = data.data.filter((stop: StopData) => stop.route_id === routeId)
      
      if (stops.length === 0) {
        toast.warning(`No stops found for route ${routeId}, loading sample data instead`)
        setRouteStops(data.data.slice(0, 20))
      } else {
        setRouteStops(stops)
        toast.success(`Loaded ${stops.length} stops for route ${routeId}`)
      }
    } catch (error) {
      console.error("Error loading route:", error)
      toast.error("Failed to load route data")
    }
  }

  return (
    <div className="space-y-6">
      {/* Route Selection */}
      <Card>
        <CardHeader>
          <CardTitle>1. Select Route</CardTitle>
          <CardDescription>
            Choose a route to simulate
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label>Route ID</Label>
            <Select
              value={selectedRoute?.toString() || ""}
              onValueChange={(value) => handleLoadRoute(parseInt(value))}
            >
              <SelectTrigger>
                <SelectValue placeholder="Select a route..." />
              </SelectTrigger>
              <SelectContent>
                {availableRoutes.slice(0, 20).map((route) => (
                  <SelectItem key={route.route_id} value={route.route_id.toString()}>
                    Route {route.route_id} ({route.stop_count} stops)
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {selectedRoute && (
            <div className="p-3 bg-muted rounded-lg">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium">Selected Route: {selectedRoute}</p>
                  <p className="text-xs text-muted-foreground">{routeStops.length} stops loaded</p>
                </div>
                <Badge variant="outline">{routeStops.length > 0 ? "Ready" : "Loading..."}</Badge>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Scenario Selection */}
      <Card>
        <CardHeader>
          <CardTitle>2. Select Simulation Condition</CardTitle>
          <CardDescription>
            Choose a predefined scenario or create a custom one
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-3">
            {SCENARIO_TEMPLATES.map((scenario) => {
              const Icon = scenario.icon
              const isSelected = selectedScenario === scenario.id
              
              return (
                <button
                  key={scenario.id}
                  onClick={() => setSelectedScenario(scenario.id)}
                  className={`p-4 border rounded-lg text-left transition-all hover:shadow-md ${
                    isSelected 
                      ? 'border-primary bg-primary/5 shadow-sm' 
                      : 'border-border hover:border-primary/50'
                  }`}
                >
                  <div className="flex items-start gap-3">
                    <Icon className={`h-5 w-5 mt-0.5 ${scenario.color}`} />
                    <div className="flex-1 min-w-0">
                      <p className="font-medium text-sm">{scenario.name}</p>
                      <p className="text-xs text-muted-foreground mt-1">
                        {scenario.description}
                      </p>
                    </div>
                  </div>
                </button>
              )
            })}
          </div>
        </CardContent>
      </Card>

      {/* Parameter Adjustment */}
      {currentScenario && (
        <Card>
          <CardHeader>
            <CardTitle>3. Adjust Parameters</CardTitle>
            <CardDescription>
              Fine-tune the simulation parameters for {currentScenario.name}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {selectedScenario === "traffic_congestion" && (
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <Label>Traffic Multiplier</Label>
                  <Badge variant="outline">{customParams.traffic_factor?.toFixed(1)}x</Badge>
                </div>
                <Slider
                  value={[customParams.traffic_factor || 1.5]}
                  onValueChange={(value) => setCustomParams({ ...customParams, traffic_factor: value[0] })}
                  min={1.0}
                  max={3.0}
                  step={0.1}
                />
                <p className="text-xs text-muted-foreground">
                  1.0 = Normal, 1.5 = Heavy traffic, 3.0 = Severe congestion
                </p>
              </div>
            )}

            {selectedScenario === "delay_at_stop" && (
              <div className="space-y-3">
                <div className="space-y-2">
                  <Label>Delay Duration (minutes)</Label>
                  <Input
                    type="number"
                    value={customParams.delay_minutes || 20}
                    onChange={(e) => setCustomParams({ ...customParams, delay_minutes: parseInt(e.target.value) })}
                    min={5}
                    max={120}
                  />
                </div>
                <div className="space-y-2">
                  <Label>Target Stop ID (optional)</Label>
                  <Input
                    type="number"
                    placeholder="Leave empty for random stop"
                    value={customParams.stop_id || ""}
                    onChange={(e) => setCustomParams({ ...customParams, stop_id: parseInt(e.target.value) || undefined })}
                  />
                </div>
              </div>
            )}

            {selectedScenario === "driver_slowdown" && (
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <Label>Slowdown Factor</Label>
                  <Badge variant="outline">{customParams.slowdown_factor?.toFixed(1)}x</Badge>
                </div>
                <Slider
                  value={[customParams.slowdown_factor || 1.3]}
                  onValueChange={(value) => setCustomParams({ ...customParams, slowdown_factor: value[0] })}
                  min={1.0}
                  max={2.0}
                  step={0.1}
                />
                <p className="text-xs text-muted-foreground">
                  1.0 = Experienced driver, 2.0 = Very slow/inexperienced
                </p>
              </div>
            )}

            {selectedScenario === "increased_workload" && (
              <div className="space-y-3">
                <Label>Additional Minutes Per Stop</Label>
                <Input
                  type="number"
                  value={customParams.workload_increase || 10}
                  onChange={(e) => setCustomParams({ ...customParams, workload_increase: parseInt(e.target.value) })}
                  min={0}
                  max={60}
                />
                <p className="text-xs text-muted-foreground">
                  Extra time needed at each stop due to complexity
                </p>
              </div>
            )}

            {selectedScenario === "random_faults" && (
              <div className="space-y-3">
                <Label>Number of Random Faults</Label>
                <Input
                  type="number"
                  value={customParams.num_faults || 3}
                  onChange={(e) => setCustomParams({ ...customParams, num_faults: parseInt(e.target.value) })}
                  min={1}
                  max={10}
                />
                <p className="text-xs text-muted-foreground">
                  Random delays, distance increases, and order swaps
                </p>
              </div>
            )}

            {selectedScenario === "custom" && (
              <div className="space-y-4">
                <div className="space-y-2">
                  <Label>Artificial Delay (minutes)</Label>
                  <Input
                    type="number"
                    value={customParams.artificial_delay_minutes || 0}
                    onChange={(e) => setCustomParams({ 
                      ...customParams, 
                      artificial_delay_minutes: parseInt(e.target.value) 
                    })}
                    min={0}
                    max={120}
                  />
                </div>
                
                <div className="space-y-2">
                  <Label>Distance Deviation (km)</Label>
                  <Input
                    type="number"
                    step="0.1"
                    value={customParams.distance_deviation || 0}
                    onChange={(e) => setCustomParams({ 
                      ...customParams, 
                      distance_deviation: parseFloat(e.target.value) 
                    })}
                  />
                </div>

                <div className="space-y-2">
                  <Label>Traffic Multiplier</Label>
                  <Input
                    type="number"
                    step="0.1"
                    value={customParams.traffic_factor || 1.0}
                    onChange={(e) => setCustomParams({ 
                      ...customParams, 
                      traffic_factor: parseFloat(e.target.value) 
                    })}
                    min={0.5}
                    max={3.0}
                  />
                </div>
              </div>
            )}

            {/* Run Simulation Button */}
            <div className="pt-4 border-t">
              <Button 
                onClick={handleRunSimulation}
                disabled={loading || !selectedRoute || routeStops.length === 0}
                className="w-full"
                size="lg"
              >
                {loading ? (
                  <>
                    <div className="mr-2 h-4 w-4 animate-spin rounded-full border-2 border-background border-t-transparent" />
                    Running Simulation...
                  </>
                ) : (
                  <>
                    <Play className="mr-2 h-4 w-4" />
                    Run Simulation
                  </>
                )}
              </Button>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}

