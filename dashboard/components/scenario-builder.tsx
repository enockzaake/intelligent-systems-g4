"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Badge } from "@/components/ui/badge"
import { Slider } from "@/components/ui/slider"
import { StopData, ScenarioModifications, ScenarioResult, simulateScenario } from "@/lib/api"
import { Play, RefreshCw, Trash2, Users, MapPin, Clock, TrendingUp } from "lucide-react"
import { toast } from "sonner"

interface ScenarioBuilderProps {
  initialStops: StopData[]
  onSimulationComplete?: (result: ScenarioResult) => void
}

export function ScenarioBuilder({ initialStops, onSimulationComplete }: ScenarioBuilderProps) {
  const [stops, setStops] = useState<StopData[]>(initialStops)
  const [modifications, setModifications] = useState<ScenarioModifications>({
    driver_reassignments: {},
    traffic_multiplier: 1.0,
    distance_multiplier: 1.0,
    time_shift_minutes: {},
    stops_to_remove: []
  })
  const [loading, setLoading] = useState(false)
  const [selectedStop, setSelectedStop] = useState<number | null>(null)

  useEffect(() => {
    setStops(initialStops)
  }, [initialStops])

  const handleDriverReassignment = (stopId: number, newDriverId: number) => {
    setModifications(prev => ({
      ...prev,
      driver_reassignments: {
        ...prev.driver_reassignments,
        [stopId]: newDriverId
      }
    }))
    toast.success(`Stop ${stopId} reassigned to Driver ${newDriverId}`)
  }

  const handleTimeShift = (stopId: number, minutes: number) => {
    setModifications(prev => ({
      ...prev,
      time_shift_minutes: {
        ...prev.time_shift_minutes,
        [stopId]: minutes
      }
    }))
    toast.info(`Stop ${stopId} time shifted by ${minutes} minutes`)
  }

  const handleRemoveStop = (stopId: number) => {
    setModifications(prev => ({
      ...prev,
      stops_to_remove: [...(prev.stops_to_remove || []), stopId]
    }))
    toast.warning(`Stop ${stopId} marked for removal`)
  }

  const handleTrafficMultiplier = (value: number[]) => {
    setModifications(prev => ({
      ...prev,
      traffic_multiplier: value[0]
    }))
  }

  const handleDistanceMultiplier = (value: number[]) => {
    setModifications(prev => ({
      ...prev,
      distance_multiplier: value[0]
    }))
  }

  const handleRunSimulation = async () => {
    setLoading(true)
    try {
      toast.info("Running scenario simulation...")
      const result = await simulateScenario(stops, modifications)
      onSimulationComplete?.(result)
      toast.success("Scenario simulation complete!")
    } catch (error) {
      console.error("Simulation error:", error)
      toast.error(`Simulation failed: ${error instanceof Error ? error.message : 'Unknown error'}`)
    } finally {
      setLoading(false)
    }
  }

  const handleReset = () => {
    setModifications({
      driver_reassignments: {},
      traffic_multiplier: 1.0,
      distance_multiplier: 1.0,
      time_shift_minutes: {},
      stops_to_remove: []
    })
    setSelectedStop(null)
    toast.info("Modifications reset")
  }

  const modificationCount = 
    Object.keys(modifications.driver_reassignments || {}).length +
    Object.keys(modifications.time_shift_minutes || {}).length +
    (modifications.stops_to_remove?.length || 0) +
    (modifications.traffic_multiplier !== 1.0 ? 1 : 0) +
    (modifications.distance_multiplier !== 1.0 ? 1 : 0)

  const availableDrivers = Array.from(new Set(stops.map(s => s.driver_id)))

  return (
    <div className="space-y-6">
      {/* Global Modifications */}
      <Card>
        <CardHeader>
          <CardTitle>Global Scenario Parameters</CardTitle>
          <CardDescription>
            Adjust these parameters to simulate different conditions
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="space-y-4">
            <div>
              <div className="flex items-center justify-between mb-2">
                <Label>Traffic Multiplier</Label>
                <Badge variant="outline">{modifications.traffic_multiplier?.toFixed(2)}x</Badge>
              </div>
              <Slider
                value={[modifications.traffic_multiplier || 1.0]}
                onValueChange={handleTrafficMultiplier}
                min={0.5}
                max={2.0}
                step={0.1}
                className="w-full"
              />
              <p className="text-xs text-muted-foreground mt-1">
                Simulates traffic conditions (0.5 = light, 1.0 = normal, 2.0 = heavy)
              </p>
            </div>

            <div>
              <div className="flex items-center justify-between mb-2">
                <Label>Distance Multiplier</Label>
                <Badge variant="outline">{modifications.distance_multiplier?.toFixed(2)}x</Badge>
              </div>
              <Slider
                value={[modifications.distance_multiplier || 1.0]}
                onValueChange={handleDistanceMultiplier}
                min={0.5}
                max={2.0}
                step={0.1}
                className="w-full"
              />
              <p className="text-xs text-muted-foreground mt-1">
                Simulates route deviations or construction detours
              </p>
            </div>
          </div>

          <div className="flex gap-2 pt-4 border-t">
            <Button 
              onClick={handleRunSimulation} 
              disabled={loading || stops.length === 0}
              className="flex-1"
            >
              {loading ? (
                <>
                  <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
                  Simulating...
                </>
              ) : (
                <>
                  <Play className="mr-2 h-4 w-4" />
                  Run Simulation
                </>
              )}
            </Button>
            <Button 
              variant="outline" 
              onClick={handleReset}
              disabled={loading || modificationCount === 0}
            >
              <RefreshCw className="mr-2 h-4 w-4" />
              Reset
            </Button>
          </div>

          {modificationCount > 0 && (
            <div className="pt-4 border-t">
              <p className="text-sm text-muted-foreground">
                <Badge variant="secondary" className="mr-2">{modificationCount}</Badge>
                modifications applied
              </p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Stop-Level Modifications */}
      <Card>
        <CardHeader>
          <CardTitle>Stop-Level Modifications</CardTitle>
          <CardDescription>
            Modify individual stops to test specific scenarios
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4 max-h-[400px] overflow-y-auto">
            {stops
              .filter(stop => !modifications.stops_to_remove?.includes(stop.stop_id))
              .slice(0, 20)
              .map((stop, idx) => (
              <div 
                key={stop.stop_id}
                className={`p-4 border rounded-lg space-y-3 ${
                  selectedStop === stop.stop_id ? 'border-primary bg-primary/5' : ''
                }`}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Badge variant="outline">Stop {stop.stop_id}</Badge>
                    <Badge>Route {stop.route_id}</Badge>
                    <Badge variant="secondary">Driver {stop.driver_id}</Badge>
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => handleRemoveStop(stop.stop_id)}
                  >
                    <Trash2 className="h-4 w-4 text-red-500" />
                  </Button>
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <div className="space-y-2">
                    <Label className="text-xs">Reassign Driver</Label>
                    <Select
                      value={modifications.driver_reassignments?.[stop.stop_id]?.toString() || stop.driver_id.toString()}
                      onValueChange={(value) => handleDriverReassignment(stop.stop_id, parseInt(value))}
                    >
                      <SelectTrigger className="h-8 text-xs">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {availableDrivers.map(driverId => (
                          <SelectItem key={driverId} value={driverId.toString()}>
                            Driver {driverId}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-2">
                    <Label className="text-xs">Time Shift (minutes)</Label>
                    <Input
                      type="number"
                      placeholder="0"
                      className="h-8 text-xs"
                      value={modifications.time_shift_minutes?.[stop.stop_id] || 0}
                      onChange={(e) => handleTimeShift(stop.stop_id, parseInt(e.target.value) || 0)}
                    />
                  </div>
                </div>

                {(modifications.driver_reassignments?.[stop.stop_id] || modifications.time_shift_minutes?.[stop.stop_id]) && (
                  <div className="pt-2 border-t">
                    <p className="text-xs text-muted-foreground">
                      {modifications.driver_reassignments?.[stop.stop_id] && (
                        <span className="mr-2">
                          ✓ Driver reassigned
                        </span>
                      )}
                      {modifications.time_shift_minutes?.[stop.stop_id] && (
                        <span>
                          ✓ Time shifted by {modifications.time_shift_minutes[stop.stop_id]} min
                        </span>
                      )}
                    </p>
                  </div>
                )}
              </div>
            ))}
          </div>

          {stops.length > 20 && (
            <p className="text-sm text-muted-foreground mt-4 text-center">
              Showing 20 of {stops.length} stops
            </p>
          )}
        </CardContent>
      </Card>

      {/* Modification Summary */}
      {modificationCount > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Modification Summary</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-3">
              {Object.keys(modifications.driver_reassignments || {}).length > 0 && (
                <div className="flex items-center gap-2">
                  <Users className="h-4 w-4 text-blue-500" />
                  <span className="text-sm">
                    {Object.keys(modifications.driver_reassignments || {}).length} driver reassignments
                  </span>
                </div>
              )}
              
              {Object.keys(modifications.time_shift_minutes || {}).length > 0 && (
                <div className="flex items-center gap-2">
                  <Clock className="h-4 w-4 text-purple-500" />
                  <span className="text-sm">
                    {Object.keys(modifications.time_shift_minutes || {}).length} time shifts
                  </span>
                </div>
              )}
              
              {(modifications.stops_to_remove?.length || 0) > 0 && (
                <div className="flex items-center gap-2">
                  <MapPin className="h-4 w-4 text-red-500" />
                  <span className="text-sm">
                    {modifications.stops_to_remove?.length} stops removed
                  </span>
                </div>
              )}
              
              {modifications.traffic_multiplier !== 1.0 && (
                <div className="flex items-center gap-2">
                  <TrendingUp className="h-4 w-4 text-orange-500" />
                  <span className="text-sm">
                    Traffic: {modifications.traffic_multiplier}x
                  </span>
                </div>
              )}
              
              {modifications.distance_multiplier !== 1.0 && (
                <div className="flex items-center gap-2">
                  <MapPin className="h-4 w-4 text-green-500" />
                  <span className="text-sm">
                    Distance: {modifications.distance_multiplier}x
                  </span>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}

