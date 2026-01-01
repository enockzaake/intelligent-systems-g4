"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Play, Loader2, Download } from "lucide-react"
import { optimizeRoutes, predictDelays, reassignStops, getSampleData } from "@/lib/api"
import { toast } from "sonner"

interface SimulationPanelProps {
  onSimulationComplete?: (result: any) => void
}

export function SimulationPanel({ onSimulationComplete }: SimulationPanelProps) {
  const [loading, setLoading] = useState(false)
  const [params, setParams] = useState({
    numStops: 100,
    numVehicles: 10,
    avgDelay: 15,
    trafficCondition: "moderate",
    weatherCondition: "clear",
    timeOfDay: "morning",
  })

  const handleRunSimulation = async () => {
    setLoading(true)
    try {
      toast.info("Starting simulation...")
      
      // Validate parameters
      if (params.numStops < 10 || params.numStops > 500) {
        toast.error("Number of stops must be between 10 and 500")
        setLoading(false)
        return
      }
      
      if (params.numVehicles < 1 || params.numVehicles > 50) {
        toast.error("Number of vehicles must be between 1 and 50")
        setLoading(false)
        return
      }
      
      // Fetch REAL data from backend instead of generating mock data
      toast.info(`Fetching ${params.numStops} real stops from dataset...`)
      const sampleData = await getSampleData(params.numStops)
      const realStops = sampleData.data
      
      console.log("Fetched real stops sample:", realStops.slice(0, 3))
      console.log(`Total stops: ${realStops.length}, Unique routes: ${new Set(realStops.map(s => s.route_id)).size}`)
      
      toast.info("Predicting delays...")
      const predictions = await predictDelays(realStops)
      console.log("Predictions received:", predictions.length)
      
      toast.info("Optimizing routes...")
      const optimization = await optimizeRoutes(realStops)
      console.log("Optimization complete:", optimization.solution.num_vehicles_used, "vehicles")
      
      toast.info("Calculating reassignments...")
      const reassignments = await reassignStops(realStops, 0.7)
      console.log("Reassignments calculated:", reassignments.num_reassignments)
      
      const result = {
        params,
        predictions,
        optimization,
        reassignments,
        mockStops: realStops,
      }
      
      onSimulationComplete?.(result)
      toast.success(`Simulation completed! Analyzed ${realStops.length} real stops from dataset.`)
      
    } catch (error) {
      console.error("Simulation error:", error)
      toast.error(`Simulation failed: ${error instanceof Error ? error.message : 'Unknown error'}. Make sure the API server is running at http://localhost:8000`)
    } finally {
      setLoading(false)
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Route Simulation</CardTitle>
        <CardDescription>
          Test the system under different conditions to evaluate route optimization and delay predictions
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="space-y-2">
            <Label htmlFor="numStops">Number of Stops</Label>
            <Input
              id="numStops"
              type="number"
              value={params.numStops}
              onChange={(e) => setParams({ ...params, numStops: parseInt(e.target.value) })}
              min={10}
              max={500}
            />
          </div>
          
          <div className="space-y-2">
            <Label htmlFor="numVehicles">Number of Vehicles</Label>
            <Input
              id="numVehicles"
              type="number"
              value={params.numVehicles}
              onChange={(e) => setParams({ ...params, numVehicles: parseInt(e.target.value) })}
              min={1}
              max={50}
            />
          </div>
          
          <div className="space-y-2">
            <Label htmlFor="avgDelay">Average Delay (minutes)</Label>
            <Input
              id="avgDelay"
              type="number"
              value={params.avgDelay}
              onChange={(e) => setParams({ ...params, avgDelay: parseInt(e.target.value) })}
              min={0}
              max={120}
            />
          </div>
          
          <div className="space-y-2">
            <Label htmlFor="traffic">Traffic Condition</Label>
            <Select
              value={params.trafficCondition}
              onValueChange={(value) => setParams({ ...params, trafficCondition: value })}
            >
              <SelectTrigger id="traffic">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="light">Light Traffic</SelectItem>
                <SelectItem value="moderate">Moderate Traffic</SelectItem>
                <SelectItem value="heavy">Heavy Traffic</SelectItem>
              </SelectContent>
            </Select>
          </div>
          
          <div className="space-y-2">
            <Label htmlFor="weather">Weather Condition</Label>
            <Select
              value={params.weatherCondition}
              onValueChange={(value) => setParams({ ...params, weatherCondition: value })}
            >
              <SelectTrigger id="weather">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="clear">Clear</SelectItem>
                <SelectItem value="rain">Rain</SelectItem>
                <SelectItem value="snow">Snow</SelectItem>
                <SelectItem value="fog">Fog</SelectItem>
              </SelectContent>
            </Select>
          </div>
          
          <div className="space-y-2">
            <Label htmlFor="timeOfDay">Time of Day</Label>
            <Select
              value={params.timeOfDay}
              onValueChange={(value) => setParams({ ...params, timeOfDay: value })}
            >
              <SelectTrigger id="timeOfDay">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="morning">Morning (6-12)</SelectItem>
                <SelectItem value="afternoon">Afternoon (12-16)</SelectItem>
                <SelectItem value="evening">Evening (16-20)</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>
        
        <div className="flex gap-2 pt-4">
          <Button 
            onClick={handleRunSimulation} 
            disabled={loading}
            className="flex-1"
          >
            {loading ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Running Simulation...
              </>
            ) : (
              <>
                <Play className="mr-2 h-4 w-4" />
                Run Simulation
              </>
            )}
          </Button>
          <Button variant="outline" disabled={loading}>
            <Download className="mr-2 h-4 w-4" />
            Export
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}

