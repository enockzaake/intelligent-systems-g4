"use client"

import { useState, useEffect } from "react"
import { ScenarioBuilder } from "@/components/scenario-builder"
import { ScenarioComparison } from "@/components/scenario-comparison"
import { RouteTimeline } from "@/components/route-timeline"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Button } from "@/components/ui/button"
import { AlertCircle, Download, RefreshCw } from "lucide-react"
import { getSampleData, ScenarioResult, StopData } from "@/lib/api"
import { toast } from "sonner"

export default function SimulationPage() {
  const [simulationResult, setSimulationResult] = useState<ScenarioResult | null>(null)
  const [sampleStops, setSampleStops] = useState<StopData[]>([])
  const [loading, setLoading] = useState(false)

  // Load initial sample data
  useEffect(() => {
    loadSampleData()
  }, [])

  const loadSampleData = async () => {
    setLoading(true)
    try {
      toast.info("Loading sample route data...")
      const data = await getSampleData(100)
      setSampleStops(data.data)
      toast.success(`Loaded ${data.data.length} stops from dataset`)
    } catch (error) {
      console.error("Error loading sample data:", error)
      toast.error("Failed to load sample data. Make sure the API server is running.")
    } finally {
      setLoading(false)
    }
  }

  const handleSimulationComplete = (result: ScenarioResult) => {
    setSimulationResult(result)
  }

  const handleRefresh = () => {
    loadSampleData()
    setSimulationResult(null)
  }

  const handleExportResults = () => {
    if (!simulationResult) return
    
    const dataStr = JSON.stringify(simulationResult, null, 2)
    const dataBlob = new Blob([dataStr], { type: 'application/json' })
    const url = URL.createObjectURL(dataBlob)
    const link = document.createElement('a')
    link.href = url
    link.download = `scenario-simulation-${Date.now()}.json`
    link.click()
    toast.success("Results exported successfully")
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Advanced Scenario Simulation</h1>
          <p className="text-muted-foreground">
            Test what-if scenarios with interactive route modifications and AI-powered predictions
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" onClick={handleRefresh} disabled={loading}>
            <RefreshCw className={`mr-2 h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
            Refresh Data
          </Button>
          {simulationResult && (
            <Button variant="outline" onClick={handleExportResults}>
              <Download className="mr-2 h-4 w-4" />
              Export Results
            </Button>
          )}
        </div>
      </div>

      {/* Info Cards */}
      <div className="grid gap-4 md:grid-cols-3">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Available Features</CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="text-xs space-y-1 text-muted-foreground">
              <li>✓ Driver reassignment</li>
              <li>✓ Time window adjustments</li>
              <li>✓ Traffic simulation</li>
              <li>✓ Stop removal</li>
            </ul>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Current Dataset</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-1">
              <p className="text-2xl font-bold">{sampleStops.length}</p>
              <p className="text-xs text-muted-foreground">
                stops from {new Set(sampleStops.map(s => s.route_id)).size} routes
              </p>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Simulation Status</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-1">
              <p className="text-sm font-medium">
                {simulationResult ? "Results Available" : "Ready to Simulate"}
              </p>
              <p className="text-xs text-muted-foreground">
                {simulationResult 
                  ? `${simulationResult.modifications_applied.driver_reassignments + simulationResult.modifications_applied.time_shifts} modifications applied`
                  : "Configure parameters below"
                }
              </p>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Quick Guide */}
      <Card>
        <CardHeader>
          <CardTitle>How to Use the Scenario Simulator</CardTitle>
          <CardDescription>Follow these steps to test different scenarios</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-4">
            <div className="flex gap-3">
              <div className="flex h-8 w-8 items-center justify-center rounded-full bg-blue-500 text-white font-bold text-sm">
                1
              </div>
              <div>
                <p className="text-sm font-medium">Adjust Global Parameters</p>
                <p className="text-xs text-muted-foreground">
                  Set traffic and distance multipliers to simulate conditions
                </p>
              </div>
            </div>
            <div className="flex gap-3">
              <div className="flex h-8 w-8 items-center justify-center rounded-full bg-blue-500 text-white font-bold text-sm">
                2
              </div>
              <div>
                <p className="text-sm font-medium">Modify Individual Stops</p>
                <p className="text-xs text-muted-foreground">
                  Reassign drivers, shift times, or remove stops
                </p>
              </div>
            </div>
            <div className="flex gap-3">
              <div className="flex h-8 w-8 items-center justify-center rounded-full bg-blue-500 text-white font-bold text-sm">
                3
              </div>
              <div>
                <p className="text-sm font-medium">Run Simulation</p>
                <p className="text-xs text-muted-foreground">
                  AI analyzes the modified scenario
                </p>
              </div>
            </div>
            <div className="flex gap-3">
              <div className="flex h-8 w-8 items-center justify-center rounded-full bg-blue-500 text-white font-bold text-sm">
                4
              </div>
              <div>
                <p className="text-sm font-medium">Review Results</p>
                <p className="text-xs text-muted-foreground">
                  Compare before/after metrics and timelines
                </p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Scenario Builder */}
      {sampleStops.length > 0 ? (
        <ScenarioBuilder 
          initialStops={sampleStops}
          onSimulationComplete={handleSimulationComplete}
        />
      ) : (
        <Card>
          <CardContent className="flex items-center justify-center h-48">
            <div className="text-center space-y-2">
              <p className="text-muted-foreground">Loading route data...</p>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Simulation Results */}
      {simulationResult && (
        <>
          <Tabs defaultValue="comparison" className="space-y-4">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="comparison">Side-by-Side Comparison</TabsTrigger>
              <TabsTrigger value="timeline">Route Timeline</TabsTrigger>
              <TabsTrigger value="details">Detailed Results</TabsTrigger>
            </TabsList>

            <TabsContent value="comparison" className="space-y-4">
              <ScenarioComparison result={simulationResult} />
            </TabsContent>

            <TabsContent value="timeline" className="space-y-4">
              <div className="grid gap-4">
                <RouteTimeline 
                  predictions={simulationResult.original.predictions}
                  title="Original Scenario Timeline"
                />
                <RouteTimeline 
                  predictions={simulationResult.modified.predictions}
                  title="Modified Scenario Timeline"
                />
              </div>
            </TabsContent>

            <TabsContent value="details" className="space-y-4">
              <div className="grid gap-4 lg:grid-cols-2">
                {/* Original Details */}
                <Card>
                  <CardHeader>
                    <CardTitle>Original Scenario Details</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3 text-sm">
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Total Stops:</span>
                        <span className="font-medium">{simulationResult.original.total_stops}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Total Delay:</span>
                        <span className="font-medium">{simulationResult.original.total_delay_minutes.toFixed(1)} min</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Avg Delay Probability:</span>
                        <span className="font-medium">{(simulationResult.original.avg_delay_probability * 100).toFixed(1)}%</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">High Risk Stops:</span>
                        <span className="font-medium text-red-500">{simulationResult.original.high_risk_count}</span>
                      </div>
                      {simulationResult.original.optimization && (
                        <>
                          <div className="pt-2 border-t" />
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Optimized Distance:</span>
                            <span className="font-medium">{simulationResult.original.optimization.total_distance.toFixed(0)} km</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Vehicles Used:</span>
                            <span className="font-medium">{simulationResult.original.optimization.num_vehicles_used}</span>
                          </div>
                        </>
                      )}
                    </div>
                  </CardContent>
                </Card>

                {/* Modified Details */}
                <Card className="border-primary">
                  <CardHeader>
                    <CardTitle>Modified Scenario Details</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3 text-sm">
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Total Stops:</span>
                        <span className="font-medium">{simulationResult.modified.total_stops}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Total Delay:</span>
                        <span className="font-medium text-green-500">{simulationResult.modified.total_delay_minutes.toFixed(1)} min</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Avg Delay Probability:</span>
                        <span className="font-medium text-green-500">{(simulationResult.modified.avg_delay_probability * 100).toFixed(1)}%</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">High Risk Stops:</span>
                        <span className="font-medium text-green-500">{simulationResult.modified.high_risk_count}</span>
                      </div>
                      {simulationResult.modified.optimization && (
                        <>
                          <div className="pt-2 border-t" />
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Optimized Distance:</span>
                            <span className="font-medium text-green-500">{simulationResult.modified.optimization.total_distance.toFixed(0)} km</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Vehicles Used:</span>
                            <span className="font-medium text-green-500">{simulationResult.modified.optimization.num_vehicles_used}</span>
                          </div>
                        </>
                      )}
                    </div>
                  </CardContent>
                </Card>
              </div>

              {/* Modifications Summary */}
              <Card>
                <CardHeader>
                  <CardTitle>Applied Modifications</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-5">
                    <div className="text-center p-3 bg-muted rounded-lg">
                      <p className="text-2xl font-bold">{simulationResult.modifications_applied.driver_reassignments}</p>
                      <p className="text-xs text-muted-foreground">Driver Reassignments</p>
                    </div>
                    <div className="text-center p-3 bg-muted rounded-lg">
                      <p className="text-2xl font-bold">{simulationResult.modifications_applied.time_shifts}</p>
                      <p className="text-xs text-muted-foreground">Time Shifts</p>
                    </div>
                    <div className="text-center p-3 bg-muted rounded-lg">
                      <p className="text-2xl font-bold">{simulationResult.modifications_applied.stops_removed}</p>
                      <p className="text-xs text-muted-foreground">Stops Removed</p>
                    </div>
                    <div className="text-center p-3 bg-muted rounded-lg">
                      <p className="text-2xl font-bold">{simulationResult.modifications_applied.traffic_multiplier}x</p>
                      <p className="text-xs text-muted-foreground">Traffic Multiplier</p>
                    </div>
                    <div className="text-center p-3 bg-muted rounded-lg">
                      <p className="text-2xl font-bold">{simulationResult.modifications_applied.distance_multiplier}x</p>
                      <p className="text-xs text-muted-foreground">Distance Multiplier</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </>
      )}

      {!simulationResult && sampleStops.length > 0 && (
        <Card>
          <CardContent className="flex items-center justify-center h-64">
            <div className="text-center space-y-3">
              <AlertCircle className="h-12 w-12 text-muted-foreground mx-auto" />
              <div>
                <p className="text-lg font-medium">No simulation results yet</p>
                <p className="text-sm text-muted-foreground">
                  Configure your scenario parameters above and click "Run Simulation" to see results
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
