"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { 
  Download, 
  RefreshCw, 
  AlertCircle, 
  Sparkles,
  FileText,
  BarChart3
} from "lucide-react"
import { EnhancedScenarioConfigurator } from "@/components/enhanced-scenario-configurator"
import { PredictionOutputPanel } from "@/components/prediction-output-panel"
import { OptimizationRecommendationPanel } from "@/components/optimization-recommendation-panel"
import { DecisionTraceVisual } from "@/components/decision-trace-visual"
import { SimulationPlaybackTimeline } from "@/components/simulation-playback-timeline"
import { 
  simulateAndPredict, 
  simulateAndOptimize,
  SimulationScenarioConfig,
  SimulationPredictionResponse,
  SimulationOptimizationResponse,
  getRoutesData,
  getSampleData,
  StopData
} from "@/lib/api"
import { toast } from "sonner"

export default function EnhancedSimulationPage() {
  const [availableRoutes, setAvailableRoutes] = useState<Array<{ route_id: number; stop_count: number }>>([])
  const [loading, setLoading] = useState(false)
  const [simulationState, setSimulationState] = useState<{
    prediction: SimulationPredictionResponse | null
    optimization: SimulationOptimizationResponse | null
    config: SimulationScenarioConfig | null
  }>({
    prediction: null,
    optimization: null,
    config: null
  })

  // Load available routes on mount
  useEffect(() => {
    loadAvailableRoutes()
  }, [])

  const loadAvailableRoutes = async () => {
    try {
      // First try to get routes data
      const routesData = await getRoutesData(50)
      const routes = routesData.routes.map(r => ({
        route_id: r.route_id,
        stop_count: r.total_stops
      }))
      setAvailableRoutes(routes)
      
      toast.success(`Loaded ${routes.length} routes`)
    } catch (error) {
      console.error("Error loading routes:", error)
      
      // Fallback: get sample data and extract unique routes
      try {
        const sampleData = await getSampleData(500)
        const routeMap = new Map<number, number>()
        
        sampleData.data.forEach((stop: StopData) => {
          const count = routeMap.get(stop.route_id) || 0
          routeMap.set(stop.route_id, count + 1)
        })
        
        const routes = Array.from(routeMap.entries()).map(([route_id, stop_count]) => ({
          route_id,
          stop_count
        }))
        
        setAvailableRoutes(routes)
        toast.success(`Loaded ${routes.length} routes from sample data`)
      } catch (fallbackError) {
        toast.error("Failed to load route data. Make sure the API server is running.")
      }
    }
  }

  const handleRunSimulation = async (config: SimulationScenarioConfig) => {
    setLoading(true)
    
    try {
      toast.info("Running simulation analysis...")
      
      // Step 1: Get predictions
      console.log("Step 1: Getting predictions...")
      const predictionResult = await simulateAndPredict(config)
      console.log("Predictions received:", predictionResult)
      
      toast.success("Predictions complete!")
      
      // Step 2: Get optimization recommendations
      console.log("Step 2: Getting optimization recommendations...")
      const optimizationResult = await simulateAndOptimize(config)
      console.log("Optimization received:", optimizationResult)
      
      toast.success("Optimization complete!")
      
      // Update state
      setSimulationState({
        prediction: predictionResult,
        optimization: optimizationResult,
        config
      })
      
      toast.success(
        `Simulation complete! Found ${optimizationResult.actions.length} optimization opportunities.`,
        {
          duration: 5000
        }
      )
      
    } catch (error) {
      console.error("Simulation error:", error)
      toast.error(
        error instanceof Error ? error.message : "Simulation failed. Please check the API server.",
        {
          duration: 7000
        }
      )
    } finally {
      setLoading(false)
    }
  }

  const handleReset = () => {
    setSimulationState({
      prediction: null,
      optimization: null,
      config: null
    })
    toast.info("Simulation reset")
  }

  const handleExport = () => {
    if (!simulationState.prediction || !simulationState.optimization) {
      toast.error("No simulation results to export")
      return
    }

    const exportData = {
      timestamp: new Date().toISOString(),
      scenario: simulationState.config,
      prediction: simulationState.prediction,
      optimization: simulationState.optimization
    }

    const dataStr = JSON.stringify(exportData, null, 2)
    const dataBlob = new Blob([dataStr], { type: 'application/json' })
    const url = URL.createObjectURL(dataBlob)
    const link = document.createElement('a')
    link.href = url
    link.download = `simulation-analysis-${Date.now()}.json`
    link.click()
    
    toast.success("Simulation results exported")
  }

  const hasResults = simulationState.prediction && simulationState.optimization

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight flex items-center gap-2">
            <Sparkles className="h-8 w-8 text-primary" />
            Intelligent Simulation Engine
          </h1>
          <p className="text-muted-foreground mt-1">
            Test scenarios, predict outcomes, and discover optimization opportunities with AI-powered analysis
          </p>
        </div>
        <div className="flex gap-2">
          <Button 
            variant="outline" 
            onClick={handleReset}
            disabled={loading || !hasResults}
          >
            <RefreshCw className="mr-2 h-4 w-4" />
            Reset
          </Button>
          <Button 
            variant="outline" 
            onClick={handleExport}
            disabled={!hasResults}
          >
            <Download className="mr-2 h-4 w-4" />
            Export
          </Button>
        </div>
      </div>

      {/* Key Features Info */}
      <div className="grid gap-4 md:grid-cols-4">
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <FileText className="h-4 w-4 text-blue-500" />
              Scenario Testing
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-xs text-muted-foreground">
              Simulate various conditions: traffic, delays, driver changes
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <BarChart3 className="h-4 w-4 text-purple-500" />
              AI Predictions
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-xs text-muted-foreground">
              ML models predict delays with detailed explanations
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <Sparkles className="h-4 w-4 text-green-500" />
              Smart Optimization
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-xs text-muted-foreground">
              Get actionable recommendations with reasoning
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium">
              {hasResults ? (
                <Badge className="bg-green-500">Results Ready</Badge>
              ) : loading ? (
                <Badge variant="secondary">Running...</Badge>
              ) : (
                <Badge variant="outline">Ready</Badge>
              )}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-xs text-muted-foreground">
              {hasResults 
                ? `${simulationState.optimization?.actions.length} actions found` 
                : "Configure and run simulation"}
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Main Content */}
      <div className="grid gap-6">
        {/* Configuration Section */}
        {!hasResults && (
          <EnhancedScenarioConfigurator
            availableRoutes={availableRoutes}
            onRunSimulation={handleRunSimulation}
            loading={loading}
          />
        )}

        {/* Results Section */}
        {hasResults && (
          <Tabs defaultValue="overview" className="space-y-6">
            <TabsList className="grid w-full grid-cols-5">
              <TabsTrigger value="overview">Overview</TabsTrigger>
              <TabsTrigger value="predictions">Predictions</TabsTrigger>
              <TabsTrigger value="optimizations">Optimizations</TabsTrigger>
              <TabsTrigger value="comparison">Comparison</TabsTrigger>
              <TabsTrigger value="playback">Playback</TabsTrigger>
            </TabsList>

            {/* Overview Tab */}
            <TabsContent value="overview" className="space-y-6">
              <Card>
                <CardHeader>
                  <CardTitle>Simulation Overview</CardTitle>
                  <CardDescription>
                    High-level summary of the simulation results
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                  {/* Scenario Info */}
                  <div>
                    <h3 className="font-semibold mb-3">Scenario Configuration</h3>
                    <div className="grid gap-3 md:grid-cols-2">
                      <div className="p-3 bg-muted rounded-lg">
                        <p className="text-sm text-muted-foreground">Scenario Type</p>
                        <p className="font-medium">
                          {simulationState.config?.scenario_type.replace(/_/g, ' ').toUpperCase()}
                        </p>
                      </div>
                      <div className="p-3 bg-muted rounded-lg">
                        <p className="text-sm text-muted-foreground">Stops Analyzed</p>
                        <p className="font-medium">{simulationState.prediction.predictions.length}</p>
                      </div>
                    </div>
                  </div>

                  {/* Key Findings */}
                  <div>
                    <h3 className="font-semibold mb-3">Key Findings</h3>
                    <div className="grid gap-3 md:grid-cols-3">
                      <div className="p-4 border border-red-200 bg-red-50/50 rounded-lg">
                        <p className="text-sm text-muted-foreground mb-1">Predicted Issues</p>
                        <p className="text-2xl font-bold text-red-600">
                          {simulationState.prediction.route_summary.high_risk_stops} high-risk stops
                        </p>
                        <p className="text-xs text-muted-foreground mt-1">
                          {simulationState.prediction.route_summary.total_expected_delay.toFixed(1)} min total delay
                        </p>
                      </div>

                      <div className="p-4 border border-green-200 bg-green-50/50 rounded-lg">
                        <p className="text-sm text-muted-foreground mb-1">Solutions Found</p>
                        <p className="text-2xl font-bold text-green-600">
                          {simulationState.optimization.actions.length} optimizations
                        </p>
                        <p className="text-xs text-muted-foreground mt-1">
                          {simulationState.optimization.impact.delay_reduction_minutes.toFixed(1)} min delay reduction
                        </p>
                      </div>

                      <div className="p-4 border border-blue-200 bg-blue-50/50 rounded-lg">
                        <p className="text-sm text-muted-foreground mb-1">Expected Impact</p>
                        <p className="text-2xl font-bold text-blue-600">
                          +{(simulationState.optimization.impact.on_time_rate_improvement * 100).toFixed(1)}%
                        </p>
                        <p className="text-xs text-muted-foreground mt-1">
                          Reliability improvement
                        </p>
                      </div>
                    </div>
                  </div>

                  {/* Quick Actions */}
                  <div className="flex gap-2 pt-4 border-t">
                    <Button 
                      variant="outline"
                      onClick={() => {
                        const tabsTriggers = document.querySelectorAll('[role="tab"]')
                        tabsTriggers[1]?.dispatchEvent(new Event('click', { bubbles: true }))
                      }}
                    >
                      View Detailed Predictions
                    </Button>
                    <Button 
                      variant="outline"
                      onClick={() => {
                        const tabsTriggers = document.querySelectorAll('[role="tab"]')
                        tabsTriggers[2]?.dispatchEvent(new Event('click', { bubbles: true }))
                      }}
                    >
                      View Optimizations
                    </Button>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            {/* Predictions Tab */}
            <TabsContent value="predictions">
              <PredictionOutputPanel predictionResult={simulationState.prediction} />
            </TabsContent>

            {/* Optimizations Tab */}
            <TabsContent value="optimizations">
              <OptimizationRecommendationPanel optimizationResult={simulationState.optimization} />
            </TabsContent>

            {/* Comparison Tab */}
            <TabsContent value="comparison">
              <DecisionTraceVisual
                originalPrediction={simulationState.prediction}
                optimizationActions={simulationState.optimization}
              />
            </TabsContent>

            {/* Playback Tab */}
            <TabsContent value="playback">
              <SimulationPlaybackTimeline
                originalPrediction={simulationState.prediction}
                optimizationResult={simulationState.optimization}
              />
            </TabsContent>
          </Tabs>
        )}

        {/* Empty State */}
        {!hasResults && !loading && (
          <Card>
            <CardContent className="flex flex-col items-center justify-center h-64 space-y-4">
              <AlertCircle className="h-12 w-12 text-muted-foreground" />
              <div className="text-center">
                <p className="text-lg font-medium">No simulation results yet</p>
                <p className="text-sm text-muted-foreground mt-1">
                  Configure a scenario above and click "Run Simulation" to begin
                </p>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Loading State */}
        {loading && (
          <Card>
            <CardContent className="flex flex-col items-center justify-center h-64 space-y-4">
              <div className="h-12 w-12 animate-spin rounded-full border-4 border-primary border-t-transparent" />
              <div className="text-center">
                <p className="text-lg font-medium">Running Intelligent Analysis...</p>
                <p className="text-sm text-muted-foreground mt-1">
                  This may take a few moments as the AI processes your scenario
                </p>
              </div>
            </CardContent>
          </Card>
        )}
      </div>

      {/* Documentation/Help Section */}
      <Card className="border-blue-200 bg-blue-50/50">
        <CardHeader>
          <CardTitle className="text-base">How This Works</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-4">
            <div>
              <div className="flex h-8 w-8 items-center justify-center rounded-full bg-blue-500 text-white font-bold mb-2">
                1
              </div>
              <p className="text-sm font-medium">Configure Scenario</p>
              <p className="text-xs text-muted-foreground mt-1">
                Select route and simulation conditions
              </p>
            </div>
            <div>
              <div className="flex h-8 w-8 items-center justify-center rounded-full bg-blue-500 text-white font-bold mb-2">
                2
              </div>
              <p className="text-sm font-medium">AI Analysis</p>
              <p className="text-xs text-muted-foreground mt-1">
                ML models predict delays and risks
              </p>
            </div>
            <div>
              <div className="flex h-8 w-8 items-center justify-center rounded-full bg-blue-500 text-white font-bold mb-2">
                3
              </div>
              <p className="text-sm font-medium">Generate Solutions</p>
              <p className="text-xs text-muted-foreground mt-1">
                Optimization engine suggests improvements
              </p>
            </div>
            <div>
              <div className="flex h-8 w-8 items-center justify-center rounded-full bg-blue-500 text-white font-bold mb-2">
                4
              </div>
              <p className="text-sm font-medium">Review & Implement</p>
              <p className="text-xs text-muted-foreground mt-1">
                Examine recommendations and apply changes
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

