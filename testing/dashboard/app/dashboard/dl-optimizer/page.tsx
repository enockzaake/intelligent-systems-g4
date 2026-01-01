"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Loader2, Brain, TrendingUp, Target, CheckCircle2, XCircle, ArrowRight } from "lucide-react"

interface Stop {
  stop_id: string
  planned_position: number
  actual_position: number
  predicted_position: number
  is_depot: boolean
  is_delivery: boolean
  distance_planned: number
  distance_actual: number
  earliest_time: string
  latest_time: string
  confidence: number
}

interface RouteInfo {
  route_id: number
  num_stops: number
  driver_id: string
  country: string
  day_of_week: string
  total_distance_planned?: number
  total_distance_actual?: number
}

interface PredictionResult {
  route_id: number
  num_stops: number
  predicted_sequence: number[]
  planned_sequence: number[]
  actual_sequence: number[]
  stop_ids: string[]
  confidence_scores: number[]
  metrics: {
    sequence_accuracy: number
    kendall_tau_predicted: number
    kendall_tau_planned: number
    improvement_over_planned: number
    planned_total_distance: number
    actual_total_distance: number
  }
}

interface VisualizationData {
  route_id: number
  num_stops: number
  stops: Stop[]
  metrics: {
    sequence_accuracy: number
    kendall_tau_predicted: number
    kendall_tau_planned: number
    improvement: number
  }
  sequences: {
    planned: number[]
    actual: number[]
    predicted: number[]
  }
}

const API_BASE_URL = "http://localhost:8001"

export default function DLOptimizerPage() {
  const [routes, setRoutes] = useState<RouteInfo[]>([])
  const [selectedRouteId, setSelectedRouteId] = useState<number | null>(null)
  const [selectedRoute, setSelectedRoute] = useState<RouteInfo | null>(null)
  const [prediction, setPrediction] = useState<PredictionResult | null>(null)
  const [vizData, setVizData] = useState<VisualizationData | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [apiStatus, setApiStatus] = useState<'checking' | 'online' | 'offline'>('checking')

  // Check API health on mount
  useEffect(() => {
    checkAPIHealth()
    loadRandomRoutes()
  }, [])

  const checkAPIHealth = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v2/health`)
      const data = await response.json()
      setApiStatus(data.model_loaded ? 'online' : 'offline')
    } catch (err) {
      setApiStatus('offline')
    }
  }

  const loadRandomRoutes = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v2/random-routes?count=20`)
      const data = await response.json()
      setRoutes(data.routes)
    } catch (err) {
      console.error("Failed to load routes:", err)
    }
  }

  const handleRouteSelect = async (routeId: string) => {
    const id = parseInt(routeId)
    setSelectedRouteId(id)
    setError(null)
    setPrediction(null)
    setVizData(null)
    
    // Load route details
    try {
      const response = await fetch(`${API_BASE_URL}/api/v2/route/${id}`)
      const data = await response.json()
      setSelectedRoute(data)
    } catch (err) {
      setError("Failed to load route details")
    }
  }

  const handlePredict = async () => {
    if (!selectedRouteId) return

    setLoading(true)
    setError(null)

    try {
      // Get prediction
      const predResponse = await fetch(`${API_BASE_URL}/api/v2/predict/${selectedRouteId}`)
      
      if (!predResponse.ok) {
        throw new Error(`Prediction failed: ${predResponse.statusText}`)
      }
      
      const predData = await predResponse.json()
      setPrediction(predData)

      // Get visualization data
      const vizResponse = await fetch(`${API_BASE_URL}/api/v2/visualization/${selectedRouteId}`)
      const vizData = await vizResponse.json()
      setVizData(vizData)

    } catch (err: any) {
      setError(err.message || "Failed to generate prediction")
    } finally {
      setLoading(false)
    }
  }

  const getSequenceComparisonColor = (predicted: number, actual: number) => {
    if (predicted === actual) return "text-green-600 font-semibold"
    return "text-orange-600"
  }

  const formatMetric = (value: number, isPercentage: boolean = false) => {
    if (isPercentage) {
      return `${(value * 100).toFixed(1)}%`
    }
    return value.toFixed(4)
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <h1 className="text-3xl font-bold flex items-center gap-2">
            <Brain className="h-8 w-8 text-purple-600" />
            Deep Learning Route Optimizer
          </h1>
          <Badge variant={apiStatus === 'online' ? 'default' : 'destructive'}>
            {apiStatus === 'checking' ? 'Checking...' : apiStatus === 'online' ? 'Model Online' : 'Model Offline'}
          </Badge>
        </div>
        <p className="text-muted-foreground">
          AI-powered route sequence optimization using Transformer neural networks
        </p>
      </div>

      {/* API Status Alert */}
      {apiStatus === 'offline' && (
        <Alert variant="destructive">
          <AlertDescription>
            The DL model is not loaded. Please train the model first using: <code>python core/train_dl_model.py</code>
          </AlertDescription>
        </Alert>
      )}

      {/* Route Selection */}
      <Card>
        <CardHeader>
          <CardTitle>Step 1: Select a Route</CardTitle>
          <CardDescription>Choose a route from the test dataset to analyze</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex gap-4">
            <div className="flex-1">
              <Select onValueChange={handleRouteSelect}>
                <SelectTrigger>
                  <SelectValue placeholder="Select a route..." />
                </SelectTrigger>
                <SelectContent>
                  {routes.map((route) => (
                    <SelectItem key={route.route_id} value={route.route_id.toString()}>
                      Route {route.route_id} - {route.num_stops} stops - {route.driver_id} - {route.country} - {route.day_of_week}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <Button onClick={loadRandomRoutes} variant="outline">
              Load More Routes
            </Button>
          </div>

          {selectedRoute && (
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4 p-4 bg-muted rounded-lg">
              <div>
                <div className="text-sm text-muted-foreground">Route ID</div>
                <div className="text-lg font-semibold">{selectedRoute.route_id}</div>
              </div>
              <div>
                <div className="text-sm text-muted-foreground">Stops</div>
                <div className="text-lg font-semibold">{selectedRoute.num_stops}</div>
              </div>
              <div>
                <div className="text-sm text-muted-foreground">Driver</div>
                <div className="text-lg font-semibold">{selectedRoute.driver_id}</div>
              </div>
              <div>
                <div className="text-sm text-muted-foreground">Country</div>
                <div className="text-lg font-semibold">{selectedRoute.country}</div>
              </div>
              <div>
                <div className="text-sm text-muted-foreground">Day</div>
                <div className="text-lg font-semibold">{selectedRoute.day_of_week}</div>
              </div>
            </div>
          )}

          <Button 
            onClick={handlePredict} 
            disabled={!selectedRouteId || loading || apiStatus !== 'online'}
            className="w-full"
          >
            {loading ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Generating AI Prediction...
              </>
            ) : (
              <>
                <Brain className="mr-2 h-4 w-4" />
                Predict Optimal Sequence
              </>
            )}
          </Button>
        </CardContent>
      </Card>

      {/* Error Display */}
      {error && (
        <Alert variant="destructive">
          <XCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* Prediction Results */}
      {prediction && vizData && (
        <>
          {/* Performance Metrics */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Target className="h-5 w-5" />
                Model Performance Metrics
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
                <div className="text-center p-4 bg-muted rounded-lg">
                  <div className="text-3xl font-bold text-blue-600">
                    {formatMetric(prediction.metrics.sequence_accuracy, true)}
                  </div>
                  <div className="text-sm text-muted-foreground mt-1">Sequence Accuracy</div>
                  <div className="text-xs text-muted-foreground mt-1">
                    Stops in correct position
                  </div>
                </div>
                <div className="text-center p-4 bg-muted rounded-lg">
                  <div className="text-3xl font-bold text-purple-600">
                    {formatMetric(prediction.metrics.kendall_tau_predicted)}
                  </div>
                  <div className="text-sm text-muted-foreground mt-1">Kendall Tau (Predicted)</div>
                  <div className="text-xs text-muted-foreground mt-1">
                    Correlation with actual
                  </div>
                </div>
                <div className="text-center p-4 bg-muted rounded-lg">
                  <div className="text-3xl font-bold text-orange-600">
                    {formatMetric(prediction.metrics.kendall_tau_planned)}
                  </div>
                  <div className="text-sm text-muted-foreground mt-1">Kendall Tau (Planned)</div>
                  <div className="text-xs text-muted-foreground mt-1">
                    Baseline correlation
                  </div>
                </div>
                <div className="text-center p-4 bg-muted rounded-lg">
                  <div className={`text-3xl font-bold ${prediction.metrics.improvement_over_planned > 0 ? 'text-green-600' : 'text-red-600'}`}>
                    {prediction.metrics.improvement_over_planned > 0 ? '+' : ''}
                    {formatMetric(prediction.metrics.improvement_over_planned)}
                  </div>
                  <div className="text-sm text-muted-foreground mt-1">Improvement</div>
                  <div className="text-xs text-muted-foreground mt-1">
                    Over planned sequence
                  </div>
                </div>
              </div>

              {/* Improvement Badge */}
              <div className="mt-4 p-4 border rounded-lg">
                {prediction.metrics.improvement_over_planned > 0 ? (
                  <div className="flex items-center gap-2 text-green-600">
                    <CheckCircle2 className="h-5 w-5" />
                    <span className="font-semibold">AI model improves upon the planned route!</span>
                  </div>
                ) : prediction.metrics.improvement_over_planned < 0 ? (
                  <div className="flex items-center gap-2 text-orange-600">
                    <TrendingUp className="h-5 w-5" />
                    <span className="font-semibold">Planned route was already quite good for this case.</span>
                  </div>
                ) : (
                  <div className="flex items-center gap-2 text-blue-600">
                    <Target className="h-5 w-5" />
                    <span className="font-semibold">AI model matches the planned route performance.</span>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

          {/* Sequence Comparison */}
          <Card>
            <CardHeader>
              <CardTitle>Sequence Comparison</CardTitle>
              <CardDescription>Comparing planned, actual (driver), and AI-predicted sequences</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {/* Legend */}
                <div className="flex gap-4 text-sm">
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full bg-blue-500"></div>
                    <span>Planned (Algorithm)</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full bg-green-500"></div>
                    <span>Actual (Driver)</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full bg-purple-500"></div>
                    <span>Predicted (AI)</span>
                  </div>
                </div>

                {/* Stops Table */}
                <div className="border rounded-lg overflow-hidden">
                  <table className="w-full">
                    <thead className="bg-muted">
                      <tr>
                        <th className="px-4 py-2 text-left">Stop ID</th>
                        <th className="px-4 py-2 text-center">Type</th>
                        <th className="px-4 py-2 text-center">Planned</th>
                        <th className="px-4 py-2 text-center">Actual</th>
                        <th className="px-4 py-2 text-center">AI Predicted</th>
                        <th className="px-4 py-2 text-center">Confidence</th>
                        <th className="px-4 py-2 text-center">Match</th>
                      </tr>
                    </thead>
                    <tbody>
                      {vizData.stops.map((stop, idx) => (
                        <tr key={stop.stop_id} className="border-t hover:bg-muted/50">
                          <td className="px-4 py-2 font-mono text-sm">{stop.stop_id}</td>
                          <td className="px-4 py-2 text-center">
                            {stop.is_depot ? (
                              <Badge variant="outline">Depot</Badge>
                            ) : (
                              <Badge variant="secondary">Delivery</Badge>
                            )}
                          </td>
                          <td className="px-4 py-2 text-center text-blue-600 font-semibold">
                            {stop.planned_position}
                          </td>
                          <td className="px-4 py-2 text-center text-green-600 font-semibold">
                            {stop.actual_position}
                          </td>
                          <td className={`px-4 py-2 text-center font-semibold ${getSequenceComparisonColor(stop.predicted_position, stop.actual_position)}`}>
                            {stop.predicted_position}
                          </td>
                          <td className="px-4 py-2 text-center">
                            <Badge variant={stop.confidence > 0.8 ? "default" : stop.confidence > 0.6 ? "secondary" : "outline"}>
                              {(stop.confidence * 100).toFixed(0)}%
                            </Badge>
                          </td>
                          <td className="px-4 py-2 text-center">
                            {stop.predicted_position === stop.actual_position ? (
                              <CheckCircle2 className="h-5 w-5 text-green-600 mx-auto" />
                            ) : (
                              <span className="text-muted-foreground">-</span>
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                {/* Sequence Visualization */}
                <div className="space-y-3">
                  <div>
                    <div className="text-sm font-semibold mb-2 text-blue-600">Planned Sequence:</div>
                    <div className="flex gap-2 flex-wrap">
                      {vizData.sequences.planned.map((pos, idx) => (
                        <div key={idx} className="flex items-center gap-1">
                          <Badge variant="outline" className="bg-blue-50">{pos}</Badge>
                          {idx < vizData.sequences.planned.length - 1 && <ArrowRight className="h-3 w-3" />}
                        </div>
                      ))}
                    </div>
                  </div>

                  <div>
                    <div className="text-sm font-semibold mb-2 text-green-600">Actual Sequence (Driver):</div>
                    <div className="flex gap-2 flex-wrap">
                      {vizData.sequences.actual.map((pos, idx) => (
                        <div key={idx} className="flex items-center gap-1">
                          <Badge variant="outline" className="bg-green-50">{pos}</Badge>
                          {idx < vizData.sequences.actual.length - 1 && <ArrowRight className="h-3 w-3" />}
                        </div>
                      ))}
                    </div>
                  </div>

                  <div>
                    <div className="text-sm font-semibold mb-2 text-purple-600">AI Predicted Sequence:</div>
                    <div className="flex gap-2 flex-wrap">
                      {vizData.sequences.predicted.map((pos, idx) => (
                        <div key={idx} className="flex items-center gap-1">
                          <Badge 
                            variant="outline" 
                            className={vizData.sequences.predicted[idx] === vizData.sequences.actual[idx] ? "bg-green-100" : "bg-purple-50"}
                          >
                            {pos}
                          </Badge>
                          {idx < vizData.sequences.predicted.length - 1 && <ArrowRight className="h-3 w-3" />}
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Explanation */}
          <Card>
            <CardHeader>
              <CardTitle>How It Works</CardTitle>
            </CardHeader>
            <CardContent className="prose prose-sm max-w-none">
              <p>
                This Deep Learning model uses a <strong>Transformer architecture</strong> with attention mechanisms to learn optimal route sequences from historical driver behavior. 
              </p>
              <ul>
                <li><strong>Planned Sequence:</strong> Generated by traditional optimization algorithms (OR-Tools)</li>
                <li><strong>Actual Sequence:</strong> The route order that experienced drivers actually followed</li>
                <li><strong>AI Predicted:</strong> Our DL model's prediction, learned from driver patterns</li>
              </ul>
              <p>
                The model learns implicit knowledge from drivers (traffic patterns, time windows, local geography) that traditional algorithms may miss. 
                The <strong>Kendall Tau</strong> metric measures correlation between sequences (1.0 = perfect match, 0 = no correlation).
              </p>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  )
}

