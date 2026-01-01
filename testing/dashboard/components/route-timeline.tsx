"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { PredictionResponse } from "@/lib/api"
import { AlertCircle, CheckCircle, AlertTriangle } from "lucide-react"

interface RouteTimelineProps {
  predictions: PredictionResponse[]
  title?: string
}

export function RouteTimeline({ predictions, title = "Route Timeline" }: RouteTimelineProps) {
  // Group predictions by route
  const routeGroups = predictions.reduce((acc, pred) => {
    if (!acc[pred.route_id]) {
      acc[pred.route_id] = []
    }
    acc[pred.route_id].push(pred)
    return acc
  }, {} as Record<number, PredictionResponse[]>)

  const getRiskColor = (riskLevel?: string) => {
    switch (riskLevel) {
      case "HIGH": return "bg-red-500"
      case "MEDIUM": return "bg-yellow-500"
      case "LOW": return "bg-green-500"
      default: return "bg-gray-400"
    }
  }

  const getRiskIcon = (riskLevel?: string) => {
    switch (riskLevel) {
      case "HIGH": return <AlertCircle className="h-3 w-3" />
      case "MEDIUM": return <AlertTriangle className="h-3 w-3" />
      case "LOW": return <CheckCircle className="h-3 w-3" />
      default: return null
    }
  }

  // Calculate timeline statistics
  const totalStops = predictions.length
  const highRiskStops = predictions.filter(p => p.risk_level === "HIGH").length
  const mediumRiskStops = predictions.filter(p => p.risk_level === "MEDIUM").length
  const lowRiskStops = predictions.filter(p => p.risk_level === "LOW").length

  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
        <CardDescription>
          Visual timeline showing delay risks across routes
        </CardDescription>
        <div className="flex gap-2 pt-2">
          <Badge className="bg-red-500">
            <AlertCircle className="mr-1 h-3 w-3" />
            High: {highRiskStops}
          </Badge>
          <Badge className="bg-yellow-500">
            <AlertTriangle className="mr-1 h-3 w-3" />
            Medium: {mediumRiskStops}
          </Badge>
          <Badge className="bg-green-500">
            <CheckCircle className="mr-1 h-3 w-3" />
            Low: {lowRiskStops}
          </Badge>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-4 max-h-[600px] overflow-y-auto">
          {Object.entries(routeGroups).slice(0, 10).map(([routeId, stops]) => {
            const totalDelay = stops.reduce((sum, s) => sum + s.delay_minutes_pred, 0)
            const avgProbability = stops.reduce((sum, s) => sum + (s.delay_probability || 0), 0) / stops.length

            return (
              <div key={routeId} className="space-y-2">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Badge variant="outline">Route {routeId}</Badge>
                    <span className="text-xs text-muted-foreground">
                      {stops.length} stops
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-muted-foreground">
                      Total delay: {totalDelay.toFixed(0)} min
                    </span>
                    <span className="text-xs text-muted-foreground">
                      Avg prob: {(avgProbability * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>

                {/* Timeline Bar */}
                <div className="relative h-12 bg-muted rounded-lg overflow-hidden">
                  <div className="absolute inset-0 flex">
                    {stops.map((stop, idx) => {
                      const width = (1 / stops.length) * 100
                      return (
                        <div
                          key={stop.stop_id}
                          className={`relative ${getRiskColor(stop.risk_level)} border-r border-white/20`}
                          style={{ width: `${width}%` }}
                          title={`Stop ${stop.stop_id}: ${stop.risk_level} risk, ${stop.delay_minutes_pred.toFixed(1)} min delay`}
                        >
                          <div className="absolute inset-0 flex items-center justify-center">
                            {stops.length <= 20 && (
                              <span className="text-[8px] text-white font-bold">
                                {stop.stop_id}
                              </span>
                            )}
                          </div>
                          
                          {/* Delay indicator */}
                          {stop.delay_minutes_pred > 10 && (
                            <div className="absolute top-0 right-0 p-0.5">
                              <div className="w-2 h-2 bg-white rounded-full animate-pulse" />
                            </div>
                          )}
                        </div>
                      )
                    })}
                  </div>
                </div>

                {/* Stop Details for High Risk Stops */}
                {stops.some(s => s.risk_level === "HIGH") && (
                  <div className="pl-4 border-l-2 border-red-500/30 space-y-1">
                    {stops
                      .filter(s => s.risk_level === "HIGH")
                      .slice(0, 3)
                      .map(stop => (
                        <div key={stop.stop_id} className="flex items-center justify-between text-xs">
                          <div className="flex items-center gap-2">
                            <AlertCircle className="h-3 w-3 text-red-500" />
                            <span>Stop {stop.stop_id}</span>
                            <Badge variant="destructive" className="h-4 text-[10px]">
                              {((stop.delay_probability || 0) * 100).toFixed(0)}%
                            </Badge>
                          </div>
                          <span className="text-muted-foreground">
                            {stop.delay_minutes_pred.toFixed(0)} min delay
                          </span>
                        </div>
                      ))}
                    {stops.filter(s => s.risk_level === "HIGH").length > 3 && (
                      <p className="text-[10px] text-muted-foreground pl-5">
                        +{stops.filter(s => s.risk_level === "HIGH").length - 3} more high-risk stops
                      </p>
                    )}
                  </div>
                )}
              </div>
            )
          })}
        </div>

        {Object.keys(routeGroups).length > 10 && (
          <p className="text-sm text-muted-foreground text-center mt-4">
            Showing 10 of {Object.keys(routeGroups).length} routes
          </p>
        )}

        {/* Legend */}
        <div className="mt-6 pt-4 border-t">
          <p className="text-xs font-medium mb-2">Timeline Legend</p>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs">
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 bg-red-500 rounded" />
              <span>High Risk (&gt;70%)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 bg-yellow-500 rounded" />
              <span>Medium Risk (40-70%)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 bg-green-500 rounded" />
              <span>Low Risk (&lt;40%)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-white rounded-full animate-pulse" />
              <span>Delay &gt;10 min</span>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

