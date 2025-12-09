"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { 
  AlertTriangle, 
  CheckCircle2, 
  Clock, 
  TrendingUp,
  Info,
  AlertCircle
} from "lucide-react"
import { SimulationPredictionResponse } from "@/lib/api"
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip"

interface PredictionOutputPanelProps {
  predictionResult: SimulationPredictionResponse
}

export function PredictionOutputPanel({ predictionResult }: PredictionOutputPanelProps) {
  const { predictions, route_summary, scenario_applied } = predictionResult

  const getRiskBadge = (riskLevel: string) => {
    switch (riskLevel.toUpperCase()) {
      case "HIGH":
        return <Badge variant="destructive">High Risk</Badge>
      case "MEDIUM":
        return <Badge className="bg-yellow-500">Medium Risk</Badge>
      case "LOW":
        return <Badge className="bg-green-500">Low Risk</Badge>
      default:
        return <Badge variant="outline">{riskLevel}</Badge>
    }
  }

  const getDelayRiskColor = (risk: string) => {
    switch (risk) {
      case "high": return "text-red-500"
      case "medium": return "text-yellow-500"
      case "low": return "text-green-500"
      default: return "text-gray-500"
    }
  }

  const getDelayRiskIcon = (risk: string) => {
    switch (risk) {
      case "high": return <AlertTriangle className="h-5 w-5 text-red-500" />
      case "medium": return <AlertCircle className="h-5 w-5 text-yellow-500" />
      case "low": return <CheckCircle2 className="h-5 w-5 text-green-500" />
      default: return <Info className="h-5 w-5 text-gray-500" />
    }
  }

  return (
    <div className="space-y-6">
      {/* Scenario Applied Summary */}
      <Card className="border-blue-200 bg-blue-50/50">
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Scenario Applied</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium">Scenario Type:</span>
              <Badge variant="outline" className="text-blue-700">
                {scenario_applied.scenario_type.replace(/_/g, ' ').toUpperCase()}
              </Badge>
            </div>
            {scenario_applied.applied_changes.length > 0 && (
              <div className="mt-3 space-y-2">
                <p className="text-xs font-medium text-muted-foreground">Applied Changes:</p>
                {scenario_applied.applied_changes.map((change, idx) => (
                  <div key={idx} className="p-2 bg-white rounded border border-blue-100">
                    <p className="text-xs font-medium">{change.description}</p>
                    {change.reason && (
                      <p className="text-xs text-muted-foreground mt-1">
                        → {change.reason}
                      </p>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Route Summary */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            Route Performance Summary
            {getDelayRiskIcon(route_summary.delay_risk)}
          </CardTitle>
          <CardDescription>
            Overall predictions for Route {predictionResult.route_id}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            <div className="p-4 bg-muted rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <p className="text-sm font-medium text-muted-foreground">Total Expected Delay</p>
                <Clock className="h-4 w-4 text-muted-foreground" />
              </div>
              <p className="text-2xl font-bold">{route_summary.total_expected_delay.toFixed(1)} min</p>
            </div>

            <div className="p-4 bg-muted rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <p className="text-sm font-medium text-muted-foreground">Delay Risk Level</p>
              </div>
              <p className={`text-2xl font-bold capitalize ${getDelayRiskColor(route_summary.delay_risk)}`}>
                {route_summary.delay_risk}
              </p>
            </div>

            <div className="p-4 bg-muted rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <p className="text-sm font-medium text-muted-foreground">Expected Distance</p>
              </div>
              <p className="text-2xl font-bold">{route_summary.expected_total_distance.toFixed(1)} km</p>
            </div>

            <div className="p-4 bg-muted rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <p className="text-sm font-medium text-muted-foreground">Expected Duration</p>
              </div>
              <p className="text-2xl font-bold">{route_summary.expected_total_duration.toFixed(0)} min</p>
            </div>

            <div className="p-4 bg-muted rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <p className="text-sm font-medium text-muted-foreground">High Risk Stops</p>
                <AlertTriangle className="h-4 w-4 text-red-500" />
              </div>
              <p className="text-2xl font-bold text-red-500">{route_summary.high_risk_stops}</p>
            </div>

            <div className="p-4 bg-muted rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <p className="text-sm font-medium text-muted-foreground">On-Time Probability</p>
                <TrendingUp className="h-4 w-4 text-green-500" />
              </div>
              <p className="text-2xl font-bold text-green-500">
                {(route_summary.on_time_probability * 100).toFixed(1)}%
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Stop-Level Predictions */}
      <Card>
        <CardHeader>
          <CardTitle>Stop-by-Stop Predictions</CardTitle>
          <CardDescription>
            Detailed delay predictions with explanations for each stop
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-3 max-h-[600px] overflow-y-auto">
            {predictions.slice(0, 20).map((pred) => (
              <div 
                key={pred.stop_id}
                className={`p-4 border rounded-lg ${
                  pred.risk_level === 'HIGH' 
                    ? 'border-red-200 bg-red-50/50' 
                    : pred.risk_level === 'MEDIUM'
                    ? 'border-yellow-200 bg-yellow-50/50'
                    : 'border-green-200 bg-green-50/50'
                }`}
              >
                <div className="flex items-start justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <Badge variant="outline">Stop {pred.stop_id}</Badge>
                    <Badge variant="secondary">Route {pred.route_id}</Badge>
                    {getRiskBadge(pred.risk_level)}
                  </div>
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger>
                        <Info className="h-4 w-4 text-muted-foreground" />
                      </TooltipTrigger>
                      <TooltipContent>
                        <p className="max-w-xs text-xs">View contributing factors</p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </div>

                <div className="grid gap-3 md:grid-cols-2 mb-3">
                  <div>
                    <p className="text-xs text-muted-foreground mb-1">Predicted Delay</p>
                    <p className="text-lg font-bold">{pred.delay_minutes.toFixed(1)} minutes</p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground mb-1">Delay Probability</p>
                    <div className="flex items-center gap-2">
                      <Progress 
                        value={pred.delay_probability * 100} 
                        className="h-2 flex-1"
                      />
                      <span className="text-sm font-medium">
                        {(pred.delay_probability * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>
                </div>

                {/* Reason/Explanation */}
                <div className="p-3 bg-white rounded border">
                  <p className="text-xs font-medium text-muted-foreground mb-1">
                    Why this prediction?
                  </p>
                  <p className="text-sm">{pred.reason}</p>
                </div>

                {/* Contributing Factors */}
                {pred.contributing_factors && pred.contributing_factors.length > 0 && (
                  <div className="mt-3 pt-3 border-t">
                    <p className="text-xs font-medium text-muted-foreground mb-2">
                      Top Contributing Factors:
                    </p>
                    <div className="space-y-1">
                      {pred.contributing_factors.slice(0, 3).map((factor, idx) => (
                        <div key={idx} className="flex items-center justify-between text-xs">
                          <span className="text-muted-foreground">
                            {factor.feature.replace(/_/g, ' ')}
                          </span>
                          <span className="font-medium">
                            {(factor.importance * 100).toFixed(1)}% importance
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>

          {predictions.length > 20 && (
            <p className="text-sm text-muted-foreground text-center mt-4">
              Showing 20 of {predictions.length} stops
            </p>
          )}
        </CardContent>
      </Card>
    </div>
  )
}

