"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { 
  ArrowRight, 
  Lightbulb, 
  TrendingUp, 
  Users, 
  Clock,
  MapPin,
  Zap,
  CheckCircle2
} from "lucide-react"
import { SimulationOptimizationResponse } from "@/lib/api"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"

interface OptimizationRecommendationPanelProps {
  optimizationResult: SimulationOptimizationResponse
}

export function OptimizationRecommendationPanel({ 
  optimizationResult 
}: OptimizationRecommendationPanelProps) {
  const { old_sequence, new_sequence, actions, impact } = optimizationResult

  const getActionIcon = (actionType: string) => {
    switch (actionType) {
      case "swap_stops":
        return <ArrowRight className="h-5 w-5 text-blue-500" />
      case "driver_reassignment":
      case "driver_suggest":
        return <Users className="h-5 w-5 text-purple-500" />
      case "adjust_time_windows":
        return <Clock className="h-5 w-5 text-orange-500" />
      case "remove_backtracking":
        return <MapPin className="h-5 w-5 text-green-500" />
      default:
        return <Lightbulb className="h-5 w-5 text-yellow-500" />
    }
  }

  const getActionLabel = (actionType: string) => {
    switch (actionType) {
      case "swap_stops":
        return "Stop Reordering"
      case "driver_reassignment":
      case "driver_suggest":
        return "Driver Reassignment"
      case "adjust_time_windows":
        return "Time Window Adjustment"
      case "remove_backtracking":
        return "Remove Backtracking"
      default:
        return actionType.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
    }
  }

  const hasSequenceChange = JSON.stringify(old_sequence) !== JSON.stringify(new_sequence)

  return (
    <div className="space-y-6">
      {/* Overall Impact Summary */}
      <Card className="border-green-200 bg-green-50/50">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Zap className="h-5 w-5 text-green-600" />
            Expected Optimization Impact
          </CardTitle>
          <CardDescription>
            Estimated improvements from applying all recommendations
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-3">
            <div className="p-4 bg-white rounded-lg border border-green-200">
              <div className="flex items-center justify-between mb-2">
                <p className="text-sm font-medium text-muted-foreground">Delay Reduction</p>
                <Clock className="h-4 w-4 text-green-600" />
              </div>
              <p className="text-2xl font-bold text-green-600">
                {impact.delay_reduction_minutes.toFixed(1)} min
              </p>
              <p className="text-xs text-muted-foreground mt-1">
                Less time spent delayed
              </p>
            </div>

            <div className="p-4 bg-white rounded-lg border border-green-200">
              <div className="flex items-center justify-between mb-2">
                <p className="text-sm font-medium text-muted-foreground">Energy Savings</p>
                <MapPin className="h-4 w-4 text-green-600" />
              </div>
              <p className="text-2xl font-bold text-green-600">
                {impact.energy_saving_km.toFixed(1)} km
              </p>
              <p className="text-xs text-muted-foreground mt-1">
                Reduced distance traveled
              </p>
            </div>

            <div className="p-4 bg-white rounded-lg border border-green-200">
              <div className="flex items-center justify-between mb-2">
                <p className="text-sm font-medium text-muted-foreground">On-Time Rate</p>
                <TrendingUp className="h-4 w-4 text-green-600" />
              </div>
              <p className="text-2xl font-bold text-green-600">
                +{(impact.on_time_rate_improvement * 100).toFixed(1)}%
              </p>
              <p className="text-xs text-muted-foreground mt-1">
                Improvement in reliability
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Sequence Comparison (if changed) */}
      {hasSequenceChange && (
        <Card>
          <CardHeader>
            <CardTitle>Suggested Stop Sequence Changes</CardTitle>
            <CardDescription>
              Optimized order of delivery stops to reduce delays
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid gap-4 md:grid-cols-2">
              {/* Old Sequence */}
              <div>
                <p className="text-sm font-medium mb-3 text-muted-foreground">
                  Original Sequence:
                </p>
                <div className="flex flex-wrap gap-2">
                  {old_sequence.slice(0, 15).map((stopId, idx) => (
                    <Badge 
                      key={idx} 
                      variant="outline"
                      className="text-xs"
                    >
                      {stopId}
                    </Badge>
                  ))}
                  {old_sequence.length > 15 && (
                    <Badge variant="outline" className="text-xs">
                      +{old_sequence.length - 15} more
                    </Badge>
                  )}
                </div>
              </div>

              {/* New Sequence */}
              <div>
                <p className="text-sm font-medium mb-3 text-green-600">
                  Optimized Sequence:
                </p>
                <div className="flex flex-wrap gap-2">
                  {new_sequence.slice(0, 15).map((stopId, idx) => {
                    const wasReordered = old_sequence[idx] !== stopId
                    return (
                      <Badge 
                        key={idx}
                        variant={wasReordered ? "default" : "outline"}
                        className={wasReordered ? "bg-green-500 text-xs" : "text-xs"}
                      >
                        {stopId}
                      </Badge>
                    )
                  })}
                  {new_sequence.length > 15 && (
                    <Badge variant="outline" className="text-xs">
                      +{new_sequence.length - 15} more
                    </Badge>
                  )}
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Recommended Actions */}
      <Card>
        <CardHeader>
          <CardTitle>Recommended Actions</CardTitle>
          <CardDescription>
            Specific optimization steps with explanations
          </CardDescription>
        </CardHeader>
        <CardContent>
          {actions.length === 0 ? (
            <Alert>
              <CheckCircle2 className="h-4 w-4" />
              <AlertTitle>Route Already Optimal</AlertTitle>
              <AlertDescription>
                No significant improvements identified. The current route configuration is performing well.
              </AlertDescription>
            </Alert>
          ) : (
            <div className="space-y-4">
              {actions.map((action, idx) => (
                <div 
                  key={idx}
                  className="p-4 border rounded-lg hover:shadow-md transition-shadow"
                >
                  {/* Action Header */}
                  <div className="flex items-start gap-3 mb-3">
                    <div className="mt-0.5">
                      {getActionIcon(action.action)}
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <h4 className="font-semibold">
                          {getActionLabel(action.action)}
                        </h4>
                        <Badge variant="secondary" className="text-xs">
                          Action {idx + 1}
                        </Badge>
                      </div>
                    </div>
                  </div>

                  {/* Why this action? */}
                  <div className="mb-3 pl-8">
                    <p className="text-xs font-medium text-muted-foreground mb-1">
                      Why this recommendation?
                    </p>
                    <p className="text-sm">{action.reason}</p>
                  </div>

                  {/* Expected Benefit */}
                  <div className="pl-8">
                    <p className="text-xs font-medium text-muted-foreground mb-1">
                      Expected Benefit:
                    </p>
                    <p className="text-sm font-medium text-green-600">
                      {action.expected_benefit}
                    </p>
                  </div>

                  {/* Action Details */}
                  {action.details && Object.keys(action.details).length > 0 && (
                    <div className="mt-3 pt-3 border-t pl-8">
                      <p className="text-xs font-medium text-muted-foreground mb-2">
                        Details:
                      </p>
                      <div className="grid gap-2 text-xs">
                        {Object.entries(action.details).map(([key, value]) => (
                          <div key={key} className="flex justify-between">
                            <span className="text-muted-foreground">
                              {key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}:
                            </span>
                            <span className="font-medium">
                              {Array.isArray(value) 
                                ? value.join(', ') 
                                : typeof value === 'object'
                                ? JSON.stringify(value)
                                : String(value)}
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Implementation Guidance */}
      {actions.length > 0 && (
        <Card className="border-blue-200 bg-blue-50/50">
          <CardHeader>
            <CardTitle className="text-base">Implementation Guidance</CardTitle>
          </CardHeader>
          <CardContent>
            <ol className="space-y-2 text-sm">
              <li className="flex gap-2">
                <span className="font-bold text-blue-600">1.</span>
                <span>Review each recommendation and verify it matches operational constraints</span>
              </li>
              <li className="flex gap-2">
                <span className="font-bold text-blue-600">2.</span>
                <span>Apply actions in the order listed for maximum impact</span>
              </li>
              <li className="flex gap-2">
                <span className="font-bold text-blue-600">3.</span>
                <span>Monitor actual performance after implementation</span>
              </li>
              <li className="flex gap-2">
                <span className="font-bold text-blue-600">4.</span>
                <span>Fine-tune parameters based on real-world results</span>
              </li>
            </ol>
          </CardContent>
        </Card>
      )}
    </div>
  )
}

