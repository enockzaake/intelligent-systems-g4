"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { ScenarioResult } from "@/lib/api"
import { 
  TrendingUp, 
  TrendingDown, 
  ArrowRight, 
  AlertCircle,
  CheckCircle,
  Clock,
  MapPin,
  Users
} from "lucide-react"

interface ScenarioComparisonProps {
  result: ScenarioResult
}

export function ScenarioComparison({ result }: ScenarioComparisonProps) {
  const getImprovementColor = (improvement: number) => {
    if (improvement > 10) return "text-green-500"
    if (improvement > 0) return "text-green-400"
    if (improvement < -10) return "text-red-500"
    if (improvement < 0) return "text-red-400"
    return "text-gray-500"
  }

  const getImprovementIcon = (improvement: number) => {
    if (improvement > 0) return <TrendingUp className="h-4 w-4" />
    if (improvement < 0) return <TrendingDown className="h-4 w-4" />
    return <ArrowRight className="h-4 w-4" />
  }

  const getScenarioRating = () => {
    const avgImprovement = result.comparisons.reduce((sum, c) => sum + c.improvement_percent, 0) / result.comparisons.length
    if (avgImprovement > 15) return { label: "Excellent", color: "bg-green-500", icon: CheckCircle }
    if (avgImprovement > 5) return { label: "Good", color: "bg-blue-500", icon: CheckCircle }
    if (avgImprovement > -5) return { label: "Neutral", color: "bg-yellow-500", icon: AlertCircle }
    return { label: "Poor", color: "bg-red-500", icon: AlertCircle }
  }

  const rating = getScenarioRating()
  const RatingIcon = rating.icon

  return (
    <div className="space-y-6">
      {/* Overall Rating */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>Scenario Evaluation</CardTitle>
              <CardDescription>Overall impact assessment</CardDescription>
            </div>
            <Badge className={`${rating.color} text-white`}>
              <RatingIcon className="mr-1 h-3 w-3" />
              {rating.label}
            </Badge>
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-3">
            <div className="space-y-2">
              <p className="text-sm text-muted-foreground">Modifications Applied</p>
              <p className="text-2xl font-bold">
                {result.modifications_applied.driver_reassignments +
                  result.modifications_applied.time_shifts +
                  result.modifications_applied.stops_removed}
              </p>
            </div>
            <div className="space-y-2">
              <p className="text-sm text-muted-foreground">High Risk Reduction</p>
              <p className="text-2xl font-bold">
                {result.original.high_risk_count - result.modified.high_risk_count}
              </p>
            </div>
            <div className="space-y-2">
              <p className="text-sm text-muted-foreground">Avg Improvement</p>
              <p className="text-2xl font-bold">
                {(result.comparisons.reduce((sum, c) => sum + c.improvement_percent, 0) / result.comparisons.length).toFixed(1)}%
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Side-by-Side Comparison */}
      <div className="grid gap-6 lg:grid-cols-2">
        {/* Original Scenario */}
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Original Scenario</CardTitle>
            <CardDescription>Before modifications</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <MapPin className="h-4 w-4 text-muted-foreground" />
                  <span className="text-sm">Total Stops</span>
                </div>
                <span className="font-medium">{result.original.total_stops}</span>
              </div>

              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Clock className="h-4 w-4 text-muted-foreground" />
                  <span className="text-sm">Total Delay</span>
                </div>
                <span className="font-medium">{result.original.total_delay_minutes.toFixed(1)} min</span>
              </div>

              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <AlertCircle className="h-4 w-4 text-muted-foreground" />
                  <span className="text-sm">Avg Delay Probability</span>
                </div>
                <span className="font-medium">{(result.original.avg_delay_probability * 100).toFixed(1)}%</span>
              </div>

              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <AlertCircle className="h-4 w-4 text-red-500" />
                  <span className="text-sm">High Risk Stops</span>
                </div>
                <Badge variant="destructive">{result.original.high_risk_count}</Badge>
              </div>

              {result.original.optimization && (
                <>
                  <div className="flex items-center justify-between pt-2 border-t">
                    <span className="text-sm">Total Distance</span>
                    <span className="font-medium">{result.original.optimization.total_distance.toFixed(0)} km</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm">Vehicles Used</span>
                    <span className="font-medium">{result.original.optimization.num_vehicles_used}</span>
                  </div>
                </>
              )}
            </div>
          </CardContent>
        </Card>

        {/* Modified Scenario */}
        <Card className="border-primary">
          <CardHeader>
            <CardTitle className="text-lg">Modified Scenario</CardTitle>
            <CardDescription>After modifications</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <MapPin className="h-4 w-4 text-muted-foreground" />
                  <span className="text-sm">Total Stops</span>
                </div>
                <span className="font-medium">{result.modified.total_stops}</span>
              </div>

              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Clock className="h-4 w-4 text-muted-foreground" />
                  <span className="text-sm">Total Delay</span>
                </div>
                <span className="font-medium text-green-500">{result.modified.total_delay_minutes.toFixed(1)} min</span>
              </div>

              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <AlertCircle className="h-4 w-4 text-muted-foreground" />
                  <span className="text-sm">Avg Delay Probability</span>
                </div>
                <span className="font-medium text-green-500">{(result.modified.avg_delay_probability * 100).toFixed(1)}%</span>
              </div>

              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <CheckCircle className="h-4 w-4 text-green-500" />
                  <span className="text-sm">High Risk Stops</span>
                </div>
                <Badge className="bg-green-500">{result.modified.high_risk_count}</Badge>
              </div>

              {result.modified.optimization && (
                <>
                  <div className="flex items-center justify-between pt-2 border-t">
                    <span className="text-sm">Total Distance</span>
                    <span className="font-medium text-green-500">{result.modified.optimization.total_distance.toFixed(0)} km</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm">Vehicles Used</span>
                    <span className="font-medium text-green-500">{result.modified.optimization.num_vehicles_used}</span>
                  </div>
                </>
              )}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Detailed Comparisons */}
      <Card>
        <CardHeader>
          <CardTitle>Detailed Metric Comparisons</CardTitle>
          <CardDescription>Impact analysis for each metric</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {result.comparisons.map((comparison, idx) => (
              <div key={idx} className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">{comparison.metric}</span>
                  <div className="flex items-center gap-2">
                    <span className={`text-sm font-bold ${getImprovementColor(comparison.improvement_percent)}`}>
                      {comparison.improvement_percent > 0 && '+'}
                      {comparison.improvement_percent.toFixed(1)}%
                    </span>
                    <span className={getImprovementColor(comparison.improvement_percent)}>
                      {getImprovementIcon(comparison.improvement_percent)}
                    </span>
                  </div>
                </div>
                
                <div className="flex items-center gap-2 text-xs text-muted-foreground">
                  <span>{comparison.original.toFixed(2)}</span>
                  <ArrowRight className="h-3 w-3" />
                  <span className="font-medium">{comparison.modified.toFixed(2)}</span>
                </div>

                <Progress 
                  value={Math.min(Math.abs(comparison.improvement_percent), 100)} 
                  className={`h-2 ${
                    comparison.improvement_percent > 0 ? '[&>div]:bg-green-500' : '[&>div]:bg-red-500'
                  }`}
                />
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Recommendations */}
      <Card>
        <CardHeader>
          <CardTitle>Recommendations</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {result.comparisons.some(c => c.improvement_percent > 10) ? (
              <div className="flex gap-3 p-3 bg-green-500/10 rounded-lg border border-green-500/20">
                <CheckCircle className="h-5 w-5 text-green-500 mt-0.5" />
                <div className="space-y-1">
                  <p className="text-sm font-medium">Recommended Implementation</p>
                  <p className="text-sm text-muted-foreground">
                    This scenario shows significant improvements across multiple metrics. Consider implementing these modifications.
                  </p>
                </div>
              </div>
            ) : result.comparisons.some(c => c.improvement_percent < -5) ? (
              <div className="flex gap-3 p-3 bg-red-500/10 rounded-lg border border-red-500/20">
                <AlertCircle className="h-5 w-5 text-red-500 mt-0.5" />
                <div className="space-y-1">
                  <p className="text-sm font-medium">Not Recommended</p>
                  <p className="text-sm text-muted-foreground">
                    This scenario shows negative impacts. Consider alternative modifications or revert changes.
                  </p>
                </div>
              </div>
            ) : (
              <div className="flex gap-3 p-3 bg-blue-500/10 rounded-lg border border-blue-500/20">
                <AlertCircle className="h-5 w-5 text-blue-500 mt-0.5" />
                <div className="space-y-1">
                  <p className="text-sm font-medium">Neutral Impact</p>
                  <p className="text-sm text-muted-foreground">
                    This scenario shows minimal changes. Consider more significant modifications to see greater impact.
                  </p>
                </div>
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

