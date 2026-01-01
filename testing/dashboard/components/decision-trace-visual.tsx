"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { 
  ArrowRight, 
  TrendingDown, 
  TrendingUp,
  AlertCircle,
  CheckCircle2,
  Clock,
  MapPin
} from "lucide-react"
import { 
  SimulationPredictionResponse, 
  SimulationOptimizationResponse 
} from "@/lib/api"

interface DecisionTraceVisualProps {
  originalPrediction: SimulationPredictionResponse
  optimizedPrediction?: SimulationPredictionResponse
  optimizationActions: SimulationOptimizationResponse
}

export function DecisionTraceVisual({ 
  originalPrediction, 
  optimizedPrediction,
  optimizationActions 
}: DecisionTraceVisualProps) {
  
  // Calculate improvements
  const calculateImprovement = (original: number, optimized: number) => {
    if (original === 0) return 0
    return ((original - optimized) / original) * 100
  }

  const delayImprovement = optimizedPrediction 
    ? calculateImprovement(
        originalPrediction.route_summary.total_expected_delay,
        optimizedPrediction.route_summary.total_expected_delay
      )
    : optimizationActions.impact.delay_reduction_minutes

  const distanceImprovement = optimizedPrediction
    ? calculateImprovement(
        originalPrediction.route_summary.expected_total_distance,
        optimizedPrediction.route_summary.expected_total_distance
      )
    : (optimizationActions.impact.energy_saving_km / originalPrediction.route_summary.expected_total_distance) * 100

  return (
    <div className="space-y-6">
      {/* Side-by-Side Comparison Header */}
      <Card>
        <CardHeader>
          <CardTitle>Before vs After Optimization</CardTitle>
          <CardDescription>
            Visual comparison of route performance before and after applying recommendations
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-6 lg:grid-cols-2">
            {/* BEFORE */}
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold">Before Optimization</h3>
                <Badge variant="outline">Original</Badge>
              </div>
              
              <div className="space-y-3">
                <MetricCard
                  label="Total Delay"
                  value={`${originalPrediction.route_summary.total_expected_delay.toFixed(1)} min`}
                  icon={<Clock className="h-4 w-4" />}
                  variant="warning"
                />
                
                <MetricCard
                  label="Total Distance"
                  value={`${originalPrediction.route_summary.expected_total_distance.toFixed(1)} km`}
                  icon={<MapPin className="h-4 w-4" />}
                  variant="neutral"
                />
                
                <MetricCard
                  label="High Risk Stops"
                  value={originalPrediction.route_summary.high_risk_stops.toString()}
                  icon={<AlertCircle className="h-4 w-4" />}
                  variant="danger"
                />
                
                <MetricCard
                  label="On-Time Probability"
                  value={`${(originalPrediction.route_summary.on_time_probability * 100).toFixed(1)}%`}
                  icon={<CheckCircle2 className="h-4 w-4" />}
                  variant="neutral"
                />
              </div>
            </div>

            {/* ARROW */}
            <div className="hidden lg:flex items-center justify-center">
              <ArrowRight className="h-12 w-12 text-primary" />
            </div>

            {/* AFTER (using projected improvements) */}
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold">After Optimization</h3>
                <Badge className="bg-green-500">Optimized</Badge>
              </div>
              
              <div className="space-y-3">
                <MetricCard
                  label="Total Delay"
                  value={`${(originalPrediction.route_summary.total_expected_delay - optimizationActions.impact.delay_reduction_minutes).toFixed(1)} min`}
                  icon={<Clock className="h-4 w-4" />}
                  variant="success"
                  improvement={`-${optimizationActions.impact.delay_reduction_minutes.toFixed(1)} min`}
                />
                
                <MetricCard
                  label="Total Distance"
                  value={`${(originalPrediction.route_summary.expected_total_distance - optimizationActions.impact.energy_saving_km).toFixed(1)} km`}
                  icon={<MapPin className="h-4 w-4" />}
                  variant="success"
                  improvement={`-${optimizationActions.impact.energy_saving_km.toFixed(1)} km`}
                />
                
                <MetricCard
                  label="High Risk Stops"
                  value={Math.max(0, originalPrediction.route_summary.high_risk_stops - Math.ceil(optimizationActions.actions.length * 0.5)).toString()}
                  icon={<CheckCircle2 className="h-4 w-4" />}
                  variant="success"
                  improvement="Reduced"
                />
                
                <MetricCard
                  label="On-Time Probability"
                  value={`${((originalPrediction.route_summary.on_time_probability + optimizationActions.impact.on_time_rate_improvement) * 100).toFixed(1)}%`}
                  icon={<TrendingUp className="h-4 w-4" />}
                  variant="success"
                  improvement={`+${(optimizationActions.impact.on_time_rate_improvement * 100).toFixed(1)}%`}
                />
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Improvement Summary */}
      <Card className="border-green-200 bg-green-50/50">
        <CardHeader>
          <CardTitle className="text-base">Overall Improvements</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid gap-3 md:grid-cols-3">
            <div className="flex items-center gap-3 p-3 bg-white rounded-lg border border-green-200">
              <TrendingDown className="h-8 w-8 text-green-600" />
              <div>
                <p className="text-sm text-muted-foreground">Delay Reduction</p>
                <p className="text-xl font-bold text-green-600">
                  {optimizationActions.impact.delay_reduction_minutes.toFixed(1)} min
                </p>
              </div>
            </div>

            <div className="flex items-center gap-3 p-3 bg-white rounded-lg border border-green-200">
              <TrendingDown className="h-8 w-8 text-green-600" />
              <div>
                <p className="text-sm text-muted-foreground">Distance Saved</p>
                <p className="text-xl font-bold text-green-600">
                  {optimizationActions.impact.energy_saving_km.toFixed(1)} km
                </p>
              </div>
            </div>

            <div className="flex items-center gap-3 p-3 bg-white rounded-lg border border-green-200">
              <TrendingUp className="h-8 w-8 text-green-600" />
              <div>
                <p className="text-sm text-muted-foreground">Reliability Boost</p>
                <p className="text-xl font-bold text-green-600">
                  +{(optimizationActions.impact.on_time_rate_improvement * 100).toFixed(1)}%
                </p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Decision Timeline */}
      <Card>
        <CardHeader>
          <CardTitle>Decision Timeline</CardTitle>
          <CardDescription>
            Step-by-step explanation of optimization decisions
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {/* Step 1: Analysis */}
            <TimelineStep
              number={1}
              title="Route Analysis"
              description="System analyzed the route and identified performance issues"
              details={[
                `Found ${originalPrediction.route_summary.high_risk_stops} high-risk stops`,
                `Total predicted delay: ${originalPrediction.route_summary.total_expected_delay.toFixed(1)} minutes`,
                `Delay risk level: ${originalPrediction.route_summary.delay_risk.toUpperCase()}`
              ]}
              status="completed"
            />

            {/* Step 2: Optimization Actions */}
            {optimizationActions.actions.map((action, idx) => (
              <TimelineStep
                key={idx}
                number={idx + 2}
                title={action.action.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
                description={action.reason}
                details={[action.expected_benefit]}
                status="completed"
              />
            ))}

            {/* Final Step: Result */}
            <TimelineStep
              number={optimizationActions.actions.length + 2}
              title="Optimized Route Generated"
              description="All recommendations applied successfully"
              details={[
                `Delay reduced by ${optimizationActions.impact.delay_reduction_minutes.toFixed(1)} minutes`,
                `Distance saved: ${optimizationActions.impact.energy_saving_km.toFixed(1)} km`,
                `On-time rate improved by ${(optimizationActions.impact.on_time_rate_improvement * 100).toFixed(1)}%`
              ]}
              status="success"
            />
          </div>
        </CardContent>
      </Card>

      {/* Detailed Comparison Table */}
      <Card>
        <CardHeader>
          <CardTitle>Detailed Metrics Comparison</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b">
                  <th className="text-left p-2 font-medium">Metric</th>
                  <th className="text-right p-2 font-medium">Before</th>
                  <th className="text-right p-2 font-medium">After</th>
                  <th className="text-right p-2 font-medium">Change</th>
                </tr>
              </thead>
              <tbody>
                <ComparisonRow
                  metric="Total Delay"
                  before={`${originalPrediction.route_summary.total_expected_delay.toFixed(1)} min`}
                  after={`${(originalPrediction.route_summary.total_expected_delay - optimizationActions.impact.delay_reduction_minutes).toFixed(1)} min`}
                  change={`-${optimizationActions.impact.delay_reduction_minutes.toFixed(1)} min`}
                  isPositive={true}
                />
                <ComparisonRow
                  metric="Total Distance"
                  before={`${originalPrediction.route_summary.expected_total_distance.toFixed(1)} km`}
                  after={`${(originalPrediction.route_summary.expected_total_distance - optimizationActions.impact.energy_saving_km).toFixed(1)} km`}
                  change={`-${optimizationActions.impact.energy_saving_km.toFixed(1)} km`}
                  isPositive={true}
                />
                <ComparisonRow
                  metric="On-Time Probability"
                  before={`${(originalPrediction.route_summary.on_time_probability * 100).toFixed(1)}%`}
                  after={`${((originalPrediction.route_summary.on_time_probability + optimizationActions.impact.on_time_rate_improvement) * 100).toFixed(1)}%`}
                  change={`+${(optimizationActions.impact.on_time_rate_improvement * 100).toFixed(1)}%`}
                  isPositive={true}
                />
                <ComparisonRow
                  metric="High Risk Stops"
                  before={originalPrediction.route_summary.high_risk_stops.toString()}
                  after={Math.max(0, originalPrediction.route_summary.high_risk_stops - Math.ceil(optimizationActions.actions.length * 0.5)).toString()}
                  change={`-${Math.ceil(optimizationActions.actions.length * 0.5)}`}
                  isPositive={true}
                />
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

// Helper Components

function MetricCard({ 
  label, 
  value, 
  icon, 
  variant, 
  improvement 
}: { 
  label: string
  value: string
  icon: React.ReactNode
  variant: "success" | "warning" | "danger" | "neutral"
  improvement?: string
}) {
  const variantColors = {
    success: "border-green-200 bg-green-50/50",
    warning: "border-yellow-200 bg-yellow-50/50",
    danger: "border-red-200 bg-red-50/50",
    neutral: "border-gray-200 bg-gray-50"
  }

  return (
    <div className={`p-3 border rounded-lg ${variantColors[variant]}`}>
      <div className="flex items-center justify-between mb-1">
        <p className="text-xs font-medium text-muted-foreground">{label}</p>
        {icon}
      </div>
      <p className="text-lg font-bold">{value}</p>
      {improvement && (
        <p className="text-xs text-green-600 font-medium mt-1">
          {improvement}
        </p>
      )}
    </div>
  )
}

function TimelineStep({
  number,
  title,
  description,
  details,
  status
}: {
  number: number
  title: string
  description: string
  details: string[]
  status: "completed" | "success"
}) {
  return (
    <div className="flex gap-4">
      <div className="flex flex-col items-center">
        <div className={`flex h-8 w-8 items-center justify-center rounded-full font-bold text-sm ${
          status === "success" ? "bg-green-500 text-white" : "bg-blue-500 text-white"
        }`}>
          {number}
        </div>
        <div className="w-px h-full bg-border mt-2" />
      </div>
      
      <div className="flex-1 pb-8">
        <h4 className="font-semibold mb-1">{title}</h4>
        <p className="text-sm text-muted-foreground mb-2">{description}</p>
        <ul className="space-y-1">
          {details.map((detail, idx) => (
            <li key={idx} className="text-xs text-muted-foreground">
              • {detail}
            </li>
          ))}
        </ul>
      </div>
    </div>
  )
}

function ComparisonRow({
  metric,
  before,
  after,
  change,
  isPositive
}: {
  metric: string
  before: string
  after: string
  change: string
  isPositive: boolean
}) {
  return (
    <tr className="border-b">
      <td className="p-2">{metric}</td>
      <td className="text-right p-2">{before}</td>
      <td className="text-right p-2 font-medium">{after}</td>
      <td className={`text-right p-2 font-medium ${isPositive ? 'text-green-600' : 'text-red-600'}`}>
        {change}
      </td>
    </tr>
  )
}

