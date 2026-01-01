"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { 
  Play, 
  Pause, 
  SkipForward, 
  SkipBack,
  AlertTriangle,
  CheckCircle2,
  Zap,
  TrendingUp
} from "lucide-react"
import { 
  SimulationPredictionResponse, 
  SimulationOptimizationResponse 
} from "@/lib/api"

interface SimulationPlaybackTimelineProps {
  originalPrediction: SimulationPredictionResponse
  optimizationResult: SimulationOptimizationResponse
}

type PlaybackStage = 
  | "original_route"
  | "disruption"
  | "prediction"
  | "optimization"
  | "new_route"

const STAGES: Array<{
  id: PlaybackStage
  title: string
  description: string
  icon: React.ReactNode
}> = [
  {
    id: "original_route",
    title: "Original Route",
    description: "Starting with the baseline route configuration",
    icon: <CheckCircle2 className="h-5 w-5" />
  },
  {
    id: "disruption",
    title: "Simulated Disruption",
    description: "Applied scenario modifications to test system response",
    icon: <AlertTriangle className="h-5 w-5 text-orange-500" />
  },
  {
    id: "prediction",
    title: "ML Model Prediction",
    description: "AI analyzes the situation and predicts delays",
    icon: <Zap className="h-5 w-5 text-blue-500" />
  },
  {
    id: "optimization",
    title: "Optimization Decision",
    description: "System generates corrective action recommendations",
    icon: <TrendingUp className="h-5 w-5 text-purple-500" />
  },
  {
    id: "new_route",
    title: "Optimized Route",
    description: "New route configuration with improvements applied",
    icon: <CheckCircle2 className="h-5 w-5 text-green-500" />
  }
]

export function SimulationPlaybackTimeline({ 
  originalPrediction, 
  optimizationResult 
}: SimulationPlaybackTimelineProps) {
  const [currentStage, setCurrentStage] = useState<number>(0)
  const [isPlaying, setIsPlaying] = useState(false)

  const handlePlay = () => {
    if (currentStage >= STAGES.length - 1) {
      setCurrentStage(0)
    }
    setIsPlaying(true)
    
    // Auto-advance through stages
    const interval = setInterval(() => {
      setCurrentStage(prev => {
        if (prev >= STAGES.length - 1) {
          setIsPlaying(false)
          clearInterval(interval)
          return prev
        }
        return prev + 1
      })
    }, 3000) // 3 seconds per stage
  }

  const handlePause = () => {
    setIsPlaying(false)
  }

  const handleNext = () => {
    if (currentStage < STAGES.length - 1) {
      setCurrentStage(currentStage + 1)
    }
  }

  const handlePrevious = () => {
    if (currentStage > 0) {
      setCurrentStage(currentStage - 1)
    }
  }

  const handleStageClick = (index: number) => {
    setCurrentStage(index)
    setIsPlaying(false)
  }

  const currentStageInfo = STAGES[currentStage]
  const progressPercentage = ((currentStage + 1) / STAGES.length) * 100

  return (
    <div className="space-y-6">
      {/* Playback Controls */}
      <Card>
        <CardHeader>
          <CardTitle>Simulation Playback</CardTitle>
          <CardDescription>
            Step-by-step visualization of the optimization decision process
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Progress Bar */}
          <div className="space-y-2">
            <div className="flex items-center justify-between text-sm">
              <span className="text-muted-foreground">Progress</span>
              <span className="font-medium">{currentStage + 1} / {STAGES.length}</span>
            </div>
            <Progress value={progressPercentage} className="h-2" />
          </div>

          {/* Control Buttons */}
          <div className="flex items-center justify-center gap-2">
            <Button
              variant="outline"
              size="icon"
              onClick={handlePrevious}
              disabled={currentStage === 0 || isPlaying}
            >
              <SkipBack className="h-4 w-4" />
            </Button>

            {isPlaying ? (
              <Button size="icon" onClick={handlePause}>
                <Pause className="h-4 w-4" />
              </Button>
            ) : (
              <Button size="icon" onClick={handlePlay}>
                <Play className="h-4 w-4" />
              </Button>
            )}

            <Button
              variant="outline"
              size="icon"
              onClick={handleNext}
              disabled={currentStage === STAGES.length - 1 || isPlaying}
            >
              <SkipForward className="h-4 w-4" />
            </Button>
          </div>

          {/* Stage Indicators */}
          <div className="flex justify-between gap-2">
            {STAGES.map((stage, index) => (
              <button
                key={stage.id}
                onClick={() => handleStageClick(index)}
                disabled={isPlaying}
                className={`flex-1 p-3 border rounded-lg text-left transition-all ${
                  index === currentStage
                    ? 'border-primary bg-primary/5 shadow-sm'
                    : index < currentStage
                    ? 'border-green-200 bg-green-50/50'
                    : 'border-border hover:border-primary/50'
                }`}
              >
                <div className="flex items-center gap-2 mb-1">
                  {stage.icon}
                  <span className="text-xs font-medium">{index + 1}</span>
                </div>
                <p className="text-xs">{stage.title}</p>
              </button>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Current Stage Details */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-3">
            {currentStageInfo.icon}
            <div>
              <CardTitle>Stage {currentStage + 1}: {currentStageInfo.title}</CardTitle>
              <CardDescription>{currentStageInfo.description}</CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {currentStage === 0 && (
            <OriginalRouteStage prediction={originalPrediction} />
          )}
          {currentStage === 1 && (
            <DisruptionStage prediction={originalPrediction} />
          )}
          {currentStage === 2 && (
            <PredictionStage prediction={originalPrediction} />
          )}
          {currentStage === 3 && (
            <OptimizationStage optimizationResult={optimizationResult} />
          )}
          {currentStage === 4 && (
            <NewRouteStage 
              originalPrediction={originalPrediction}
              optimizationResult={optimizationResult}
            />
          )}
        </CardContent>
      </Card>
    </div>
  )
}

// Stage Components

function OriginalRouteStage({ prediction }: { prediction: SimulationPredictionResponse }) {
  return (
    <div className="space-y-4">
      <div className="p-4 bg-muted rounded-lg">
        <h4 className="font-semibold mb-3">Baseline Route Configuration</h4>
        <div className="grid gap-3 md:grid-cols-2">
          <div>
            <p className="text-sm text-muted-foreground">Total Stops</p>
            <p className="text-xl font-bold">{prediction.predictions.length}</p>
          </div>
          <div>
            <p className="text-sm text-muted-foreground">Total Distance</p>
            <p className="text-xl font-bold">
              {prediction.route_summary.expected_total_distance.toFixed(1)} km
            </p>
          </div>
        </div>
      </div>
      <p className="text-sm text-muted-foreground">
        This is the original route before any disruptions or optimizations.
      </p>
    </div>
  )
}

function DisruptionStage({ prediction }: { prediction: SimulationPredictionResponse }) {
  return (
    <div className="space-y-4">
      <div className="p-4 border border-orange-200 bg-orange-50/50 rounded-lg">
        <h4 className="font-semibold mb-3 flex items-center gap-2">
          <AlertTriangle className="h-5 w-5 text-orange-500" />
          Scenario Applied
        </h4>
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-sm">Scenario Type:</span>
            <Badge variant="outline" className="text-orange-700">
              {prediction.scenario_applied.scenario_type.toUpperCase().replace(/_/g, ' ')}
            </Badge>
          </div>
          {prediction.scenario_applied.applied_changes.map((change, idx) => (
            <div key={idx} className="text-sm p-2 bg-white rounded border">
              <p className="font-medium">{change.description}</p>
              {change.reason && (
                <p className="text-xs text-muted-foreground mt-1">→ {change.reason}</p>
              )}
            </div>
          ))}
        </div>
      </div>
      <p className="text-sm text-muted-foreground">
        External factors have been introduced to test system resilience.
      </p>
    </div>
  )
}

function PredictionStage({ prediction }: { prediction: SimulationPredictionResponse }) {
  const highRiskStops = prediction.predictions.filter(p => p.risk_level === 'HIGH')
  
  return (
    <div className="space-y-4">
      <div className="p-4 border border-blue-200 bg-blue-50/50 rounded-lg">
        <h4 className="font-semibold mb-3 flex items-center gap-2">
          <Zap className="h-5 w-5 text-blue-500" />
          ML Model Analysis
        </h4>
        <div className="grid gap-3 md:grid-cols-3">
          <div>
            <p className="text-sm text-muted-foreground">Total Delay Predicted</p>
            <p className="text-xl font-bold text-red-600">
              {prediction.route_summary.total_expected_delay.toFixed(1)} min
            </p>
          </div>
          <div>
            <p className="text-sm text-muted-foreground">Delay Risk</p>
            <p className="text-xl font-bold text-orange-600 capitalize">
              {prediction.route_summary.delay_risk}
            </p>
          </div>
          <div>
            <p className="text-sm text-muted-foreground">High Risk Stops</p>
            <p className="text-xl font-bold text-red-600">
              {highRiskStops.length}
            </p>
          </div>
        </div>
      </div>
      
      {highRiskStops.length > 0 && (
        <div>
          <p className="text-sm font-medium mb-2">Identified Problem Stops:</p>
          <div className="space-y-2">
            {highRiskStops.slice(0, 3).map((stop) => (
              <div key={stop.stop_id} className="text-sm p-2 bg-red-50 rounded border border-red-200">
                <div className="flex items-center justify-between">
                  <span className="font-medium">Stop {stop.stop_id}</span>
                  <Badge variant="destructive">High Risk</Badge>
                </div>
                <p className="text-xs text-muted-foreground mt-1">{stop.reason}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

function OptimizationStage({ optimizationResult }: { optimizationResult: SimulationOptimizationResponse }) {
  return (
    <div className="space-y-4">
      <div className="p-4 border border-purple-200 bg-purple-50/50 rounded-lg">
        <h4 className="font-semibold mb-3 flex items-center gap-2">
          <TrendingUp className="h-5 w-5 text-purple-500" />
          Optimization Engine Recommendations
        </h4>
        <p className="text-sm mb-3">
          {optimizationResult.actions.length} optimization actions identified
        </p>
        <div className="space-y-2">
          {optimizationResult.actions.map((action, idx) => (
            <div key={idx} className="p-2 bg-white rounded border">
              <p className="text-sm font-medium">
                {idx + 1}. {action.action.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
              </p>
              <p className="text-xs text-muted-foreground mt-1">{action.reason}</p>
            </div>
          ))}
        </div>
      </div>
      <p className="text-sm text-muted-foreground">
        System has analyzed the predictions and generated corrective actions.
      </p>
    </div>
  )
}

function NewRouteStage({ 
  originalPrediction, 
  optimizationResult 
}: { 
  originalPrediction: SimulationPredictionResponse
  optimizationResult: SimulationOptimizationResponse
}) {
  return (
    <div className="space-y-4">
      <div className="p-4 border border-green-200 bg-green-50/50 rounded-lg">
        <h4 className="font-semibold mb-3 flex items-center gap-2">
          <CheckCircle2 className="h-5 w-5 text-green-500" />
          Optimized Route Results
        </h4>
        <div className="grid gap-3 md:grid-cols-3">
          <div>
            <p className="text-sm text-muted-foreground">Delay Reduced By</p>
            <p className="text-xl font-bold text-green-600">
              {optimizationResult.impact.delay_reduction_minutes.toFixed(1)} min
            </p>
          </div>
          <div>
            <p className="text-sm text-muted-foreground">Distance Saved</p>
            <p className="text-xl font-bold text-green-600">
              {optimizationResult.impact.energy_saving_km.toFixed(1)} km
            </p>
          </div>
          <div>
            <p className="text-sm text-muted-foreground">Reliability Boost</p>
            <p className="text-xl font-bold text-green-600">
              +{(optimizationResult.impact.on_time_rate_improvement * 100).toFixed(1)}%
            </p>
          </div>
        </div>
      </div>
      
      <div className="p-4 bg-muted rounded-lg">
        <h4 className="font-semibold mb-2">Final Route Performance</h4>
        <div className="grid gap-2 md:grid-cols-2">
          <div className="flex justify-between">
            <span className="text-sm text-muted-foreground">New Total Delay:</span>
            <span className="font-medium">
              {(originalPrediction.route_summary.total_expected_delay - optimizationResult.impact.delay_reduction_minutes).toFixed(1)} min
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-sm text-muted-foreground">New Total Distance:</span>
            <span className="font-medium">
              {(originalPrediction.route_summary.expected_total_distance - optimizationResult.impact.energy_saving_km).toFixed(1)} km
            </span>
          </div>
        </div>
      </div>
      
      <p className="text-sm text-muted-foreground">
        ✓ Optimization complete! The route is now more efficient and reliable.
      </p>
    </div>
  )
}

