"use client"

import { useState } from "react"
import { DelayPredictionTable } from "@/components/delay-prediction-table"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { RefreshCw, Download } from "lucide-react"
import { RouteChart } from "@/components/route-chart"
import { 
  Select, 
  SelectContent, 
  SelectItem, 
  SelectTrigger, 
  SelectValue 
} from "@/components/ui/select"

export default function PredictionsPage() {
  const [loading, setLoading] = useState(false)
  const [modelType, setModelType] = useState("random_forest")

  // Mock prediction data
  const mockPredictions = Array.from({ length: 50 }, (_, i) => ({
    route_id: Math.floor(i / 5) + 1,
    stop_id: i + 1,
    driver_id: Math.floor(i / 5) + 1,
    delayed_flag_pred: Math.random() > 0.7 ? 1 : 0,
    delay_probability: Math.random(),
    delay_minutes_pred: Math.random() * 30,
  }))

  const delayDistribution = [
    { name: "0-5 min", value: 25 },
    { name: "5-10 min", value: 35 },
    { name: "10-15 min", value: 20 },
    { name: "15-20 min", value: 12 },
    { name: "20+ min", value: 8 },
  ]

  const handleRefresh = async () => {
    setLoading(true)
    // Simulate API call
    setTimeout(() => setLoading(false), 2000)
  }

  const highRiskCount = mockPredictions.filter(p => p.delay_probability > 0.7).length
  const mediumRiskCount = mockPredictions.filter(p => p.delay_probability > 0.4 && p.delay_probability <= 0.7).length
  const lowRiskCount = mockPredictions.filter(p => p.delay_probability <= 0.4).length

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Delay Predictions</h1>
          <p className="text-muted-foreground">
            AI-powered delay predictions and risk analysis
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Select value={modelType} onValueChange={setModelType}>
            <SelectTrigger className="w-[200px]">
              <SelectValue placeholder="Select model" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="random_forest">Random Forest</SelectItem>
              <SelectItem value="logistic_regression">Logistic Regression</SelectItem>
              <SelectItem value="lstm">LSTM Neural Network</SelectItem>
            </SelectContent>
          </Select>
          <Button onClick={handleRefresh} disabled={loading}>
            <RefreshCw className={`mr-2 h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
          <Button variant="outline">
            <Download className="mr-2 h-4 w-4" />
            Export
          </Button>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">High Risk</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-red-500">{highRiskCount}</div>
            <p className="text-xs text-muted-foreground">Probability &gt; 70%</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Medium Risk</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-orange-500">{mediumRiskCount}</div>
            <p className="text-xs text-muted-foreground">Probability 40-70%</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Low Risk</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-green-500">{lowRiskCount}</div>
            <p className="text-xs text-muted-foreground">Probability &lt; 40%</p>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Prediction Results</CardTitle>
          <CardDescription>
            Delay predictions for all active stops using {modelType} model
          </CardDescription>
        </CardHeader>
        <CardContent>
          <DelayPredictionTable predictions={mockPredictions} />
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Prediction Confidence</CardTitle>
          <CardDescription>Model accuracy and confidence metrics</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-2">
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-sm text-muted-foreground">Model Accuracy:</span>
                <span className="font-medium">87.3%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-muted-foreground">Precision:</span>
                <span className="font-medium">84.1%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-muted-foreground">Recall:</span>
                <span className="font-medium">89.5%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-muted-foreground">F1-Score:</span>
                <span className="font-medium">86.7%</span>
              </div>
            </div>
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-sm text-muted-foreground">Total Predictions:</span>
                <span className="font-medium">{mockPredictions.length}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-muted-foreground">Avg Confidence:</span>
                <span className="font-medium">82.4%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-muted-foreground">Last Updated:</span>
                <span className="font-medium">Just now</span>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

