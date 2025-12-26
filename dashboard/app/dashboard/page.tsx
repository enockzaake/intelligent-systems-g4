"use client";

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Activity } from "lucide-react";
import Link from "next/link";
import { Button } from "@/components/ui/button";

export default function DashboardPage() {
  // Actual results from synthetic dataset training (outputs_improved/evaluation_results_20251209_095854.json)
  const models = [
    {
      name: "Random Forest Classifier",
      description: "Production Model - Best Performance",
      accuracy: 94.77,
      precision: 68.10,
      recall: 88.15,
    f1Score: 76.84,
      rocAuc: 98.24,
    },
    {
      name: "Logistic Regression",
      description: "Baseline Model",
      accuracy: 78.47,
      precision: 28.62,
      recall: 79.59,
      f1Score: 42.10,
      rocAuc: 87.21,
    },
    {
      name: "LSTM Classifier",
      description: "Deep Learning Model",
      accuracy: 65.72,
      precision: 17.46,
      recall: 66.70,
      f1Score: 27.67,
      rocAuc: 71.93,
    },
  ];

  return (
    <div className="flex flex-col gap-6">
      {/* Header */}
      <div className="flex flex-col gap-2">
        <h1 className="text-3xl font-bold tracking-tight">
          AI-Driven Route Optimization Dashboard
        </h1>
        <p className="text-muted-foreground">
          Delay Prediction & Fleet Route Optimization using Machine Learning + Operations Research
        </p>
      </div>

      {/* Model Performance Comparison */}
        <Card>
          <CardHeader>
          <CardTitle>Model Performance Metrics</CardTitle>
            <CardDescription>
            Latest training results from actual model evaluation
            </CardDescription>
          </CardHeader>
        <CardContent>
          <div className="space-y-6">
            {models.map((model, index) => (
              <div key={index} className="space-y-3">
              <div className="flex items-center justify-between">
              <div>
                    <h4 className="text-sm font-semibold">{model.name}</h4>
                <p className="text-xs text-muted-foreground">
                      {model.description}
                </p>
              </div>
                  <Activity className="h-4 w-4 text-muted-foreground" />
                </div>
                <div className="grid grid-cols-5 gap-2">
                  <div className="space-y-1">
                    <p className="text-xs text-muted-foreground">Accuracy</p>
                    <p className="text-sm font-bold">{model.accuracy}%</p>
                  </div>
                  <div className="space-y-1">
                    <p className="text-xs text-muted-foreground">Precision</p>
                    <p className="text-sm font-bold">{model.precision}%</p>
            </div>
                  <div className="space-y-1">
                    <p className="text-xs text-muted-foreground">Recall</p>
                    <p className="text-sm font-bold">{model.recall}%</p>
              </div>
                  <div className="space-y-1">
                    <p className="text-xs text-muted-foreground">F1-Score</p>
                    <p className="text-sm font-bold">{model.f1Score}%</p>
            </div>
                  <div className="space-y-1">
                    <p className="text-xs text-muted-foreground">ROC-AUC</p>
                    <p className="text-sm font-bold">{model.rocAuc}%</p>
              </div>
            </div>
                {index < models.length - 1 && (
                  <div className="border-b pt-3" />
                )}
              </div>
            ))}
            </div>
          </CardContent>
        </Card>

      {/* System Approach */}
      <Card>
        <CardHeader>
          <CardTitle>System Approach</CardTitle>
          <CardDescription>
            Dual-task ML + OR optimization pipeline
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="space-y-2">
              <h4 className="text-sm font-semibold">1. Delay Prediction (ML)</h4>
              <p className="text-sm text-muted-foreground">
                Random Forest classifier predicts delivery delays based on 18 engineered features 
                (distance, time windows, traffic, historical patterns)
              </p>
            </div>
            <div className="space-y-2">
              <h4 className="text-sm font-semibold">2. Route Optimization (OR-Tools)</h4>
              <p className="text-sm text-muted-foreground">
                Vehicle Routing Problem (VRP) solver with time windows and capacity constraints, 
                integrating predicted delays into the optimization
              </p>
            </div>
            <div className="space-y-2">
              <h4 className="text-sm font-semibold">3. Driver Reassignment</h4>
              <p className="text-sm text-muted-foreground">
                Intelligent reassignment logic identifies high-risk stops and suggests 
                alternative drivers to minimize delays
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Call to Action */}
      <div className="grid gap-4 md:grid-cols-2">
      <Card className="bg-primary/5 border-primary/20">
        <CardHeader>
            <CardTitle>Try Route Simulation</CardTitle>
          <CardDescription>
              Interactive demo with preset scenarios and manual input
          </CardDescription>
        </CardHeader>
        <CardContent>
            <Link href="/dashboard/route-simulation">
              <Button size="lg" className="w-full">
                Open Route Simulation
            </Button>
          </Link>
        </CardContent>
      </Card>

        <Card className="bg-primary/5 border-primary/20">
        <CardHeader>
            <CardTitle>Advanced Testing</CardTitle>
          <CardDescription>
              Test different scenarios and optimization parameters
          </CardDescription>
        </CardHeader>
        <CardContent>
            <Link href="/dashboard/optimization">
              <Button size="lg" variant="outline" className="w-full">
                Open Optimization & Testing
              </Button>
            </Link>
        </CardContent>
      </Card>
      </div>
    </div>
  );
}
