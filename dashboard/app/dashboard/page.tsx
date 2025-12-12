"use client";

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Activity,
  TrendingUp,
  AlertCircle,
  CheckCircle2,
  Clock,
  MapPin,
} from "lucide-react";
import Link from "next/link";
import { Button } from "@/components/ui/button";

export default function DashboardPage() {
  // Simple dashboard stats
  const stats = {
    totalRoutes: 20000,
    totalStops: 240000,
    delayRate: 23.0,
    avgDelay: 8.5,
    modelAccuracy: 94.77,
    f1Score: 76.84,
    recall: 88.15,
    optimizationSavings: 17.3,
  };

  return (
    <div className="flex flex-col gap-6">
      {/* Header */}
      <div className="flex flex-col gap-2">
        <h1 className="text-3xl font-bold tracking-tight">
          Dashboard Overview
        </h1>
        <p className="text-muted-foreground">
          AI-Driven Fleet Route Optimization & Delay Prediction System
        </p>
      </div>

      {/* Key Performance Indicators */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Routes</CardTitle>
            <MapPin className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {stats.totalRoutes.toLocaleString()}
            </div>
            <p className="text-xs text-muted-foreground">
              {stats.totalStops.toLocaleString()} total stops
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Delay Rate</CardTitle>
            <AlertCircle className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.delayRate}%</div>
            <p className="text-xs text-muted-foreground">
              Avg delay: {stats.avgDelay} minutes
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">
              Model Accuracy
            </CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.modelAccuracy}%</div>
            <p className="text-xs text-muted-foreground">
              F1-Score: {stats.f1Score}%
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">
              Optimization Impact
            </CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {stats.optimizationSavings}%
            </div>
            <p className="text-xs text-muted-foreground">Distance reduction</p>
          </CardContent>
        </Card>
      </div>

      {/* Model Performance Details */}
      <div className="grid gap-4 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Model Performance</CardTitle>
            <CardDescription>
              Random Forest Classifier (Production Model)
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Accuracy</span>
                <span className="text-sm font-bold">
                  {stats.modelAccuracy}%
                </span>
              </div>
              <div className="h-2 bg-secondary rounded-full overflow-hidden">
                <div
                  className="h-full bg-green-500"
                  style={{ width: `${stats.modelAccuracy}%` }}
                />
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">
                  Recall (Delay Detection)
                </span>
                <span className="text-sm font-bold">{stats.recall}%</span>
              </div>
              <div className="h-2 bg-secondary rounded-full overflow-hidden">
                <div
                  className="h-full bg-blue-500"
                  style={{ width: `${stats.recall}%` }}
                />
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">F1-Score (Balanced)</span>
                <span className="text-sm font-bold">{stats.f1Score}%</span>
              </div>
              <div className="h-2 bg-secondary rounded-full overflow-hidden">
                <div
                  className="h-full bg-purple-500"
                  style={{ width: `${stats.f1Score}%` }}
                />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>System Status</CardTitle>
            <CardDescription>Current operational status</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center gap-2">
              <CheckCircle2 className="h-5 w-5 text-green-500" />
              <div>
                <p className="text-sm font-medium">Models Loaded</p>
                <p className="text-xs text-muted-foreground">
                  All 5 models operational
                </p>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <CheckCircle2 className="h-5 w-5 text-green-500" />
              <div>
                <p className="text-sm font-medium">Validation Complete</p>
                <p className="text-xs text-muted-foreground">
                  4 validation methods passed
                </p>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <CheckCircle2 className="h-5 w-5 text-green-500" />
              <div>
                <p className="text-sm font-medium">API Status</p>
                <p className="text-xs text-muted-foreground">
                  Backend running on port 8000
                </p>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <Clock className="h-5 w-5 text-blue-500" />
              <div>
                <p className="text-sm font-medium">Optimization Ready</p>
                <p className="text-xs text-muted-foreground">
                  Scenario testing available
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Key Features */}
      <Card>
        <CardHeader>
          <CardTitle>System Capabilities</CardTitle>
          <CardDescription>What this system can do</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            <div className="space-y-2">
              <h4 className="text-sm font-semibold">Delay Prediction</h4>
              <p className="text-sm text-muted-foreground">
                Predicts delivery delays with 88% recall rate using Random
                Forest and LSTM models
              </p>
            </div>

            <div className="space-y-2">
              <h4 className="text-sm font-semibold">Route Optimization</h4>
              <p className="text-sm text-muted-foreground">
                Optimizes routes using OR-Tools VRP solver with 17.3% distance
                reduction
              </p>
            </div>

            <div className="space-y-2">
              <h4 className="text-sm font-semibold">Scenario Testing</h4>
              <p className="text-sm text-muted-foreground">
                Test different conditions (traffic, weather, delays) before
                deployment
              </p>
            </div>

            <div className="space-y-2">
              <h4 className="text-sm font-semibold">Driver Reassignment</h4>
              <p className="text-sm text-muted-foreground">
                Intelligent driver-route matching based on performance and
                conditions
              </p>
            </div>

            <div className="space-y-2">
              <h4 className="text-sm font-semibold">Real-Time Insights</h4>
              <p className="text-sm text-muted-foreground">
                Live predictions and recommendations with transparent
                explanations
              </p>
            </div>

            <div className="space-y-2">
              <h4 className="text-sm font-semibold">Validated Models</h4>
              <p className="text-sm text-muted-foreground">
                Statistically proven improvements (p &lt; 0.001) with 95%
                confidence intervals
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Call to Action */}
      <Card className="bg-primary/5 border-primary/20">
        <CardHeader>
          <CardTitle>Ready to Optimize Routes?</CardTitle>
          <CardDescription>
            Test different scenarios and see how the AI optimizes routes and
            predicts delays
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Link href="/dashboard/optimization">
            <Button size="lg" className="w-full sm:w-auto">
              Go to Optimization & Scenario Testing
            </Button>
          </Link>
        </CardContent>
      </Card>

      {/* Validation Results Summary */}
      <Card>
        <CardHeader>
          <CardTitle>Validation Results</CardTitle>
          <CardDescription>
            Comprehensive validation proves system effectiveness
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-2">
            <div className="space-y-2">
              <h4 className="text-sm font-semibold">Baseline Comparison</h4>
              <p className="text-sm text-muted-foreground">
                ML models outperform best baseline by{" "}
                <strong className="text-foreground">81%</strong>
              </p>
            </div>

            <div className="space-y-2">
              <h4 className="text-sm font-semibold">Cross-Validation</h4>
              <p className="text-sm text-muted-foreground">
                F1-Score:{" "}
                <strong className="text-foreground">76.84% ± 2.98%</strong> (95%
                CI)
              </p>
            </div>

            <div className="space-y-2">
              <h4 className="text-sm font-semibold">Temporal Validation</h4>
              <p className="text-sm text-muted-foreground">
                Future data performance drop:{" "}
                <strong className="text-foreground">only 2.53%</strong>
              </p>
            </div>

            <div className="space-y-2">
              <h4 className="text-sm font-semibold">Statistical Tests</h4>
              <p className="text-sm text-muted-foreground">
                P-value: <strong className="text-foreground">&lt; 0.001</strong>{" "}
                (highly significant)
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
