"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { OptimizationResult } from "@/lib/api"
import { ArrowRight, TrendingDown, Truck, MapPin, Clock } from "lucide-react"

interface OptimizationResultsProps {
  result: OptimizationResult
}

export function OptimizationResults({ result }: OptimizationResultsProps) {
  const { solution, reassignments } = result

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle>Optimization Summary</CardTitle>
          <CardDescription>Route optimization and efficiency improvements</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="space-y-1">
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <Truck className="h-4 w-4" />
                Vehicles Used
              </div>
              <div className="text-2xl font-bold">{solution.num_vehicles_used}</div>
            </div>
            <div className="space-y-1">
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <MapPin className="h-4 w-4" />
                Total Distance
              </div>
              <div className="text-2xl font-bold">{solution.total_distance.toFixed(0)} km</div>
            </div>
            <div className="space-y-1">
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <Clock className="h-4 w-4" />
                Total Time
              </div>
              <div className="text-2xl font-bold">{solution.total_time.toFixed(0)} min</div>
            </div>
            <div className="space-y-1">
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <TrendingDown className="h-4 w-4" />
                Reassignments
              </div>
              <div className="text-2xl font-bold">{reassignments.length}</div>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Optimized Routes</CardTitle>
          <CardDescription>Detailed route assignments for each vehicle</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {solution.routes.slice(0, 10).map((route) => (
              <div key={route.vehicle_id} className="flex items-center justify-between p-3 border rounded-lg">
                <div className="flex items-center gap-3">
                  <div className="flex items-center justify-center w-10 h-10 rounded-full bg-primary/10">
                    <Truck className="h-5 w-5 text-primary" />
                  </div>
                  <div>
                    <div className="font-medium">Vehicle {route.vehicle_id}</div>
                    <div className="text-sm text-muted-foreground">
                      {route.stops.length} stops
                    </div>
                  </div>
                </div>
                <div className="text-right">
                  <div className="font-medium">{route.distance.toFixed(0)} km</div>
                  <div className="text-sm text-muted-foreground">{route.time.toFixed(0)} min</div>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {reassignments.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Driver Reassignments</CardTitle>
            <CardDescription>Suggested stop reassignments to minimize delays</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {reassignments.slice(0, 10).map((reassignment, index) => (
                <div key={index} className="flex items-center justify-between p-3 border rounded-lg">
                  <div className="flex items-center gap-3">
                    <Badge variant="outline">Stop {reassignment.stop_id}</Badge>
                    <div className="flex items-center gap-2">
                      <span className="text-sm">Driver {reassignment.original_driver}</span>
                      <ArrowRight className="h-4 w-4 text-muted-foreground" />
                      <span className="text-sm font-medium">Driver {reassignment.new_driver}</span>
                    </div>
                  </div>
                  <div className="text-right">
                    <Badge className="bg-green-500">
                      +{(reassignment.expected_improvement * 100).toFixed(1)}%
                    </Badge>
                    <div className="text-xs text-muted-foreground mt-1">improvement</div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}

