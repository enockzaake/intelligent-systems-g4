"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { CheckCircle2, TrendingUp, Truck, MapPin, Clock, Users } from "lucide-react"

interface SimulationSummaryProps {
  mockStops: any[]
  params: {
    numStops: number
    numVehicles: number
    avgDelay: number
    trafficCondition: string
    weatherCondition: string
    timeOfDay: string
  }
}

export function SimulationSummary({ mockStops, params }: SimulationSummaryProps) {
  const uniqueRoutes = new Set(mockStops.map(s => s.route_id)).size
  const uniqueDrivers = new Set(mockStops.map(s => s.driver_id)).size
  const totalDistance = mockStops.reduce((sum, s) => sum + s.distancep, 0)
  const avgDistancePerStop = totalDistance / mockStops.length
  
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle>Simulation Data Generated</CardTitle>
            <CardDescription>Mock data created based on your parameters</CardDescription>
          </div>
          <Badge className="bg-green-500">
            <CheckCircle2 className="mr-1 h-3 w-3" />
            Ready
          </Badge>
        </div>
      </CardHeader>
      <CardContent>
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          <div className="flex items-start gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-full bg-blue-500/10">
              <MapPin className="h-5 w-5 text-blue-500" />
            </div>
            <div>
              <p className="text-sm text-muted-foreground">Total Stops</p>
              <p className="text-2xl font-bold">{mockStops.length}</p>
            </div>
          </div>
          
          <div className="flex items-start gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-full bg-purple-500/10">
              <Truck className="h-5 w-5 text-purple-500" />
            </div>
            <div>
              <p className="text-sm text-muted-foreground">Routes Created</p>
              <p className="text-2xl font-bold">{uniqueRoutes}</p>
            </div>
          </div>
          
          <div className="flex items-start gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-full bg-green-500/10">
              <Users className="h-5 w-5 text-green-500" />
            </div>
            <div>
              <p className="text-sm text-muted-foreground">Drivers Assigned</p>
              <p className="text-2xl font-bold">{uniqueDrivers}</p>
            </div>
          </div>
          
          <div className="flex items-start gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-full bg-orange-500/10">
              <TrendingUp className="h-5 w-5 text-orange-500" />
            </div>
            <div>
              <p className="text-sm text-muted-foreground">Total Distance</p>
              <p className="text-2xl font-bold">{totalDistance.toFixed(0)} km</p>
            </div>
          </div>
          
          <div className="flex items-start gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-full bg-cyan-500/10">
              <MapPin className="h-5 w-5 text-cyan-500" />
            </div>
            <div>
              <p className="text-sm text-muted-foreground">Avg Distance/Stop</p>
              <p className="text-2xl font-bold">{avgDistancePerStop.toFixed(1)} km</p>
            </div>
          </div>
          
          <div className="flex items-start gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-full bg-pink-500/10">
              <Clock className="h-5 w-5 text-pink-500" />
            </div>
            <div>
              <p className="text-sm text-muted-foreground">Time Period</p>
              <p className="text-2xl font-bold capitalize">{params.timeOfDay}</p>
            </div>
          </div>
        </div>
        
        <div className="mt-6 pt-6 border-t">
          <h4 className="text-sm font-medium mb-3">Simulation Conditions</h4>
          <div className="grid gap-2 md:grid-cols-3">
            <div className="flex justify-between items-center p-2 bg-muted rounded-md">
              <span className="text-sm text-muted-foreground">Traffic</span>
              <Badge variant="outline" className="capitalize">{params.trafficCondition}</Badge>
            </div>
            <div className="flex justify-between items-center p-2 bg-muted rounded-md">
              <span className="text-sm text-muted-foreground">Weather</span>
              <Badge variant="outline" className="capitalize">{params.weatherCondition}</Badge>
            </div>
            <div className="flex justify-between items-center p-2 bg-muted rounded-md">
              <span className="text-sm text-muted-foreground">Expected Delay</span>
              <Badge variant="outline">{params.avgDelay} min</Badge>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

