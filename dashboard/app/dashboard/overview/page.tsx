"use client"

import { useState, useEffect } from "react"
import { DashboardStats } from "@/components/dashboard-stats"
import { RouteChart } from "@/components/route-chart"
import { generateDriverData, generateRouteData } from "@/lib/mock-data"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

export default function OverviewPage() {
  const [stats, setStats] = useState({
    totalRoutes: 45,
    activeDrivers: 38,
    totalStops: 1248,
    avgDelayRate: 0.23,
    totalDistance: 4567.8,
    avgDelayMinutes: 12.5,
    onTimeRate: 0.77,
    vehicleUtilization: 0.85,
  })

  const chartData = [
    { name: "Mon", routes: 42, delays: 8, onTime: 34 },
    { name: "Tue", routes: 48, delays: 12, onTime: 36 },
    { name: "Wed", routes: 45, delays: 10, onTime: 35 },
    { name: "Thu", routes: 52, delays: 15, onTime: 37 },
    { name: "Fri", routes: 49, delays: 11, onTime: 38 },
    { name: "Sat", routes: 38, delays: 6, onTime: 32 },
    { name: "Sun", routes: 35, delays: 5, onTime: 30 },
  ]

  const delayTrendData = [
    { name: "Week 1", avgDelay: 15.2, predicted: 14.8 },
    { name: "Week 2", avgDelay: 14.1, predicted: 13.5 },
    { name: "Week 3", avgDelay: 13.8, predicted: 13.2 },
    { name: "Week 4", avgDelay: 12.5, predicted: 12.0 },
  ]

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Dashboard Overview</h1>
        <p className="text-muted-foreground">
          Monitor your fleet performance and key metrics in real-time
        </p>
      </div>

      <DashboardStats stats={stats} />

      <div className="grid gap-4 lg:grid-cols-2">
        <RouteChart data={chartData} type="area" />
        
        <Card>
          <CardHeader>
            <CardTitle>Delay Trends</CardTitle>
            <CardDescription>Actual vs Predicted delays over time</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-[350px] flex items-center justify-center text-muted-foreground">
              <div className="text-center space-y-2">
                <p>Delay trend analysis</p>
                <p className="text-sm">Average delay reduced by 18% this month</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      <Tabs defaultValue="daily" className="space-y-4">
        <TabsList>
          <TabsTrigger value="daily">Daily</TabsTrigger>
          <TabsTrigger value="weekly">Weekly</TabsTrigger>
          <TabsTrigger value="monthly">Monthly</TabsTrigger>
        </TabsList>
        <TabsContent value="daily" className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium">Routes Completed</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">245</div>
                <p className="text-xs text-muted-foreground">+12% from yesterday</p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium">On-Time Delivery</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">89.2%</div>
                <p className="text-xs text-muted-foreground">+2.1% from yesterday</p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium">Fuel Efficiency</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">8.2 km/L</div>
                <p className="text-xs text-muted-foreground">+0.3 improvement</p>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
        <TabsContent value="weekly" className="space-y-4">
          <RouteChart data={chartData} type="bar" />
        </TabsContent>
        <TabsContent value="monthly" className="space-y-4">
          <RouteChart data={chartData} type="line" />
        </TabsContent>
      </Tabs>
    </div>
  )
}

