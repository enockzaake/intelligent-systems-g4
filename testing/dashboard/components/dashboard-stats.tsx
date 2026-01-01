"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { TrendingUp, TrendingDown, Truck, AlertCircle, MapPin, Clock } from "lucide-react"

interface DashboardStatsProps {
  stats: {
    totalRoutes: number
    activeDrivers: number
    totalStops: number
    avgDelayRate: number
    totalDistance: number
    avgDelayMinutes: number
    onTimeRate: number
    vehicleUtilization: number
  }
}

export function DashboardStats({ stats }: DashboardStatsProps) {
  const statCards = [
    {
      title: "Total Routes",
      value: stats.totalRoutes,
      description: "Active delivery routes",
      icon: MapPin,
      trend: "+12% from last week",
      trendUp: true,
    },
    {
      title: "Active Drivers",
      value: stats.activeDrivers,
      description: "Currently on duty",
      icon: Truck,
      trend: "85% utilization",
      trendUp: true,
    },
    {
      title: "Total Stops",
      value: stats.totalStops.toLocaleString(),
      description: "Delivery locations",
      icon: MapPin,
      trend: "+8% from yesterday",
      trendUp: true,
    },
    {
      title: "Avg Delay Rate",
      value: `${(stats.avgDelayRate * 100).toFixed(1)}%`,
      description: "Routes with delays",
      icon: AlertCircle,
      trend: "-5% improvement",
      trendUp: false,
      alert: stats.avgDelayRate > 0.3,
    },
    {
      title: "Total Distance",
      value: `${stats.totalDistance.toFixed(0)} km`,
      description: "Covered today",
      icon: TrendingUp,
      trend: "12% optimized",
      trendUp: true,
    },
    {
      title: "Avg Delay Time",
      value: `${stats.avgDelayMinutes.toFixed(0)} min`,
      description: "Per delayed stop",
      icon: Clock,
      trend: "-3 min from average",
      trendUp: false,
    },
  ]

  return (
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
      {statCards.map((stat, index) => {
        const Icon = stat.icon
        return (
          <Card key={index} className={stat.alert ? "border-orange-500" : ""}>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">{stat.title}</CardTitle>
              <Icon className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{stat.value}</div>
              <p className="text-xs text-muted-foreground">{stat.description}</p>
              <div className="flex items-center pt-1">
                {stat.trendUp ? (
                  <TrendingUp className="mr-1 h-3 w-3 text-green-500" />
                ) : (
                  <TrendingDown className="mr-1 h-3 w-3 text-green-500" />
                )}
                <span className="text-xs text-muted-foreground">{stat.trend}</span>
              </div>
            </CardContent>
          </Card>
        )
      })}
    </div>
  )
}

