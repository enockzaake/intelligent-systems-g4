"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Activity, TrendingUp, TrendingDown } from "lucide-react"

interface RealTimeMonitorProps {
  title: string
  description?: string
}

export function RealTimeMonitor({ title, description }: RealTimeMonitorProps) {
  const [liveData, setLiveData] = useState({
    activeRoutes: 45,
    delayedRoutes: 8,
    avgSpeed: 42.5,
    efficiency: 87.3,
  })

  // Simulate real-time updates
  useEffect(() => {
    const interval = setInterval(() => {
      setLiveData(prev => ({
        activeRoutes: prev.activeRoutes + Math.floor(Math.random() * 3 - 1),
        delayedRoutes: Math.max(0, prev.delayedRoutes + Math.floor(Math.random() * 3 - 1)),
        avgSpeed: +(prev.avgSpeed + (Math.random() * 4 - 2)).toFixed(1),
        efficiency: +(Math.min(100, Math.max(0, prev.efficiency + (Math.random() * 2 - 1)))).toFixed(1),
      }))
    }, 5000)

    return () => clearInterval(interval)
  }, [])

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle>{title}</CardTitle>
            {description && <CardDescription>{description}</CardDescription>}
          </div>
          <Badge variant="outline" className="gap-1">
            <Activity className="h-3 w-3 animate-pulse" />
            Live
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-1">
            <p className="text-sm text-muted-foreground">Active Routes</p>
            <div className="flex items-center gap-2">
              <p className="text-2xl font-bold">{liveData.activeRoutes}</p>
              <TrendingUp className="h-4 w-4 text-green-500" />
            </div>
          </div>
          <div className="space-y-1">
            <p className="text-sm text-muted-foreground">Delayed Routes</p>
            <div className="flex items-center gap-2">
              <p className="text-2xl font-bold">{liveData.delayedRoutes}</p>
              <TrendingDown className="h-4 w-4 text-red-500" />
            </div>
          </div>
          <div className="space-y-1">
            <p className="text-sm text-muted-foreground">Avg Speed</p>
            <p className="text-2xl font-bold">{liveData.avgSpeed} km/h</p>
          </div>
          <div className="space-y-1">
            <p className="text-sm text-muted-foreground">Efficiency</p>
            <p className="text-2xl font-bold">{liveData.efficiency}%</p>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

