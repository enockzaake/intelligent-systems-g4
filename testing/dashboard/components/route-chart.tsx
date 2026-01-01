"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { 
  LineChart, 
  Line, 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer,
  Area,
  AreaChart
} from "recharts"

interface RouteChartProps {
  data: Array<{
    name: string
    routes: number
    delays: number
    onTime: number
  }>
  type?: "line" | "bar" | "area"
}

export function RouteChart({ data, type = "area" }: RouteChartProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Route Performance</CardTitle>
        <CardDescription>Daily route completion and delay trends</CardDescription>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={350}>
          {type === "area" && (
            <AreaChart data={data}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Area 
                type="monotone" 
                dataKey="routes" 
                stackId="1" 
                stroke="#8884d8" 
                fill="#8884d8" 
              />
              <Area 
                type="monotone" 
                dataKey="delays" 
                stackId="2" 
                stroke="#ef4444" 
                fill="#ef4444" 
              />
              <Area 
                type="monotone" 
                dataKey="onTime" 
                stackId="3" 
                stroke="#22c55e" 
                fill="#22c55e" 
              />
            </AreaChart>
          )}
          {type === "line" && (
            <LineChart data={data}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="routes" stroke="#8884d8" />
              <Line type="monotone" dataKey="delays" stroke="#ef4444" />
              <Line type="monotone" dataKey="onTime" stroke="#22c55e" />
            </LineChart>
          )}
          {type === "bar" && (
            <BarChart data={data}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="routes" fill="#8884d8" />
              <Bar dataKey="delays" fill="#ef4444" />
              <Bar dataKey="onTime" fill="#22c55e" />
            </BarChart>
          )}
        </ResponsiveContainer>
      </CardContent>
    </Card>
  )
}

