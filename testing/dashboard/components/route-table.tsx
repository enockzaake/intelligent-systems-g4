"use client"

import {
  Table,
  TableBody,
  TableCaption,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { MapPin, MoreHorizontal } from "lucide-react"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"

interface Route {
  id: number
  driver_id: number
  status: string
  total_stops: number
  completed_stops: number
  total_distance: number
  estimated_time: number
  delay_probability: number
  avg_delay_minutes: number
}

interface RouteTableProps {
  routes: Route[]
}

export function RouteTable({ routes }: RouteTableProps) {
  const getStatusBadge = (status: string) => {
    if (status === 'completed') return <Badge className="bg-green-500">Completed</Badge>
    if (status === 'in_progress') return <Badge className="bg-blue-500">In Progress</Badge>
    if (status === 'delayed') return <Badge variant="destructive">Delayed</Badge>
    return <Badge variant="outline">Pending</Badge>
  }

  const getRiskBadge = (probability: number) => {
    if (probability > 0.7) return <Badge variant="destructive">High Risk</Badge>
    if (probability > 0.4) return <Badge className="bg-orange-500">Medium</Badge>
    return <Badge className="bg-green-500">Low Risk</Badge>
  }

  return (
    <div className="rounded-md border">
      <Table>
        <TableCaption>Active and completed delivery routes</TableCaption>
        <TableHeader>
          <TableRow>
            <TableHead>Route</TableHead>
            <TableHead>Driver</TableHead>
            <TableHead>Status</TableHead>
            <TableHead>Progress</TableHead>
            <TableHead>Distance</TableHead>
            <TableHead>Est. Time</TableHead>
            <TableHead>Delay Risk</TableHead>
            <TableHead className="text-right">Actions</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {routes.map((route) => {
            const progress = (route.completed_stops / route.total_stops) * 100
            return (
              <TableRow key={route.id}>
                <TableCell className="font-medium">
                  <div className="flex items-center gap-2">
                    <MapPin className="h-4 w-4 text-muted-foreground" />
                    R-{route.id}
                  </div>
                </TableCell>
                <TableCell>D-{route.driver_id}</TableCell>
                <TableCell>{getStatusBadge(route.status)}</TableCell>
                <TableCell>
                  <div className="space-y-1">
                    <div className="text-sm">
                      {route.completed_stops}/{route.total_stops} stops
                    </div>
                    <Progress value={progress} className="w-20" />
                  </div>
                </TableCell>
                <TableCell>{route.total_distance.toFixed(0)} km</TableCell>
                <TableCell>{route.estimated_time.toFixed(0)} min</TableCell>
                <TableCell>{getRiskBadge(route.delay_probability)}</TableCell>
                <TableCell className="text-right">
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button variant="ghost" className="h-8 w-8 p-0">
                        <span className="sr-only">Open menu</span>
                        <MoreHorizontal className="h-4 w-4" />
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end">
                      <DropdownMenuLabel>Actions</DropdownMenuLabel>
                      <DropdownMenuItem>View details</DropdownMenuItem>
                      <DropdownMenuItem>Optimize route</DropdownMenuItem>
                      <DropdownMenuSeparator />
                      <DropdownMenuItem>Reassign driver</DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                </TableCell>
              </TableRow>
            )
          })}
        </TableBody>
      </Table>
    </div>
  )
}

