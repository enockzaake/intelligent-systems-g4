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
import { MoreHorizontal, User } from "lucide-react"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"

interface Driver {
  id: number
  name: string
  status: string
  current_route: number | null
  completed_stops: number
  total_distance: number
  avg_delay: number
  rating: string
}

interface DriverTableProps {
  drivers: Driver[]
}

export function DriverTable({ drivers }: DriverTableProps) {
  const getStatusBadge = (status: string) => {
    if (status === 'available') return <Badge className="bg-green-500">Available</Badge>
    if (status === 'busy') return <Badge className="bg-blue-500">On Route</Badge>
    return <Badge variant="outline">Offline</Badge>
  }

  return (
    <div className="rounded-md border">
      <Table>
        <TableCaption>Fleet driver management and status</TableCaption>
        <TableHeader>
          <TableRow>
            <TableHead>Driver</TableHead>
            <TableHead>Status</TableHead>
            <TableHead>Current Route</TableHead>
            <TableHead>Completed Stops</TableHead>
            <TableHead>Total Distance</TableHead>
            <TableHead>Avg Delay</TableHead>
            <TableHead>Rating</TableHead>
            <TableHead className="text-right">Actions</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {drivers.map((driver) => (
            <TableRow key={driver.id}>
              <TableCell className="font-medium">
                <div className="flex items-center gap-2">
                  <div className="flex items-center justify-center w-8 h-8 rounded-full bg-primary/10">
                    <User className="h-4 w-4" />
                  </div>
                  {driver.name}
                </div>
              </TableCell>
              <TableCell>{getStatusBadge(driver.status)}</TableCell>
              <TableCell>
                {driver.current_route ? `R-${driver.current_route}` : '-'}
              </TableCell>
              <TableCell>{driver.completed_stops}</TableCell>
              <TableCell>{driver.total_distance.toFixed(0)} km</TableCell>
              <TableCell>{driver.avg_delay.toFixed(1)} min</TableCell>
              <TableCell>⭐ {driver.rating}</TableCell>
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
                    <DropdownMenuItem>Assign route</DropdownMenuItem>
                    <DropdownMenuSeparator />
                    <DropdownMenuItem>View history</DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  )
}

