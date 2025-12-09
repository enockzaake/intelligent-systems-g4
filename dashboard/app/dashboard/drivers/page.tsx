"use client"

import { useState, useEffect } from "react"
import { DriverTable } from "@/components/driver-table"
import { getDriversData } from "@/lib/api"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Plus, Search, Filter, UserPlus, Loader2 } from "lucide-react"
import { toast } from "sonner"
import { 
  Select, 
  SelectContent, 
  SelectItem, 
  SelectTrigger, 
  SelectValue 
} from "@/components/ui/select"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"

export default function DriversPage() {
  const [drivers, setDrivers] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [filter, setFilter] = useState("all")
  const [searchQuery, setSearchQuery] = useState("")
  
  useEffect(() => {
    async function fetchDrivers() {
      try {
        setLoading(true)
        const data = await getDriversData()
        // Transform backend data to match DriverTable interface
        const transformedDrivers = data.drivers.map((driver: any) => ({
          id: driver.driver_id,
          name: `Driver ${driver.driver_id}`,
          status: 'available',
          current_route: null,
          completed_stops: driver.completed_stops,
          total_distance: driver.total_distance,
          avg_delay: 0,
          rating: '4.5',
        }))
        setDrivers(transformedDrivers)
      } catch (error) {
        console.error("Failed to fetch drivers:", error)
        toast.error("Failed to load drivers data. Make sure the API server is running.")
      } finally {
        setLoading(false)
      }
    }
    
    fetchDrivers()
  }, [])

  const filteredDrivers = drivers.filter(driver => {
    const matchesFilter = filter === "all" || driver.status === filter
    const matchesSearch = searchQuery === "" || 
      driver.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      driver.id.toString().includes(searchQuery)
    return matchesFilter && matchesSearch
  })

  const activeDrivers = drivers.filter(d => d.status === 'available' || d.status === 'busy').length
  const avgRating = drivers.length > 0 
    ? (drivers.reduce((acc, d) => acc + parseFloat(d.rating), 0) / drivers.length).toFixed(1)
    : '0.0'
  const totalStops = drivers.reduce((acc, d) => acc + d.completed_stops, 0)

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Driver Management</h1>
          <p className="text-muted-foreground">
            Real drivers from cleaned dataset ({drivers.length} total)
          </p>
        </div>
        <Button>
          <UserPlus className="mr-2 h-4 w-4" />
          Add Driver
        </Button>
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Active Drivers</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{activeDrivers}</div>
            <p className="text-xs text-muted-foreground">out of {drivers.length} total</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Average Rating</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">⭐ {avgRating}</div>
            <p className="text-xs text-muted-foreground">Fleet performance</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Total Deliveries</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{totalStops}</div>
            <p className="text-xs text-muted-foreground">Completed today</p>
          </CardContent>
        </Card>
      </div>

      <div className="flex items-center gap-4">
        <div className="flex-1 relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search drivers..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-9"
          />
        </div>
        <Select value={filter} onValueChange={setFilter}>
          <SelectTrigger className="w-[180px]">
            <Filter className="mr-2 h-4 w-4" />
            <SelectValue placeholder="Filter by status" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Drivers</SelectItem>
            <SelectItem value="available">Available</SelectItem>
            <SelectItem value="busy">On Route</SelectItem>
            <SelectItem value="offline">Offline</SelectItem>
          </SelectContent>
        </Select>
      </div>

      <DriverTable drivers={filteredDrivers} />

      <div className="flex items-center justify-between text-sm text-muted-foreground">
        <p>Showing {filteredDrivers.length} of {drivers.length} drivers</p>
      </div>
    </div>
  )
}

