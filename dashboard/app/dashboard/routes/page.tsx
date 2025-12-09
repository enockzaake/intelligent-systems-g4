"use client"

import { useState, useEffect } from "react"
import { RouteTable } from "@/components/route-table"
import { getRoutesData } from "@/lib/api"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Plus, Search, Filter, Loader2 } from "lucide-react"
import { 
  Select, 
  SelectContent, 
  SelectItem, 
  SelectTrigger, 
  SelectValue 
} from "@/components/ui/select"
import { toast } from "sonner"

export default function RoutesPage() {
  const [routes, setRoutes] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [filter, setFilter] = useState("all")
  const [searchQuery, setSearchQuery] = useState("")
  
  useEffect(() => {
    async function fetchRoutes() {
      try {
        setLoading(true)
        const data = await getRoutesData(50)
        // Transform backend data to match RouteTable interface
        const transformedRoutes = data.routes.map((route: any) => ({
          id: route.route_id,
          driver_id: route.driver_id,
          status: 'completed',
          total_stops: route.total_stops,
          completed_stops: route.total_stops,
          total_distance: route.total_distance,
          estimated_time: route.total_distance * 2, // Estimate based on distance
          delay_probability: Math.random(), // Will be replaced with real predictions
          avg_delay_minutes: 0,
        }))
        setRoutes(transformedRoutes)
      } catch (error) {
        console.error("Failed to fetch routes:", error)
        toast.error("Failed to load routes data. Make sure the API server is running.")
      } finally {
        setLoading(false)
      }
    }
    
    fetchRoutes()
  }, [])

  const filteredRoutes = routes.filter(route => {
    const matchesFilter = filter === "all" || route.status === filter
    const matchesSearch = searchQuery === "" || 
      route.id.toString().includes(searchQuery) ||
      route.driver_id.toString().includes(searchQuery)
    return matchesFilter && matchesSearch
  })

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
          <h1 className="text-3xl font-bold tracking-tight">Routes Management</h1>
          <p className="text-muted-foreground">
            Real routes from cleaned dataset ({routes.length} total)
          </p>
        </div>
        <Button>
          <Plus className="mr-2 h-4 w-4" />
          Create Route
        </Button>
      </div>

      <div className="flex items-center gap-4">
        <div className="flex-1 relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search routes..."
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
            <SelectItem value="all">All Routes</SelectItem>
            <SelectItem value="in_progress">In Progress</SelectItem>
            <SelectItem value="completed">Completed</SelectItem>
            <SelectItem value="delayed">Delayed</SelectItem>
          </SelectContent>
        </Select>
      </div>

      <RouteTable routes={filteredRoutes} />

      <div className="flex items-center justify-between text-sm text-muted-foreground">
        <p>Showing {filteredRoutes.length} of {routes.length} routes</p>
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm">Previous</Button>
          <Button variant="outline" size="sm">Next</Button>
        </div>
      </div>
    </div>
  )
}

