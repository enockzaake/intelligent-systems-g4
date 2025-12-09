"use client";

import { AnalyticsChart } from "@/components/analytics-chart";
import { RealTimeMonitor } from "@/components/real-time-monitor";
import { RouteChart } from "@/components/route-chart";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

export default function AnalyticsPage() {
  const delayDistribution = [
    { name: "Traffic", value: 35 },
    { name: "Weather", value: 20 },
    { name: "Vehicle Issues", value: 15 },
    { name: "Driver Delays", value: 18 },
    { name: "Other", value: 12 },
  ];

  const routeEfficiency = [
    { name: "Optimized", value: 68 },
    { name: "Standard", value: 25 },
    { name: "Needs Improvement", value: 7 },
  ];

  const performanceTrend = [
    { name: "Jan", routes: 420, delays: 85, onTime: 335 },
    { name: "Feb", routes: 445, delays: 78, onTime: 367 },
    { name: "Mar", routes: 480, delays: 92, onTime: 388 },
    { name: "Apr", routes: 465, delays: 71, onTime: 394 },
    { name: "May", routes: 510, delays: 68, onTime: 442 },
    { name: "Jun", routes: 495, delays: 59, onTime: 436 },
  ];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">
          Analytics & Insights
        </h1>
        <p className="text-muted-foreground">
          Deep insights into fleet performance and optimization opportunities
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <RealTimeMonitor
          title="Real-Time Monitoring"
          description="Live fleet status updates"
        />
        <Card>
          <CardHeader>
            <CardTitle>Performance Summary</CardTitle>
            <CardDescription>Current period overview</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-sm text-muted-foreground">
                Routes Completed
              </span>
              <span className="font-bold">2,847</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-muted-foreground">
                Average Efficiency
              </span>
              <span className="font-bold text-green-500">92.3%</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-muted-foreground">
                Cost Savings
              </span>
              <span className="font-bold text-green-500">$24,567</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-muted-foreground">
                CO2 Reduction
              </span>
              <span className="font-bold text-green-500">1,234 kg</span>
            </div>
          </CardContent>
        </Card>
      </div>

      <RouteChart data={performanceTrend} type="area" />

      <div className="grid gap-4 md:grid-cols-2">
        <AnalyticsChart
          data={delayDistribution}
          title="Delay Causes Distribution"
          description="Primary factors contributing to delays"
        />
        <AnalyticsChart
          data={routeEfficiency}
          title="Route Efficiency Status"
          description="Current optimization levels"
        />
      </div>

      <Tabs defaultValue="efficiency" className="space-y-4">
        <TabsList>
          <TabsTrigger value="efficiency">Efficiency Metrics</TabsTrigger>
          <TabsTrigger value="costs">Cost Analysis</TabsTrigger>
          <TabsTrigger value="environmental">Environmental Impact</TabsTrigger>
        </TabsList>

        <TabsContent value="efficiency" className="space-y-4">
          <div className="grid gap-4 md:grid-cols-3">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium">
                  Route Optimization Rate
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">89.4%</div>
                <p className="text-xs text-muted-foreground">
                  +5.2% from last month
                </p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium">
                  Fuel Efficiency
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">8.7 km/L</div>
                <p className="text-xs text-muted-foreground">
                  +0.4 improvement
                </p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium">
                  Vehicle Utilization
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">87.2%</div>
                <p className="text-xs text-muted-foreground">
                  +3.1% from last month
                </p>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="costs" className="space-y-4">
          <div className="grid gap-4 md:grid-cols-3">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium">
                  Total Fuel Cost
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">$15,234</div>
                <p className="text-xs text-green-500">
                  -8.3% reduction
                </p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium">
                  Maintenance Cost
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">$4,567</div>
                <p className="text-xs text-muted-foreground">Within budget</p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium">
                  Cost per Delivery
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">$6.87</div>
                <p className="text-xs text-green-500">
                  -$0.42 decrease
                </p>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="environmental" className="space-y-4">
          <div className="grid gap-4 md:grid-cols-3">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium">
                  CO2 Emissions
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">3,456 kg</div>
                <p className="text-xs text-green-500">
                  -12.4% reduction
                </p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium">
                  Distance Optimized
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">2,347 km</div>
                <p className="text-xs text-muted-foreground">
                  Saved through optimization
                </p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium">
                  Green Score
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">A-</div>
                <p className="text-xs text-muted-foreground">
                  Industry average: B+
                </p>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
