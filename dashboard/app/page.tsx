import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import Link from "next/link";
import {
  Truck,
  Users,
  Brain,
  ArrowRight,
  TrendingUp,
  MapPin,
} from "lucide-react";

export default function Home() {
  const teamMembers = [
    "Member Name 1",
    "Member Name 2",
    "Member Name 3",
    "Member Name 4",
  ];

  return (
    <main className="min-h-screen bg-gradient-to-b from-background to-muted">
      {/* Hero Section */}
      <div className="container mx-auto px-4 py-16">
        <div className="flex flex-col items-center text-center space-y-8">
          {/* University Logo */}
          <div className="flex flex-col items-center space-y-4">
            <div className="w-32 h-32 bg-primary/10 rounded-full flex items-center justify-center">
              <Brain className="w-16 h-16 text-primary" />
            </div>
            <div className="space-y-2">
              <h2 className="text-2xl font-bold">
                Deggendorf Institute of Technology
              </h2>
            </div>
          </div>

          {/* Project Title */}
          <div className="space-y-4 max-w-4xl">
            <Badge className="mb-2" variant="outline">
              Intelligent Systems Project
            </Badge>
            <h1 className="text-5xl md:text-6xl font-bold tracking-tight">
              Fleet Management & <br />
              Route Optimization System
            </h1>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
              An AI-driven platform for intelligent route optimization,
              real-time delay prediction, and driver reassignment using machine
              learning and operations research
            </p>
          </div>

          {/* Key Features */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 w-full max-w-4xl mt-8">
            <Card>
              <CardHeader className="text-center">
                <Truck className="w-8 h-8 mx-auto mb-2 text-primary" />
                <CardTitle className="text-lg">Route Optimization</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground text-center">
                  OR-Tools powered optimization for efficient route planning
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="text-center">
                <Brain className="w-8 h-8 mx-auto mb-2 text-primary" />
                <CardTitle className="text-lg">AI Predictions</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground text-center">
                  Machine learning models for delay forecasting and risk
                  analysis
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="text-center">
                <TrendingUp className="w-8 h-8 mx-auto mb-2 text-primary" />
                <CardTitle className="text-lg">Real-time Analytics</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground text-center">
                  Comprehensive dashboards for fleet performance monitoring
                </p>
              </CardContent>
            </Card>
          </div>

          {/* CTA Button */}
          <Button size="lg" asChild className="mt-8">
            <Link href="/dashboard">
              Launch Dashboard
              <ArrowRight className="ml-2 h-4 w-4" />
            </Link>
          </Button>

          {/* Team & Professor Section */}
          <div className="mt-16 w-full max-w-4xl space-y-8">
            <div className="grid md:grid-cols-2 gap-8">
              {/* Team Members */}
              <Card>
                <CardHeader>
                  <div className="flex items-center gap-2">
                    <Users className="w-5 h-5 text-primary" />
                    <CardTitle>Project Team</CardTitle>
                  </div>
                  <CardDescription>Group Members</CardDescription>
                </CardHeader>
                <CardContent>
                  <ul className="space-y-2">
                    {teamMembers.map((member, index) => (
                      <li key={index} className="flex items-center gap-2">
                        <div className="w-2 h-2 rounded-full bg-primary" />
                        <span className="text-sm font-medium">{member}</span>
                      </li>
                    ))}
                  </ul>
                </CardContent>
              </Card>

              {/* Professor */}
              <Card>
                <CardHeader>
                  <div className="flex items-center gap-2">
                    <Brain className="w-5 h-5 text-primary" />
                    <CardTitle>Academic Guidance</CardTitle>
                  </div>
                  <CardDescription>Under the supervision of</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div>
                    <p className="font-semibold text-lg">Professor Name</p>
                    <p className="text-sm text-muted-foreground">
                      Department of Computer Science
                    </p>
                    <p className="text-sm text-muted-foreground">
                      Deggendorf Institute of Technology, Bhubaneswar
                    </p>
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Tech Stack */}
            <Card>
              <CardHeader>
                <CardTitle className="text-center">Technology Stack</CardTitle>
                <CardDescription className="text-center">
                  Built with modern technologies and frameworks
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="flex flex-wrap gap-2 justify-center">
                  <Badge variant="secondary">Next.js 15</Badge>
                  <Badge variant="secondary">FastAPI</Badge>
                  <Badge variant="secondary">Python</Badge>
                  <Badge variant="secondary">TensorFlow</Badge>
                  <Badge variant="secondary">OR-Tools</Badge>
                  <Badge variant="secondary">shadcn/ui</Badge>
                  <Badge variant="secondary">Tailwind CSS</Badge>
                  <Badge variant="secondary">TypeScript</Badge>
                  <Badge variant="secondary">Recharts</Badge>
                  <Badge variant="secondary">scikit-learn</Badge>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Footer */}
          <div className="mt-16 text-center text-sm text-muted-foreground">
            <p>
              © 2024 Deggendorf Institute of Technology - Intelligent Systems
              Project
            </p>
            <p className="mt-2">
              Fleet Management & Route Optimization Dashboard
            </p>
          </div>
        </div>
      </div>
    </main>
  );
}
