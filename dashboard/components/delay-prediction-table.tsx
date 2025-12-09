"use client";

import {
  Table,
  TableBody,
  TableCaption,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { PredictionResponse } from "@/lib/api"
import { AlertCircle, AlertTriangle, CheckCircle } from "lucide-react";

interface DelayPredictionTableProps {
  predictions: PredictionResponse[];
}

export function DelayPredictionTable({
  predictions,
}: DelayPredictionTableProps) {
  const getRiskIcon = (riskLevel?: string) => {
    switch (riskLevel) {
      case "HIGH": return <AlertCircle className="h-3 w-3" />
      case "MEDIUM": return <AlertTriangle className="h-3 w-3" />
      case "LOW": return <CheckCircle className="h-3 w-3" />
      default: return null
    }
  }

  const getDelayBadge = (riskLevel?: string, probability?: number | null) => {
    // Use risk_level if available, otherwise fall back to probability
    if (riskLevel) {
      switch (riskLevel) {
        case "HIGH":
          return <Badge variant="destructive" className="gap-1">{getRiskIcon(riskLevel)} High Risk</Badge>
        case "MEDIUM":
          return <Badge className="bg-orange-500 gap-1">{getRiskIcon(riskLevel)} Medium Risk</Badge>
        case "LOW":
          return <Badge className="bg-green-500 gap-1">{getRiskIcon(riskLevel)} Low Risk</Badge>
      }
    }
    
    // Fallback to probability-based classification
    if (probability === null || probability === undefined) return <Badge variant="outline">N/A</Badge>;
    if (probability > 0.7)
      return <Badge variant="destructive" className="gap-1"><AlertCircle className="h-3 w-3" /> High Risk</Badge>;
    if (probability > 0.4)
      return <Badge className="bg-orange-500 gap-1"><AlertTriangle className="h-3 w-3" /> Medium Risk</Badge>;
    return <Badge className="bg-green-500 gap-1"><CheckCircle className="h-3 w-3" /> Low Risk</Badge>;
  };

  return (
    <div className="rounded-md border">
      <Table>
        <TableCaption>Delay predictions for active routes</TableCaption>
        <TableHeader>
          <TableRow>
            <TableHead>Route ID</TableHead>
            <TableHead>Stop ID</TableHead>
            <TableHead>Driver ID</TableHead>
            <TableHead>Delay Risk</TableHead>
            <TableHead>Probability</TableHead>
            <TableHead>Predicted Delay (min)</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {predictions.slice(0, 20).map((pred, index) => (
            <TableRow key={index}>
              <TableCell className="font-medium">R-{pred.route_id}</TableCell>
              <TableCell>S-{pred.stop_id}</TableCell>
              <TableCell>D-{pred.driver_id}</TableCell>
              <TableCell>{getDelayBadge(pred.risk_level, pred.delay_probability)}</TableCell>
              <TableCell>
                {pred.delay_probability !== null
                  ? `${(pred.delay_probability * 100).toFixed(1)}%`
                  : "N/A"}
              </TableCell>
              <TableCell>{pred.delay_minutes_pred.toFixed(1)}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}
