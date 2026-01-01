import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatDistance(km: number): string {
  return `${km.toFixed(1)} km`
}

export function formatDuration(minutes: number): string {
  const hours = Math.floor(minutes / 60)
  const mins = Math.floor(minutes % 60)
  
  if (hours === 0) return `${mins}m`
  if (mins === 0) return `${hours}h`
  return `${hours}h ${mins}m`
}

export function formatPercentage(value: number): string {
  return `${(value * 100).toFixed(1)}%`
}

export function formatCurrency(amount: number): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
  }).format(amount)
}

export function formatDate(date: Date | string): string {
  const d = typeof date === 'string' ? new Date(date) : date
  return new Intl.DateTimeFormat('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  }).format(d)
}

export function formatTime(time: string): string {
  // Convert "HH:MM" or "H:MM" to "HH:MM AM/PM"
  const [hours, minutes] = time.split(':').map(Number)
  const period = hours >= 12 ? 'PM' : 'AM'
  const displayHours = hours % 12 || 12
  return `${displayHours}:${minutes.toString().padStart(2, '0')} ${period}`
}

export function getDelayRiskColor(probability: number): string {
  if (probability > 0.7) return 'text-red-500'
  if (probability > 0.4) return 'text-orange-500'
  return 'text-green-500'
}

export function getDelayRiskBadgeVariant(probability: number): "default" | "destructive" | "outline" | "secondary" {
  if (probability > 0.7) return 'destructive'
  if (probability > 0.4) return 'secondary'
  return 'outline'
}
