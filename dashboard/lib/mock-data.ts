// Mock data generator for simulation and testing
import { StopData } from './api';

export function generateMockStops(count: number = 100): StopData[] {
  const stops: StopData[] = [];
  const daysOfWeek = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];
  
  for (let i = 0; i < count; i++) {
    const hour = Math.floor(Math.random() * 12) + 6;
    const minute = Math.floor(Math.random() * 60);
    const earliestHour = Math.floor(Math.random() * 4) + 6;
    const latestHour = Math.floor(Math.random() * 4) + 14;
    
    stops.push({
      route_id: Math.floor(i / 10) + 1,
      driver_id: Math.floor(i / 10) + 1,
      stop_id: i + 1,
      address_id: Math.floor(Math.random() * 500) + 1,
      week_id: Math.floor(Math.random() * 52) + 1,
      country: Math.floor(Math.random() * 5) + 1,
      day_of_week: daysOfWeek[Math.floor(Math.random() * 7)],
      indexp: i % 20,
      indexa: i % 20,
      arrived_time: `${hour}:${minute.toString().padStart(2, '0')}`,
      earliest_time: `${earliestHour}:00`,
      latest_time: `${latestHour}:00`,
      distancep: parseFloat((Math.random() * 50 + 5).toFixed(2)),
      distancea: parseFloat((Math.random() * 50 + 5).toFixed(2)),
      depot: 0,
      delivery: Math.random() > 0.5 ? 1 : 0,
    });
  }
  
  return stops;
}

export function generateMockStopsWithParams(params: {
  numStops: number;
  numVehicles: number;
  avgDelay: number;
  trafficCondition: string;
  weatherCondition: string;
  timeOfDay: string;
}): StopData[] {
  const stops: StopData[] = [];
  const daysOfWeek = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'];
  
  // Calculate stops per vehicle
  const stopsPerVehicle = Math.ceil(params.numStops / params.numVehicles);
  
  for (let i = 0; i < params.numStops; i++) {
    const vehicleId = Math.floor(i / stopsPerVehicle) + 1;
    const stopInRoute = i % stopsPerVehicle;
    
    // Base distance calculation
    let baseDistance = Math.random() * 30 + 10;
    
    // Adjust distances based on traffic
    const trafficMultiplier = 
      params.trafficCondition === 'heavy' ? 1.5 :
      params.trafficCondition === 'moderate' ? 1.2 : 1.0;
    
    // Adjust for weather
    const weatherMultiplier = 
      params.weatherCondition === 'snow' ? 1.4 :
      params.weatherCondition === 'rain' ? 1.2 :
      params.weatherCondition === 'fog' ? 1.15 : 1.0;
    
    const totalMultiplier = trafficMultiplier * weatherMultiplier;
    
    // Calculate time windows based on time of day
    let earliestHour: number, latestHour: number, arrivedHour: number;
    
    if (params.timeOfDay === 'morning') {
      earliestHour = 6 + Math.floor(Math.random() * 2);
      latestHour = 10 + Math.floor(Math.random() * 2);
      arrivedHour = earliestHour + Math.floor(Math.random() * (latestHour - earliestHour));
    } else if (params.timeOfDay === 'afternoon') {
      earliestHour = 12 + Math.floor(Math.random() * 2);
      latestHour = 16 + Math.floor(Math.random() * 2);
      arrivedHour = earliestHour + Math.floor(Math.random() * (latestHour - earliestHour));
    } else { // evening
      earliestHour = 16 + Math.floor(Math.random() * 2);
      latestHour = 20 + Math.floor(Math.random() * 2);
      arrivedHour = earliestHour + Math.floor(Math.random() * (latestHour - earliestHour));
    }
    
    const arrivedMinute = Math.floor(Math.random() * 60);
    
    stops.push({
      route_id: vehicleId,
      driver_id: vehicleId,
      stop_id: i + 1,
      address_id: Math.floor(Math.random() * 500) + 1,
      week_id: Math.floor(Math.random() * 4) + 1,
      country: Math.floor(Math.random() * 3) + 1,
      day_of_week: daysOfWeek[Math.floor(Math.random() * 5)],
      indexp: stopInRoute,
      indexa: stopInRoute,
      arrived_time: `${arrivedHour}:${arrivedMinute.toString().padStart(2, '0')}`,
      earliest_time: `${earliestHour}:00`,
      latest_time: `${latestHour}:00`,
      distancep: parseFloat((baseDistance * totalMultiplier).toFixed(2)),
      distancea: parseFloat((baseDistance * totalMultiplier * 1.05).toFixed(2)),
      depot: 0,
      delivery: Math.random() > 0.2 ? 1 : 0, // 80% deliveries
    });
  }
  
  return stops;
}

export function generateDriverData(count: number = 20) {
  return Array.from({ length: count }, (_, i) => ({
    id: i + 1,
    name: `Driver ${i + 1}`,
    status: Math.random() > 0.2 ? 'available' : 'busy',
    current_route: Math.random() > 0.3 ? Math.floor(Math.random() * 10) + 1 : null,
    completed_stops: Math.floor(Math.random() * 50),
    total_distance: Math.floor(Math.random() * 500) + 100,
    avg_delay: Math.random() * 30,
    rating: (Math.random() * 2 + 3).toFixed(1),
  }));
}

export function generateRouteData(count: number = 10) {
  return Array.from({ length: count }, (_, i) => ({
    id: i + 1,
    driver_id: i + 1,
    status: Math.random() > 0.3 ? 'in_progress' : 'completed',
    total_stops: Math.floor(Math.random() * 20) + 5,
    completed_stops: Math.floor(Math.random() * 15),
    total_distance: Math.floor(Math.random() * 200) + 50,
    estimated_time: Math.floor(Math.random() * 300) + 60,
    delay_probability: Math.random(),
    avg_delay_minutes: Math.random() * 20,
  }));
}

