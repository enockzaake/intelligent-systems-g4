# Fleet Management & Route Optimization Dashboard

AI-Driven Route Optimization and Delay Prediction System built with Next.js 15 and shadcn/ui.

## Features

### 🎯 Core Functionality

- **Overview Dashboard**: Real-time KPIs, metrics, and fleet performance monitoring
- **Route Management**: View, manage, and optimize delivery routes
- **Driver Management**: Track driver performance and availability
- **Delay Predictions**: AI-powered predictions for potential delays
- **Route Simulation**: Test different scenarios under various conditions
- **Analytics & Insights**: Deep analysis of fleet performance and optimization

### 🚀 Key Features

1. **Real-Time Monitoring**
   - Live fleet status updates
   - Active route tracking
   - Performance metrics

2. **AI-Powered Predictions**
   - Delay probability analysis
   - Route optimization suggestions
   - Driver reassignment recommendations

3. **Interactive Simulation**
   - Configure custom parameters
   - Test traffic conditions
   - Weather impact analysis
   - Time-of-day optimization

4. **Comprehensive Analytics**
   - Performance trends
   - Cost analysis
   - Environmental impact tracking
   - Efficiency metrics

## Tech Stack

- **Framework**: Next.js 15.5.6
- **UI Library**: shadcn/ui
- **Styling**: Tailwind CSS 4
- **Charts**: Recharts
- **Icons**: Lucide React & Tabler Icons
- **State Management**: React Hooks
- **API Integration**: Fetch API

## Getting Started

### Prerequisites

- Node.js 20+ installed
- Backend API running on `http://localhost:8000`

### Installation

1. Install dependencies:
```bash
npm install
```

2. Create `.env.local` file:
```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
```

3. Run the development server:
```bash
npm run dev
```

4. Open [http://localhost:3000](http://localhost:3000) in your browser

## Project Structure

```
dashboard/
├── app/
│   ├── dashboard/
│   │   ├── overview/       # Dashboard overview page
│   │   ├── routes/         # Routes management
│   │   ├── drivers/        # Driver management
│   │   ├── predictions/    # Delay predictions
│   │   ├── simulation/     # Route simulation
│   │   ├── analytics/      # Analytics & insights
│   │   └── layout.tsx      # Dashboard layout
│   ├── globals.css
│   ├── layout.tsx
│   └── page.tsx
├── components/
│   ├── ui/                 # shadcn/ui components
│   ├── dashboard-stats.tsx
│   ├── route-chart.tsx
│   ├── delay-prediction-table.tsx
│   ├── simulation-panel.tsx
│   ├── optimization-results.tsx
│   ├── driver-table.tsx
│   ├── route-table.tsx
│   ├── analytics-chart.tsx
│   ├── real-time-monitor.tsx
│   └── ...
├── lib/
│   ├── api.ts             # API integration functions
│   ├── mock-data.ts       # Mock data generators
│   └── utils.ts
└── package.json
```

## API Integration

The dashboard connects to the FastAPI backend with the following endpoints:

- `POST /predict/delays` - Get delay predictions for stops
- `POST /predict/routes` - Get route-level aggregates
- `POST /optimize/routes` - Optimize routes with AI
- `POST /reassign/stops` - Get driver reassignment suggestions
- `GET /models` - List available AI models
- `GET /optimization/status` - Get optimization system status

## Pages

### Overview (`/dashboard/overview`)
- Real-time KPI cards
- Performance charts
- Delay trends
- Daily/weekly/monthly metrics

### Routes (`/dashboard/routes`)
- Route list with filtering
- Status tracking
- Progress monitoring
- Route optimization

### Drivers (`/dashboard/drivers`)
- Driver performance tracking
- Availability status
- Statistics and ratings
- Assignment management

### Predictions (`/dashboard/predictions`)
- AI delay predictions
- Risk analysis
- Model selection
- Confidence metrics

### Simulation (`/dashboard/simulation`)
- Parameter configuration
- Scenario testing
- Optimization results
- Before/after comparison

### Analytics (`/dashboard/analytics`)
- Performance insights
- Cost analysis
- Environmental impact
- Real-time monitoring

## Backend Setup

Make sure the backend API is running:

1. Navigate to the core directory:
```bash
cd ../core
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Start the API server:
```bash
cd api
fastapi dev main.py
```

The API will be available at `http://localhost:8000`

## Features in Detail

### Simulation Panel

The simulation feature allows you to test the route optimization system under different conditions:

- **Parameters**:
  - Number of stops (10-500)
  - Number of vehicles (1-50)
  - Average delay (0-120 minutes)
  - Traffic conditions (light/moderate/heavy)
  - Weather conditions (clear/rain/snow/fog)
  - Time of day (morning/afternoon/evening)

- **Process**:
  1. Configure simulation parameters
  2. Generate mock data based on conditions
  3. Run AI delay predictions
  4. Optimize routes
  5. Calculate driver reassignments
  6. Display comprehensive results

### Real-Time Features

- Live fleet status updates every 5 seconds
- Active route monitoring
- Performance metrics tracking
- Delay alerts and notifications

## Development

### Adding New Components

1. Create component in `components/` directory
2. Use shadcn/ui primitives
3. Follow TypeScript conventions
4. Import and use in pages

### Adding New Pages

1. Create page in `app/dashboard/[page-name]/page.tsx`
2. Add navigation link in `components/app-sidebar.tsx`
3. Implement page layout and functionality

## Production Build

```bash
npm run build
npm start
```

## License

This project is part of the CS Intelligent Systems application.

## Support

For issues or questions, please refer to the main project documentation.
