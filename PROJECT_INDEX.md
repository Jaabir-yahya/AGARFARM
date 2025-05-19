# AGARFARM Project Index

## Project Overview
AGARFARM is an AI-driven simulation and real-time control system for optimized greenhouse environments. The project combines physics-based simulation, machine learning control, and a modern web interface for monitoring and control.

## Directory Structure

### Frontend (`agarfarm-frontend/`)
```
agarfarm-frontend/
├── app/                    # Next.js app directory (pages and layouts)
├── components/            # Reusable React components
├── hooks/                 # Custom React hooks
├── lib/                   # Utility functions and shared code
│   └── config.js         # Environment configuration
├── styles/               # Global styles and Tailwind config
├── next.config.mjs       # Next.js configuration
├── package.json          # Dependencies and scripts
├── tailwind.config.ts    # Tailwind CSS configuration
└── tsconfig.json         # TypeScript configuration
```

### Backend
```
├── app.py                # FastAPI main application
├── requirements.txt      # Python dependencies
├── simulator/           # Greenhouse simulator core
├── controllers/         # Control system implementations
├── ml_training/         # Machine learning training code
├── notebooks/           # Jupyter notebooks for analysis
├── evaluation/          # Evaluation scripts and results
├── figures/             # Generated figures and visualizations
└── logs/                # System logs
```

## Key Components

### Frontend
- **Next.js Application**: Modern React framework with TypeScript
- **UI Components**: Built with Radix UI and Tailwind CSS
- **Real-time Updates**: WebSocket integration for live data
- **Data Visualization**: Recharts for time-series data
- **Environment Support**: Configurable for both local and Render deployment

### Backend
- **FastAPI Server**: High-performance Python web framework
- **Greenhouse Simulator**: Physics-based environment simulation
- **ML Control System**: Reinforcement learning agents for control
- **WebSocket Server**: Real-time data streaming
- **API Endpoints**: RESTful interface for system control

### Machine Learning
- **Training Framework**: Stable-Baselines3 with PPO algorithm
- **Environment Wrapper**: Custom Gymnasium environment
- **Evaluation Tools**: Performance analysis and visualization

## Environment Configuration

### Frontend Environment Variables
```env
# API Configuration
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000

# Environment
NODE_ENV=development

# Feature Flags
NEXT_PUBLIC_ENABLE_DEBUG_MODE=true
NEXT_PUBLIC_ENABLE_ML_CONTROLS=true

# Analytics
NEXT_PUBLIC_ANALYTICS_ID=

# Theme
NEXT_PUBLIC_DEFAULT_THEME=light
```

### Backend Dependencies
- FastAPI >= 0.95.0
- Uvicorn >= 0.20.0
- Gymnasium >= 0.26.0
- Stable-Baselines3 >= 2.0.0
- Pandas >= 1.5.0
- NumPy >= 1.21.0
- Matplotlib >= 3.5.0
- Seaborn >= 0.11.0

## Development Setup

### Frontend
1. Navigate to frontend directory:
   ```bash
   cd agarfarm-frontend
   ```

2. Install dependencies:
   ```bash
   pnpm install
   ```

3. Configure environment:
   - Edit `lib/config.js` to select environment (local/Render)
   - Create `.env` file with required variables

4. Start development server:
   ```bash
   pnpm dev
   ```

### Backend
1. Create Python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # or `venv\Scripts\activate` on Windows
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Start FastAPI server:
   ```bash
   uvicorn app:app --reload
   ```

## Deployment

### Frontend (Render)
- Configured for Render deployment
- Environment variables set in Render dashboard
- Automatic builds on git push

### Backend (Render)
- FastAPI application deployed on Render
- Environment variables configured in Render dashboard
- Automatic deployment on git push

## Testing
- Frontend: Next.js built-in testing
- Backend: FastAPI test client
- ML: Custom evaluation scripts in `evaluation/` directory

## Documentation
- API Documentation: Available at `/docs` when running backend
- Component Documentation: In respective directories
- ML Documentation: In `ml_training/` directory

## Monitoring and Logging
- Backend logs: `logs/` directory
- Frontend logs: Browser console and Render logs
- ML training logs: TensorBoard integration

## Contributing
1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## License
[Add your license information here] 