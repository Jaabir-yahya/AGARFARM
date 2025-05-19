# AGARFARM: Intelligent Greenhouse Control System

## Overview

AGARFARM is a full-stack application that simulates and controls greenhouse environments using Machine Learning. The system combines a physics-based greenhouse simulator with ML-powered controllers to optimize environmental conditions while minimizing resource consumption.

### Key Features

- **Real-time Simulation**: Physics-based greenhouse environment simulation
- **ML-Powered Control**: Reinforcement Learning (PPO) controllers for optimal control
- **Multiple Climate Support**: Configurable for different climate conditions (Oslo, Riyadh)
- **Dual Control Modes**: 
  - Normal Mode: Tight environmental control
  - Eco Mode: Resource-efficient operation
- **Interactive Dashboard**: Real-time monitoring and control interface
- **RESTful API**: Well-documented API for integration and automation

## Tech Stack

### Backend
- Python 3.8+
- FastAPI
- NumPy
- Stable-Baselines3 (PPO)
- PyYAML

### Frontend
- Next.js 14
- React
- Tailwind CSS
- TypeScript
- Chart.js

## Project Structure

```
AGARFARM/
├── api/                    # FastAPI backend
│   ├── routes.py          # API endpoints
│   ├── models.py          # Pydantic models
│   ├── errors.py          # Error handling
│   └── docs.py            # API documentation
├── simulator/             # Greenhouse simulation
│   ├── core.py           # Core simulator
│   └── city_configs/     # Climate configurations
├── controllers/          # Control algorithms
│   ├── baseline.py      # Rule-based controller
│   └── smart_ml_agent.py # ML controller
├── ml_training/         # ML model training
│   ├── custom_env_wrapper.py
│   ├── reward_functions.py
│   └── config/         # Training configurations
├── evaluation/         # Model evaluation
│   ├── scenarios.py
│   └── metrics.py
├── agarfarm-frontend/  # Next.js frontend
│   ├── app/           # App router
│   ├── components/    # React components
│   └── lib/          # Utility functions
└── requirements.txt   # Python dependencies
```

## Setup Instructions

### Backend Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Jaabir-yahya/AGARFARM.git
   cd AGARFARM
   ```

2. **Create Virtual Environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   # .\venv\Scripts\activate  # Windows
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Frontend Setup

1. **Navigate to Frontend Directory:**
   ```bash
   cd agarfarm-frontend
   ```

2. **Install Dependencies:**
   ```bash
   npm install
   ```

3. **Build Frontend:**
   ```bash
   npm run build
   ```

## Running the Application

1. **Start Backend Server:**
   ```bash
   # From project root
   uvicorn app:app --reload
   ```

2. **Start Frontend Development Server:**
   ```bash
   # From agarfarm-frontend directory
   npm run dev
   ```

3. **Access the Application:**
   - Frontend: http://localhost:3000
   - API Documentation: http://localhost:8000/docs

## API Documentation

The API is fully documented using OpenAPI/Swagger. Access the interactive documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

Jaabir Yahya - [@Jaabir-yahya](https://github.com/Jaabir-yahya)

Project Link: [https://github.com/Jaabir-yahya/AGARFARM](https://github.com/Jaabir-yahya/AGARFARM)

