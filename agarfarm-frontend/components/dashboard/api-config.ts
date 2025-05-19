export const API_ENDPOINTS = {
  BASE_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
  WS_URL: process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000/ws',
  API: {
    START: '/api/simulation/start',
    STOP: '/api/simulation/stop',
    RESET: '/api/simulation/reset',
    SET_CONTROL_MODE: '/api/simulation/set-control-mode',
    SET_MANUAL_CONTROLS: '/api/simulation/set-manual-controls',
    SET_CITY: '/api/simulation/set-city',
    SET_SIMULATOR_PARAMS: '/api/simulation/set-simulator-params',
    RUN_BATCH: '/api/simulation/run-batch'
  }
}

export const CONNECTION_RETRY = {
  MAX_RETRIES: 5,
  INITIAL_DELAY_MS: 1000,
  BACKOFF_FACTOR: 1.5
} 