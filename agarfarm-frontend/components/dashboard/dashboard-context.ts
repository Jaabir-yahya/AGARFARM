import { createContext, useContext, useState, useEffect, ReactNode } from 'react'
import { wsService } from '@/lib/websocket-service'

export interface ControlState {
  fan: boolean
  ac: boolean
  vent: boolean
  irrigation: boolean
}

export interface EnvironmentData {
  t_c: number
  rh: number
  sm: number
  vpd: number
  rain_tank_l: number
  current_datetime_iso: string
  outside_temp: number
  outside_rh: number
}

export interface SnapshotData {
  environment: EnvironmentData
  controls_applied: ControlState
  controls_requested: ControlState
  resources: {
    kwh: number
    water_l: number
  }
  config: {
    city: string
    control_mode: string
    sim_is_running: boolean
    model_name: string
    model_metadata: Record<string, any>
    current_sim_datetime_iso_config: string
    user_simulator_params: Record<string, number>
  }
  targets: {
    t_c_min: number
    t_c_max: number
    rh_min: number
    rh_max: number
    sm_min: number
    sm_max: number
  }
  user_display_targets: Record<string, number> | null
  events: {
    is_raining: boolean
  }
  history: {
    actuators_requested: ControlState[]
    sensors: EnvironmentData[]
  }
  simulation_error: string | null
}

interface DashboardContextType {
  snapshot: SnapshotData | null
  connectionStatus: 'connected' | 'disconnected' | 'connecting'
  setDropdownState: (isOpen: boolean) => void
}

const DashboardContext = createContext<DashboardContextType | undefined>(undefined)

export function DashboardProvider({ children }: { children: ReactNode }) {
  const [snapshot, setSnapshot] = useState<SnapshotData | null>(null)
  const [connectionStatus, setConnectionStatus] = useState<'connected' | 'disconnected' | 'connecting'>('disconnected')

  useEffect(() => {
    wsService.registerMessageHandler(setSnapshot)
    wsService.registerStatusHandler(setConnectionStatus)

    return () => {
      wsService.unregisterMessageHandler(setSnapshot)
      wsService.unregisterStatusHandler(setConnectionStatus)
    }
  }, [])

  return (
    <DashboardContext.Provider
      value={{
        snapshot,
        connectionStatus,
        setDropdownState: wsService.setDropdownState.bind(wsService)
      }}
    >
      {children}
    </DashboardContext.Provider>
  )
}

export function useDashboard() {
  const context = useContext(DashboardContext)
  if (context === undefined) {
    throw new Error('useDashboard must be used within a DashboardProvider')
  }
  return context
} 