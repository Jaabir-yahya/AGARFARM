import math
import random
import yaml
from dataclasses import dataclass, field
from typing import Dict, Optional, Any
import datetime
import logging

logger = logging.getLogger(__name__)

# --- Constants ---
DAY_LENGTH_H = 18
SUN_HEAT_GAIN = 0.07
NIGHT_COOL_LOSS = 0.035
TEMP_EQUALIZATION_FACTOR = 0.0075
FAN_COOL_RATE = 0.25
AC_COOL_RATE = 0.55
AC_HEAT_RATE = 0.40
VENT_COOL_RATE = 0.20
PLANT_TRANSPIRATION_RH = 0.03
VENT_HUM_RATE = 0.05
VENT_DRY_RATE = 0.08
FAN_RH_INCREASE = 0.03
EVAP_RATE_COEFF = 0.0004
SM_RETENTION_FACTOR = 0.0008
SM_BASE_LEVEL = 20.0
IRRIGATION_RATE = 12.0
RAIN_FILL_RATE_LPH = 25
TANK_CAPACITY_L = 500
FAN_POWER_KW = 0.04
AC_POWER_KW = 0.45
PUMP_POWER_KW = 0.03
TEMP_CAP = 45.0
TEMP_FLOOR = 0.0
RH_CAP = 100.0
RH_FLOOR = 10.0
SM_CAP = 100.0
SM_FLOOR = 0.0

# Constants for Stricter Termination (to be used by GreenhouseEnv)
RH_TERMINATE_MIN = 5.0
RH_TERMINATE_MAX = 100.0
SM_TERMINATE_MIN = 0.0

MIN_FAN_ON_DURATION = datetime.timedelta(minutes=15)
MIN_FAN_OFF_DURATION = datetime.timedelta(minutes=10)
MIN_AC_ON_DURATION = datetime.timedelta(minutes=20)
MIN_AC_OFF_DURATION = datetime.timedelta(minutes=10)
MIN_VENT_ON_DURATION = datetime.timedelta(minutes=10)
MIN_VENT_OFF_DURATION = datetime.timedelta(minutes=5)
MIN_IRRIGATION_ON_DURATION = datetime.timedelta(minutes=5)
MIN_IRRIGATION_OFF_DURATION = datetime.timedelta(minutes=30)


@dataclass
class SensorSnapshot:
    t_c: float
    rh: float
    sm: float
    vpd: float
    rain_tank_l: float
    current_datetime_iso: str
    # --- Add optional outside_temp and outside_rh with default values ---
    outside_temp: Optional[float] = field(default=None)
    outside_rh: Optional[float] = field(default=None)

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "t_c": self.t_c, "rh": self.rh, "sm": self.sm, "vpd": self.vpd,
            "rain_tank_l": self.rain_tank_l, "current_datetime_iso": self.current_datetime_iso
        }
        # Only include outside conditions if they have been set (i.e., not None)
        if self.outside_temp is not None:
            data["outside_temp"] = self.outside_temp
        if self.outside_rh is not None:
            data["outside_rh"] = self.outside_rh
        return data


@dataclass
class ControlState:
    fan: bool = False
    ac: bool = False
    vent: bool = False
    irrigation: bool = False

    def to_dict(self) -> Dict[str, bool]:
        return {"fan": self.fan, "ac": self.ac, "vent": self.vent, "irrigation": self.irrigation}

    def copy(self):
        return ControlState(fan=self.fan, ac=self.ac, vent=self.vent, irrigation=self.irrigation)


def saturation_vapor_pressure(temp_c: float) -> float:
    return 0.61094 * math.exp((17.625 * temp_c) / (temp_c + 243.04))


def vapor_pressure_deficit(temp_c: float, rh: float) -> float:
    rh_calc = max(0.0, min(100.0, rh));
    es = saturation_vapor_pressure(temp_c)
    ea = es * (rh_calc / 100.0);
    return max(es - ea, 0.0)


class GreenhouseSimulator:
    def __init__(
            self,
            city_config_path: str,
            init_state: Optional[Dict[str, float]] = None,
            start_year: int = 2025,
            start_month: int = 7,
            start_day_of_month: int = 1,
            start_hour: int = 0,
            rain_intensity_multiplier: float = 1.0,
            rain_probability_multiplier: float = 1.0,
            plant_transpiration_multiplier: float = 1.0,
            soil_drying_multiplier: float = 1.0
    ):
        try:
            with open(city_config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            raise ValueError(f"Error loading city config from {city_config_path}: {e}")

        initial = init_state or {}
        self.temp_c = min(initial.get("t_c", self.config.get("default_init_temp", 20.0)), TEMP_CAP)
        self.rh = initial.get("rh", self.config.get("default_init_rh", 60.0))
        self.sm = initial.get("sm", self.config.get("default_init_sm", 50.0))
        self.rain_tank = initial.get("rain_tank_l", self.config.get("default_init_rain_tank_l", 300.0))

        self.climate_name = self.config.get("climate_name", "unknown")
        self.outside_rh_base = self.config.get("default_outside_rh", 70.0)

        try:
            self.current_datetime = datetime.datetime(start_year, start_month, start_day_of_month, start_hour)
        except ValueError as e:
            raise ValueError(
                f"Invalid start date/time provided: {start_year}-{start_month}-{start_day_of_month} {start_hour}:00. Error: {e}")

        if "monthly_temps" not in self.config or "monthly_rain_prob" not in self.config:
            raise ValueError("City config missing 'monthly_temps' or 'monthly_rain_prob'")
        self.monthly_temps_config = self.config["monthly_temps"]
        self.monthly_rain_prob_config = self.config["monthly_rain_prob"]
        self.monthly_outside_rh_config = self.config.get("monthly_outside_rh", {})

        self.rain_active_for_current_hour = False
        self.rain_hour_last_checked = -1
        self.cumulative_water_l = 0.0
        self.cumulative_kwh = 0.0

        self.rain_intensity_multiplier = max(0.0, rain_intensity_multiplier)
        self.rain_probability_multiplier = max(0.0, rain_probability_multiplier)
        self.plant_transpiration_multiplier = max(0.0, plant_transpiration_multiplier)
        self.soil_drying_multiplier = max(0.0, soil_drying_multiplier)
        logger.info(
            f"Simulator multipliers: RainInt={self.rain_intensity_multiplier:.2f}, "
            f"RainProb={self.rain_probability_multiplier:.2f}, PlantTransp={self.plant_transpiration_multiplier:.2f}, "
            f"SoilDry={self.soil_drying_multiplier:.2f}"
        )

        self.actual_fan_on = initial.get("fan_on", False)
        self.fan_last_toggle_datetime = self.current_datetime - datetime.timedelta(days=1)
        self.actual_ac_on = initial.get("ac_on", False)
        self.ac_last_toggle_datetime = self.current_datetime - datetime.timedelta(days=1)
        self.actual_vent_on = initial.get("vent_on", False)
        self.vent_last_toggle_datetime = self.current_datetime - datetime.timedelta(days=1)
        self.actual_irrigation_on = initial.get("irrigation_on", False)
        self.irrigation_last_toggle_datetime = self.current_datetime - datetime.timedelta(days=1)

    def get_outside_temp(self) -> float:
        current_month = self.current_datetime.month
        base_temp = self.monthly_temps_config.get(current_month, 15.0)
        time_fraction_of_day = (self.current_datetime.hour * 60 + self.current_datetime.minute) / 1440.0
        daily_temp_amplitude = self.config.get("monthly_temp_amplitude", {}).get(current_month, 5.0)
        daily_variation = daily_temp_amplitude * math.sin(2 * math.pi * (time_fraction_of_day - 0.25))
        return base_temp + daily_variation

    def get_outside_rh(self) -> float:
        current_month = self.current_datetime.month
        return self.monthly_outside_rh_config.get(current_month, self.outside_rh_base)

    def _apply_actuator_cooldowns(self, requested_controls: ControlState) -> ControlState:
        effective_controls = requested_controls.copy();
        now = self.current_datetime
        if requested_controls.fan != self.actual_fan_on:
            time_since_last_toggle = now - self.fan_last_toggle_datetime
            if self.actual_fan_on and time_since_last_toggle < MIN_FAN_ON_DURATION:
                effective_controls.fan = True
            elif not self.actual_fan_on and time_since_last_toggle < MIN_FAN_OFF_DURATION:
                effective_controls.fan = False
        if requested_controls.ac != self.actual_ac_on:
            time_since_last_toggle = now - self.ac_last_toggle_datetime
            if self.actual_ac_on and time_since_last_toggle < MIN_AC_ON_DURATION:
                effective_controls.ac = True
            elif not self.actual_ac_on and time_since_last_toggle < MIN_AC_OFF_DURATION:
                effective_controls.ac = False
        if requested_controls.vent != self.actual_vent_on:
            time_since_last_toggle = now - self.vent_last_toggle_datetime
            if self.actual_vent_on and time_since_last_toggle < MIN_VENT_ON_DURATION:
                effective_controls.vent = True
            elif not self.actual_vent_on and time_since_last_toggle < MIN_VENT_OFF_DURATION:
                effective_controls.vent = False
        if requested_controls.irrigation != self.actual_irrigation_on:
            time_since_last_toggle = now - self.irrigation_last_toggle_datetime
            if self.actual_irrigation_on and time_since_last_toggle < MIN_IRRIGATION_ON_DURATION:
                effective_controls.irrigation = True
            elif not self.actual_irrigation_on and time_since_last_toggle < MIN_IRRIGATION_OFF_DURATION:
                effective_controls.irrigation = False
        return effective_controls

    def step(self, dt_min: float, requested_controls: ControlState,
             target_temp_midpoint: float = 18.0) -> SensorSnapshot:
        if dt_min <= 0: raise ValueError("dt_min must be positive")
        if not isinstance(requested_controls, ControlState): raise TypeError("controls must be ControlState type")

        effective_controls = self._apply_actuator_cooldowns(requested_controls)
        now = self.current_datetime
        if effective_controls.fan != self.actual_fan_on: self.fan_last_toggle_datetime = now; self.actual_fan_on = effective_controls.fan
        if effective_controls.ac != self.actual_ac_on: self.ac_last_toggle_datetime = now; self.actual_ac_on = effective_controls.ac
        if effective_controls.vent != self.actual_vent_on: self.vent_last_toggle_datetime = now; self.actual_vent_on = effective_controls.vent
        if effective_controls.irrigation != self.actual_irrigation_on: self.irrigation_last_toggle_datetime = now; self.actual_irrigation_on = effective_controls.irrigation

        if self.current_datetime.hour != self.rain_hour_last_checked:
            self.rain_hour_last_checked = self.current_datetime.hour
            current_month_rain_prob = self.monthly_rain_prob_config.get(self.current_datetime.month, 0) / 100.0
            actual_rain_prob = min(max(current_month_rain_prob * self.rain_probability_multiplier, 0.0), 1.0)
            self.rain_active_for_current_hour = random.random() < actual_rain_prob

        outside_temp = self.get_outside_temp();
        current_outside_rh = self.get_outside_rh()  # Get current outside conditions
        delta_t_natural = 0.0
        if self._is_daytime():
            gain = SUN_HEAT_GAIN * dt_min
            if self.temp_c > outside_temp: gain *= max(0, 1 - (self.temp_c - outside_temp) / 20.0)
            delta_t_natural += gain
        else:
            loss = NIGHT_COOL_LOSS * dt_min
            temp_diff_factor = max(0.1, (self.temp_c - outside_temp) / 10.0 if self.temp_c > outside_temp else 0.1)
            delta_t_natural -= loss * temp_diff_factor
        temp_diff_to_outside = outside_temp - self.temp_c
        delta_t_natural += temp_diff_to_outside * TEMP_EQUALIZATION_FACTOR * dt_min

        delta_t_actuators = 0.0
        if self.actual_fan_on: delta_t_actuators -= FAN_COOL_RATE * dt_min; self._add_energy(FAN_POWER_KW, dt_min)
        if self.actual_ac_on:
            if self.temp_c < target_temp_midpoint:
                delta_t_actuators += AC_HEAT_RATE * dt_min
            else:
                delta_t_actuators -= AC_COOL_RATE * dt_min
            self._add_energy(AC_POWER_KW, dt_min)
        if self.actual_vent_on:
            if self.temp_c > outside_temp: delta_t_actuators -= min(self.temp_c - outside_temp, VENT_COOL_RATE * dt_min)

        delta_rh_natural = PLANT_TRANSPIRATION_RH * dt_min * self.plant_transpiration_multiplier
        delta_rh_actuators = 0.0
        if self.actual_fan_on: delta_rh_actuators += FAN_RH_INCREASE * dt_min
        if self.actual_ac_on and self.temp_c >= target_temp_midpoint: delta_rh_actuators -= VENT_DRY_RATE * 0.5 * dt_min
        if self.actual_vent_on:
            rh_diff_to_outside = current_outside_rh - self.rh
            delta_rh_actuators += rh_diff_to_outside * VENT_HUM_RATE * dt_min
            delta_rh_actuators -= VENT_DRY_RATE * dt_min
        if self.actual_irrigation_on: delta_rh_actuators += 0.15 * dt_min

        delta_sm_natural = 0.0
        temp_effect_sm_evap = max((self.temp_c - 20.0) * EVAP_RATE_COEFF, 0)
        retention_decay_sm = max(0, (self.sm - SM_BASE_LEVEL)) * SM_RETENTION_FACTOR
        delta_sm_natural -= (temp_effect_sm_evap + retention_decay_sm) * dt_min * self.soil_drying_multiplier

        delta_sm_actuators = 0.0
        if self.actual_irrigation_on:
            irrigation_amount_this_step = IRRIGATION_RATE * (dt_min / 60.0)
            delta_sm_actuators += irrigation_amount_this_step * 0.5
            water_used = self._use_water(irrigation_amount_this_step)
            if water_used > 0: self._add_energy(PUMP_POWER_KW, dt_min)

        self.temp_c += (delta_t_natural + delta_t_actuators)
        self.rh += (delta_rh_natural + delta_rh_actuators)
        self.sm += (delta_sm_natural + delta_sm_actuators)

        if self.rain_active_for_current_hour:
            added_water = RAIN_FILL_RATE_LPH * (dt_min / 60.0) * self.rain_intensity_multiplier
            self.rain_tank = min(TANK_CAPACITY_L, self.rain_tank + added_water)

        self.temp_c = max(TEMP_FLOOR, min(TEMP_CAP, self.temp_c))
        self.rh = max(RH_FLOOR, min(RH_CAP, self.rh))
        self.sm = max(SM_FLOOR, min(SM_CAP, self.sm))
        self.rain_tank = max(0.0, min(TANK_CAPACITY_L, self.rain_tank))

        self.current_datetime += datetime.timedelta(minutes=dt_min)
        vpd_kpa = vapor_pressure_deficit(self.temp_c, self.rh)

        return SensorSnapshot(self.temp_c, self.rh, self.sm, vpd_kpa, self.rain_tank,
                              self.current_datetime.isoformat(),
                              outside_temp=outside_temp,
                              outside_rh=current_outside_rh)  # Pass current outside conditions

    def _is_daytime(self) -> bool:
        hour = self.current_datetime.hour
        day_start_hour = self.config.get("day_start_hour", 5)
        day_end_hour = day_start_hour + DAY_LENGTH_H
        if day_end_hour >= 24: return hour >= day_start_hour or hour < (day_end_hour % 24)
        return day_start_hour <= hour < day_end_hour

    def _is_currently_raining(self) -> bool:
        return self.rain_active_for_current_hour

    def _add_energy(self, kw: float, dt_min: float):
        self.cumulative_kwh += kw * (dt_min / 60.0)

    def _use_water(self, litres: float) -> float:
        used_from_tank = min(litres, self.rain_tank)
        self.rain_tank -= used_from_tank
        self.cumulative_water_l += used_from_tank
        return used_from_tank

    def resource_totals(self) -> Dict[str, float]:
        return {"water_l": self.cumulative_water_l, "kwh": self.cumulative_kwh}

    def get_actual_actuator_states(self) -> ControlState:
        return ControlState(fan=self.actual_fan_on, ac=self.actual_ac_on, vent=self.actual_vent_on,
                            irrigation=self.actual_irrigation_on)

