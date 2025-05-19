from typing import Dict, Any

#import simulator types for hinting
try:
    #simulator is a sibling directory
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from simulator.core import SensorSnapshot, ControlState, IRRIGATION_RATE
except ImportError:
    # Define dummy types if import fails
    SensorSnapshot = Dict[str, float]
    ControlState = Any # Use Any if dataclass not available
    IRRIGATION_RATE = 12.0 # Provide a default if import fails
    print("WARN: Could not import simulator types for type hinting in baseline.")


class BaselineController:
    """
    A simple rule-based controller for the greenhouse environment.
    Includes logic differentiation for 'Normal' and 'Eco' modes.
    """
    def __init__(self, city_config: Dict[str, Any], target_ranges: Dict[str, float], eco_mode: bool = False):
        """
        Initializes the baseline controller.

        Args:
            city_config: Loaded configuration for the city (e.g., from oslo.yaml). Used for context.
            target_ranges: Dictionary with keys like 't_c_min', 't_c_max', etc.
            eco_mode: If True, controller uses wider thresholds and prioritizes low-energy actions.
        """
        self.city = city_config.get("climate_name", "unknown")
        self.eco_mode = eco_mode

        # Store target ranges
        self.t_c_target_min = target_ranges['t_c_min']
        self.t_c_target_max = target_ranges['t_c_max']
        self.rh_target_min = target_ranges['rh_min']
        self.rh_target_max = target_ranges['rh_max']
        self.sm_target_min = target_ranges['sm_min']
        self.sm_target_max = target_ranges['sm_max']

        # --- Define Trigger Thresholds based on Mode ---
        # Eco mode uses wider bands before taking action
        temp_buffer = 1.0 if eco_mode else 0.2 # Smaller buffer for tighter control in normal
        rh_buffer = 5.0 if eco_mode else 2.0
        sm_buffer = 5.0 if eco_mode else 2.0

        self.temp_low_trigger = self.t_c_target_min + temp_buffer
        self.temp_high_trigger = self.t_c_target_max - temp_buffer
        self.rh_low_trigger = self.rh_target_min + rh_buffer
        self.rh_high_trigger = self.rh_target_max - rh_buffer
        self.sm_low_trigger = self.sm_target_min + sm_buffer
        # No high trigger for SM baseline,  overwatering isn't actively prevented by baseline

        # Simple state memory (optional, e.g., prevent rapid cycling)
        self.last_controls = ControlState()
        # Could add timers here to implement hysteresis (e.g., keep AC on for min 10 mins)

    def get_controls(self, snapshot: SensorSnapshot, outside_temp: float, outside_rh: float, is_raining: bool) -> ControlState:
        """Applies simple rule-based logic based on current state and mode."""
        controls = ControlState()

        # --- Temperature Control ---
        # Decide if AC system needs to be ON (heating or cooling intent)
        # Simulator's step function needs the target_temp_midpoint, which the evaluation engine provides.
        # The baseline just decides if AC should be ON.
        if snapshot.t_c < self.temp_low_trigger:
            controls.ac = True # Request AC (will act as heater)
        elif snapshot.t_c > self.temp_high_trigger:
            controls.ac = True # Request AC (will act as cooler)

        # Fan: Use primarily for cooling assistance when warm
        if snapshot.t_c > self.temp_high_trigger + 0.5: # Activate fan if noticeably above high trigger
            controls.fan = True
            # Eco mode might disable fan if AC is already handling it
            if self.eco_mode and controls.ac and snapshot.t_c > self.temp_low_trigger: # Check if AC is cooling
                 controls.fan = False

        # Vent: Use for passive cooling or dehumidification if advantageous
        can_vent_cool = snapshot.t_c > outside_temp + (3.0 if self.eco_mode else 1.5) # Need bigger difference in eco
        can_vent_dry = snapshot.rh > outside_rh + (15.0 if self.eco_mode else 8.0) # Need bigger difference in eco

        should_vent = False
        if snapshot.t_c > self.temp_high_trigger and can_vent_cool:
            should_vent = True
        if snapshot.rh > self.rh_high_trigger + 5.0 and can_vent_dry: # Vent if significantly humid & outside drier
             should_vent = True

        # Eco mode prioritizes Vent over AC for moderate conditions
        if self.eco_mode and should_vent and controls.ac:
             # If temp is only moderately high, prefer venting
             if self.temp_high_trigger < snapshot.t_c < self.temp_high_trigger + 2.0 and can_vent_cool:
                 controls.ac = False # Turn off AC, let vent handle cooling
             # If humidity is high but temp okay, prefer venting
             elif self.rh_high_trigger < snapshot.rh < self.rh_high_trigger + 10.0 and can_vent_dry and not (snapshot.t_c < self.temp_low_trigger):
                  controls.ac = False # Turn off AC, let vent handle drying

        # Set vent state based on final decision
        if should_vent:
            controls.vent = True
            # Force fan on with vent for better air exchange? (Maybe only normal mode?)
            if not self.eco_mode:
                controls.fan = True

        # --- Humidity Control ---
        # Primarily handled by vent/AC logic above. Could add rules like:

        # --- Soil Moisture Control ---
        if snapshot.sm < self.sm_low_trigger and not is_raining:
            # Basic check if tank has *some* water (more robust check in simulator)
            if snapshot.rain_tank_l > 1.0: # Need at least a little water
                controls.irrigation = True

        # Store current controls for potential future hysteresis logic
        self.last_controls = controls
        return controls