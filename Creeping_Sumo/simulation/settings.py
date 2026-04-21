"""
simulation/settings.py
----------------------
Physical constants shared between environment and agent.
"""

MAX_ACCEL  = 2.0    # m/s^2  -- ego max acceleration
MAX_DECEL  = 4.0    # m/s^2  -- ego max deceleration
MAX_SPEED  = 13.89  # m/s    -- road speed limit (50 km/h)
STEP_LENGTH = 0.05  # s      -- simulation step