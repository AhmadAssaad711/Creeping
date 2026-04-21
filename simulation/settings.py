'''
All the parameters used in the Simulation has been documented here.

Easily modifiable paramters with the quick access in this settings.py file \
    to achieve quick modifications especially during the training sessions.

Names of the parameters are self-explanatory therefore elimating the use of further comments.
'''


HOST = "localhost"
PORT = 2000
TIMEOUT = 20.0

CAR_NAME = 'vehicle.tesla.model3'
EPISODE_LENGTH = 120
NUMBER_OF_VEHICLES = 30
NUMBER_OF_PEDESTRIAN = 10
CONTINUOUS_ACTION = True
VISUAL_DISPLAY = True


RGB_CAMERA = 'sensor.camera.rgb'
SSC_CAMERA = 'sensor.camera.semantic_segmentation'

REALTIME_DEFAULT = True      # sleep(dt) each tick for wall-time pacing

# NOTE: Tune all the parameters based on your country!

# ---------- Phase 1 ----------

# Accel/Decel bounds/limits
MAX_ACCEL = 2.0
MAX_DECEL = 3.0

# Time margin
NEAR_MISS_THRESHOLD = 1.5     # secs
UNSAFE_COMMIT_THRESHOLD = 2.5 # secs

# Hard brake
HARD_BRAKE_THRESHOLD = 3.0    # m/s^2

# Relevance around conflict zone
RELATIVE_RADIUS = 60.0 # m
INNER_RADIUS = 6.0 # m
EXIT_RADIUS = 12.0 # m

# Weights
W_SAFETY     = 5.0
W_NEGOTIATION= 2.0
W_EFFICIENCY = 1.0
W_SMOOTHNESS = 0.2
