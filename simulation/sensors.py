import math
import numpy as np
import weakref
from collections import deque

from .connection import carla
from .settings import RGB_CAMERA, SSC_CAMERA


# Keep only the latest N samples to avoid memory growth
MAX_QUEUE_LEN = 4          # for image obs buffer
MAX_COLLISION_LEN = 256    # for collision history


def _is_listening(sensor_actor) -> bool:
    """
    CARLA versions differ:
      - some use sensor.is_listening() method
      - some expose sensor.is_listening boolean
    """
    if sensor_actor is None:
        return False
    try:
        attr = getattr(sensor_actor, "is_listening", None)
        if callable(attr):
            return bool(attr())
        return bool(attr)
    except Exception:
        return False


def _safe_stop(sensor_actor) -> None:
    """Stop sensor stream if possible; never throw."""
    if sensor_actor is None:
        return
    try:
        if _is_listening(sensor_actor):
            sensor_actor.stop()
    except Exception:
        # already stopped / destroyed / disconnected -> ignore
        pass


def _safe_destroy(sensor_actor) -> None:
    """Destroy actor if possible; never throw."""
    if sensor_actor is None:
        return
    try:
        # stopping first is fine; if it's already stopped, it does nothing
        _safe_stop(sensor_actor)
        sensor_actor.destroy()
    except Exception:
        # already destroyed / disconnected -> ignore
        pass


# ---------------------------------------------------------------------
# ------------------------------- CAMERA (semantic segmentation)
# ---------------------------------------------------------------------

class CameraSensor:
    def __init__(self, vehicle):
        self.sensor_name = SSC_CAMERA
        self.parent = vehicle

        # Use deque to cap memory growth
        self.front_camera = deque(maxlen=MAX_QUEUE_LEN)

        world = self.parent.get_world()
        self.sensor = self._set_camera_sensor(world)

        weak_self = weakref.ref(self)
        self.sensor.listen(lambda image: CameraSensor._get_front_camera_data(weak_self, image))

    def _set_camera_sensor(self, world):
        front_camera_bp = world.get_blueprint_library().find(self.sensor_name)
        front_camera_bp.set_attribute('image_size_x', '160')
        front_camera_bp.set_attribute('image_size_y', '80')
        front_camera_bp.set_attribute('fov', '125')
        front_camera = world.spawn_actor(
            front_camera_bp,
            carla.Transform(carla.Location(x=2.4, z=1.5), carla.Rotation(pitch=-10)),
            attach_to=self.parent
        )
        return front_camera

    @staticmethod
    def _get_front_camera_data(weak_self, image):
        self = weak_self()
        if not self:
            return

        # If the sensor is being torn down, ignore late callbacks
        if getattr(self, "sensor", None) is None:
            return

        image.convert(carla.ColorConverter.CityScapesPalette)
        placeholder = np.frombuffer(image.raw_data, dtype=np.uint8)

        # NOTE: CARLA image is typically (height, width, 4). Your old code used (width, height, 4).
        # To avoid changing behavior unexpectedly, we keep your shape, but if you see rotated frames,
        # swap the reshape order.
        try:
            placeholder1 = placeholder.reshape((image.width, image.height, 4))
        except Exception:
            # fallback to common CARLA ordering
            placeholder1 = placeholder.reshape((image.height, image.width, 4))

        target = placeholder1[:, :, :3]
        self.front_camera.append(target)

    def stop(self):
        _safe_stop(getattr(self, "sensor", None))

    def destroy(self):
        _safe_destroy(getattr(self, "sensor", None))
        self.sensor = None

        # clear buffer
        try:
            self.front_camera.clear()
        except Exception:
            pass

    def __del__(self):
        # best-effort cleanup
        try:
            self.destroy()
        except Exception:
            pass


# ---------------------------------------------------------------------
# ------------------------------- ENV CAMERA (RGB spectator view)
# ---------------------------------------------------------------------

class CameraSensorEnv:
    def __init__(self, vehicle):
        self.sensor_name = RGB_CAMERA
        self.parent = vehicle

        world = self.parent.get_world()
        self.sensor = self._set_camera_sensor(world)

        weak_self = weakref.ref(self)
        self.sensor.listen(lambda image: CameraSensorEnv._get_third_person_camera(weak_self, image))

    def _set_camera_sensor(self, world):
        thrid_person_camera_bp = world.get_blueprint_library().find(self.sensor_name)
        thrid_person_camera_bp.set_attribute('image_size_x', '720')
        thrid_person_camera_bp.set_attribute('image_size_y', '720')
        third_camera = world.spawn_actor(
            thrid_person_camera_bp,
            carla.Transform(
                carla.Location(x=-4.0, z=2.0),
                carla.Rotation(pitch=-12.0)
            ),
            attach_to=self.parent
        )
        return third_camera

    @staticmethod
    def _get_third_person_camera(weak_self, image):
        self = weak_self()
        if not self:
            return

        # If the sensor is being torn down, ignore late callbacks
        if getattr(self, "sensor", None) is None:
            return

        try:
            world = self.parent.get_world()
            spectator = world.get_spectator()
            camera_transform = self.sensor.get_transform()
            spectator.set_transform(camera_transform)
        except Exception:
            pass

    def stop(self):
        _safe_stop(getattr(self, "sensor", None))

    def destroy(self):
        _safe_destroy(getattr(self, "sensor", None))
        self.sensor = None

    def __del__(self):
        try:
            self.destroy()
        except Exception:
            pass


# ---------------------------------------------------------------------
# ------------------------------- COLLISION SENSOR
# ---------------------------------------------------------------------

class CollisionSensor:
    def __init__(self, vehicle) -> None:
        self.sensor_name = 'sensor.other.collision'
        self.parent = vehicle

        # Cap history to avoid growth
        self.collision_data = deque(maxlen=MAX_COLLISION_LEN)

        world = self.parent.get_world()
        self.sensor = self._set_collision_sensor(world)

        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def _set_collision_sensor(self, world) -> object:
        collision_sensor_bp = world.get_blueprint_library().find(self.sensor_name)
        sensor_relative_transform = carla.Transform(carla.Location(x=1.3, z=0.5))
        collision_sensor = world.spawn_actor(
            collision_sensor_bp, sensor_relative_transform, attach_to=self.parent
        )
        return collision_sensor

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return

        # If the sensor is being torn down, ignore late callbacks
        if getattr(self, "sensor", None) is None:
            return

        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self.collision_data.append(float(intensity))

    def stop(self):
        _safe_stop(getattr(self, "sensor", None))

    def destroy(self):
        _safe_destroy(getattr(self, "sensor", None))
        self.sensor = None
        try:
            self.collision_data.clear()
        except Exception:
            pass

    def __del__(self):
        try:
            self.destroy()
        except Exception:
            pass
