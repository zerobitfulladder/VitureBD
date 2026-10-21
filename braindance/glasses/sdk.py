"""sdk.py — ctypes binding to the parts of VITURE's libglasses braindance uses.

Loaded only inside the helper process (helper.py). The SDK's teardown can hang
inside the library, so it has to live where the session can kill it.

The SDK is not redistributable and is not in this repository: unzip VITURE's
Linux x86_64 SDK into viture-sdk/ at the repo root, or point BRAINDANCE_VITURE_SDK
at another copy. Its x86_64/ dir must be on LD_LIBRARY_PATH for libglasses' own
dependencies (control.py sets that for the helper).
"""
import ctypes
import os

VITURE_VID = 0x35CA
DEVICE_TYPE_CARINA = 2          # Luma Ultra
CALLBACK_DISPLAY_MODE = 2       # VITURE_CALLBACK_ID_DISPLAY_MODE

_ERRORS = {
    -1: "invalid parameter", -2: "USB unavailable", -3: "USB command failed",
    -4: "not supported", -5: "no data", -6: "data parse error",
    -7: "device rejected", -8: "calibration init failed",
    -9: "serial fetch failed", -10: "invalid state", -99: "unknown error",
}

_STATE_CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_int)


def sdk_dir() -> str:
    default = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "viture-sdk")
    return os.environ.get("BRAINDANCE_VITURE_SDK", default)


def lib_dir() -> str:
    return os.path.join(sdk_dir(), "x86_64")


class SdkError(RuntimeError):
    def __init__(self, call: str, rc: int):
        super().__init__(f"{call}: {_ERRORS.get(rc, rc)} ({rc})")
        self.rc = rc


def usb_pids() -> list[int]:
    """Product IDs of every VITURE USB device present (glasses, their microphone)."""
    base, pids = "/sys/bus/usb/devices", []
    for d in os.listdir(base):
        try:
            with open(os.path.join(base, d, "idVendor")) as f:
                if int(f.read(), 16) != VITURE_VID:
                    continue
            with open(os.path.join(base, d, "idProduct")) as f:
                pids.append(int(f.read(), 16))
        except (OSError, ValueError):
            continue
    return pids


class Glasses:
    """An open SDK session on the glasses: create -> initialize -> start.
    `six_dof` picks the Luma Ultra's tracking for the whole session: 3DOF (the
    IMU) or 6DOF (its stereo cameras too, with VIO running on this machine)."""

    def __init__(self, on_state, six_dof: bool = False):
        L = self._lib = ctypes.CDLL(os.path.join(lib_dir(), "libglasses.so"))
        H = ctypes.c_void_p
        L.xr_device_provider_create.restype = H
        L.xr_device_provider_create.argtypes = [ctypes.c_int]
        L.xr_device_provider_initialize.argtypes = [H, ctypes.c_char_p, ctypes.c_char_p]
        for f in ("start", "stop", "shutdown", "get_device_type", "get_display_mode"):
            getattr(L, f"xr_device_provider_{f}").argtypes = [H]
        L.xr_device_provider_destroy.argtypes = [H]
        L.xr_device_provider_destroy.restype = None
        L.xr_device_provider_set_display_mode.argtypes = [H, ctypes.c_int]
        L.xr_device_provider_set_dof_type_carina.argtypes = [H, ctypes.c_int]
        L.xr_device_provider_get_gl_pose_carina.argtypes = [
            H, ctypes.POINTER(ctypes.c_float), ctypes.c_double, ctypes.POINTER(ctypes.c_int)]
        L.xr_device_provider_register_state_callback.argtypes = [H, _STATE_CALLBACK]
        L.xr_device_provider_get_glasses_version.argtypes = [
            H, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)]
        L.xr_device_provider_get_market_name.argtypes = [
            ctypes.c_int, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)]
        L.xr_device_provider_set_log_level(1)       # errors only

        self.pid = next((p for p in usb_pids() if L.xr_device_provider_is_product_id_valid(p)),
                        None)
        if self.pid is None:
            raise RuntimeError("no VITURE glasses on USB")
        self._callback = _STATE_CALLBACK(on_state)  # kept alive for the SDK's thread
        h = self._h = L.xr_device_provider_create(self.pid)
        if not h:
            raise RuntimeError(f"cannot open VITURE device 0x{self.pid:04x}")
        if L.xr_device_provider_get_device_type(h) == DEVICE_TYPE_CARINA:
            # The SDK's default is 6DOF; display control alone needs neither
            # the cameras nor VIO, so they run only when asked for.
            L.xr_device_provider_set_dof_type_carina(h, 1 if six_dof else 0)
        self.six_dof = six_dof
        self._pose = (ctypes.c_float * 7)()
        self._pose_status = ctypes.c_int()
        self._call("initialize", L.xr_device_provider_initialize(h, None, None))
        L.xr_device_provider_register_state_callback(h, self._callback)
        self._call("start", L.xr_device_provider_start(h))

    @staticmethod
    def _call(name: str, rc: int) -> int:
        if rc < 0:
            raise SdkError(name, rc)
        return rc

    @property
    def name(self) -> str:
        buf, n = ctypes.create_string_buffer(64), ctypes.c_int(64)
        self._lib.xr_device_provider_get_market_name(self.pid, buf, ctypes.byref(n))
        return buf.value.decode(errors="replace")

    @property
    def firmware(self) -> str:
        buf, n = ctypes.create_string_buffer(128), ctypes.c_int(128)
        self._call("get_glasses_version",
                   self._lib.xr_device_provider_get_glasses_version(self._h, buf, ctypes.byref(n)))
        return buf.value[:n.value].decode(errors="replace")

    def get_mode(self) -> int:
        return self._call("get_display_mode", self._lib.xr_device_provider_get_display_mode(self._h))

    def set_mode(self, code: int) -> None:
        self._call("set_display_mode", self._lib.xr_device_provider_set_display_mode(self._h, code))

    def pose(self, predict: float) -> tuple[tuple, int]:
        """The head pose `predict` seconds ahead: (px, py, pz, qw, qx, qy, qz) in
        OpenGL axes (x right, y up, z backward), and a status, 0 when steady."""
        self._call("get_gl_pose_carina", self._lib.xr_device_provider_get_gl_pose_carina(
            self._h, self._pose, predict, ctypes.byref(self._pose_status)))
        return tuple(self._pose), self._pose_status.value

    def close(self) -> None:
        L, h = self._lib, self._h
        L.xr_device_provider_stop(h)
        L.xr_device_provider_shutdown(h)
        L.xr_device_provider_destroy(h)
