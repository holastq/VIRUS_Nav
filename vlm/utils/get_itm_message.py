import os
import cv2
import numpy as np
import socket
from vlm.itm.blip2itm import BLIP2ITMClient


def _port_from_env(env_name, default):
    raw = os.getenv(env_name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"[WARN] Invalid {env_name}={raw!r}, fallback to {default}")
        return default


BLIP2_PORT = _port_from_env("APEXNAV_BLIP2_PORT", 12182)
itmclient = BLIP2ITMClient(port=BLIP2_PORT)
_warned_server_unavailable = False


def _itm_server_ready(host="127.0.0.1", port=BLIP2_PORT, timeout=0.05):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _warn_once():
    global _warned_server_unavailable
    if not _warned_server_unavailable:
        print(
            f"[WARN] BLIP2 ITM server is unavailable on localhost:{BLIP2_PORT}, fallback score=0.0"
        )
        _warned_server_unavailable = True

def get_itm_message(rgb_image, label):
    txt = f"Is there a {label} in the image?"
    if not _itm_server_ready():
        _warn_once()
        return 0.0, 0.0
    cosine = itmclient.cosine(rgb_image, txt)
    itm_score = itmclient.itm_score(rgb_image, txt)
    return cosine, itm_score

def get_itm_message_cosine(rgb_image, label, room):
    if room != "everywhere":
        txt = f"Seems like there is a {room} or a {label} ahead?"
    else:
        txt = f"Seems like there is a {label} ahead?"
    if not _itm_server_ready():
        _warn_once()
        return 0.0
    cosine = itmclient.cosine(rgb_image, txt)
    return cosine
