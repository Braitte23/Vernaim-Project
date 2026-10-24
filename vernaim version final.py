import argparse
import base64
import binascii
import collections
import getpass
import hashlib
import hmac
import io
import json
import logging
import os
import pickle
import queue
import re
import secrets
import shutil
import signal
import socket
import sys
import tempfile
import threading
import time
import unicodedata
from datetime import datetime, timedelta, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

import cv2
import numpy as np
from flask import (Flask, Response, jsonify, redirect, render_template_string,
                   request, send_from_directory, session)
from PIL import Image, ImageDraw, ImageFont

# Consola de Windows (cp1252) no soporta todos los símbolos: nunca lanzar error por imprimir.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except Exception:
        pass

IS_WINDOWS = os.name == "nt"
IS_LINUX = sys.platform.startswith("linux")
IS_MAC = sys.platform == "darwin"


#conf. of our variables
def _env(name, default=None):
    v = os.environ.get(name)
    return v if v not in (None, "") else default


def _env_num(name, default, cast=float):
    try:
        return cast(_env(name, default))
    except (TypeError, ValueError):
        return default


def _env_bool(name, default):
    v = _env(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "si", "sí", "on")


THRESHOLD = _env_num("VERNAIM_THRESHOLD", 0.45)                   # cosin minumum
MIN_DET_SCORE = _env_num("VERNAIM_MIN_DET_SCORE", 0.5)            # we discard susp. detections

CAMERA_SOURCE = _env("VERNAIM_CAMERA", #ESP32 IP STREAM)
HOST = _env("VERNAIM_HOST", "0.0.0.0")
PORT = _env_num("VERNAIM_PORT", 5000, int)

UNKNOWN_COOLDOWN = 4.0           
KNOWN_LOG_COOLDOWN = 15.0        
PRE_EVENT_SECONDS = _env_num("VERNAIM_PRE_SECONDS", 3.0)    # previous secs. in the clip
POST_EVENT_SECONDS = _env_num("VERNAIM_POST_SECONDS", 5.0)  #post secs in the clip
RETENTION_DAYS = _env_num("VERNAIM_RETENTION_DAYS", 30, int)
MAX_UNKNOWN_MB = _env_num("VERNAIM_MAX_UNKNOWN_MB", 2048, int)
MAX_EMB_PER_USER = 20
MAX_LIVE_VIEWERS = 4
STREAM_MAX_FPS = 15
SESSION_HOURS = _env_num("VERNAIM_SESSION_HOURS", 12, int)
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

LIVENESS_MODE = _env("VERNAIM_LIVENESS", "motion").lower()
LIVENESS_DEBUG = _env_bool("VERNAIM_LIVENESS_DEBUG", False)

EMB_DIM = 512


class _State:
    base = db_json = db_pkl = unknown = thumbs = icons = log_file = auth_file = None
    cert = key = None
    db = engine = hub = worker = auth = None
    use_https = False
    ui_queue = queue.Queue(maxsize=500)


S = _State()
logger = logging.getLogger("vernaim")


def resolve_base_path():
    
    forced = _env("VERNAIM_HOME")
    if forced:
        return Path(forced).expanduser()
    candidates = [Path.home() / "Desktop" / "folder"]
    for od in (os.environ.get("OneDrive"), os.environ.get("OneDriveConsumer")):
        if od:
            candidates.append(Path(od) / "Desktop" / "folder")
    for c in candidates:
        if c.exists():
            return c
    return Path.home() / "vernaim_data"

class _UiHandler(logging.Handler):
    def emit(self, record):
        if getattr(record, "ui", False):
            try:
                S.ui_queue.put_nowait(f"[{datetime.now().strftime('%H:%M')}] {record.getMessage()}")
            except queue.Full:
                pass


def setup_logging():
    logger.setLevel(logging.INFO)
    for h in list(logger.handlers):
        logger.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = RotatingFileHandler(S.log_file, maxBytes=2 * 1024 * 1024, backupCount=5, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.addHandler(_UiHandler())
    logger.propagate = False
    logging.getLogger("werkzeug").setLevel(logging.WARNING)


def ui_event(msg):
    logger.info(msg, extra={"ui": True})


def atomic_write(path, data, backup=False, private=False):
    path = Path(path)
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    if private:
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
    if backup and path.exists():
        try:
            shutil.copy2(path, path.with_name(path.name + ".bak"))
        except OSError:
            pass
    for attempt in range(6):                       # Windows: PermissionError in case we have a different thread working
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.05 * (attempt + 1))


def imwrite_safe(path, img, quality=90):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return False
    atomic_write(path, buf.tobytes())
    return True


def imread_safe(path):
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    except OSError:
        return None


NAME_RE = re.compile(r"^[^\W_][\w .'\-]{0,47}$")
_TS = r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}"
FACE_RE = re.compile(rf"FACE_({_TS})\.jpg")
CLIP_RE = re.compile(rf"CLIP_({_TS})\.mp4")


def clean_name(raw):
    if not isinstance(raw, str):
        return None
    s = re.sub(r"\s+", " ", unicodedata.normalize("NFC", raw)).strip()
    return s if NAME_RE.fullmatch(s) else None


def safe_child(base, filename, regex):
    if not isinstance(filename, str):
        return None
    m = regex.fullmatch(filename)
    if not m:
        return None
    base = Path(base).resolve()
    p = (base / filename).resolve()
    return (p, m.group(1)) if p.parent == base else None


def clip_files(ts):
    return (f"FACE_{ts}.jpg", f"CLIP_{ts}.mp4", f"EMB_{ts}.npy")


def fold_ascii(text):
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii") or "?"


def crop_square(img, bbox, margin=0.5, out=224):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(x2 - x1, y2 - y1) * (1 + margin) / 2
    half = max(half, 16)
    l, t, r, b = int(cx - half), int(cy - half), int(cx + half), int(cy + half)
    pad = max(0, -l, -t, r - w, b - h)
    if pad:
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        l, t, r, b = l + pad, t + pad, r + pad, b + pad
    return cv2.resize(img[t:b, l:r], (out, out), interpolation=cv2.INTER_AREA)


def get_ip():
    targets = []
    m = re.match(r"^https?://([^/:]+)", str(CAMERA_SOURCE))
    if m:
        targets.append(m.group(1))
    targets.append("8.8.8.8")
    for t in targets:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.settimeout(1)
            s.connect((t, 80))
            ip = s.getsockname()[0]
            s.close()
            if ip and not ip.startswith("0."):
                return ip
        except OSError:
            continue
    return "127.0.0.1"


class AuthStore:
    ITER = 600_000

    def __init__(self, path):
        self.path = Path(path)
        self.data = {}
        self.initial_password = None

    def _derive(self, pw, salt, iters):
        return hashlib.pbkdf2_hmac("sha256", pw.encode("utf-8"), salt, iters)

    def _store(self, pw):
        salt = os.urandom(16)
        self.data = {
            "iter": self.ITER, "salt": salt.hex(),
            "hash": self._derive(pw, salt, self.ITER).hex(),
            "secret": secrets.token_hex(32),   #we change passwords 
        }
        atomic_write(self.path, json.dumps(self.data).encode(), private=True)

    def load_or_create(self):
        env_pw = _env("VERNAIM_PASSWORD")
        try:
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
            assert all(k in self.data for k in ("iter", "salt", "hash", "secret"))
        except FileNotFoundError:
            self.data = {}
        except Exception:
            bad = self.path.with_name(f"{self.path.name}.corrupt-{int(time.time())}")
            try:
                self.path.rename(bad)
            except OSError:
                pass
            logger.error("auth.json ilegible; se movió a %s y se crea una contraseña nueva.", bad)
            self.data = {}
        if not self.data:
            pw = env_pw or secrets.token_urlsafe(12)
            self._store(pw)
            if not env_pw:
                self.initial_password = pw
        elif env_pw and not self.verify(env_pw):
            self._store(env_pw)

    def set_password(self, pw):
        if len(pw) < 8:
            raise ValueError("La contraseña debe tener al menos 8 caracteres.")
        self._store(pw)

    def verify(self, pw):
        if not isinstance(pw, str) or not pw or len(pw) > 256:
            return False
        try:
            calc = self._derive(pw, bytes.fromhex(self.data["salt"]), int(self.data["iter"]))
            return hmac.compare_digest(calc.hex(), self.data["hash"])
        except Exception:
            return False

    @property
    def secret_key(self):
        return bytes.fromhex(self.data["secret"])


class LoginLimiter:

    def __init__(self, max_fail=5, window=300, lock=300):
        self.max_fail, self.window, self.lock = max_fail, window, lock
        self._d = {}
        self._m = threading.Lock()

    def locked_for(self, ip):
        with self._m:
            rec = self._d.get(ip)
            if rec and rec["until"] > time.time():
                return int(rec["until"] - time.time()) + 1
            return 0

    def fail(self, ip):
        now = time.time()
        with self._m:
            rec = self._d.setdefault(ip, {"fails": [], "until": 0})
            rec["fails"] = [t for t in rec["fails"] if now - t < self.window] + [now]
            if len(rec["fails"]) >= self.max_fail:
                rec["until"], rec["fails"] = now + self.lock, []

    def ok(self, ip):
        with self._m:
            self._d.pop(ip, None)


limiter = LoginLimiter()


class _RestrictedUnpickler(pickle.Unpickler):
    _OK_MODULES = {"numpy", "numpy.core.multiarray", "numpy._core.multiarray",
                   "numpy.core.numeric", "numpy._core.numeric"}
    _OK_NAMES = {"_reconstruct", "ndarray", "dtype", "scalar", "_frombuffer"}

    def find_class(self, module, name):
        if module in self._OK_MODULES and name in self._OK_NAMES:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"Objeto no permitido en el pickle: {module}.{name}")


def _valid_emb(e):
    try:
        a = np.asarray(e, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    return a if a.shape[0] == EMB_DIM and np.isfinite(a).all() and np.linalg.norm(a) > 0 else None


def thumb_id(name):
    return hashlib.sha256(name.encode("utf-8")).hexdigest()[:24]


class FaceDB:
    def __init__(self, json_path, legacy_pkl, thumbs_dir):
        self.json_path, self.legacy_pkl, self.thumbs_dir = Path(json_path), Path(legacy_pkl), Path(thumbs_dir)
        self.lock = threading.RLock()
        self.users = {}
        self._snap = ((), np.empty((0, EMB_DIM), np.float32))

    def _read_json(self, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        users = {}
        for name, embs in data["users"].items():
            good = [v for v in (_valid_emb(e) for e in embs) if v is not None]
            if good:
                users[str(name)] = good
        return users

    def load(self):
        with self.lock:
            bak = self.json_path.with_name(self.json_path.name + ".bak")
            if self.json_path.exists():
                for cand, label in ((self.json_path, "principal"), (bak, "respaldo .bak")):
                    try:
                        self.users = self._read_json(cand)
                        if cand is bak:
                            logger.error("faces_db.json ilegible: se restauró desde el respaldo.")
                            self._save()
                        break
                    except Exception as e:
                        logger.error("No se pudo leer la base (%s): %s", label, e)
                        if cand is self.json_path:
                            q = self.json_path.with_name(f"{self.json_path.name}.corrupt-{int(time.time())}")
                            try:
                                self.json_path.rename(q)
                                logger.error("Archivo dañado conservado en %s", q)
                            except OSError:
                                pass
                else:
                    self.users = {}
            elif self.legacy_pkl.exists():
                self._migrate_pickle()
            self._migrate_thumbs()
            self._rebuild()

    def _migrate_pickle(self):
        try:
            with open(self.legacy_pkl, "rb") as f:
                raw = _RestrictedUnpickler(f).load()
            users = {}
            for name, embs in dict(raw).items():
                good = [v for v in (_valid_emb(e) for e in embs) if v is not None]
                if good and isinstance(name, str):
                    users[name] = good
            self.users = users
            self._save()
            logger.info("Migrados %d usuarios desde %s (el .pkl original no se tocó).",
                        len(users), self.legacy_pkl.name)
        except Exception as e:
            self.users = {}
            logger.error("No se pudo migrar %s (%s). Se deja intacto; la base nueva empieza vacía.",
                         self.legacy_pkl, e)

    def _migrate_thumbs(self):
        for name in list(self.users):
            if "/" in name or "\\" in name or ".." in name:
                continue
            old = self.thumbs_dir / f"{name}.jpg"
            new = self.thumb_path(name)
            try:
                if old.exists() and not new.exists() and old.resolve().parent == self.thumbs_dir.resolve():
                    os.replace(old, new)
            except OSError:
                pass

    def _save(self):
        payload = {"version": 2, "users": {
            n: [np.round(e, 6).tolist() for e in embs] for n, embs in self.users.items()}}
        atomic_write(self.json_path, json.dumps(payload).encode("utf-8"), backup=True)
        bak = self.json_path.with_name(self.json_path.name + ".bak")
        if not bak.exists():                       
            try:
                shutil.copy2(self.json_path, bak)
            except OSError:
                pass

    def _rebuild(self):
        names, rows = [], []
        for n, embs in self.users.items():
            for e in embs:
                names.append(n)
                rows.append(e / np.linalg.norm(e))
        mat = np.vstack(rows).astype(np.float32) if rows else np.empty((0, EMB_DIM), np.float32)
        self._snap = (tuple(names), mat)      

    def match(self, emb):
        """(nombre|None, similitud). Lee un snapshot inmutable: seguro entre hilos."""
        names, mat = self._snap
        if not len(names):
            return None, 0.0
        v = np.asarray(emb, dtype=np.float32).reshape(-1)
        n = np.linalg.norm(v)
        if n == 0:
            return None, 0.0
        sims = mat @ (v / n)
        i = int(np.argmax(sims))
        return names[i], float(sims[i])

    
    def thumb_path(self, name):
        return self.thumbs_dir / f"{thumb_id(name)}.jpg"

    def find(self, name):
        low = name.casefold()
        for n in self.users:
            if n.casefold() == low:
                return n
        return None

    def add(self, name, emb, thumb=None):
        a = _valid_emb(emb)
        if a is None:
            raise ValueError("Embedding inválido.")
        with self.lock:
            canon = self.find(name) or name
            lst = self.users.setdefault(canon, [])
            lst.append(a)
            del lst[:-MAX_EMB_PER_USER]
            if thumb is not None:
                imwrite_safe(self.thumb_path(canon), thumb)
            self._save()
            self._rebuild()
            return canon

    def rename(self, old, new):
        with self.lock:
            if old not in self.users:
                return "missing"
            clash = self.find(new)
            if clash and clash != old:
                return "exists"
            self.users[new] = self.users.pop(old)
            try:
                if self.thumb_path(old).exists():
                    os.replace(self.thumb_path(old), self.thumb_path(new))
            except OSError:
                pass
            self._save()
            self._rebuild()
            return "ok"

    def delete(self, name):
        with self.lock:
            if name not in self.users:
                return False
            del self.users[name]
            try:
                self.thumb_path(name).unlink()
            except OSError:
                pass
            self._save()
            self._rebuild()
            return True

    def list_users(self):
        with self.lock:
            return [{"name": n, "thumb_url": f"/thumbs/{thumb_id(n)}.jpg" if self.thumb_path(n).exists() else None}
                    for n in sorted(self.users, key=str.casefold)]


class FaceEngine:
    def __init__(self):
        self.app = None
        self.ctx_id = -1
        self.lock = threading.Lock()

    def load(self):
        from insightface.app import FaceAnalysis
        providers, ctx = ["CPUExecutionProvider"], -1
        try:
            import onnxruntime as ort
            if "CUDAExecutionProvider" in ort.get_available_providers():
                providers, ctx = ["CUDAExecutionProvider", "CPUExecutionProvider"], 0
        except Exception:
            pass
        self.ctx_id = ctx
        self.app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"],
                                providers=providers)
        self.app.prepare(ctx_id=ctx, det_size=(640, 640))
        logger.info("Motor IA listo (%s).", "GPU CUDA" if ctx == 0 else "CPU")

    def detect(self, img):
        with self.lock:
            return self.app.get(img)

    def detect_hires(self, img):
        h, w = img.shape[:2]
        scale = min(1.0, 1280.0 / max(h, w))
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            h, w = img.shape[:2]
        size = (max(160, w - w % 32), max(160, h - h % 32))
        with self.lock:
            try:
                self.app.prepare(ctx_id=self.ctx_id, det_size=size)
                return self.app.get(img), img
            finally:
                self.app.prepare(ctx_id=self.ctx_id, det_size=(640, 640))


def yaw_ratio(kps):
    k = np.asarray(kps, dtype=np.float32)
    eye_mid = (k[0] + k[1]) / 2
    d = float(np.linalg.norm(k[1] - k[0])) or 1.0
    return float((k[2][0] - eye_mid[0]) / d)


class LivenessChecker:
    def __init__(self, mode=LIVENESS_MODE, window=4.0, ttl=30.0, min_range=None):
        self.mode = mode if mode in ("off", "motion", "challenge") else "motion"
        self.window, self.ttl = window, ttl
        self.min_range = min_range if min_range is not None else _env_num(
            "VERNAIM_LIVENESS_RANGE", 0.30 if self.mode == "challenge" else 0.10)
        self._tracks = {}

    def update(self, key, kps, now=None):
        """Devuelve (verificado, rango_actual)."""
        if self.mode == "off":
            return True, 0.0
        now = time.time() if now is None else now
        tr = self._tracks.setdefault(key, {"s": collections.deque(), "until": 0.0, "seen": now})
        tr["seen"] = now
        tr["s"].append((now, yaw_ratio(kps)))
        while tr["s"] and now - tr["s"][0][0] > self.window:
            tr["s"].popleft()
        ys = [y for _, y in tr["s"]]
        rng = (max(ys) - min(ys)) if len(ys) >= 4 else 0.0
        if rng >= self.min_range:
            tr["until"] = now + self.ttl
        for k in [k for k, v in self._tracks.items() if now - v["seen"] > 120]:
            del self._tracks[k]
        return now < tr["until"], rng



def parse_source(src):
    s = str(src).strip()
    return int(s) if s.isdigit() else s


def open_capture(src):
    if isinstance(src, int):                       # webcam local
        backend = cv2.CAP_DSHOW if IS_WINDOWS else (cv2.CAP_V4L2 if IS_LINUX else cv2.CAP_ANY)
        cap = cv2.VideoCapture(src, backend)
        return cap if cap.isOpened() else cv2.VideoCapture(src)
    if src.lower().startswith("rtsp://"):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|max_delay;500000"
    else:                                          # HTTP/MJPEG (ESP32-CAM)
        os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)
    return cv2.VideoCapture(src, cv2.CAP_FFMPEG)


class VideoStreamer:
    STALE_SECONDS = 8.0

    def __init__(self, src):
        self.src = parse_source(src)
        self.frame = None
        self.seq = 0
        self.last_frame_time = 0.0
        self.running = True
        self._lock = threading.Lock()
        self.cap = None
        self.thread = threading.Thread(target=self._update, daemon=True, name="video-reader")
        self.thread.start()

    @property
    def connected(self):
        return self.frame is not None and (time.time() - self.last_frame_time) < self.STALE_SECONDS

    def _update(self):
        backoff, fails = 1.0, 0
        while self.running:
            try:
                if self.cap is None or not self.cap.isOpened():
                    if self.cap is not None:
                        self.cap.release()
                    self.cap = open_capture(self.src)
                    if not self.cap.isOpened():
                        time.sleep(backoff)
                        backoff = min(backoff * 2, 10.0)
                        continue
                    backoff, fails = 1.0, 0
                ok, frame = self.cap.read()
                if ok and frame is not None:
                    with self._lock:
                        self.frame, self.seq = frame, self.seq + 1
                        self.last_frame_time = time.time()
                    fails = 0
                else:
                    fails += 1
                    time.sleep(0.01)
                    if fails > 100:                # ESP32 
                        self.cap.release()
                        self.cap, fails = None, 0
                        time.sleep(0.5)
            except Exception as e:
                logger.warning("Lector de video: %s", e)
                time.sleep(0.5)

    def read_seq(self):
        with self._lock:
            return self.frame, self.seq

    def read(self):                                 
        f, _ = self.read_seq()
        return f is not None, f

    def release(self):
        self.running = False
        self.thread.join(timeout=2.0)
        if not self.thread.is_alive() and self.cap is not None:
            self.cap.release()



class FrameHub:
    def __init__(self):
        self.cond = threading.Condition()
        self.frame, self.seq = None, 0
        self.viewers = 0
        self._enc_lock = threading.Lock()
        self._jpeg, self._jpeg_seq = None, -1

    def publish(self, frame):
        with self.cond:
            self.frame, self.seq = frame, self.seq + 1
            self.cond.notify_all()

    def viewer_enter(self):
        with self.cond:
            if self.viewers >= MAX_LIVE_VIEWERS:
                return False
            self.viewers += 1
            return True

    def viewer_leave(self):
        with self.cond:
            self.viewers = max(0, self.viewers - 1)

    def wait_jpeg(self, last_seq, timeout=5.0):
        """Espera un frame NUEVO y devuelve (seq, bytes|None). Codifica una vez por frame."""
        with self.cond:
            if not self.cond.wait_for(lambda: self.seq != last_seq and self.frame is not None, timeout):
                return last_seq, None
            frame, seq = self.frame, self.seq
        with self._enc_lock:
            if self._jpeg_seq != seq:
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                self._jpeg, self._jpeg_seq = (buf.tobytes() if ok else None), seq
            return seq, self._jpeg



_clip_slots = threading.Semaphore(2)


_bad_codecs = set()


def _open_writer(path, fps, size):
    order = [c for c in ("avc1", "mp4v") if c not in _bad_codecs]   
    for i, code in enumerate(order):
        w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*code), fps, size)
        if w.isOpened():
            _bad_codecs.update(order[:i])         
            return w
        w.release()
    return None


def write_clip(path, frames, fps):
    if not frames:
        return False
    h, w = frames[0].shape[:2]
    path = Path(path)
    for target in (path, Path(tempfile.gettempdir()) / f"vernaim_{os.getpid()}_{path.name}"):
        writer = _open_writer(target, fps, (w, h))
        if writer is None:
            continue
        for f in frames:
            if f.shape[:2] == (h, w):
                writer.write(f)
        writer.release()
        if target.exists() and target.stat().st_size > 0:
            if target != path:
                shutil.move(str(target), str(path))
            return True
    return False


def record_clip(ts, pre_frames, streamer, post_seconds=None):
    post_seconds = POST_EVENT_SECONDS if post_seconds is None else post_seconds
    if not _clip_slots.acquire(blocking=False):
        logger.warning("Demasiadas grabaciones simultáneas; se omite el clip %s", ts)
        return
    try:
        stamped = list(pre_frames)                 # [(t, frame)]
        last_seq = -1
        start = time.time()
        while time.time() - start < post_seconds and streamer.running:
            f, seq = streamer.read_seq()
            if f is not None and seq != last_seq:
                stamped.append((time.time(), f))
                last_seq = seq
            time.sleep(0.02)
        if len(stamped) < 2:
            return
        span = max(0.1, stamped[-1][0] - stamped[0][0])
        fps = float(min(30, max(1, round((len(stamped) - 1) / span))))
        out = Path(S.unknown) / f"CLIP_{ts}.mp4"
        if not write_clip(out, [f for _, f in stamped], fps):
            logger.error("No se pudo escribir el clip %s", out.name)
    except Exception as e:
        logger.error("Error grabando clip: %s", e)
    finally:
        _clip_slots.release()


class CameraWorker(threading.Thread):
    def __init__(self, source, engine, db, hub, liveness=None):
        super().__init__(daemon=True, name="camera-worker")
        self.source, self.engine, self.db, self.hub = source, engine, db, hub
        self.liveness = liveness or LivenessChecker()
        self.stop_event = threading.Event()
        self.streamer = None
        self.ring = collections.deque()
        self.fps = 0.0
        self.last_unknown = 0.0
        self.last_known_log = {}
        self.last_unverified_log = {}
        self.connected = False

    def status(self):
        return {"connected": self.connected, "fps": round(self.fps, 1), "viewers": self.hub.viewers,
                "liveness": self.liveness.mode, "users": len(self.db.users)}

    def stop(self):
        self.stop_event.set()
        if self.streamer:
            self.streamer.release()

    # foto, embedding y clip 
    def _unknown_event(self, frame, face):
        now = time.time()
        if now - self.last_unknown < UNKNOWN_COOLDOWN:
            return None
        self.last_unknown = now
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        face_f, _, emb_f = clip_files(ts)
        imwrite_safe(Path(S.unknown) / face_f, crop_square(frame, face.bbox, margin=0.6))
        try:
            buf = io.BytesIO()
            np.save(buf, np.asarray(face.embedding, dtype=np.float32), allow_pickle=False)
            atomic_write(Path(S.unknown) / emb_f, buf.getvalue())
        except Exception as e:
            logger.warning("No se guardó el embedding del evento: %s", e)
        pre = list(self.ring)
        threading.Thread(target=record_clip, args=(ts, pre, self.streamer), daemon=True,
                         name="clip-recorder").start()
        return "ALERTA: Desconocido (foto + clip guardados)"

    def run(self):
        self.streamer = VideoStreamer(self.source)
        last_seq, was_connected, t_fps, n_fps = -1, None, time.time(), 0
        ui_event(f"Conectando a la cámara: {self.source}")
        while not self.stop_event.is_set():
            conn = self.streamer.connected
            self.connected = conn
            if conn != was_connected:
                if was_connected is not None or conn:
                    ui_event("Cámara conectada" if conn else "Se perdió la conexión con la cámara")
                was_connected = conn
            frame, seq = self.streamer.read_seq()
            if frame is None or seq == last_seq:
                time.sleep(0.005)
                continue
            last_seq = seq
            try:
                self._process(frame)
            except Exception as e:                 
                logger.exception("Error procesando frame: %s", e)
                time.sleep(0.05)
            n_fps += 1
            if time.time() - t_fps >= 2:
                self.fps, t_fps, n_fps = n_fps / (time.time() - t_fps), time.time(), 0

    def _process(self, frame):
        now = time.time()
        self.ring.append((now, frame))
        while self.ring and now - self.ring[0][0] > PRE_EVENT_SECONDS:
            self.ring.popleft()

        proc = frame.copy()
        for face in self.engine.detect(proc):
            if float(getattr(face, "det_score", 1.0)) < MIN_DET_SCORE:
                continue
            name, sim = self.db.match(face.embedding)
            if name is not None and sim > THRESHOLD:
                verified, rng = self.liveness.update(name, face.kps, now)
                if verified:
                    label, color = name, (136, 180, 95)           # green
                    if now - self.last_known_log.get(name, 0) > KNOWN_LOG_COOLDOWN:
                        ui_event(f"Acceso: {name}")
                        self.last_known_log[name] = now
                else:
                    label, color = f"{name} (verificando)", (104, 171, 211)   # yellow
                    if now - self.last_unverified_log.get(name, 0) > 30:
                        ui_event(f"Reconocido SIN verificar vida: {name}")
                        self.last_unverified_log[name] = now
                if LIVENESS_DEBUG:
                    label += f" [r={rng:.2f}]"
            else:
                label, color = "DESCONOCIDO", (70, 70, 209)        # red
                msg = self._unknown_event(frame, face) if float(
                    getattr(face, "det_score", 1.0)) >= MIN_DET_SCORE else None
                if msg:
                    ui_event(msg)
            x1, y1, x2, y2 = face.bbox.astype(int)
            cv2.rectangle(proc, (x1, y1), (x2, y2), color, 2)
            cv2.putText(proc, fold_ascii(label), (x1, max(15, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        self.hub.publish(proc)



def purge_unknown(days=None, max_mb=None):
    days = RETENTION_DAYS if days is None else days
    max_mb = MAX_UNKNOWN_MB if max_mb is None else max_mb
    base, groups = Path(S.unknown), {}
    for p in base.iterdir():
        m = re.fullmatch(rf"(?:FACE|CLIP|EMB)_({_TS})\.(?:jpg|mp4|npy)", p.name)
        if m:
            groups.setdefault(m.group(1), []).append(p)
    removed = 0

    def drop(ts):
        nonlocal removed
        for p in groups.pop(ts, []):
            try:
                p.unlink()
                removed += 1
            except OSError:
                pass

    cutoff = time.time() - days * 86400
    if days > 0:
        for ts in [t for t, ps in groups.items() if max(p.stat().st_mtime for p in ps) < cutoff]:
            drop(ts)
    if max_mb > 0:
        total = lambda: sum(p.stat().st_size for ps in groups.values() for p in ps if p.exists())
        for ts in sorted(groups):                   # el timestamp goes from the older to the newest
            if total() <= max_mb * 1024 * 1024:
                break
            drop(ts)
    return removed


def janitor_loop(stop_event):
    while not stop_event.wait(600):
        try:
            n = purge_unknown()
            if n:
                logger.info("Retención: %d archivos antiguos eliminados.", n)
        except Exception as e:
            logger.warning("Retención: %s", e)



def generate_pwa_icons():
    ACCENT, DARK = "#7a93b8", "#17181b"
    specs = [("icon-192.png", 192, 0.0), ("icon-512.png", 512, 0.0), ("icon-512-maskable.png", 512, 0.16)]
    icons = Path(S.icons)
    if all((icons / fn).exists() for fn, _, _ in specs):
        return
    fonts = ["DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
             "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
             "/System/Library/Fonts/Helvetica.ttc", "arialbd.ttf", "Arial Bold.ttf",
             r"C:\Windows\Fonts\arialbd.ttf"]
    for filename, size, pad in specs:
        path = icons / filename
        if path.exists():
            continue
        img = Image.new("RGB", (size, size), ACCENT)
        draw = ImageDraw.Draw(img)
        fsize, font = int(size * (1 - pad * 2) * 0.6), None
        for fn in fonts:
            try:
                font = ImageFont.truetype(fn, fsize)
                break
            except Exception:
                continue
        font = font or ImageFont.load_default()
        bbox = draw.textbbox((0, 0), "V", font=font)
        pos = ((size - (bbox[2] - bbox[0])) / 2 - bbox[0], (size - (bbox[3] - bbox[1])) / 2 - bbox[1])
        draw.text(pos, "V", font=font, fill=DARK)
        img.save(path, "PNG")


def _load_cert(cert_path):
    from cryptography import x509
    return x509.load_pem_x509_certificate(Path(cert_path).read_bytes())


def _cert_is_ours(cert):
    from cryptography.x509.oid import NameOID
    try:
        return cert.subject == cert.issuer and cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value == "VERNAiM"
    except Exception:
        return False


def _cert_ok(cert, ip):
    import ipaddress
    from cryptography import x509
    exp = getattr(cert, "not_valid_after_utc", None) or cert.not_valid_after.replace(tzinfo=timezone.utc)
    if exp - datetime.now(timezone.utc) < timedelta(days=30):
        return False
    san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    return ipaddress.ip_address(ip) in san.get_values_for_type(x509.IPAddress)


def ensure_certificate(ip):
    cert_p, key_p = Path(S.cert), Path(S.key)
    have = cert_p.exists() and key_p.exists()
    if not _env_bool("VERNAIM_AUTOCERT", True):
        return have
    if have:
        try:
            cert = _load_cert(cert_p)
            if not _cert_is_ours(cert) or _cert_ok(cert, ip):
                return True                       # user's certificate
        except ImportError:
            return True
        except Exception:
            stamp = int(time.time())               
            for p in (cert_p, key_p):
                try:
                    p.rename(p.with_name(f"{p.name}.invalid-{stamp}"))
                except OSError:
                    pass
            logger.warning("cert.pem ilegible: se apartó como *.invalid-%d y se genera uno nuevo.", stamp)
    try:
        import ipaddress
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID
        k = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "VERNAiM")])
        sans = [x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
                x509.IPAddress(ipaddress.ip_address(ip))]
        try:
            sans.append(x509.DNSName(socket.gethostname()))
        except Exception:
            pass
        now = datetime.now(timezone.utc)
        c = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(k.public_key())
             .serial_number(x509.random_serial_number()).not_valid_before(now - timedelta(days=1))
             .not_valid_after(now + timedelta(days=365))
             .add_extension(x509.SubjectAlternativeName(sans), critical=False)
             .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
             .sign(k, hashes.SHA256()))
        atomic_write(key_p, k.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL,
                                          serialization.NoEncryption()), private=True)
        atomic_write(cert_p, c.public_bytes(serialization.Encoding.PEM))
        logger.info("Certificado HTTPS autofirmado generado para %s.", ip)
        return True
    except ImportError:
        logger.warning("Falta 'cryptography' (pip install cryptography): no se puede generar HTTPS automático.")
    except Exception as e:
        logger.warning("No se pudo generar el certificado: %s", e)
    return cert_p.exists() and key_p.exists()



app_web = Flask(__name__)

PWA_MANIFEST = {
    "name": "Vernaim", "short_name": "Vernaim",
    "description": "Identity & Security — reconocimiento facial y monitoreo en tiempo real.",
    "start_url": "/", "scope": "/", "display": "standalone",
    "background_color": "#17181b", "theme_color": "#17181b", "orientation": "portrait-primary",
    "icons": [
        {"src": "/icons/icon-192.png", "sizes": "192x192", "type": "image/png", "purpose": "any"},
        {"src": "/icons/icon-512.png", "sizes": "512x512", "type": "image/png", "purpose": "any"},
        {"src": "/icons/icon-512-maskable.png", "sizes": "512x512", "type": "image/png", "purpose": "maskable"},
    ],
}


SERVICE_WORKER_JS = r"""
const CACHE_NAME = 'vernaim-shell-v2';
const APP_SHELL = ['/manifest.json', '/icons/icon-192.png', '/icons/icon-512.png'];

self.addEventListener('install', (event) => {
    self.skipWaiting();
    event.waitUntil(caches.open(CACHE_NAME).then((c) => c.addAll(APP_SHELL).catch(() => {})));
});
self.addEventListener('activate', (event) => {
    event.waitUntil(
        caches.keys()
            .then((keys) => Promise.all(keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k))))
            .then(() => self.clients.claim())
    );
});
self.addEventListener('fetch', (event) => {
    if (event.request.method !== 'GET') return;
    const url = new URL(event.request.url);
    if (!APP_SHELL.includes(url.pathname)) return;      
    event.respondWith(
        fetch(event.request)
            .then((r) => { if (r && r.ok) { const c = r.clone(); caches.open(CACHE_NAME).then((ch) => ch.put(event.request, c)); } return r; })
            .catch(() => caches.match(event.request))
    );
});
"""

LOGIN_HTML = """<!DOCTYPE html>
<html lang="es"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>VERNAiM — Acceso</title>
<link rel="manifest" href="/manifest.json"><meta name="theme-color" content="#17181b">
<link rel="icon" href="/icons/icon-192.png">
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#17181b;color:#f1f2f4;font-family:system-ui,-apple-system,'Segoe UI',sans-serif;min-height:100vh;display:flex;align-items:center;justify-content:center;padding:20px}
.card{background:#212226;border:1px solid #2f3034;border-radius:14px;padding:28px;width:100%;max-width:360px}
h1{font-size:1.2rem;letter-spacing:.02em}.tag{font-size:.75rem;color:#6b6c72;margin:2px 0 22px}
label{display:block;font-size:.8rem;color:#9a9ba1;margin-bottom:7px;font-weight:500}
input{width:100%;padding:11px 14px;background:#26272b;border:1px solid #2f3034;color:#f1f2f4;border-radius:6px;font-size:.95rem;outline:none}
input:focus{border-color:#7a93b8}
button{margin-top:16px;width:100%;padding:11px;background:#7a93b8;color:#14151a;border:0;border-radius:6px;font-weight:600;font-size:.95rem;cursor:pointer}
.err{margin-top:14px;font-size:.85rem;color:#d1706a}
</style></head><body>
<form class="card" method="post" action="/login" autocomplete="on">
<h1>VERNAiM</h1><div class="tag">Identity &amp; Security</div>
<label for="password">Contraseña de acceso</label>
<input id="password" name="password" type="password" autocomplete="current-password" autofocus required maxlength="256">
<button type="submit">Entrar</button>
{% if error %}<div class="err">{{ error }}</div>{% endif %}
</form></body></html>"""

HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VERNAiM — Panel de control</title>

    <!-- PWA: manifest e iconos -->
    <link rel="manifest" href="/manifest.json">
    <meta name="theme-color" content="#17181b">
    <meta name="csrf-token" content="{{ csrf_token }}">
    <link rel="icon" href="/icons/icon-192.png">
    <link rel="apple-touch-icon" href="/icons/icon-192.png">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
    <meta name="apple-mobile-web-app-title" content="Vernaim">

    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-app: #17181b;
            --bg-panel: #1c1d21;
            --bg-card: #212226;
            --bg-elevated: #26272b;
            --border: #2f3034;
            --border-soft: rgba(255,255,255,0.06);
            --text-primary: #f1f2f4;
            --text-secondary: #9a9ba1;
            --text-muted: #6b6c72;
            --accent: #7a93b8;
            --accent-soft: rgba(122,147,184,0.12);
            --accent-strong: #91a8c9;
            --success: #5fb488;
            --success-soft: rgba(95,180,136,0.12);
            --danger: #d1706a;
            --danger-soft: rgba(209,112,106,0.12);
            --warning: #d3ab68;
            --warning-soft: rgba(211,171,104,0.12);
            --radius-sm: 6px;
            --radius-md: 10px;
            --radius-lg: 14px;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        html, body {
            background-color: var(--bg-app);
            color: var(--text-primary);
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            min-height: 100vh;
            -webkit-font-smoothing: antialiased;
        }

        /* ===== NAV ===== */
        .nav {
            background: var(--bg-panel);
            border-bottom: 1px solid var(--border);
            padding: 14px 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 12px;
            position: sticky;
            top: 0;
            z-index: 50;
        }
        .brand { display: flex; flex-direction: column; }
        .brand-name {
            font-size: 1.15rem;
            font-weight: 700;
            letter-spacing: 0.02em;
            color: var(--text-primary);
        }
        .brand-tag {
            font-size: 0.72rem;
            color: var(--text-muted);
            font-weight: 500;
        }
        .nav-links {
            display: flex;
            gap: 4px;
            background: var(--bg-card);
            padding: 4px;
            border-radius: var(--radius-md);
            border: 1px solid var(--border);
        }
        .nav-btn {
            padding: 8px 16px;
            background: transparent;
            border: none;
            color: var(--text-secondary);
            font-family: inherit;
            font-size: 0.85rem;
            font-weight: 500;
            cursor: pointer;
            border-radius: var(--radius-sm);
            transition: background 0.15s ease, color 0.15s ease;
        }
        .nav-btn:hover { color: var(--text-primary); background: var(--bg-elevated); }
        .nav-btn.active { background: var(--accent-soft); color: var(--accent-strong); }

        .install-btn {
            padding: 9px 14px;
            background: var(--accent);
            color: #14151a;
            border: none;
            font-family: inherit;
            font-size: 0.85rem;
            font-weight: 600;
            cursor: pointer;
            border-radius: var(--radius-sm);
            white-space: nowrap;
            transition: filter 0.15s ease;
        }
        .install-btn:hover { filter: brightness(1.08); }

        /* ===== LAYOUT ===== */
        .container {
            max-width: 960px;
            margin: 0 auto;
            padding: 28px 20px 60px;
        }
        .section { display: none; }
        .section.active { display: block; animation: fadeIn 0.2s ease; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(4px);} to { opacity: 1; transform: translateY(0);} }

        .page-header { margin-bottom: 22px; }
        .page-title { font-size: 1.3rem; font-weight: 700; color: var(--text-primary); }
        .page-subtitle { font-size: 0.88rem; color: var(--text-secondary); margin-top: 4px; }

        /* ===== STATUS STRIP ===== */
        .status-strip {
            display: flex;
            gap: 18px;
            flex-wrap: wrap;
            margin-bottom: 22px;
            padding: 12px 16px;
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
        }
        .status-item { display: flex; align-items: center; gap: 8px; font-size: 0.82rem; color: var(--text-secondary); }
        .status-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--success); flex-shrink: 0; }

        /* ===== CARDS ===== */
        .card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius-lg);
            padding: 24px;
            margin-bottom: 16px;
        }

        /* ===== FORM ===== */
        .form-group { margin-bottom: 18px; }
        label {
            display: block;
            margin-bottom: 7px;
            font-size: 0.8rem;
            color: var(--text-secondary);
            font-weight: 500;
        }
        input[type="text"] {
            width: 100%;
            padding: 11px 14px;
            background: var(--bg-elevated);
            border: 1px solid var(--border);
            color: var(--text-primary);
            font-family: inherit;
            font-size: 0.95rem;
            border-radius: var(--radius-sm);
            outline: none;
            transition: border-color 0.15s ease;
        }
        input[type="text"]:focus { border-color: var(--accent); }
        input[type="text"]::placeholder { color: var(--text-muted); }

        .media-box {
            width: 100%;
            aspect-ratio: 4/3;
            max-height: 340px;
            background: var(--bg-elevated);
            border: 1px solid var(--border);
            margin-bottom: 16px;
            display: flex;
            justify-content: center;
            align-items: center;
            position: relative;
            overflow: hidden;
            border-radius: var(--radius-md);
        }
        video, .preview-img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            display: none;
        }
        .media-placeholder { color: var(--text-muted); font-size: 0.85rem; text-align: center; padding: 20px; }

        .tabs {
            display: flex;
            margin-bottom: 18px;
            background: var(--bg-elevated);
            border-radius: var(--radius-sm);
            padding: 3px;
            gap: 3px;
        }
        .tab-btn {
            flex: 1;
            padding: 9px;
            background: transparent;
            border: none;
            color: var(--text-secondary);
            font-family: inherit;
            font-size: 0.85rem;
            font-weight: 500;
            cursor: pointer;
            border-radius: 5px;
            transition: all 0.15s ease;
        }
        .tab-btn.active { color: var(--text-primary); background: var(--bg-card); }
        .tab-content { display: none; }
        .tab-content.active { display: block; }

        .file-label {
            padding: 12px;
            background: var(--bg-elevated);
            border: 1px dashed var(--border);
            border-radius: var(--radius-sm);
            cursor: pointer;
            display: block;
            text-align: center;
            color: var(--text-secondary);
            font-size: 0.88rem;
            font-weight: 500;
            transition: border-color 0.15s ease, color 0.15s ease;
        }
        .file-label:hover { border-color: var(--accent); color: var(--text-primary); }
        input[type="file"] { display: none; }

        /* ===== BUTTONS ===== */
        .btn {
            width: 100%;
            padding: 12px;
            font-family: inherit;
            font-size: 0.9rem;
            font-weight: 600;
            border: 1px solid transparent;
            border-radius: var(--radius-sm);
            cursor: pointer;
            margin-top: 10px;
            transition: filter 0.15s ease, background 0.15s ease;
        }
        .btn:hover { filter: brightness(1.08); }
        .btn-primary { background: var(--accent); color: #14151a; }
        .btn-success { background: var(--success); color: #14151a; }
        .btn-danger { background: var(--danger); color: #14151a; }
        .btn-outline {
            background: transparent;
            border: 1px solid var(--border);
            color: var(--text-secondary);
        }
        .btn-outline:hover { border-color: var(--accent); color: var(--text-primary); }
        .btn-small { padding: 7px 12px; font-size: 0.78rem; width: auto; margin: 0; }

        /* ===== TOASTS ===== */
        #toast-container {
            position: fixed;
            bottom: 20px;
            right: 20px;
            display: flex;
            flex-direction: column;
            gap: 10px;
            z-index: 2000;
        }
        .toast {
            min-width: 240px;
            max-width: 340px;
            padding: 13px 16px;
            border-radius: var(--radius-md);
            background: var(--bg-elevated);
            border: 1px solid var(--border);
            border-left: 3px solid var(--accent);
            color: var(--text-primary);
            font-size: 0.85rem;
            box-shadow: 0 6px 20px rgba(0,0,0,0.3);
            animation: toastIn 0.2s ease;
        }
        .toast.success { border-left-color: var(--success); }
        .toast.error { border-left-color: var(--danger); }
        .toast.info { border-left-color: var(--accent); }
        @keyframes toastIn { from { opacity: 0; transform: translateX(12px);} to { opacity: 1; transform: translateX(0);} }
        @keyframes toastOut { from { opacity: 1; } to { opacity: 0; transform: translateY(-4px);} }
        .toast.hide { animation: toastOut 0.2s ease forwards; }

        /* ===== USERS GRID ===== */
        .users-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(170px, 1fr));
            gap: 14px;
        }
        .user-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius-lg);
            padding: 18px;
            text-align: center;
        }
        .user-thumb {
            width: 84px;
            height: 84px;
            object-fit: cover;
            border-radius: 50%;
            border: 1px solid var(--border);
            margin-bottom: 12px;
        }
        .user-name {
            font-size: 0.92rem;
            font-weight: 600;
            margin-bottom: 12px;
            color: var(--text-primary);
            word-break: break-word;
        }
        .user-actions { display: flex; gap: 6px; justify-content: center; }
        .no-thumb {
            width: 84px;
            height: 84px;
            background: var(--bg-elevated);
            border: 1px solid var(--border);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 12px auto;
            font-size: 1.5rem;
            color: var(--text-muted);
        }

        /* ===== CLIPS / DETECTIONS ===== */
        .clip-item {
            display: flex;
            align-items: center;
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius-lg);
            padding: 14px;
            margin-bottom: 12px;
            flex-wrap: wrap;
            gap: 16px;
        }
        .clip-thumb {
            width: 84px;
            height: 84px;
            object-fit: cover;
            border-radius: var(--radius-sm);
            border: 1px solid var(--border);
            flex-shrink: 0;
        }
        .clip-info { flex: 1; min-width: 160px; }
        .clip-label {
            font-size: 0.7rem;
            font-weight: 600;
            letter-spacing: 0.06em;
            color: var(--text-muted);
            margin-bottom: 4px;
        }
        .clip-time { font-size: 0.95rem; font-weight: 600; color: var(--text-primary); margin-bottom: 2px; }
        .clip-desc { font-size: 0.82rem; color: var(--text-secondary); }
        .clip-actions { display: flex; flex-wrap: wrap; gap: 6px; }

        /* ===== EMPTY STATES ===== */
        .empty-state {
            text-align: center;
            padding: 50px 20px;
            color: var(--text-muted);
            background: var(--bg-card);
            border: 1px dashed var(--border);
            border-radius: var(--radius-lg);
            font-size: 0.9rem;
        }

        /* ===== MODAL ===== */
        .modal-overlay {
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(10,10,12,0.6);
            display: none;
            justify-content: center;
            align-items: center;
            z-index: 1000;
            padding: 20px;
        }
        .modal-overlay.active { display: flex; }
        .modal-content {
            background: var(--bg-panel);
            border: 1px solid var(--border);
            border-radius: var(--radius-lg);
            padding: 26px;
            max-width: 420px;
            width: 100%;
            text-align: left;
        }
        .modal-title { font-size: 1.05rem; font-weight: 700; margin-bottom: 6px; color: var(--text-primary); }
        .modal-subtitle { font-size: 0.85rem; color: var(--text-secondary); margin-bottom: 18px; line-height: 1.5; }
        .modal-content img {
            max-width: 100%;
            max-height: 300px;
            width: 100%;
            object-fit: contain;
            border-radius: var(--radius-md);
            margin-bottom: 16px;
            background: var(--bg-elevated);
        }
        .modal-actions { display: flex; gap: 10px; margin-top: 6px; }
        .modal-actions .btn { margin-top: 0; }

        /* ===== QR SECTION (embed reference styling) ===== */
        .qr-card { text-align: center; }

        /* ===== LIVE FEED ===== */
        .live-box {
            max-height: none;
            aspect-ratio: 16/9;
        }

        /* ===== RESPONSIVE ===== */
        @media (max-width: 640px) {
            .nav { padding: 12px 16px; flex-direction: column; align-items: stretch; }
            .nav-links { justify-content: stretch; }
            .nav-btn { flex: 1; text-align: center; }
            .install-btn { width: 100%; }
            .container { padding: 20px 14px 50px; }
            .card { padding: 18px; }
            .users-grid { grid-template-columns: repeat(auto-fill, minmax(130px, 1fr)); gap: 10px; }
            .clip-item { flex-direction: column; align-items: flex-start; }
            .clip-thumb { width: 100%; height: 160px; }
            .clip-actions { width: 100%; }
            .clip-actions .btn-small { flex: 1; }
            .toast { max-width: none; width: 100%; }
        }
    </style>
</head>
<body>
    <nav class="nav">
        <div class="brand">
            <span class="brand-name">VERNAiM</span>
            <span class="brand-tag">Identity &amp; Security</span>
        </div>
        <div class="nav-links">
            <button class="nav-btn active" onclick="showSection('register', event)">Registro</button>
            <button class="nav-btn" onclick="showSection('live', event)">En vivo</button>
            <button class="nav-btn" onclick="showSection('users', event)">Usuarios</button>
            <button class="nav-btn" onclick="showSection('clips', event)">Detecciones</button>
        </div>
        <button id="btn-install-pwa" class="install-btn" style="display:none;" onclick="handleInstallClick()">📲 Instalar Vernaim</button>
        <button class="nav-btn" onclick="logout()">Salir</button>
    </nav>

    <div class="container">
        <!-- SECCIÓN REGISTRO -->
        <div id="section-register" class="section active">
            <div class="page-header">
                <div class="page-title">Registrar nuevo usuario</div>
                <div class="page-subtitle">Captura un rostro con la cámara o sube una fotografía para registrarlo en el sistema.</div>
            </div>
            <div class="card">
                <div class="form-group">
                    <label>Nombre completo</label>
                    <input type="text" id="subject-name" placeholder="Ej. Bryan Pérez" autocomplete="off">
                </div>
                <div class="tabs">
                    <button class="tab-btn active" onclick="switchTab('cam-tab')">Cámara</button>
                    <button class="tab-btn" onclick="switchTab('file-tab')">Subir foto</button>
                </div>
                <div id="cam-tab" class="tab-content active">
                    <div class="media-box">
                        <span id="cam-placeholder" class="media-placeholder">La cámara está apagada</span>
                        <video id="video" autoplay playsinline></video>
                        <img id="cam-preview" class="preview-img" alt="Vista previa">
                    </div>
                    <button class="btn btn-primary" id="btn-init-cam" onclick="initCamera()">Iniciar cámara</button>
                    <button class="btn btn-success" id="btn-capture" style="display:none;" onclick="registerViaCamera()">Capturar y registrar</button>
                </div>
                <div id="file-tab" class="tab-content">
                    <div class="media-box">
                        <span id="file-placeholder" class="media-placeholder">Ningún archivo seleccionado</span>
                        <img id="file-preview" class="preview-img" alt="Vista previa de archivo">
                    </div>
                    <label class="file-label" for="file-input">Seleccionar imagen</label>
                    <input type="file" id="file-input" accept="image/*" onchange="previewFile()">
                    <button class="btn btn-success" onclick="registerViaFile()">Registrar usuario</button>
                </div>
                <canvas id="canvas" style="display:none;"></canvas>
            </div>
        </div>

        <!-- SECCIÓN EN VIVO -->
        <div id="section-live" class="section">
            <div class="page-header">
                <div class="page-title">Cámara en vivo</div>
                <div class="page-subtitle">Transmisión en tiempo real con detección de rostros superpuesta.</div>
            </div>
            <div class="card">
                <div class="media-box live-box">
                    <img id="live-feed" class="preview-img" alt="Cámara en vivo">
                    <span id="live-placeholder" class="media-placeholder">Presiona "Iniciar" para ver la transmisión</span>
                </div>
                <button class="btn btn-primary" id="btn-live-toggle" onclick="toggleLive()">Iniciar</button>
            </div>
        </div>

        <!-- SECCIÓN USUARIOS -->
        <div id="section-users" class="section">
            <div class="page-header">
                <div class="page-title">Usuarios registrados</div>
                <div class="page-subtitle">Consulta, renombra o elimina las identidades registradas en el sistema.</div>
            </div>
            <div id="users-container" class="users-grid"></div>
        </div>

        <!-- SECCIÓN DETECCIONES -->
        <div id="section-clips" class="section">
            <div class="page-header">
                <div class="page-title">Detecciones recientes</div>
                <div class="page-subtitle">Eventos de personas no reconocidas capturados por el sistema.</div>
            </div>
            <div id="clips-container"></div>
        </div>
    </div>

    <div id="toast-container"></div>

    <!-- MODALS -->
    <div id="modal-photo" class="modal-overlay" onclick="closeModal('modal-photo')">
        <div class="modal-content" onclick="event.stopPropagation()">
            <div class="modal-title">Fotografía</div>
            <img id="modal-photo-img" src="" alt="Foto">
            <div class="modal-actions">
                <button class="btn btn-outline" onclick="closeModal('modal-photo')">Cerrar</button>
            </div>
        </div>
    </div>

    <div id="modal-rename" class="modal-overlay" onclick="closeModal('modal-rename')">
        <div class="modal-content" onclick="event.stopPropagation()">
            <div class="modal-title">Renombrar usuario</div>
            <div class="modal-subtitle">Este cambio se aplicará a todos los registros asociados.</div>
            <input type="text" id="rename-input" placeholder="Nuevo nombre">
            <input type="hidden" id="rename-old">
            <div class="modal-actions">
                <button class="btn btn-outline" onclick="closeModal('modal-rename')">Cancelar</button>
                <button class="btn btn-primary" onclick="confirmRename()">Guardar</button>
            </div>
        </div>
    </div>

    <div id="modal-delete" class="modal-overlay" onclick="closeModal('modal-delete')">
        <div class="modal-content" onclick="event.stopPropagation()">
            <div class="modal-title">Eliminar usuario</div>
            <div class="modal-subtitle">Esta acción no se puede deshacer. Se eliminará permanentemente a <strong id="delete-user-name"></strong>.</div>
            <input type="hidden" id="delete-target">
            <div class="modal-actions">
                <button class="btn btn-outline" onclick="closeModal('modal-delete')">Cancelar</button>
                <button class="btn btn-danger" onclick="confirmDelete()">Sí, eliminar</button>
            </div>
        </div>
    </div>

    <div id="modal-register-clip" class="modal-overlay" onclick="closeModal('modal-register-clip')">
        <div class="modal-content" onclick="event.stopPropagation()">
            <div class="modal-title">Registrar desde detección</div>
            <div class="modal-subtitle">Asigna un nombre para registrar a esta persona en el sistema.</div>
            <img id="clip-reg-img" src="" alt="Detección">
            <input type="text" id="clip-reg-name" placeholder="Nombre de la persona">
            <input type="hidden" id="clip-reg-file">
            <div class="modal-actions">
                <button class="btn btn-outline" onclick="closeModal('modal-register-clip')">Cancelar</button>
                <button class="btn btn-success" onclick="confirmClipRegister()">Registrar</button>
            </div>
        </div>
    </div>

    <!-- PWA: instrucciones manuales para iOS (no soporta instalación automática) -->
    <div id="modal-ios-install" class="modal-overlay" onclick="closeModal('modal-ios-install')">
        <div class="modal-content" onclick="event.stopPropagation()">
            <div class="modal-title">Instalar Vernaim</div>
            <div class="modal-subtitle">
                En iPhone/iPad la instalación se hace manualmente desde Safari:<br><br>
                1. Toca el ícono de <strong>Compartir</strong> (cuadrado con flecha hacia arriba).<br>
                2. Selecciona <strong>"Agregar a pantalla de inicio"</strong>.<br>
                3. Confirma tocando <strong>"Agregar"</strong>.<br><br>
                Después de esto, Vernaim aparecerá como un ícono en tu pantalla de inicio y abrirá directamente, sin pasar por el navegador.
            </div>
            <div class="modal-actions">
                <button class="btn btn-primary" style="width:100%;" onclick="closeModal('modal-ios-install')">Entendido</button>
            </div>
        </div>
    </div>

    <script>
        let stream = null;
        let activeTab = 'cam-tab';
        let base64FileString = null;
        let liveOn = false;

        const CSRF = document.querySelector('meta[name="csrf-token"]').content;

        async function api(url, body) {
            const opts = { credentials: 'same-origin', headers: { 'X-CSRF-Token': CSRF } };
            if (body !== undefined) {
                opts.method = 'POST';
                opts.headers['Content-Type'] = 'application/json';
                opts.body = JSON.stringify(body);
            }
            const res = await fetch(url, opts);
            if (res.status === 401) { window.location.href = '/login'; throw new Error('auth'); }
            let data = {};
            try { data = await res.json(); } catch (e) {}
            if (res.status === 403) showToast(data.message || 'Sesión inválida. Recarga la página.', 'error');
            return data;
        }

        function el(tag, cls, text) {
            const e = document.createElement(tag);
            if (cls) e.className = cls;
            if (text !== undefined && text !== null) e.textContent = text;
            return e;
        }

        async function logout() {
            try { await api('/logout', {}); } catch (e) {}
            window.location.href = '/login';
        }

        function showSection(name, evt) {
            document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
            document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
            document.getElementById('section-' + name).classList.add('active');
            if (evt && evt.target) evt.target.classList.add('active');

            if (name === 'users') loadUsers();
            if (name === 'clips') loadClips();
            if (name !== 'live' && liveOn) toggleLive(); // apaga el stream si sales de "En vivo"
        }

        function switchTab(tabId) {
            activeTab = tabId;
            document.querySelectorAll('.tab-btn').forEach((b, i) => {
                b.classList.toggle('active', (i === 0 && tabId === 'cam-tab') || (i === 1 && tabId === 'file-tab'));
            });
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            document.getElementById(tabId).classList.add('active');

            if (tabId !== 'cam-tab' && stream) {
                stream.getTracks().forEach(track => track.stop());
                document.getElementById('video').style.display = 'none';
                document.getElementById('cam-placeholder').style.display = 'block';
                document.getElementById('btn-init-cam').style.display = 'block';
                document.getElementById('btn-capture').style.display = 'none';
                document.getElementById('cam-preview').style.display = 'none';
                stream = null;
            }
        }

        async function initCamera() {
            if (!window.isSecureContext || !navigator.mediaDevices) {
                showToast('El navegador bloquea la cámara sin HTTPS. Usa "Subir foto" o abre el panel por https://.', 'error');
                switchTab('file-tab');
                return;
            }
            try {
                stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" }, audio: false });
                const video = document.getElementById('video');
                video.srcObject = stream;
                document.getElementById('cam-placeholder').style.display = 'none';
                document.getElementById('cam-preview').style.display = 'none';
                video.style.display = 'block';
                document.getElementById('btn-init-cam').style.display = 'none';
                document.getElementById('btn-capture').style.display = 'block';
            } catch (err) {
                showToast("No se pudo acceder a la cámara.", "error");
            }
        }

        function previewFile() {
            const file = document.getElementById('file-input').files[0];
            if (!file) return;
            const reader = new FileReader();
            reader.onloadend = function() {
                base64FileString = reader.result;
                document.getElementById('file-placeholder').style.display = 'none';
                const imgPrev = document.getElementById('file-preview');
                imgPrev.src = reader.result;
                imgPrev.style.display = 'block';
            };
            reader.readAsDataURL(file);
        }

        // ===== CÁMARA EN VIVO (MJPEG) =====
        function toggleLive() {
            const img = document.getElementById('live-feed');
            const placeholder = document.getElementById('live-placeholder');
            const btn = document.getElementById('btn-live-toggle');
            liveOn = !liveOn;
            if (liveOn) {
                img.onerror = () => { if (liveOn) { showToast('Sin señal de la cámara o sesión expirada.', 'error'); toggleLive(); } };
                img.src = '/video_feed?t=' + Date.now();
                img.style.display = 'block';
                placeholder.style.display = 'none';
                btn.textContent = 'Detener';
            } else {
                img.onerror = null;
                img.src = '';
                img.style.display = 'none';
                placeholder.style.display = 'block';
                btn.textContent = 'Iniciar';
            }
        }

        // ===== TOASTS =====
        function showToast(msg, type) {
            const container = document.getElementById('toast-container');
            const toast = document.createElement('div');
            toast.className = 'toast ' + (type || 'info');
            toast.textContent = msg;
            container.appendChild(toast);
            setTimeout(() => {
                toast.classList.add('hide');
                setTimeout(() => toast.remove(), 200);
            }, 3000);
        }

        function sendRegister(name, base64Str, callback) {
            showToast("Procesando registro…", "info");
            api('/register_web', { name: name, image: base64Str })
            .then(data => {
                if (data.success) {
                    showToast(data.message, "success");
                    document.getElementById('subject-name').value = '';
                    if (callback) callback();
                } else {
                    showToast(data.message || 'No se pudo registrar.', "error");
                }
            })
            .catch(() => showToast("No se pudo conectar con el servidor.", "error"));
        }

        function registerViaCamera() {
            const name = document.getElementById('subject-name').value.trim();
            if (!name) { showToast("Ingresa el nombre antes de continuar.", "error"); return; }
            const video = document.getElementById('video');
            const canvas = document.getElementById('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
            const dataUrl = canvas.toDataURL('image/jpeg');
            video.style.display = 'none';
            const camPrev = document.getElementById('cam-preview');
            camPrev.src = dataUrl;
            camPrev.style.display = 'block';
            sendRegister(name, dataUrl, () => {
                camPrev.style.display = 'none';
                video.style.display = 'block';
            });
        }

        function registerViaFile() {
            const name = document.getElementById('subject-name').value.trim();
            if (!name) { showToast("Ingresa el nombre antes de continuar.", "error"); return; }
            if (!base64FileString) { showToast("Selecciona una imagen primero.", "error"); return; }
            sendRegister(name, base64FileString, () => {
                document.getElementById('file-preview').style.display = 'none';
                document.getElementById('file-placeholder').style.display = 'block';
                document.getElementById('file-input').value = '';
                base64FileString = null;
            });
        }

        // ===== USUARIOS =====
        async function loadUsers() {
            let data;
            try { data = await api('/api/users'); } catch (e) { return; }
            const container = document.getElementById('users-container');
            container.replaceChildren();
            if (!data.users || data.users.length === 0) {
                container.appendChild(el('div', 'empty-state', 'No hay usuarios registrados'));
                return;
            }
            data.users.forEach(u => {
                const card = el('div', 'user-card');
                if (u.thumb_url) {
                    const img = el('img', 'user-thumb');
                    img.src = u.thumb_url + '?t=' + Date.now();
                    img.alt = u.name;
                    card.appendChild(img);
                } else {
                    card.appendChild(el('div', 'no-thumb', '?'));
                }
                card.appendChild(el('div', 'user-name', u.name));
                const actions = el('div', 'user-actions');
                const bRen = el('button', 'btn btn-outline btn-small', 'Renombrar');
                bRen.addEventListener('click', () => openRename(u.name));
                const bDel = el('button', 'btn btn-outline btn-small', 'Eliminar');
                bDel.addEventListener('click', () => openDelete(u.name));
                actions.append(bRen, bDel);
                card.appendChild(actions);
                container.appendChild(card);
            });
        }

        function openRename(name) {
            document.getElementById('rename-old').value = name;
            document.getElementById('rename-input').value = '';
            document.getElementById('modal-rename').classList.add('active');
        }

        async function confirmRename() {
            const oldName = document.getElementById('rename-old').value;
            const newName = document.getElementById('rename-input').value.trim();
            if (!newName) { showToast("Ingresa un nombre válido.", "error"); return; }
            let data;
            try { data = await api('/api/rename_user', { old_name: oldName, new_name: newName }); } catch (e) { return; }
            if (data.success) {
                closeModal('modal-rename');
                showToast("Usuario renombrado.", "success");
                loadUsers();
            } else {
                showToast(data.message || 'No se pudo renombrar.', "error");
            }
        }

        function openDelete(name) {
            document.getElementById('delete-target').value = name;
            document.getElementById('delete-user-name').textContent = name;
            document.getElementById('modal-delete').classList.add('active');
        }

        async function confirmDelete() {
            const name = document.getElementById('delete-target').value;
            let data;
            try { data = await api('/api/delete_user', { name: name }); } catch (e) { return; }
            closeModal('modal-delete');
            if (data.success) { showToast("Usuario eliminado.", "success"); loadUsers(); }
            else showToast(data.message || 'No se pudo eliminar.', "error");
        }

        // ===== DETECCIONES / CLIPS =====
        async function loadClips() {
            let data;
            try { data = await api('/api/clips'); } catch (e) { return; }
            const container = document.getElementById('clips-container');
            container.replaceChildren();
            if (!data.clips || data.clips.length === 0) {
                container.appendChild(el('div', 'empty-state', 'No hay detecciones recientes'));
                return;
            }
            data.clips.forEach(c => {
                const item = el('div', 'clip-item');
                const img = el('img', 'clip-thumb');
                img.src = '/unknown/' + encodeURIComponent(c.img) + '?t=' + Date.now();
                img.alt = 'Detección';
                const info = el('div', 'clip-info');
                info.appendChild(el('div', 'clip-label', 'DETECCIÓN'));
                info.appendChild(el('div', 'clip-time', c.timestamp));
                info.appendChild(el('div', 'clip-desc', 'Persona no reconocida' + (c.video ? ' · video disponible' : '')));
                const actions = el('div', 'clip-actions');
                const bView = el('button', 'btn btn-outline btn-small', 'Ver');
                bView.addEventListener('click', () => viewPhoto('/unknown/' + encodeURIComponent(c.img)));
                actions.appendChild(bView);
                if (c.video) {
                    const a = el('a', 'btn btn-outline btn-small', 'Video');
                    a.href = '/unknown/' + encodeURIComponent(c.video);
                    a.target = '_blank';
                    a.rel = 'noopener';
                    a.style.textDecoration = 'none';
                    actions.appendChild(a);
                }
                const bReg = el('button', 'btn btn-outline btn-small', 'Registrar');
                bReg.addEventListener('click', () => openClipRegister(c.img));
                const bDel = el('button', 'btn btn-outline btn-small', 'Eliminar');
                bDel.addEventListener('click', () => deleteClip(c.img));
                actions.append(bReg, bDel);
                item.append(img, info, actions);
                container.appendChild(item);
            });
        }

        function viewPhoto(url) {
            document.getElementById('modal-photo-img').src = url + '?t=' + Date.now();
            document.getElementById('modal-photo').classList.add('active');
        }

        function openClipRegister(imgFile) {
            document.getElementById('clip-reg-img').src = '/unknown/' + encodeURIComponent(imgFile) + '?t=' + Date.now();
            document.getElementById('clip-reg-file').value = imgFile;
            document.getElementById('clip-reg-name').value = '';
            document.getElementById('modal-register-clip').classList.add('active');
        }

        async function confirmClipRegister() {
            const name = document.getElementById('clip-reg-name').value.trim();
            const imgFile = document.getElementById('clip-reg-file').value;
            if (!name) { showToast("Ingresa un nombre válido.", "error"); return; }
            let data;
            try { data = await api('/api/register_from_clip', { name: name, img_file: imgFile }); } catch (e) { return; }
            if (data.success) {
                closeModal('modal-register-clip');
                showToast("Registrado con éxito: " + name, "success");
                loadClips();
            } else {
                showToast(data.message || 'No se pudo registrar.', "error");
            }
        }

        async function deleteClip(img) {
            if (!confirm("¿Eliminar esta detección?")) return;
            let data;
            try { data = await api('/api/delete_clip', { img: img }); } catch (e) { return; }
            if (data.success) { showToast("Detección eliminada.", "success"); loadClips(); }
            else showToast(data.message || 'No se pudo eliminar.', "error");
        }

        function closeModal(id) {
            document.getElementById(id).classList.remove('active');
        }


        let deferredInstallPrompt = null;
        const installBtn = document.getElementById('btn-install-pwa');

        function isStandaloneDisplay() {
            return window.matchMedia('(display-mode: standalone)').matches
                || window.navigator.standalone === true; // iOS Safari
        }

        function isIOSDevice() {
            return /iphone|ipad|ipod/i.test(navigator.userAgent)
                || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1); // iPadOS 13+
        }

        function refreshInstallButtonVisibility() {
            if (!installBtn) return;
            if (isStandaloneDisplay()) {
                // Ya está instalada y corriendo como app: no mostramos el botón.
                installBtn.style.display = 'none';
                return;
            }
            installBtn.style.display = (deferredInstallPrompt || isIOSDevice()) ? 'inline-flex' : 'none';
        }

        if ('serviceWorker' in navigator) {
            window.addEventListener('load', () => {
                navigator.serviceWorker.register('/sw.js').catch(() => {
                });
            });
        }

        window.addEventListener('beforeinstallprompt', (event) => {
            event.preventDefault();
            deferredInstallPrompt = event;
            refreshInstallButtonVisibility();
        });

        window.addEventListener('appinstalled', () => {
            deferredInstallPrompt = null;
            refreshInstallButtonVisibility();
            showToast('Vernaim se instaló correctamente.', 'success');
        });

        async function handleInstallClick() {
            if (deferredInstallPrompt) {
                deferredInstallPrompt.prompt();
                await deferredInstallPrompt.userChoice;
                deferredInstallPrompt = null;
                refreshInstallButtonVisibility();
                return;
            }
            if (isIOSDevice()) {
                document.getElementById('modal-ios-install').classList.add('active');
                return;
            }
            showToast('Busca "Agregar a pantalla de inicio" en el menú de tu navegador.', 'info');
        }

        refreshInstallButtonVisibility();
    </script>
</body>
</html>
"""

CSP = ("default-src 'self'; script-src 'self' 'unsafe-inline'; "
       "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src https://fonts.gstatic.com; "
       "img-src 'self' data: blob:; media-src 'self' blob:; connect-src 'self'; manifest-src 'self'; "
       "worker-src 'self'; object-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'")
PUBLIC_ENDPOINTS = {"login", "manifest", "service_worker", "serve_icon"}
ICON_RE = re.compile(r"[\w\-]+\.png")
THUMB_RE = re.compile(r"[0-9a-f]{24}\.jpg")


def configure_app(use_https):
    S.use_https = bool(use_https)
    app_web.config.update(
        SECRET_KEY=S.auth.secret_key, MAX_CONTENT_LENGTH=MAX_UPLOAD_BYTES,
        SESSION_COOKIE_NAME="vernaim_session", SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax", SESSION_COOKIE_SECURE=S.use_https,
        PERMANENT_SESSION_LIFETIME=timedelta(hours=SESSION_HOURS), JSON_AS_ASCII=False)


def jerr(msg, code=400):
    return jsonify({"success": False, "message": msg}), code


def json_body():
    b = request.get_json(silent=True)
    return b if isinstance(b, dict) else None


@app_web.before_request
def guard():
    if request.endpoint in PUBLIC_ENDPOINTS:
        return None
    if not session.get("auth"):
        if request.path == "/":
            return redirect("/login")
        return jerr("No autenticado.", 401)
    if request.method not in ("GET", "HEAD", "OPTIONS"):
        sent = request.headers.get("X-CSRF-Token", "")
        if not hmac.compare_digest(sent, session.get("csrf", "")):
            return jerr("Token CSRF inválido. Recarga la página.", 403)
    return None


@app_web.after_request
def secure_headers(resp):
    resp.headers["Content-Security-Policy"] = CSP
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "no-referrer"
    resp.headers["Permissions-Policy"] = "camera=(self), microphone=(), geolocation=()"
    if request.path not in ("/video_feed", "/sw.js") and not request.path.startswith(("/icons/", "/manifest")):
        resp.headers.setdefault("Cache-Control", "no-store")
    return resp


@app_web.errorhandler(Exception)
def on_error(e):
    from werkzeug.exceptions import HTTPException
    if isinstance(e, HTTPException):
        if request.path.startswith("/api/") or request.method != "GET":
            return jerr(e.description or e.name, e.code or 500)
        return e
    logger.exception("Error no controlado en %s", request.path)
    return jerr("Error interno del servidor.", 500)



def _login_page(error=None, code=200):
    return render_template_string(LOGIN_HTML, error=error), code


@app_web.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return redirect("/") if session.get("auth") else _login_page()
    ip = request.remote_addr or "?"
    wait = limiter.locked_for(ip)
    if wait:
        return _login_page(f"Demasiados intentos. Espera {wait} s.", 429)
    if S.auth.verify(request.form.get("password", "")):
        limiter.ok(ip)
        session.clear()
        session.permanent = True
        session["auth"], session["csrf"] = True, secrets.token_urlsafe(32)
        logger.info("Inicio de sesión correcto desde %s", ip)
        return redirect("/")
    limiter.fail(ip)
    logger.warning("Intento de acceso FALLIDO desde %s", ip)
    return _login_page("Contraseña incorrecta.", 401)


@app_web.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"success": True})


@app_web.route("/")
def index():
    return render_template_string(HTML_TEMPLATE, csrf_token=session["csrf"])


@app_web.route("/manifest.json")
def manifest():
    return Response(json.dumps(PWA_MANIFEST), mimetype="application/manifest+json")


@app_web.route("/sw.js")
def service_worker():
    resp = Response(SERVICE_WORKER_JS, mimetype="application/javascript")
    resp.headers["Service-Worker-Allowed"] = "/"
    return resp


@app_web.route("/icons/<path:filename>")
def serve_icon(filename):
    if not ICON_RE.fullmatch(filename):
        return jerr("No encontrado.", 404)
    return send_from_directory(S.icons, filename)


@app_web.route("/thumbs/<path:filename>")
def serve_thumb(filename):
    if not THUMB_RE.fullmatch(filename):
        return jerr("No encontrado.", 404)
    return send_from_directory(S.thumbs, filename)


@app_web.route("/unknown/<path:filename>")
def serve_unknown(filename):
    hit = safe_child(S.unknown, filename, FACE_RE) or safe_child(S.unknown, filename, CLIP_RE)
    if not hit or not hit[0].exists():
        return jerr("No encontrado.", 404)
    return send_from_directory(S.unknown, filename, conditional=True)



def gen_mjpeg(hub):
    last, idle, next_t = -1, 0, 0.0
    while True:
        seq, jpeg = hub.wait_jpeg(last, timeout=10.0)
        if jpeg is None:
            idle += 1
            if idle >= 3:                          
                return
            continue
        idle, last = 0, seq
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        next_t = max(next_t, time.time() - 0.5) + 1.0 / STREAM_MAX_FPS
        delay = next_t - time.time()
        if delay > 0:
            time.sleep(delay)


@app_web.route("/video_feed")
def video_feed():
    if not S.hub.viewer_enter():
        return jerr("Demasiados espectadores simultáneos.", 503)
    resp = Response(gen_mjpeg(S.hub), mimetype="multipart/x-mixed-replace; boundary=frame")
    resp.call_on_close(S.hub.viewer_leave)         
    resp.headers["Cache-Control"] = "no-store"
    resp.headers["X-Accel-Buffering"] = "no"
    return resp



def decode_data_url(s):
    if not isinstance(s, str) or len(s) > 14_000_000:
        return None
    m = re.match(r"^data:image/(?:jpeg|jpg|png|webp);base64,", s)
    if not m:
        return None
    try:
        raw = base64.b64decode(s[m.end():], validate=True)
        with Image.open(io.BytesIO(raw)) as im:
            if im.size[0] * im.size[1] > 36_000_000:
                return None
        return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    except (binascii.Error, ValueError, OSError):
        return None


def pick_face(faces):
    if not faces:
        return None, "No se detectó ningún rostro."
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    if float(getattr(face, "det_score", 1.0)) < 0.6 or (face.bbox[2] - face.bbox[0]) < 60:
        return None, "Rostro muy pequeño o borroso. Acércate y mejora la luz."
    return face, None


@app_web.route("/register_web", methods=["POST"])
def register_web():
    body = json_body()
    if body is None:
        return jerr("Solicitud inválida.")
    name = clean_name(body.get("name"))
    if not name:
        return jerr("Nombre inválido (letras, números, espacio . ' - ; máx. 48).")
    img = decode_data_url(body.get("image"))
    if img is None:
        return jerr("Imagen inválida o demasiado grande.")
    faces, img = S.engine.detect_hires(img)
    face, err = pick_face(faces)
    if err:
        return jerr(err)
    canon = S.db.add(name, face.embedding, thumb=crop_square(img, face.bbox, margin=0.35))
    ui_event(f"WEB: {canon} registrado")
    return jsonify({"success": True, "message": f"'{canon}' registrado."})



@app_web.route("/api/status")
def api_status():
    return jsonify(S.worker.status() if S.worker else {"connected": False})


@app_web.route("/api/users")
def api_users():
    return jsonify({"users": S.db.list_users()})


@app_web.route("/api/rename_user", methods=["POST"])
def api_rename_user():
    body = json_body()
    if body is None:
        return jerr("Solicitud inválida.")
    old, new = body.get("old_name"), clean_name(body.get("new_name"))
    if not isinstance(old, str) or old not in S.db.users:
        return jerr("El usuario no existe.", 404)
    if not new:
        return jerr("Nombre nuevo inválido.")
    res = S.db.rename(old, new)
    if res == "exists":
        return jerr("Ya existe un usuario con ese nombre.", 409)
    ui_event(f"Renombrado: {old} → {new}")
    return jsonify({"success": True})


@app_web.route("/api/delete_user", methods=["POST"])
def api_delete_user():
    body = json_body()
    name = body.get("name") if body else None
    if not isinstance(name, str) or not S.db.delete(name):
        return jerr("El usuario no existe.", 404)
    ui_event(f"Eliminado: {name}")
    return jsonify({"success": True})


@app_web.route("/api/clips")
def api_clips():
    base, clips = Path(S.unknown), []
    names = sorted((n for n in os.listdir(base) if FACE_RE.fullmatch(n)), reverse=True)[:200]
    for f_img in names:
        ts = FACE_RE.fullmatch(f_img).group(1)
        vid = f"CLIP_{ts}.mp4"
        pretty = ts.replace("_", " ", 1)
        pretty = pretty[:11] + pretty[11:].replace("-", ":")
        clips.append({"img": f_img, "video": vid if (base / vid).exists() else None, "timestamp": pretty})
    return jsonify({"clips": clips})


def _clip_ts(body):
    for key in ("img", "img_file", "vid"):
        v = body.get(key)
        if isinstance(v, str) and v:
            hit = safe_child(S.unknown, v, FACE_RE) or safe_child(S.unknown, v, CLIP_RE)
            return hit[1] if hit else None
    return None


def _delete_clip_family(ts):
    for fn in clip_files(ts):
        try:
            (Path(S.unknown) / fn).unlink()
        except OSError:
            pass


@app_web.route("/api/register_from_clip", methods=["POST"])
def api_register_from_clip():
    body = json_body()
    if body is None:
        return jerr("Solicitud inválida.")
    name, ts = clean_name(body.get("name")), _clip_ts({"img": body.get("img_file")})
    if not name or not ts:
        return jerr("Datos inválidos.")
    face_f, _, emb_f = clip_files(ts)
    img = imread_safe(Path(S.unknown) / face_f)
    if img is None:
        return jerr("No se pudo leer la imagen.", 404)
    emb = None
    try:                                           # embedding original del evento (mejor calidad)
        emb = np.load(Path(S.unknown) / emb_f, allow_pickle=False)
    except Exception:
        pass
    if _valid_emb(emb) is None:
        faces, _ = S.engine.detect_hires(img)
        face, err = pick_face(faces)
        if err:
            return jerr(err)
        emb = face.embedding
    canon = S.db.add(name, emb, thumb=cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA))
    _delete_clip_family(ts)
    ui_event(f"Clip → {canon}")
    return jsonify({"success": True, "message": f"'{canon}' registrado desde clip."})


@app_web.route("/api/delete_clip", methods=["POST"])
def api_delete_clip():
    body = json_body()
    ts = _clip_ts(body) if body else None
    if not ts:
        return jerr("Archivo inválido.")
    _delete_clip_family(ts)
    logger.info("Detección %s eliminada", ts)
    return jsonify({"success": True})


def init_runtime(base=None):
    S.base = Path(base).expanduser() if base else resolve_base_path()
    S.unknown, S.thumbs, S.icons = S.base / "unknown", S.base / "thumbs", S.base / "icons"
    for p in (S.base, S.unknown, S.thumbs, S.icons):
        p.mkdir(parents=True, exist_ok=True)
    S.db_json, S.db_pkl = S.base / "faces_db.json", S.base / "faces_db.pkl"
    S.log_file, S.auth_file = S.base / "access_log.txt", S.base / "auth.json"
    S.cert, S.key = S.base / "cert.pem", S.base / "key.pem"
    setup_logging()
    S.auth = AuthStore(S.auth_file)
    S.auth.load_or_create()
    S.db = FaceDB(S.db_json, S.db_pkl, S.thumbs)
    S.db.load()
    S.hub = FrameHub()
    generate_pwa_icons()
    return S

DESK = dict(bg="#17181b", sidebar="#1c1d21", card="#212226", border="#2c2d31",
            text="#f1f2f4", muted="#9a9ba1", accent="#7a93b8")


def gui_available():
    if IS_LINUX and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return False
    try:
        import customtkinter  # noqa: F401
        return True
    except Exception:
        return False


def run_gui(url, on_close):
    import customtkinter as ctk
    import qrcode
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")

    class VernaimApp(ctk.CTk):
        def __init__(self):
            super().__init__()
            self.title("VERNAiM — Monitor")
            self.geometry("980x640")
            self.configure(fg_color=DESK["bg"])
            self.grid_columnconfigure(1, weight=1)
            self.grid_rowconfigure(0, weight=1)
            sb = ctk.CTkFrame(self, width=240, corner_radius=0, fg_color=DESK["sidebar"])
            sb.grid(row=0, column=0, sticky="nsew")
            font = lambda s, b=False: ("Segoe UI" if IS_WINDOWS else "DejaVu Sans", s, "bold" if b else "normal")
            ctk.CTkLabel(sb, text="VERNAiM", font=font(22, True), text_color=DESK["text"]).pack(pady=(24, 0), padx=20, anchor="w")
            ctk.CTkLabel(sb, text="Identity & Security", font=font(11), text_color=DESK["muted"]).pack(pady=(0, 14), padx=20, anchor="w")
            ctk.CTkLabel(sb, text="PANEL WEB", font=font(10, True), text_color=DESK["muted"]).pack(pady=(0, 4), padx=20, anchor="w")
            ctk.CTkLabel(sb, text=url, font=font(12), text_color=DESK["accent"]).pack(pady=(0, 10), padx=20, anchor="w")
            qr = qrcode.QRCode(box_size=4, border=2)
            qr.add_data(url)
            qr.make(fit=True)
            buf = io.BytesIO()
            qr.make_image(fill_color="#f1f2f4", back_color=DESK["sidebar"]).save(buf, format="PNG")
            buf.seek(0)
            pil = Image.open(buf).convert("RGB")
            self.qr_img = ctk.CTkImage(light_image=pil, dark_image=pil, size=(130, 130))
            ctk.CTkLabel(sb, text="", image=self.qr_img).pack(pady=(0, 4))
            ctk.CTkLabel(sb, text="Escanea para abrir el panel", font=font(10), text_color=DESK["muted"]).pack(pady=(0, 12))
            self.status = ctk.CTkLabel(sb, text="Cámara: conectando…", font=font(11), text_color=DESK["muted"])
            self.status.pack(pady=(0, 8), padx=20, anchor="w")
            ctk.CTkLabel(sb, text="ACTIVIDAD", font=font(10, True), text_color=DESK["muted"]).pack(pady=(0, 6), padx=20, anchor="w")
            self.log_box = ctk.CTkTextbox(sb, width=200, height=230, font=font(11), fg_color=DESK["card"],
                                          text_color=DESK["text"], border_color=DESK["border"], border_width=1, corner_radius=8)
            self.log_box.pack(pady=(0, 16), padx=20)
            main = ctk.CTkFrame(self, fg_color=DESK["bg"])
            main.grid(row=0, column=1, sticky="nsew", padx=16, pady=16)
            vf = ctk.CTkFrame(main, fg_color=DESK["card"], corner_radius=10, border_width=1, border_color=DESK["border"])
            vf.pack(expand=True, fill="both")
            self.video = ctk.CTkLabel(vf, text="Iniciando cámara…", font=font(13), text_color=DESK["muted"])
            self.video.pack(expand=True, fill="both", padx=8, pady=8)
            self.vf, self.last_seq = vf, -1
            if S.auth.initial_password:
                self.log(f"Contraseña inicial: {S.auth.initial_password}")
            self.protocol("WM_DELETE_WINDOW", self.close)
            self.after(50, self.tick)

        def log(self, msg):
            self.log_box.insert("end", msg + "\n")
            self.log_box.see("end")

        def tick(self):
            try:
                while True:                                    # logs
                    self.log(S.ui_queue.get_nowait())
            except queue.Empty:
                pass
            hub, w = S.hub, S.worker
            if w:
                self.status.configure(text=f"Cámara: {'conectada' if w.connected else 'sin señal'} · {w.fps:.0f} fps")
            if hub.seq != self.last_seq and hub.frame is not None:
                self.last_seq, frame = hub.seq, hub.frame
                aw, ah = max(200, self.vf.winfo_width() - 24), max(150, self.vf.winfo_height() - 24)
                h, wd = frame.shape[:2]
                sc = min(aw / wd, ah / h)
                size = (max(1, int(wd * sc)), max(1, int(h * sc)))
                rgb = cv2.cvtColor(cv2.resize(frame, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
                img = ctk.CTkImage(light_image=Image.fromarray(rgb), dark_image=Image.fromarray(rgb), size=size)
                self.video.configure(image=img, text="")
                self.video.image = img
            self.after(40, self.tick)

        def close(self):
            on_close()
            self.destroy()

    VernaimApp().mainloop()

def serve_web(use_https):
    kwargs = dict(host=HOST, port=PORT, debug=False, use_reloader=False, threaded=True)
    if use_https:
        kwargs["ssl_context"] = (str(S.cert), str(S.key))
    app_web.run(**kwargs)


def main(argv=None):
    ap = argparse.ArgumentParser(description="VERNAiM — control de acceso facial")
    ap.add_argument("--headless", action="store_true", help="sin ventana de escritorio (servidores Linux)")
    ap.add_argument("--no-https", action="store_true", help="forzar HTTP")
    ap.add_argument("--set-password", action="store_true", help="cambiar la contraseña del panel y salir")
    ap.add_argument("--camera", help="fuente de video (URL o índice de webcam)")
    ap.add_argument("--port", type=int)
    args = ap.parse_args(argv)

    global CAMERA_SOURCE, PORT
    if args.camera:
        CAMERA_SOURCE = args.camera
    if args.port:
        PORT = args.port

    init_runtime()
    if args.set_password:
        pw = getpass.getpass("Nueva contraseña (mín. 8 caracteres): ")
        if pw != getpass.getpass("Repite la contraseña: "):
            sys.exit("Las contraseñas no coinciden.")
        try:
            S.auth.set_password(pw)
        except ValueError as e:
            sys.exit(str(e))
        print("Contraseña actualizada. Las sesiones abiertas quedaron invalidadas.")
        return

    ip = get_ip()
    use_https = (not args.no_https) and ensure_certificate(ip)
    configure_app(use_https)
    url = f"{'https' if use_https else 'http'}://{ip}:{PORT}"

    S.engine = FaceEngine()
    S.engine.load()
    S.worker = CameraWorker(CAMERA_SOURCE, S.engine, S.db, S.hub, LivenessChecker())
    S.worker.start()
    stop = threading.Event()
    threading.Thread(target=janitor_loop, args=(stop,), daemon=True, name="janitor").start()

    def shutdown(*_):
        stop.set()
        if S.worker:
            S.worker.stop()

    signal.signal(signal.SIGINT, lambda *a: (shutdown(), os._exit(0)))
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, lambda *a: (shutdown(), os._exit(0)))

    banner = [f"Panel: {url}", f"Datos: {S.base}",
              f"Liveness: {S.worker.liveness.mode} | Cámara: {CAMERA_SOURCE}"]
    if not use_https:
        banner.append("AVISO: sin HTTPS. El registro con la cámara del celular será bloqueado por el navegador "
                      "(usa 'Subir foto') y la PWA no se podrá instalar. Instala 'cryptography' o coloca cert.pem/key.pem.")
    elif _env_bool("VERNAIM_AUTOCERT", True):
        banner.append("HTTPS con certificado autofirmado: el navegador mostrará una advertencia la primera vez.")
    if S.auth.initial_password:
        banner.append(f"CONTRASEÑA INICIAL (guárdala; no se vuelve a mostrar): {S.auth.initial_password}")
    for line in banner:
        logger.info(line)

    if not args.headless and gui_available():
        threading.Thread(target=serve_web, args=(use_https,), daemon=True, name="flask").start()
        run_gui(url, shutdown)
    else:
        if not args.headless:
            logger.info("Sin entorno gráfico o sin customtkinter: modo headless.")
        try:
            serve_web(use_https)
        finally:
            shutdown()


if __name__ == "__main__":
    main()