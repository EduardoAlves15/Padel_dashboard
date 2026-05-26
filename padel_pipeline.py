from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.models import load_model
from ultralytics import YOLO


APP_TITLE = "Padel 3D Analysis Dashboard"
DEFAULT_OUTPUT_DIR = Path("dashboard_output")
DEFAULT_BALL_MODEL = Path(r"c:\Users\edual\Desktop\github\Padel\data\padeltracker100\runs\detect\~\runs\detect\ball_4080_m_1280_final\weights\best.pt")
DEFAULT_POSE_MODEL = Path(r"c:\Users\edual\Desktop\github\Padel\data\padelvic\yolov8m-pose.pt")
DEFAULT_ACTION_MODEL = Path(r"c:\Users\edual\Desktop\github\Padel\modelo_padel_action_v2.h5")

K_MAIN = np.array([[3000, 0, 1813], [0, 3000, 980], [0, 0, 1]], dtype=np.float32)
K_SECOND = np.array([[2500, 0, 1403], [0, 2500, 935], [0, 0, 1]], dtype=np.float32)
METROS_REAIS_2D = np.ascontiguousarray(
    np.array([[0, 20], [10, 20], [10, 0], [0, 0]], dtype=np.float32)
)
PTS_MAIN_2D = np.ascontiguousarray(
    np.array([[1289, 681], [2349, 681], [3354, 1897], [280, 1897]], dtype=np.float32)
)
PTS_SECOND_2D = np.ascontiguousarray(
    np.array([[939, 625], [1805, 625], [2575, 1834], [216, 1834]], dtype=np.float32)
)
FIELD_WIDTH_M = 10.0
FIELD_LENGTH_M = 20.0
ACTION_CLASSES = ["Backhand", "Dropshot", "Forehand", "Other", "Serve", "Smash"]

# Collect geometry issues for debugging (triangulation / homography failures)
GEOMETRY_ISSUES: list[dict[str, Any]] = []


def save_geometry_issues(path: Path) -> None:
    try:
        if not GEOMETRY_ISSUES:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(GEOMETRY_ISSUES, fh, indent=2, ensure_ascii=False)
    except Exception:
        # never crash the pipeline for logging failures
        pass


def validate_calibration() -> dict[str, Any]:
    """Validate homography by mapping the configured image corner points
    (`PTS_MAIN_2D`) through the computed homography and comparing to
    `METROS_REAIS_2D`. Returns a dict with RMSE, mapped points and expected.
    """
    P1, P2, H = get_geometry_matrices()
    pts_img = np.asarray(PTS_MAIN_2D, dtype=np.float32)
    expected = np.asarray(METROS_REAIS_2D, dtype=np.float32)

    mapped = []
    for p in pts_img:
        mx, my = to_meters_from_homography(float(p[0]), float(p[1]), H)
        mapped.append([mx, my])

    mapped_arr = np.asarray(mapped, dtype=np.float32)
    # compute RMSE only on finite mapped values
    mask = np.isfinite(mapped_arr).all(axis=1) & np.isfinite(expected).all(axis=1)
    if not mask.any():
        rmse = float("nan")
    else:
        dif = mapped_arr[mask] - expected[mask]
        rmse = float(np.sqrt((dif * dif).sum(axis=1).mean()))

    return {
        "rmse": rmse,
        "mapped": mapped_arr.tolist(),
        "expected": expected.tolist(),
        "H": H.tolist(),
    }


def get_calibration() -> dict[str, Any]:
    return {
        "PTS_MAIN_2D": PTS_MAIN_2D.tolist(),
        "PTS_SECOND_2D": PTS_SECOND_2D.tolist(),
        "METROS_REAIS_2D": METROS_REAIS_2D.tolist(),
    }


def set_calibration(pts_main: Iterable[Iterable[float]] | None = None,
                    pts_second: Iterable[Iterable[float]] | None = None,
                    metros: Iterable[Iterable[float]] | None = None) -> None:
    """Update global calibration arrays and clear cached geometry."""
    global PTS_MAIN_2D, PTS_SECOND_2D, METROS_REAIS_2D, _GEOMETRY_CACHE
    if pts_main is not None:
        PTS_MAIN_2D = np.ascontiguousarray(np.array(pts_main, dtype=np.float32)[:, :2])
    if pts_second is not None:
        PTS_SECOND_2D = np.ascontiguousarray(np.array(pts_second, dtype=np.float32)[:, :2])
    if metros is not None:
        METROS_REAIS_2D = np.ascontiguousarray(np.array(metros, dtype=np.float32)[:, :2])
    _GEOMETRY_CACHE.clear()


def auto_fix_calibration_permutations() -> dict[str, Any]:
    """Try permutations of METROS_REAIS_2D order to minimize RMSE.

    Returns a dict with keys: 'improved' (bool), 'best_rmse', 'best_permutation', 'best_metros'.
    If an improvement is found, updates global METROS_REAIS_2D and clears cache.
    """
    from itertools import permutations

    current = np.asarray(METROS_REAIS_2D, dtype=np.float32)
    orig = current.copy()
    try:
        base_res = validate_calibration()
        base_rmse = float(base_res.get("rmse", float("nan")))
    except Exception:
        base_rmse = float("inf")

    best_rmse = base_rmse
    best_perm = None
    best_metros = None

    indices = list(range(current.shape[0]))
    for perm in permutations(indices):
        candidate = current[list(perm), :]
        # set temporarily
        set_calibration(metros=candidate)
        try:
            res = validate_calibration()
            rmse = float(res.get("rmse", float("inf")))
        except Exception:
            rmse = float("inf")
        if rmse < best_rmse - 1e-9:
            best_rmse = rmse
            best_perm = perm
            best_metros = candidate.copy()

    # if improvement found, keep it; otherwise restore
    if best_perm is not None:
        set_calibration(metros=best_metros)
        improved = True
    else:
        set_calibration(metros=orig)
        improved = False

    return {
        "improved": improved,
        "best_rmse": best_rmse,
        "best_permutation": list(best_perm) if best_perm is not None else None,
        "best_metros": best_metros.tolist() if best_metros is not None else None,
    }

def load_ball_model(model_path: str):
    return YOLO(model_path)


def load_pose_model(model_path: str):
    return YOLO(model_path)


def load_action_model(model_path: str):
    return load_model(model_path)


def projection_matrix(points_2d: np.ndarray, points_3d: np.ndarray, camera_matrix: np.ndarray) -> np.ndarray:
    _, rvec, tvec = cv2.solvePnP(points_3d, points_2d, camera_matrix, None)
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    return camera_matrix @ np.hstack((rotation_matrix, tvec))


_GEOMETRY_CACHE: dict[str, Any] = {}


def _prepare_pts_for_homography(pts: np.ndarray) -> np.ndarray:
    arr = np.asarray(pts)
    if arr.ndim != 2 or arr.shape[0] < 4:
        raise ValueError("homography points must be a (4,2) or (4,3) array-like")
    # take first two columns as OpenCV expects 2D points
    two = arr[:, :2].astype(np.float32)
    return np.ascontiguousarray(two)


def get_geometry_matrices() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lazily compute and cache projection matrices P1,P2 and homography H_main.

    Validates shapes/dtypes and converts to the formats OpenCV expects (float32, contiguous).
    """
    if _GEOMETRY_CACHE:
        return _GEOMETRY_CACHE["P1"], _GEOMETRY_CACHE["P2"], _GEOMETRY_CACHE["H_MAIN"]

    pts_main = _prepare_pts_for_homography(PTS_MAIN_2D)
    pts_second = _prepare_pts_for_homography(PTS_SECOND_2D)
    metros = _prepare_pts_for_homography(METROS_REAIS_2D)

    P1 = projection_matrix(pts_main, np.pad(metros, ((0, 0), (0, 1))), K_MAIN)
    P2 = projection_matrix(pts_second, np.pad(metros, ((0, 0), (0, 1))), K_SECOND)

    try:
        H_MAIN = cv2.getPerspectiveTransform(pts_main, metros)
    except cv2.error as exc:
        raise RuntimeError(f"failed to compute homography: {exc}") from exc

    _GEOMETRY_CACHE.update({"P1": P1, "P2": P2, "H_MAIN": H_MAIN})
    return P1, P2, H_MAIN


def triangulate_point(point_1: Iterable[float], point_2: Iterable[float]) -> np.ndarray:
    p1 = np.asarray(point_1, dtype=np.float32).reshape(2, 1)
    p2 = np.asarray(point_2, dtype=np.float32).reshape(2, 1)
    P1, P2, _ = get_geometry_matrices()
    point_3d = cv2.triangulatePoints(P1, P2, p1, p2)
    # validate homogeneous coordinate
    w = point_3d[3]
    if not np.isfinite(w) or np.isclose(w, 0.0):
        return np.array([np.nan, np.nan, np.nan], dtype=np.float32)
    return (point_3d[:3] / w).flatten()


def reprojection_error_2view(point_3d: Iterable[float], p1_px: Iterable[float], p2_px: Iterable[float]) -> tuple[float, float]:
    """Compute reprojection error (pixels) for a 3D point against the two camera projection matrices."""
    P1, P2, _ = get_geometry_matrices()
    X = np.asarray([point_3d[0], point_3d[1], point_3d[2], 1.0], dtype=np.float32)
    proj1 = P1 @ X
    proj2 = P2 @ X
    if not np.isfinite(proj1).all() or not np.isfinite(proj2).all() or np.isclose(proj1[2], 0.0) or np.isclose(proj2[2], 0.0):
        return float('inf'), float('inf')
    u1 = proj1[0] / proj1[2]
    v1 = proj1[1] / proj1[2]
    u2 = proj2[0] / proj2[2]
    v2 = proj2[1] / proj2[2]
    e1 = float(np.hypot(u1 - float(p1_px[0]), v1 - float(p1_px[1])))
    e2 = float(np.hypot(u2 - float(p2_px[0]), v2 - float(p2_px[1])))
    return e1, e2


def to_meters_from_homography(x: float, y: float, homography: np.ndarray) -> tuple[float, float]:
    point = np.array([x, y, 1], dtype=np.float32)
    result = homography @ point
    if not np.isfinite(result).all() or np.isclose(result[2], 0.0):
        return (float('nan'), float('nan'))
    return float(result[0] / result[2]), float(result[1] / result[2])


def extract_ball_xyz(frame: dict[str, Any]) -> list[float] | None:
    if frame.get("ball_3d") is not None:
        return [float(v) for v in frame["ball_3d"]]
    if frame.get("ball") is not None:
        return [float(v) for v in frame["ball"]]
    ball_metrics = frame.get("ball_metrics")
    if isinstance(ball_metrics, dict) and ball_metrics.get("pos") is not None:
        return [float(v) for v in ball_metrics["pos"]]
    return None


def extract_ball_speed(frame: dict[str, Any]) -> float | None:
    ball_metrics = frame.get("ball_metrics")
    if isinstance(ball_metrics, dict) and ball_metrics.get("speed_kmh") is not None:
        return float(ball_metrics["speed_kmh"])
    return None


def extract_player_position(player: dict[str, Any]) -> tuple[float, float] | None:
    if player.get("pos_m") is not None:
        return float(player["pos_m"][0]), float(player["pos_m"][1])
    if player.get("pos") is not None:
        pos = player["pos"]
        if isinstance(pos, (list, tuple)) and len(pos) >= 2:
            return float(pos[0]), float(pos[1])
    if player.get("pose_2d") is not None:
        pose = np.asarray(player["pose_2d"], dtype=np.float32)
        if pose.shape[0] >= 17:
            # Prefer ankles when available; otherwise fall back to the two lowest visible keypoints.
            candidates = []
            for idx in (15, 16):
                if idx < pose.shape[0]:
                    candidates.append(pose[idx])

            if len(candidates) >= 2 and all(np.isfinite(c).all() and c[0] > 0 and c[1] > 0 for c in candidates):
                foot_x = float((candidates[0][0] + candidates[1][0]) / 2)
                foot_y = float((candidates[0][1] + candidates[1][1]) / 2)
            else:
                # Use the lowest points in the image as a fallback; this is more stable than the centroid.
                finite = pose[np.isfinite(pose).all(axis=1)]
                if finite.shape[0] == 0:
                    return None
                lowest = finite[np.argsort(finite[:, 1])[-2:]] if finite.shape[0] >= 2 else finite
                foot_x = float(np.mean(lowest[:, 0]))
                foot_y = float(np.mean(lowest[:, 1]))

            _, _, H_MAIN = get_geometry_matrices()
            pos_m = to_meters_from_homography(foot_x, foot_y, H_MAIN)
            if not np.isfinite(pos_m[0]) or not np.isfinite(pos_m[1]):
                return None
            # Reject extreme outliers instead of propagating nonsense into the final report.
            if pos_m[0] < -1.0 or pos_m[0] > FIELD_WIDTH_M + 1.0 or pos_m[1] < -1.0 or pos_m[1] > FIELD_LENGTH_M + 1.0:
                return None
            return pos_m
    return None


def normalize_pose_for_action(pose_2d: np.ndarray) -> np.ndarray:
    arr = np.asarray(pose_2d, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 2:
        return np.asarray([], dtype=np.float32)
    root = arr[0, :2]
    normalized = arr[:, :2] - root
    return normalized.reshape(-1)


def run_detection_stage(
    video_main_path: str,
    video_second_path: str,
    ball_model_path: str,
    pose_model_path: str,
    max_frames: int,
    ball_conf: float,
    imgsz: int,
) -> list[dict[str, Any]]:
    ball_model = load_ball_model(ball_model_path)
    pose_model = load_pose_model(pose_model_path)

    cap_main = cv2.VideoCapture(video_main_path)
    cap_second = cv2.VideoCapture(video_second_path)
    results: list[dict[str, Any]] = []

    for frame_idx in range(max_frames):
        ret_main, img_main = cap_main.read()
        ret_second, img_second = cap_second.read()
        if not ret_main or not ret_second:
            break

        frame_data: dict[str, Any] = {"frame": frame_idx, "ball_3d": None, "players": []}

        ball_res_1 = ball_model.predict(img_main, conf=ball_conf, imgsz=imgsz, verbose=False)[0]
        ball_res_2 = ball_model.predict(img_second, conf=ball_conf, imgsz=imgsz, verbose=False)[0]

        if len(ball_res_1.boxes) > 0 and len(ball_res_2.boxes) > 0:
            p1 = ball_res_1.boxes.xywh.cpu().numpy()[0][:2]
            p2 = ball_res_2.boxes.xywh.cpu().numpy()[0][:2]
            try:
                ball_3d = triangulate_point(p1, p2)
            except Exception:
                ball_3d = np.array([np.nan, np.nan, np.nan], dtype=np.float32)

            # validate by reprojection error for logging, but keep a finite triangulation
            # so the refinement stage still has data to smooth/interpolate.
            accept = np.isfinite(ball_3d).all() and 0 <= float(ball_3d[2]) <= 15
            e1 = e2 = float('nan')
            if accept:
                try:
                    e1, e2 = reprojection_error_2view(ball_3d, p1, p2)
                except Exception:
                    e1 = e2 = float('nan')

            if accept:
                frame_data["ball_3d"] = ball_3d.tolist()
            else:
                # leave as None to be interpolated by refinement stage
                frame_data["ball_3d"] = None
                # record details for debugging
                try:
                    GEOMETRY_ISSUES.append({
                        "type": "triangulation_invalid",
                        "frame": int(frame_idx),
                        "p1": [float(p1[0]), float(p1[1])],
                        "p2": [float(p2[0]), float(p2[1])],
                        "triangulated": [None if not np.isfinite(x) else float(x) for x in ball_3d.tolist()],
                        "reproj_error": [None if not np.isfinite(ball_3d).all() else float(e1), None if not np.isfinite(ball_3d).all() else float(e2)],
                    })
                except Exception:
                    pass

        pose_res = pose_model.track(img_main, persist=True, imgsz=imgsz, verbose=False)[0]
        if pose_res.boxes.id is not None and pose_res.keypoints is not None:
            ids = pose_res.boxes.id.int().cpu().tolist()
            poses = pose_res.keypoints.xy.cpu().numpy()
            for player_id, pose in zip(ids, poses):
                frame_data["players"].append({
                    "id": int(player_id),
                    "pose_2d": pose.tolist(),
                    "action": "A aguardar...",
                })

        results.append(frame_data)

    cap_main.release()
    cap_second.release()
    # attempt to persist geometry issues to a temp file for later inspection
    try:
        tmp = Path(tempfile.gettempdir()) / "padel_geometry_issues.json"
        save_geometry_issues(tmp)
    except Exception:
        pass
    return results


def refine_ball_stage(raw_data: list[dict[str, Any]], max_speed_kmh: float, window: int) -> list[dict[str, Any]]:
    issue_lookup: dict[int, list[float]] = {}
    for issue in GEOMETRY_ISSUES:
        if issue.get("type") != "triangulation_invalid":
            continue
        frame_idx = issue.get("frame")
        triangulated = issue.get("triangulated")
        if frame_idx is None or not isinstance(triangulated, list) or len(triangulated) < 3:
            continue
        if all(v is not None and np.isfinite(v) for v in triangulated[:3]):
            issue_lookup[int(frame_idx)] = [float(triangulated[0]), float(triangulated[1]), float(triangulated[2])]

    # also recover issues persisted to temp from a previous detection run
    try:
        tmp_issues = Path(tempfile.gettempdir()) / "padel_geometry_issues.json"
        if tmp_issues.exists():
            for issue in json.loads(tmp_issues.read_text(encoding="utf-8")):
                if issue.get("type") != "triangulation_invalid":
                    continue
                frame_idx = issue.get("frame")
                triangulated = issue.get("triangulated")
                if frame_idx is None or not isinstance(triangulated, list) or len(triangulated) < 3:
                    continue
                if all(v is not None and np.isfinite(v) for v in triangulated[:3]):
                    issue_lookup.setdefault(int(frame_idx), [float(triangulated[0]), float(triangulated[1]), float(triangulated[2])])
    except Exception:
        pass

    ball_coords = []
    for frame in raw_data:
        ball = extract_ball_xyz(frame)
        if ball is None:
            ball = issue_lookup.get(int(frame.get("frame", -1)))
        ball_coords.append([np.nan, np.nan, np.nan] if ball is None else ball)

    df_ball = pd.DataFrame(ball_coords, columns=["x", "y", "z"])
    # In the current calibration, ball Y is often returned with the opposite sign / origin.
    # Shift it into the court frame before applying bounds filters if it is consistently negative.
    finite_y = df_ball["y"].replace([np.inf, -np.inf], np.nan).dropna()
    if not finite_y.empty and float(finite_y.median()) < 0:
        # Move the median into the court range instead of dropping the trajectory.
        y_shift = FIELD_LENGTH_M - float(finite_y.median())
        df_ball["y"] = df_ball["y"] + y_shift

    df_ball.loc[(df_ball.x < -1) | (df_ball.x > FIELD_WIDTH_M + 1), "x"] = np.nan
    # Keep a usable trajectory even if some recovered Y values fall slightly outside the court.
    df_ball["y"] = df_ball["y"].clip(lower=0.0, upper=FIELD_LENGTH_M)
    df_ball.loc[(df_ball.z < 0) | (df_ball.z > 7), "z"] = np.nan

    fps = 50
    max_dist_per_frame = (max_speed_kmh / 3.6) / fps
    for idx in range(1, len(df_ball)):
        prev = df_ball.iloc[idx - 1][["x", "y", "z"]].values
        curr = df_ball.iloc[idx][["x", "y", "z"]].values
        if not np.isnan(prev).any() and not np.isnan(curr).any() and np.linalg.norm(curr - prev) > max_dist_per_frame:
            df_ball.iloc[idx] = [np.nan, np.nan, np.nan]

    df_ball = df_ball.interpolate(method="linear").ffill().bfill()
    df_ball = df_ball.rolling(window=window, center=True, min_periods=1).mean()

    # Post-process: remove implausible instantaneous speeds caused by interpolation over gaps.
    fps = 50
    cap_kmh = max_speed_kmh
    # compute instantaneous speed per frame (based on smoothed positions)
    diffs = np.sqrt(df_ball["x"].diff().fillna(0) ** 2 + df_ball["y"].diff().fillna(0) ** 2 + df_ball["z"].diff().fillna(0) ** 2)
    speeds = diffs * fps * 3.6
    spike_mask = speeds > cap_kmh
    if spike_mask.any():
        df_ball.loc[spike_mask, ["x", "y", "z"]] = np.nan
        # re-interpolate + smooth after removing spikes
        df_ball = df_ball.interpolate(method="linear").ffill().bfill()
        df_ball = df_ball.rolling(window=window, center=True, min_periods=1).mean()

    all_ids = [player["id"] for frame in raw_data for player in frame["players"]]
    top_ids = [item[0] for item in pd.Series(all_ids).value_counts().head(4).items()]

    refined: list[dict[str, Any]] = []
    for idx, frame in enumerate(raw_data):
        frame_obj = {
            "frame": int(frame["frame"]),
            "ball": [round(float(df_ball.iloc[idx].x), 3), round(float(df_ball.iloc[idx].y), 3), round(float(df_ball.iloc[idx].z), 3)],
            "players": [],
        }
        for player_id in top_ids:
            player_pos = [np.nan, np.nan]
            # preserve pose_2d where available; prefer explicit pos/pos_m
            pose_2d = None
            for player in frame["players"]:
                if player["id"] == player_id:
                    pose_2d = player.get("pose_2d")
                    # compute position (prefer existing pos/pos_m if present)
                    position = None
                    if player.get("pos_m") is not None:
                        posm = player.get("pos_m")
                        if isinstance(posm, (list, tuple)) and len(posm) >= 2:
                            position = (float(posm[0]), float(posm[1]))
                    elif player.get("pos") is not None:
                        pos = player.get("pos")
                        if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                            position = (float(pos[0]), float(pos[1]))
                    else:
                        position = extract_player_position(player)
                    if position is not None and not (np.isnan(position[0]) or np.isnan(position[1])):
                        player_pos = [round(float(position[0]), 3), round(float(position[1]), 3)]
                    else:
                        # record homography issue for debugging
                        try:
                            GEOMETRY_ISSUES.append({
                                "type": "homography_invalid",
                                "frame": int(frame["frame"]),
                                "player_id": int(player_id),
                                "pose_2d_sample": player.get("pose_2d")[:2] if player.get("pose_2d") is not None else None,
                                "mapped": [None if position is None or not np.isfinite(position[0]) else float(position[0]), None if position is None or not np.isfinite(position[1]) else float(position[1])],
                            })
                        except Exception:
                            pass
                    break
            frame_obj["players"].append({"id": int(player_id), "pos": player_pos, "pose_2d": pose_2d})
        refined.append(frame_obj)

    return refined


def analyze_stage(clean_data: list[dict[str, Any]], action_model_path: str, threshold: float, fps: int) -> list[dict[str, Any]]:
    action_model = load_action_model(action_model_path)

    all_ids = [player["id"] for frame in clean_data for player in frame["players"]]
    top_ids = [item[0] for item in pd.Series(all_ids).value_counts().head(4).items()]
    buffers = {player_id: [] for player_id in top_ids}

    final_report: list[dict[str, Any]] = []
    for idx, frame in enumerate(clean_data):
        ball = np.asarray(frame["ball"], dtype=np.float32)
        ball_prev = np.asarray(clean_data[max(idx - 1, 0)]["ball"], dtype=np.float32)
        speed = float(np.linalg.norm(ball - ball_prev) * fps * 3.6)
        # Keep the final report physically plausible; extreme spikes are usually
        # interpolation or tracking artifacts rather than real padel shots.
        if not np.isfinite(speed):
            speed = float("nan")
        elif speed > 160.0:
            speed = 160.0

        current = {
            "frame": int(frame["frame"]),
            "ball_metrics": {
                "pos": [float(ball[0]), float(ball[1]), float(ball[2])],
                "speed_kmh": speed,
                "is_out": bool(ball[0] < 0 or ball[0] > FIELD_WIDTH_M or ball[1] < 0 or ball[1] > FIELD_LENGTH_M),
            },
            "players": [],
        }

        for player in frame["players"]:
            if player["id"] not in top_ids:
                continue

            # prefer explicit metric positions produced by the refinement stage
            position = None
            if player.get("pos_m") is not None:
                posm = player.get("pos_m")
                if isinstance(posm, (list, tuple)) and len(posm) >= 2:
                    position = (float(posm[0]), float(posm[1]))
            elif player.get("pos") is not None:
                pos = player.get("pos")
                if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                    position = (float(pos[0]), float(pos[1]))

            pose_data = player.get("pose_2d")
            pose = np.asarray(pose_data, dtype=np.float32) if pose_data is not None else np.empty((0, 2), dtype=np.float32)
            if pose.ndim != 2 or pose.shape[0] == 0 or pose.shape[1] < 2:
                pose = np.empty((0, 2), dtype=np.float32)

            if position is None and pose.size > 0:
                position = extract_player_position(player)
            if position is None:
                position = (np.nan, np.nan)

            action = "A aguardar..."
            confidence = 0.0
            if pose.size > 0:
                buffers[player["id"]].append(normalize_pose_for_action(pose))
                if len(buffers[player["id"]]) > 30:
                    buffers[player["id"]].pop(0)
                if len(buffers[player["id"]]) == 30:
                    prediction = action_model.predict(np.array([buffers[player["id"]]]), verbose=0)[0]
                    if float(np.max(prediction)) >= threshold:
                        action = ACTION_CLASSES[int(np.argmax(prediction))]
                        confidence = float(np.max(prediction))
                    else:
                        action = "Neutro"

            current["players"].append({
                "id": int(player["id"]),
                "pos_m": [round(float(position[0]), 2), round(float(position[1]), 2)],
                "action": action,
                "confidence": confidence,
            })

        final_report.append(current)

    return final_report


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=4, ensure_ascii=False)


def ball_metrics_from_any_schema(data: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for frame in data:
        ball = extract_ball_xyz(frame)
        if ball is None:
            continue
        speed = extract_ball_speed(frame)
        rows.append({
            "frame": int(frame.get("frame", len(rows))),
            "x": float(ball[0]),
            "y": float(ball[1]),
            "z": float(ball[2]),
            "speed_kmh": float(speed) if speed is not None else np.nan,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if "speed_kmh" not in df or df["speed_kmh"].isna().all():
        dist = np.sqrt(df[["x", "y", "z"]].diff().pow(2).sum(axis=1))
        df["speed_kmh"] = dist * 50 * 3.6
    return df


def court_figure(clean_data: list[dict[str, Any]]) -> plt.Figure:
    df = ball_metrics_from_any_schema(clean_data)
    fig, ax = plt.subplots(figsize=(7, 10))
    ax.plot([0, FIELD_WIDTH_M, FIELD_WIDTH_M, 0, 0], [0, 0, FIELD_LENGTH_M, FIELD_LENGTH_M, 0], color="black", lw=2)
    ax.axhline(FIELD_LENGTH_M / 2, color="red", ls="--", alpha=0.4)

    if not df.empty:
        ax.plot(df["x"], df["y"], color="#0066cc", lw=2, label="Bola")

    for frame in clean_data:
        for player in frame.get("players", []):
            pos = extract_player_position(player)
            if pos is not None:
                ax.scatter(pos[0], pos[1], c="#2a9d8f" if pos[1] < FIELD_LENGTH_M / 2 else "#264653", s=40)

    ax.set_xlim(-1, FIELD_WIDTH_M + 1)
    ax.set_ylim(-1, FIELD_LENGTH_M + 1)
    ax.set_aspect("equal")
    ax.set_title("Radar tático")
    ax.set_xlabel("Largura (m)")
    ax.set_ylabel("Comprimento (m)")
    ax.grid(alpha=0.15)
    return fig


def preview_image(uploaded_file) -> Image.Image | None:
    if uploaded_file is None:
        return None
    try:
        return Image.open(uploaded_file)
    except Exception:
        return None


def persist_uploaded_video(uploaded_file, suffix: str) -> str:
    if uploaded_file is None:
        return ""
    temp_dir = Path(tempfile.mkdtemp(prefix="padel_dashboard_"))
    target = temp_dir / f"{uploaded_file.name.rsplit('.', 1)[0]}{suffix}"
    with target.open("wb") as handle:
        handle.write(uploaded_file.getbuffer())
    return str(target)