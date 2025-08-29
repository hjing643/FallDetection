#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Headless fall detection demo (YOLOv8-Pose + OpenCV)
- 输入: 文件路径 或 RTSP URL
- 输出: 事件 JSON (fall_events.json)，可选保存标注视频 (--save-video)
依赖: ultralytics, opencv-python-headless, numpy
"""

import argparse, json, math, time, os
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

# ---------- 参数与默认阈值 ----------
DEF_MIN_FALLEN_SEC = 1.0      # Fallen状态持续时间阈值(秒)
DEF_RATIO_THR      = 0.85     # 人框高宽比阈值: H/W < 0.6 -> 更像躺
DEF_AXIS_THR_DEG   = 35.0     # 身体主轴与水平夹角阈值(度): 越小越平躺
DEF_DHDT_THR       = -0.25    # 骨盆相对高度(0~1)每秒下降速率阈值(负值)
SUPPRESS_SEC       = 8.0      # 同一ID告警抑制窗口

# ---------- 工具函数 ----------
def pca_aspect_ratio(kpts):
    """
    用关键点的 PCA 计算“短轴/长轴”比(0~1)，与旋转无关。
    躺/趴时通常更“扁”(值更小)；站立时更“瘦”(值更大)。
    """
    if kpts is None: return None
    vis = kpts[kpts[:,2] > 0.25, :2]#get all the visible keypoints, score > 0.25
    if vis.shape[0] < 4:
        return None
    X = vis - vis.mean(axis=0, keepdims=True)
    cov = np.cov(X.T)
    w, v = np.linalg.eigh(cov)
    idx = np.argsort(w)         # [小, 大]
    v_min, v_max = v[:, idx[0]], v[:, idx[1]]
    s_min = (X @ v_min)
    s_max = (X @ v_max)
    minor = s_min.max() - s_min.min()
    major = s_max.max() - s_max.min()
    return float(minor / (major + 1e-6))  # 越小越“扁”

def body_axis_angle_deg(kpts):
    """
    身体主轴与水平夹角(度)，0≈水平(更像躺)，90≈竖直(更像站)。
    先用可见关键点的 PCA，缺失时退化肩-胯。
    """
    if kpts is None or len(kpts) == 0:
        return None
    vis = kpts[kpts[:,2] > 0.25, :2]
    if vis.shape[0] >= 4:
        x = vis[:,0] - vis[:,0].mean()
        y = vis[:,1] - vis[:,1].mean()
        cov = np.cov(np.vstack([x, y]))
        w, v = np.linalg.eigh(cov)
        vx, vy = v[:, np.argmax(w)]
        ang = abs(math.degrees(math.atan2(vy, vx)))   # 0~180
        if ang > 90: ang = 180 - ang                  # 折叠到 0~90
        return ang
    if kpts.shape[0] >= 13:
        ls, rs = kpts[5], kpts[6]
        lh, rh = kpts[11], kpts[12]
        if min(ls[2], rs[2], lh[2], rh[2]) >= 0.20:
            c_sh = ((ls[0]+rs[0])/2, (ls[1]+rs[1])/2)
            c_hp = ((lh[0]+rh[0])/2, (lh[1]+rh[1])/2)
            dx, dy = c_sh[0]-c_hp[0], c_sh[1]-c_hp[1]
            ang = abs(math.degrees(math.atan2(dy, dx)))
            if ang > 90: ang = 180 - ang
            return ang
    return None

def norm_pelvis_height(kpts, img_h):
    """骨盆相对高度: 0=顶端, 1=底部"""
    if kpts.shape[0] < 13:
        return None
    lh, rh = kpts[11], kpts[12]
    if min(lh[2], rh[2]) < 0.20:
        return None
    y = (lh[1] + rh[1]) / 2.0
    return float(np.clip(y / (img_h + 1e-6), 0.0, 1.0))

# ---------- 简易多目标“跟踪ID” ----------
# 为了无依赖，使用基于IOU的简易关联；生产可换成 OC-SORT/DeepSORT
class IdTracker:
    def __init__(self, iou_thr=0.3, keep_sec=1.0):
        self.next_id = 1
        self.tracks = {}  # id -> {box, ts_last}
        self.iou_thr = iou_thr
        self.keep_sec = keep_sec

    @staticmethod
    def iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
        inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
        inter = iw * ih
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter + 1e-6
        return inter / union

    def update(self, boxes, now_ts):
        assigned = {}
        # 清理过期track
        self.tracks = {tid: t for tid, t in self.tracks.items() if now_ts - t["ts_last"] <= self.keep_sec}

        for bi, box in enumerate(boxes):
            # 选最佳IOU
            best_id, best_iou = None, 0.0
            for tid, t in self.tracks.items():
                iouv = self.iou(box, t["box"])
                if iouv > best_iou:
                    best_iou, best_id = iouv, tid
            if best_iou >= self.iou_thr:
                assigned[bi] = best_id
                self.tracks[best_id] = {"box": box, "ts_last": now_ts}
            else:
                tid = self.next_id
                self.next_id += 1
                assigned[bi] = tid
                self.tracks[tid] = {"box": box, "ts_last": now_ts}
        return assigned

# ---------- 跌倒状态机 ----------
class FallJudge:
    def __init__(self, min_fallen_sec=DEF_MIN_FALLEN_SEC, ratio_thr=DEF_RATIO_THR,
                 axis_thr_deg=DEF_AXIS_THR_DEG, dhdt_thr=DEF_DHDT_THR, suppress_sec=SUPPRESS_SEC):
        self.min_fallen_sec = min_fallen_sec
        self.ratio_thr = ratio_thr
        self.axis_thr_deg = axis_thr_deg
        self.dhdt_thr = dhdt_thr
        self.suppress_sec = suppress_sec
        self.mem = {}  # id -> state dict

    def update(self, pid, ratio, axis_deg, pelvis_h, dt, now_ts):
        st = self.mem.setdefault(pid, {"state":"Standing","t_fall":0.0,"fallen_dur":0.0,"last_h":None,"last_alert":0.0})
        v = 0.0
        if pelvis_h is not None and st["last_h"] is not None and dt > 0:
            v = (pelvis_h - st["last_h"]) / dt  # 下降为正
        if pelvis_h is not None:
            st["last_h"] = pelvis_h

        low_posture = (ratio is not None and ratio < self.ratio_thr) or (axis_deg is not None and axis_deg < self.axis_thr_deg)
        rapid_down = v < self.dhdt_thr

        if st["state"] == "Standing":
            if low_posture and rapid_down:#如果之前是站立，现在低姿态且快速下降，则认为正在跌倒
                st["state"] = "Falling"
                st["t_fall"] = 0.0
        elif st["state"] == "Falling":
            st["t_fall"] += dt
            if low_posture and st["t_fall"] > 0.4:#如果之前是跌倒，现在低姿态且持续时间超过0.4秒，则认为已经跌倒
                st["state"] = "Fallen"
                st["fallen_dur"] = 0.0
            elif st["t_fall"] > 1.2:#如果之前是跌倒，现在持续时间超过1.2秒，则认为已经站起来了
                st["state"] = "Standing"
        elif st["state"] == "Fallen":
            st["fallen_dur"] += dt
            if not low_posture:#如果之前是跌倒，现在不是低姿态，则认为已经站起来了
                st["state"] = "Standing"
                st["fallen_dur"] = 0.0

        alert = False
        if st["state"] == "Fallen" and st["fallen_dur"] >= self.min_fallen_sec:
            if now_ts - st["last_alert"] >= self.suppress_sec:
                alert = True
                st["last_alert"] = now_ts
        return st["state"], alert, v

# ---------- 主流程 ----------
def main():
    ap = argparse.ArgumentParser(description="Headless Fall Detection (YOLOv8-Pose + OpenCV)")
    #ap.add_argument("--source", required=True, help="视频文件路径或 RTSP URL")
    ap.add_argument("--model", default="yolov8n-pose.pt", help="Ultralytics pose 模型权重")
    ap.add_argument("--save-video", default="./FallDetection/fileDepends/Caida_Elcasar (1)_pos_output.mp4", help="保存标注视频到此路径(留空则不保存)")
    ap.add_argument("--events-json", default="fall_events.json", help="事件输出JSON文件")
    ap.add_argument("--ratio-thr", type=float, default=DEF_RATIO_THR)
    ap.add_argument("--axis-thr-deg", type=float, default=DEF_AXIS_THR_DEG)
    ap.add_argument("--dhdt-thr", type=float, default=DEF_DHDT_THR)
    ap.add_argument("--min-fallen-sec", type=float, default=DEF_MIN_FALLEN_SEC)
    ap.add_argument("--cpu", action="store_true", help="强制使用CPU")
    args = ap.parse_args()

    # 模型
    #det_person = YOLO("yolov8n.pt")  # 只要 person 类
    pos_model = YOLO(args.model)
    if not args.cpu and hasattr(pos_model, "to"):
        try:
            import torch
            if torch.cuda.is_available():
                pos_model.to("cuda")
        except Exception:
            pass

    # 输入
    file_path = "./FallDetection/fileDepends/Caida_Elcasar (1).MP4"

    f1 = open(file_path, "r")
    f1.close()

    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开输入: {file_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 输出视频（可选）
    writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save_video, fourcc, fps, (w, h))
        if not writer.isOpened():
            raise RuntimeError(f"无法创建输出视频: {args.save_video}")

    tracker = IdTracker(iou_thr=0.3, keep_sec=1.0)
    judge = FallJudge(min_fallen_sec=args.min_fallen_sec,
                      ratio_thr=DEF_RATIO_THR,
                      axis_thr_deg=DEF_AXIS_THR_DEG,
                      dhdt_thr=DEF_DHDT_THR)

    events = []
    last_ts = time.time()
    frame_id = 0

    # 确保事件目录存在
    Path(os.path.dirname(args.events_json) or ".").mkdir(parents=True, exist_ok=True)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        now_ts = time.time()
        dt = max(1.0/fps, now_ts - last_ts)
        last_ts = now_ts

        # 推理（关闭verbose以便服务端打印简洁）
        pos_res = pos_model(frame, conf=0.01, iou=0.85, imgsz=1280, classes=[0], max_det=100, verbose=False)[0]
        pos_boxes = pos_res.boxes.xyxy.cpu().numpy() if pos_res.boxes is not None else np.empty((0,4))
        kpts_all = pos_res.keypoints.data.cpu().numpy() if pos_res.keypoints is not None else np.empty((0,17,3))

        # 关联ID
        assigned = tracker.update(pos_boxes, now_ts)

        # 遍历检测到的人
        for i in range(len(pos_boxes)):
            box = pos_boxes[i]
            kpts = kpts_all[i] if i < len(kpts_all) else np.zeros((17,3), dtype=np.float32)
            pid = assigned.get(i, -1)

            x1, y1, x2, y2 = box
            bw, bh = max(1.0, x2-x1), max(1.0, y2-y1)
            ratio_pca = pca_aspect_ratio(kpts)  # 0~1，越小越躺
            ratio_box = bh / bw
            ratio = ratio_pca if ratio_pca is not None else ratio_box
            ang   = body_axis_angle_deg(kpts)
            ph    = norm_pelvis_height(kpts, h)

            state, alert, v = judge.update(pid, ratio, ang, ph, dt, now_ts)

            # 写标注视频（可选）
            if writer is not None:
                color = (0,0,255) if state=="Fallen" or state=="Falling" else (0,255,0)
                cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), color, 2)
                pid = assigned.get(i, -1)
                label = f"id={pid} {state} r={ratio:.2f} rbox={ratio_box:.2f} box_conf={pos_res.boxes.conf[i]:.2f} ang={-1 if ang is None else ang:.1f} v={v:.2f}"
                cv2.putText(frame, label, (int(x1), max(0,int(y1)-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                # 画关键点(简易)
                for (kx,ky,kc) in kpts:
                    if kc > 0.00:
                        cv2.circle(frame, (int(kx), int(ky)), 2, (255,255,255), -1)

            if alert:
                evt = {
                    "ts": now_ts,
                    "frame": frame_id,
                    "id": int(pid),
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "ratio": float(ratio),
                    "axis_deg": None if ang is None else float(ang),
                    "pelvis_h": None if ph is None else float(ph),
                    "state": state
                }
                events.append(evt)
                # 也可边跑边写入行式JSON，避免长任务丢失
                # print(json.dumps({"event":"fall", **evt}, ensure_ascii=False))

        if writer is not None:
            writer.write(frame)
        frame_id += 1

    cap.release()
    if writer is not None:
        writer.release()

    with open(args.events_json, "w", encoding="utf-8") as f:
        json.dump(events, f, ensure_ascii=False, indent=2)

    print(f"[done] events={len(events)} saved to {args.events_json}")
    if args.save_video:
        print(f"[done] annotated video saved to {args.save_video}")

if __name__ == "__main__":
    print("current work space:", os.getcwd())
    main()
