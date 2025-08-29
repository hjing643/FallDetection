# pip install torch torchvision pytorchvideo opencv-python
import cv2, torch, numpy as np
from collections import deque
from torchvision.transforms import functional as F
import argparse, os

# -------- Kinetics-400 归一化参数 --------
MEAN = [0.45, 0.45, 0.45]
STD  = [0.225, 0.225, 0.225]

def pack_slowfast(frames_tensor, alpha=4):
    """
    frames_tensor: (T, H, W, C) in [0,1] float32
    返回 SlowFast 两条路径的输入: [slow, fast]
    slow: (B,C,T/alpha,H,W), fast: (B,C,T,H,W)
    """
    fast = frames_tensor.permute(3,0,1,2).unsqueeze(0)  # (1,C,T,H,W)
    idx = torch.linspace(0, fast.shape[2]-1, max(1, fast.shape[2]//alpha)).long()
    slow = fast.index_select(2, idx)
    return [slow, fast]

def preprocess_clip(frames_rgb, size=224):
    """
    frames_rgb: list[np.ndarray(H,W,3) RGB 0..255]
    输出 clip: (T, size, size, 3) 0..1 normalized
    """
    tensors = []
    for img in frames_rgb:
        t = torch.from_numpy(img).float() / 255.0  # (H,W,3)
        t = t.permute(2,0,1)                       # (3,H,W)
        # 短边缩放到256，中心裁剪到 size×size
        h, w = t.shape[1], t.shape[2]
        short = min(h, w)
        scale = 256 / short
        t = F.resize(t, [int(h*scale), int(w*scale)])
        t = F.center_crop(t, [size, size])
        # 归一化
        for c in range(3):
            t[c] = (t[c] - MEAN[c]) / STD[c]
        t = t.permute(1,2,0)  # (size,size,3)
        tensors.append(t)
    clip = torch.stack(tensors, dim=0)  # (T,size,size,3)
    return clip

def load_labels(label_map_path):
    if label_map_path and os.path.isfile(label_map_path):
        with open(label_map_path, "r", encoding="utf-8") as f:
            return [l.strip() for l in f.readlines()]
    return None

def draw_text_panel(frame, lines, top_left=(10,10), alpha=0.5):
    """
    在左上角画半透明信息面板并写入多行文本
    """
    x, y = top_left
    h = 22
    pad = 8
    width = max(300, max([cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0][0] for s in lines] + [0]) + 2*pad)
    height = h*len(lines) + 2*pad
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x+width, y+height), (0,0,0), -1)
    cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)
    yy = y + pad + 16
    for s in lines:
        cv2.putText(frame, s, (x+pad, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
        yy += h

def main(args):
    # 线程数/设备
    torch.set_num_threads(max(1, args.threads))
    device = "cpu"  # CPU 跑 SlowFast；若有 GPU 可改 "cuda"

    # 模型
    model = torch.hub.load("facebookresearch/pytorchvideo", "slowfast_r50", pretrained=True)
    model = model.eval().to(device)

    # 标签（可选）
    labels = load_labels(args.labels)

    # 视频读写
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{args.source}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, fps, (W, H))

    # 滑窗缓冲：存最近 T*stride 帧（原始 RGB）
    raw_buf = deque(maxlen=args.clip_len * args.frame_stride)
    # 平滑：存最近 N 次的概率向量
    prob_buf = deque(maxlen=args.smooth_win)

    frame_idx = 0
    roi = None
    if args.roi and len(args.roi) == 4:
        x,y,w,h = args.roi
        # 裁剪范围合法化
        x = max(0, min(x, W-1)); y = max(0, min(y, H-1))
        w = max(1, min(w, W-x)); h = max(1, min(h, H-y))
        roi = (x,y,w,h)

    with torch.no_grad():
        while True:
            ret, bgr = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            # 可选：只在 ROI 做分类，然后把标签叠加回整帧
            vis_frame = bgr.copy()
            crop_rgb = rgb
            if roi is not None:
                x,y,w,h = roi
                crop_rgb = rgb[y:y+h, x:x+w].copy()
                # 画 ROI 边框
                cv2.rectangle(vis_frame, (x,y), (x+w, y+h), (0,255,255), 2)

            raw_buf.append(crop_rgb)

            # 当缓冲足够时做一次推理（每 infer_every 帧做一次）
            do_infer = (len(raw_buf) == raw_buf.maxlen) and (frame_idx % args.infer_every == 0)

            top_lines = []
            if do_infer:
                # 采样出 T 帧
                sampled = list(raw_buf)[::args.frame_stride][-args.clip_len:]  # 长度=T
                clip = preprocess_clip(sampled, size=args.resize)
                inputs = pack_slowfast(clip.to(device), alpha=args.alpha)
                logits = model(inputs)
                probs = torch.softmax(logits, dim=1)[0].cpu()
                prob_buf.append(probs)

            # 平滑后的概率
            if len(prob_buf) > 0:
                probs_mean = torch.stack(list(prob_buf), dim=0).mean(dim=0)
                topk = torch.topk(probs_mean, k=min(5, probs_mean.numel()))
                idxs = topk.indices.numpy().tolist()
                scores = topk.values.numpy().tolist()
                if labels:
                    names = [labels[i] for i in idxs]
                else:
                    names = [f"class_{i}" for i in idxs]

                # Top-1 行 + Top-5 列表
                top1 = f"Top1: {names[0]}  {scores[0]*100:.1f}%"
                top_lines.append(top1)
                for n, s in zip(names[1:], scores[1:]):
                    top_lines.append(f"- {n}  {s*100:.1f}%")

            # 叠加到视频帧并写出
            if top_lines:
                draw_text_panel(vis_frame, top_lines, top_left=(10,10), alpha=0.5)
            writer.write(vis_frame)

            frame_idx += 1

    cap.release()
    writer.release()
    print(f"✅ 已保存带叠加文字的结果视频：{args.out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("SlowFast 推理并叠加到视频输出")
    parser.add_argument("--source", type=str, default="./FallDetection/fileDepends/Caida_Elcasar (1).MP4", help="输入视频路径")
    parser.add_argument("--out", type=str, default="./FallDetection/fileDepends/Caida_Elcasar (1)_slowfast_output.mp4", help="输出视频路径")
    parser.add_argument("--labels", type=str, default=None, help="Kinetics-400 标签文件（可选）")
    parser.add_argument("--threads", type=int, default=8, help="CPU 线程数")
    # 模型输入相关
    parser.add_argument("--clip_len", type=int, default=32, help="每次推理的帧数 T")
    parser.add_argument("--frame_stride", type=int, default=2, help="缓冲内采样步长（原始帧间隔）")
    parser.add_argument("--infer_every", type=int, default=4, help="每隔多少帧触发一次推理（越大越省算力）")
    parser.add_argument("--resize", type=int, default=224, help="中心裁剪尺寸")
    parser.add_argument("--alpha", type=int, default=4, help="SlowFast 慢支时间降采样比")
    # ROI: x y w h（可选）
    parser.add_argument("--roi", type=int, nargs=4, default=None, help="只在 ROI 做分类，格式: x y w h")
    args = parser.parse_args()
    main(args)
