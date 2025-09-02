from ultralytics import YOLO
import cv2, numpy as np
import os, glob

def pca_minor_major_from_mask(binmask: np.ndarray):
    ys, xs = np.where(binmask > 0)
    if xs.size < 200:
        return None, None #如果小于20，肯定画不出轮廓，没有这么小的人像素，其实一般是几万个像素
    X = np.stack([xs, ys], axis=1).astype(np.float32)
    X -= X.mean(axis=0, keepdims=True) #去中心化，计算协方差矩阵。得到的值是2*x的矩阵，x越大代表像素越多。
    cov = np.cov(X.T) #协方差矩阵，计算后用于判断是水平还是竖直，比例。
    w, v = np.linalg.eigh(cov)        #特征值w，特征向量V， w[0]是较小的特征值，w[1]是较大的特征值，方向由v决定。
    if w[1] < 1e-6:
        return None, None
    minor, major = np.sqrt(w[0]), np.sqrt(w[1])
    ratio = float(minor / (major + 1e-6))     # (0,1] 越小越扁，横轴除以纵轴。

    # 主轴方向与水平的夹角（标准化到 [0,90]）
    vx, vy = float(v[0,1]), float(v[1,1])     # 主轴向量 v_max
    ang = np.degrees(np.arctan2(vy, vx))      # [-180,180]
    ang = abs(ang) % 180.0
    angle_h = ang if ang <= 90 else 180 - ang # [0,90], 0=水平,90=竖直
    return ratio, float(angle_h)


def min_area_box_ratio_and_angle(binmask: np.ndarray):
    # 确保是 uint8 的 0/255
    m = (binmask > 0).astype(np.uint8) * 255

    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None
    c = max(cnts, key=cv2.contourArea)

    (cx, cy), (w, h), theta = cv2.minAreaRect(c)  # theta in (-90, 0], degrees
    if w < 1 or h < 1:
        return None, None

    # 细长度：短边/长边 ∈ (0,1]
    box_ratio = float(min(w, h) / max(w, h))

    # —— 关键：把角度稳健归一化到 [0,90]，表示“长边相对水平”的夹角 ——
    # 1) 让 theta 表示“长边”的角度
    if w < h:
        ang = theta + 90.0   # 宽<高，原始角度是短边的，要+90转到长边
    else:
        ang = theta          # 宽>=高，已经是长边的角度

    # 2) 规范到 [0,180)
    ang = (ang % 180.0 + 180.0) % 180.0

    # 3) 压缩到 [0,90]（因为 0° 和 180° 等价）
    angle_h = ang if ang <= 90.0 else 180.0 - ang

    return box_ratio, float(angle_h)


def run_inference(source="rail.mp4", model_path="yolov8n-seg.pt", out_path="out.mp4"):
    model = YOLO(model_path)
    print("task=", model.task)
    print("names=", model.names)

    cap = cv2.VideoCapture(source)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 定义视频输出
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 或 'XVID'
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))

    # 用 generator 流式推理
    results = model.predict(
        source=source,
        imgsz=1280, #640,960,1280
        conf=0.3,
        iou=0.6,
        classes=[0],
        stream=True
    )

    for res in results: #多少帧图像就有多少个res  
        frame = res.orig_img.copy()
        H, W = frame.shape[:2]

        if res.masks is not None:
            #i是masks有几个人的意思。m是每个人的mask的值
            #mask默认是基于imgsiz得到的矩阵，需要先resize到frame的尺寸
            for i, m in enumerate(res.masks.data.cpu().numpy()): 
                #resize到frame的尺寸后，mask里面的值是0~1的一个范围，越大代表越有可能是元素(目前classes=0，只有人)
                #所以需要设置一个阈值则认为是有效元素，大于阈值的值为1，否则为0
                binmask = (cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST) > 0.55).astype(np.uint8) * 255

                #binmask是一个1920*1080的矩阵，里面描述的1代表是元素，0代表不是元素，如果绘制出来，则是一个轮廓
                pca_ratio, pca_ang_h = pca_minor_major_from_mask(binmask)
                box_ratio, box_ang_h = min_area_box_ratio_and_angle(binmask)

                ratio = 0.0
                ang = 0.0
                method = "unknow"

                # 任选其一或做融合
                lying = False
                if pca_ratio is not None and pca_ratio < 0.6:
                    method = "pca"
                    #细长目标用angel判断
                    ratio = pca_ratio
                    ang = pca_ang_h
                    if pca_ang_h is not None and pca_ang_h < 60:
                        lying = True
                    else:
                        lying = False
                else:
                    # 物体不是细长的，无法通过角度判断
                     # 1) 形状不够细长 → 无法判定方向
                    method = "min"
                    ratio = box_ratio
                    ang = box_ang_h
                    if box_ratio >= 0.95:
                        lying = False
                    # 2) 根据角度判定
                    else:
                        if box_ang_h < 60:
                            lying = True
                        else:
                            lying = False

                x1, y1, x2, y2 = map(int, res.boxes.xyxy[i].cpu().numpy())
                mask_area = binmask.sum() / 255
                bbox_area = (x2 - x1) * (y2 - y1)
                solidity = mask_area / (bbox_area + 1e-6)

                if solidity > 0.8:
                    pass
                    #lying = True

                color = (0, 200, 0) if not lying else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                reasons = []
                txt = ""
                if method != "unknown":
                    reasons.append(f"a:{pca_ang_h:.1f},{box_ang_h:.1f},r:{pca_ratio:.1f},{box_ratio:.1f},m:{method},s:{solidity:.1f}")
                if reasons:
                    txt += " [" + ",".join(reasons) + "]"
                txt += f"conf:{res.boxes.conf[i]:.2f}"

                #在矩形框的左上角绘制文字
                cv2.putText(frame, txt, (x1, max(20, y1-8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # 写入视频
        writer.write(frame)

    cap.release()
    writer.release()
    print(f"✅ 结果已保存到 {out_path}")

def runAll():
    in_dir = "./FallDetection/fileDepends"
    model_path = "yolov8m-seg.pt"
    # 匹配 .mp4 / .MP4（glob 默认大小写敏感，要分别写）
    video_files = glob.glob(os.path.join(in_dir, "*.mp4")) + \
                  glob.glob(os.path.join(in_dir, "*.MP4"))

    for source in video_files:
        # 拆分目录、文件名、扩展名
        base, ext = os.path.splitext(source)
        out_path = f"{base}_seg_out{ext}"

        print(f"▶️ 处理文件: {source}")
        run_inference(source, model_path, out_path)
        
def runOnce():
    source = "./FallDetection/fileDepends/Caida_Elcasar (1).MP4"
    # 拆分目录、文件名、扩展名
    base, ext = os.path.splitext(source)
    # 拼接新的输出路径
    out_path = f"{base}_seg_out{ext}"
    run_inference(source, "yolov8m-seg", out_path)

if __name__ == "__main__":
    runAll()
    #runOnce()
    

