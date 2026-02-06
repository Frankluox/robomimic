import pickle
import argparse
import numpy as np
import cv2
import os
import glob

# ==========================================
# 绘图配置 (基于之前的 V3 版本)
# ==========================================
SIDE_PANEL_W = 450
CHART_H = 280
BG_COLOR = (245, 245, 245) 
TEXT_COLOR = (20, 20, 20)
AXIS_COLOR = (120, 120, 120)
LINE_COLOR = (255, 80, 80) 
HIGHLIGHT_COLOR = (40, 40, 255) 
FONT = cv2.FONT_HERSHEY_SIMPLEX

def draw_line_chart_cv2(img, top_left, size, xs, ys, title, y_label):
    x_start, y_start = top_left
    w, h = size
    margin_L, margin_R, margin_T, margin_B = 70, 30, 60, 60
    plot_x0, plot_y0 = x_start + margin_L, y_start + h - margin_B
    plot_w, plot_h = w - margin_L - margin_R, h - margin_T - margin_B
    
    cv2.rectangle(img, top_left, (x_start + w, y_start + h), BG_COLOR, -1)
    cv2.putText(img, title, (x_start + 15, y_start + 25), FONT, 0.6, TEXT_COLOR, 2)
    
    if not xs:
        min_x, max_x, min_y, max_y = 0, 10, 0, 10
    else:
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = 0, max(ys) + 2
        if max_x == min_x: max_x += 1

    def to_screen(x_val, y_val):
        px = plot_x0 + int((x_val - min_x) / (max_x - min_x) * plot_w)
        py = plot_y0 - int((y_val - min_y) / (max_y - min_y) * plot_h)
        return (px, py)

    cv2.line(img, (plot_x0, plot_y0), (plot_x0 + plot_w, plot_y0), AXIS_COLOR, 2)
    cv2.line(img, (plot_x0, plot_y0), (plot_x0, plot_y0 - plot_h), AXIS_COLOR, 2)
    cv2.putText(img, str(int(max_y)), (x_start + 10, plot_y0 - plot_h + 5), FONT, 0.5, TEXT_COLOR, 1)
    cv2.putText(img, str(int(min_y)), (x_start + 10, plot_y0 + 5), FONT, 0.5, TEXT_COLOR, 1)
    cv2.putText(img, y_label, (x_start + 10, plot_y0 - plot_h - 15), FONT, 0.5, AXIS_COLOR, 1)
    cv2.putText(img, str(int(max_x)), (plot_x0 + plot_w - 20, plot_y0 + 25), FONT, 0.4, TEXT_COLOR, 1)

    if len(xs) > 1:
        for i in range(len(xs) - 1):
            p1, p2 = to_screen(xs[i], ys[i]), to_screen(xs[i+1], ys[i+1])
            cv2.line(img, p1, p2, LINE_COLOR, 2)
        cv2.circle(img, to_screen(xs[-1], ys[-1]), 6, HIGHLIGHT_COLOR, -1)

def draw_dist_panel_cv2(img, top_left, size, dist, chosen_k, title):
    x_start, y_start = top_left
    w, h = size
    margin_L, margin_R, margin_T, margin_B = 50, 30, 50, 50
    cv2.rectangle(img, top_left, (x_start + w, y_start + h), BG_COLOR, -1)
    cv2.putText(img, title, (x_start + 15, y_start + 25), FONT, 0.6, TEXT_COLOR, 2)
    if dist is None: return

    plot_x0, plot_y0 = x_start + margin_L, y_start + h - margin_B
    plot_w, plot_h = w - margin_L - margin_R, h - margin_T - margin_B
    num_bins = len(dist)
    bin_w = plot_w // num_bins
    step = 1 if num_bins <= 10 else (5 if num_bins <= 30 else 10)

    for i, p in enumerate(dist):
        bar_h = int(p * plot_h)
        color = HIGHLIGHT_COLOR if i == chosen_k else (180, 180, 180)
        cv2.rectangle(img, (plot_x0 + i*bin_w + 1, plot_y0 - bar_h), (plot_x0 + (i+1)*bin_w - 1, plot_y0), color, -1)
        if i % step == 0 or i == num_bins - 1:
            cv2.putText(img, str(i), (plot_x0 + i*bin_w, plot_y0 + 20), FONT, 0.4, TEXT_COLOR, 1)
    cv2.line(img, (plot_x0, plot_y0), (plot_x0 + plot_w, plot_y0), AXIS_COLOR, 2)

# ==========================================
# 核心逻辑
# ==========================================

def process_single_pkl(pkl_path, output_dir):
    filename = os.path.basename(pkl_path)
    # 根据原文件名提取成功/失败状态
    status = "success" if "success" in filename.lower() else "failed"
    video_name = filename.replace(".pkl", ".mp4")
    if "success" not in video_name.lower() and "failed" not in video_name.lower():
        video_name = video_name.replace(".mp4", f"_{status}.mp4")
    
    output_path = os.path.join(output_dir, video_name)
    
    print(f"[PROCESS] {filename} -> {video_name}")
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    traj = data.get('trajectory', [])
    if not traj or 'image' not in traj[0]:
        print(f"  [SKIP] No images in {filename}")
        return

    decisions_map = {d['step_start']: d for d in data.get('decisions', [])}
    img_h, img_w = traj[0]['image'].shape[:2]
    out_w, out_h = img_w + SIDE_PANEL_W, max(img_h, CHART_H * 2)

    video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (out_w, out_h))
    
    history_steps, history_ks = [], []
    last_decision_step = -1

    for idx, step_data in enumerate(traj):
        canvas = np.ones((out_h, out_w, 3), dtype=np.uint8) * 255
        img_bgr = cv2.cvtColor(step_data['image'], cv2.COLOR_RGB2BGR)
        canvas[(out_h-img_h)//2 : (out_h-img_h)//2 + img_h, 0:img_w] = img_bgr
        
        curr_step = step_data['step']
        active_d = None
        for start_s, d in decisions_map.items():
            if start_s <= curr_step < start_s + d['chosen_k']:
                active_d = d
                break
        
        if active_d and active_d['step_start'] != last_decision_step:
            history_steps.append(active_d['step_start'])
            history_ks.append(active_d['chosen_k'])
            last_decision_step = active_d['step_start']

        draw_dist_panel_cv2(canvas, (img_w, 0), (SIDE_PANEL_W, CHART_H), 
                            active_d.get('policy_dist') if active_d else None, 
                            active_d['chosen_k'] if active_d else -1, "Policy Dist")
        draw_line_chart_cv2(canvas, (img_w, CHART_H), (SIDE_PANEL_W, CHART_H),
                            history_steps, history_ks, "Chunk Size History", "k")

        cv2.putText(canvas, f"Step: {curr_step} | Status: {status.upper()}", (10, 30), FONT, 0.7, (0,0,0), 2)
        video.write(canvas)

    video.release()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="存放PKL文件的文件夹")
    parser.add_argument("--output_dir", type=str, default="vis_videos", help="视频输出文件夹")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 扫描所有 pkl 文件
    pkl_files = glob.glob(os.path.join(args.input_dir, "*.pkl"))
    print(f"[INFO] 找到 {len(pkl_files)} 个 PKL 文件。")

    for pkl_path in sorted(pkl_files):
        try:
            process_single_pkl(pkl_path, args.output_dir)
        except Exception as e:
            print(f"  [ERROR] 处理 {pkl_path} 失败: {e}")

if __name__ == "__main__":
    main()