import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

# 替换为你实际的文件路径
file_path = "/home/wuhao/jobspace/robomimic/wuhao/logs/log_rollout_0.pkl"

def load_and_view_pkl(path):
    if not os.path.exists(path):
        print(f"文件不存在: {path}")
        return

    with open(path, "rb") as f:
        # 修正之前的 typo: data = pickle.dump = pickle.load(f) 是错误的
        data = pickle.load(f)
    
    print(f"成功加载文件，记录条数: {len(data)}")
    
    for i, entry in enumerate(data):
        step = entry['step']
        image = entry['image']
        # print(image)
        
        # --- 调试与修复逻辑 ---
        # 1. 如果 image 是列表，尝试取第一个元素或转换为数组
        if isinstance(image, list):
            image = np.array(image)
            
        # 2. 检查 dtype。如果是 object，通常是因为里面嵌套了其他数组
        if hasattr(image, 'dtype') and image.dtype == object:
            print(f"警告: 发现 object 类型数据，尝试强制转换...")
            # 尝试解包：如果数组里只有一个元素
            if image.size == 1:
                image = image.item()
            else:
                # 尝试强制转换为 uint8
                image = np.array(list(image), dtype=np.uint8)

        # 3. 最终检查：确保是 numpy 数组且类型正确
        image = np.asanyarray(image).astype(np.uint8)
        
        print(f"Step {step}: 图片形状 {image.shape}, 类型 {image.dtype}")

        # 可视化
        plt.figure(figsize=(6, 4))
        plt.imshow(image)
        plt.title(f"Step: {step} | Truncation: {entry['truncation_length']}")
        plt.axis('off')

        
        
        # 如果在服务器上，建议保存为文件
        plt.savefig(f"debug_step_{step}.png")
        # plt.show()

        if i >= 2: break # 只看前几张

if __name__ == "__main__":
    load_and_view_pkl(file_path)