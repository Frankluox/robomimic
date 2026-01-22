import numpy as np
from typing import Optional, Tuple
from collections import deque







class OODMonitor:
    """
    OOD 检测器与执行下界生成器。
    
    逻辑：
    通过计算 [当前预测 chunk] 与 [上一帧预测 chunk 的剩余部分] 之间的
    时域一致性 (Temporal Consistency)，来评估当前的 OOD 程度。
    
    - 一致性低 (差异大) -> 认为处于 OOD/不稳定区域 -> 提高执行下界 (Bridging)。
    - 一致性高 (差异小) -> 认为处于 IID/稳定区域 -> 下界由系统默认决定。
    """
    
    def __init__(self, 
                 max_prediction_len: int,
                 max_ood_bound: int = 16):
        """
        Args:
            max_prediction_len (int): 模型输出的最大长度 (用于对齐缓存)。
            consistency_threshold (float): 判定为显著不一致的 L2 距离阈值。
                                           超过此值，OOD 强度开始增加。
            max_ood_bound (int): 当 OOD 程度满格时，强制执行的最小步数 (下界)。
            smooth_factor (float): OOD 分数的平滑因子 (EMA)，防止下界剧烈跳变。
        """
        self.max_len = max_prediction_len
        # self.threshold = consistency_threshold
        self.max_bound = max_ood_bound
        self.loss_history = deque(maxlen=10) # 限制长度
        self._last_metrics = {} # 用于存储内部指标
        
        # 缓存上一帧预测的完整 Chunk (未经过截断的 raw prediction)
        # 形状: (T, D)
        self.prev_prediction_buffer: Optional[np.ndarray] = None
        
        # 缓存当前的 OOD Score (用于平滑)
        self.current_ood_score = 0.0

        # 在 __init__ 中定义两个因子
        self.attack_alpha = 0.8  # 变差时反应快
        self.release_alpha = 0.2 # 变好时回落慢

        # 3. 内部指标缓存（解决 NameError）
        self._last_metrics = {
            "c0_loss": 0.0,
            "trend_loss": 0.0
        }

    def compute_consistency_loss(self, 
                                 current_action: np.ndarray, 
                                 executed_steps: int) -> float:
        """
        高级一致性损失函数：融合运动学高阶导数连续性与重叠段形状相似度。
        """
        if self.prev_prediction_buffer is None:
            return 0.0
            
        prev_pred = self.prev_prediction_buffer
        T_prev, D = prev_pred.shape
        T_curr = current_action.shape[0]
        
        # 确定旧计划在当前时间点的“锚点”索引
        # anchor_idx 指向上一帧预测中，对应当前时刻 t 的那个点
        anchor_idx = min(executed_steps, T_prev - 1)

        # --- 1. 高阶边界连续性 (Boundary Kinematic Continuity) ---
        # 考察新旧轨迹在衔接处的 0阶(位置), 1阶(速度), 2阶(加速度) 跳变
        
        # 0阶：位置偏移
        loss_c0 = np.linalg.norm(current_action[0] - prev_pred[anchor_idx])
        
        # 1阶：速度向量一致性 (Velocity Jump)
        # 计算旧轨迹在锚点处的瞬时速度，与新轨迹起点的瞬时速度对比
        if anchor_idx > 0 and T_curr > 1:
            v_prev = prev_pred[anchor_idx] - prev_pred[anchor_idx - 1]
            v_curr = current_action[1] - current_action[0]
            loss_c1 = np.linalg.norm(v_curr - v_prev)
        else:
            loss_c1 = 0.0

        # 2阶：加速度一致性 (Acceleration Jump)
        if anchor_idx > 1 and T_curr > 2:
            a_prev = (prev_pred[anchor_idx] - 2 * prev_pred[anchor_idx-1] + prev_pred[anchor_idx-2])
            a_curr = (current_action[2] - 2 * current_action[1] + current_action[0])
            loss_c2 = np.linalg.norm(a_curr - a_prev)
        else:
            loss_c2 = 0.0

        # 边界综合损失 (赋予不同权重，位置最重)
        boundary_loss = loss_c0 + 0.5 * loss_c1 + 0.2 * loss_c2

        # --- 2. 形状相干性 (Overlap Shape Consistency) ---
        # 考察重叠段的整体轨迹形态是否发生“语义”上的突变
        shape_loss = 0.0
        trend_dist = 0.0 # 初始化，防止分支未进入
        overlap_len = 0
        
        if executed_steps < T_prev:
            prev_overlap = prev_pred[executed_steps:]
            overlap_len = min(prev_overlap.shape[0], T_curr)
            
            if overlap_len > 1:
                chunk_old = prev_overlap[:overlap_len]
                chunk_new = current_action[:overlap_len]
                
                # A. 均方根误差 (RMSE) - 基础位移差异
                rmse = np.sqrt(np.mean(np.square(chunk_old - chunk_new)))
                
                # B. 趋势一致性 (Trend Similarity)
                # 通过计算一阶导数（向量）的余弦距离，看轨迹是不是往一个方向走的
                diff_old = np.diff(chunk_old, axis=0)
                diff_new = np.diff(chunk_new, axis=0)
                
                # 防止除零的极小值
                norm_old = np.linalg.norm(diff_old, axis=1, keepdims=True) + 1e-6
                norm_new = np.linalg.norm(diff_new, axis=1, keepdims=True) + 1e-6
                
                # 计算每一步方向的余弦相似度，并转化为距离 (0~2)
                cos_sim = np.sum((diff_old / norm_old) * (diff_new / norm_new), axis=1)
                trend_dist = np.mean(1.0 - cos_sim)
                
                # 融合 RMSE 与 趋势
                shape_loss = rmse + 0.5 * trend_dist

        # --- 更新指标缓存供 step 使用 ---
        self._last_metrics["c0_loss"] = float(loss_c0)
        self._last_metrics["trend_loss"] = float(trend_dist)

        # --- 3. 动态融合 ---
        # 计算重叠权重：重叠越多，越看重 shape_loss；重叠越少，越看重边界跳变
        # 使用 sigmoid 化的比例函数，让权重平滑切换
        ratio = overlap_len / float(self.max_len)
        overlap_weight = 1.0 / (1.0 + np.exp(-10 * (ratio - 0.5))) # 在 0.5 处快速切换的 Sigmoid
        
        # 最终损失 = 边界损失 + 形状损失 (由重叠度决定)
        # 如果没有重叠，则完全退化为 boundary_loss
        total_loss = (1.0 - overlap_weight) * boundary_loss + overlap_weight * shape_loss
        
        return float(total_loss)

    def get_ood_score(self, consistency_loss, dynamic_threshold):
        # k 值决定了响应的锐利度
        # 建议设为 10/threshold，这样在 0.8*threshold 以下几乎为 0
        k = 10.0 / max(dynamic_threshold, 1e-6)
        # 数值稳定性：限制 k 的大小，防止 np.exp 溢出
        k = min(k, 1000.0)
        
        # Sigmoid 映射
        raw_score = 1.0 / (1.0 + np.exp(-k * (consistency_loss - dynamic_threshold)))
        
        # 相比 tanh，这样在 loss < threshold 时更安静，在 loss > threshold 时更果断
        return raw_score

    def step(self, 
             current_action_raw: np.ndarray, 
             last_execution_len: int) -> Tuple[int, float]:
        """
        Args:
            current_action_raw: 当前原始预测 (T, D)
            last_execution_len: 上一步实际执行了多少步才轮到当前这一步。
                                (如果是刚初始化，传 0 或任意值即可)
        
        Returns:
            lower_bound (int): 建议的最小执行长度
            ood_score (float): 当前计算出的 OOD 分数 (0~1)
        """

        # 1. 第一帧特殊处理
        if self.prev_prediction_buffer is None:
            self.prev_prediction_buffer = current_action_raw.copy()
            return {
                "suggested_bound": 1,
                "ood_score": 0.0,
                "metrics": {"c0_loss": 0.0, "trend_loss": 0.0, "is_bridging": False}
            }



        # 2. 计算 Loss
        consistency_loss = self.compute_consistency_loss(current_action_raw, last_execution_len)
        self.loss_history.append(consistency_loss)


        # 3. 计算动态阈值（增加微小偏置防止全 0）
        history_arr = np.array(self.loss_history)
        dynamic_threshold = np.mean(history_arr) + 2.0 * np.std(history_arr)+ 0.05
        
        # 4. 计算 Score
        raw_score = self.get_ood_score(consistency_loss, dynamic_threshold)
        
        # 简单线性版：
        # raw_score = np.clip(consistency_loss / self.threshold, 0.0, 1.0)

        # 5. 平滑（非对称 EMA）
        if raw_score > self.current_ood_score:
            self.current_ood_score = self.attack_alpha * raw_score + (1 - self.attack_alpha) * self.current_ood_score
        else:
            self.current_ood_score = self.release_alpha * raw_score + (1 - self.release_alpha) * self.current_ood_score

        # 死区处理：极小分值直接归零
        if self.current_ood_score < 0.01:
            self.current_ood_score = 0.0
        
        # # 3. 平滑 Score (避免下界频繁跳动)
        # self.current_ood_score = (self.alpha * raw_score + 
        #                           (1 - self.alpha) * self.current_ood_score)
        
        # 下界 = 最小保证长度 + (最大下界 - 最小保证) * score
        # 最小保证长度通常为 1
        # 6. 计算下界
        suggested_bound = 1 + (self.max_bound - 1) * self.current_ood_score
        self.prev_prediction_buffer = current_action_raw.copy()
        
        # 5. 更新缓存 (必须存储当前的原始预测，供下一步对比)
        self.prev_prediction_buffer = current_action_raw.copy()
        
        
        return {
            "suggested_bound": int(suggested_bound),
            "ood_score": float(self.current_ood_score),
            "metrics": {
                **self._last_metrics,
                "is_bridging": self.current_ood_score > 0.5
            }
        }

# ==========================================
# 集成示例 (结合你之前的 ActionScheduler)
# ==========================================
if __name__ == "__main__":
    # 假设动作维度 D=7, 预测长度 T=50
    # 初始化你的调度器
    from scipy.fftpack import dct # 假设你的类在上面
    scheduler = ActionScheduler(safety_lambda=1.5)
    
    # 初始化 OOD 监测器
    # threshold 设定建议：先跑一段正常数据，看平均 L2 误差是多少，设为那个值的 2-3 倍
    ood_monitor = OODMonitor(max_prediction_len=50, 
                             consistency_threshold=0.15, 
                             max_ood_bound=15) # OOD 时最少也要走 15 步
    
    # 模拟循环
    # -------------------------------------------------
    # 时刻 t=0: IID 区域
    pred_t0 = np.zeros((50, 7)) # 预测静止
    # 决策 t=0
    # 假设这是第一帧，last_exec_len 无所谓
    lb_0, score_0 = ood_monitor.step(pred_t0, last_execution_len=0)
    target_len_0, _, _ = scheduler.step(pred_t0)
    
    final_len_0 = max(target_len_0, lb_0)
    print(f"[T=0] IID | Complexity Len: {target_len_0}, OOD LB: {lb_0}, Final: {final_len_0}")
    
    # 假设 t=0 执行了 5 步就被截断了 (比如因为复杂度高)
    executed_len_t0 = 5
    
    # -------------------------------------------------
    # 时刻 t=1 (实际上是 t=5): 进入 OOD，模型开始抖动
    # 模型突然预测要向右猛冲 (与 t=0 的预测不符)
    pred_t1 = np.ones((50, 7)) * 0.5 
    
    # 计算 OOD 下界
    # 注意：这里传入 executed_len_t0，告诉 monitor 距离上次预测过去了 5 步
    lb_1, score_1 = ood_monitor.step(pred_t1, last_execution_len=executed_len_t0)
    
    # 计算复杂度长度
    target_len_1, _, info = scheduler.step(pred_t1, prev_action=pred_t0[:executed_len_t0])
    
    # 最终决策
    final_len_1 = max(target_len_1, lb_1)
    
    print(f"[T=1] OOD | Loss: {ood_monitor.current_ood_score:.2f}")
    print(f"       -> Complexity Len: {target_len_1} (因高频可能很短)")
    print(f"       -> OOD LowerBound: {lb_1} (因不一致而变长)")
    print(f"       -> Final Decision: {final_len_1} (Triggered Bridging!)")