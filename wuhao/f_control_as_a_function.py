import numpy as np
from scipy.fftpack import dct
from typing import Tuple, Optional, List

class ActionScheduler:
    """
    动作序列动态调度器 (Action Chunk Scheduler)。
    
    基于动作序列的频域复杂度和时域能量分布，动态决定当前动作块的执行长度。
    """

    def __init__(self, 
                 safety_lambda: float = 1.0, 
                 power_threshold: float = 0.95, 
                 dct_scale: float = 10.0,
                 trend_ratio: float = 0.2,
                 freq_smoothing_factor: float = 2.0):
        """
        Args:
            safety_lambda (float): 安全系数 lambda。
                                   值越大，调度越保守（执行步数越短）；
                                   值越小，调度越激进（执行步数越长）。
            power_threshold (float): 频域带宽计算的能量截断阈值 (0.0~1.0)。
            dct_scale (float): DCT 噪声门限的缩放因子。
            trend_ratio (float): 时域有效动作起始点的能量占比阈值。
            freq_smoothing_factor (float): 频率指数的平滑除数 (原代码中的 / 2.0)。
        """
        self.safety_lambda = safety_lambda
        self.power_threshold = power_threshold
        self.dct_scale = dct_scale
        self.trend_ratio = trend_ratio
        self.freq_smoothing_factor = freq_smoothing_factor

        # [新增] 初始化内部历史缓存
        self.prev_action_buffer: Optional[np.ndarray] = None

    # def _apply_dct_noise_gate(self, dct_coeffs: np.ndarray) -> np.ndarray:
    #     """
    #     内部辅助函数：对 DCT 系数应用噪声门限，过滤微小的频率波动。
    #     """
    #     noise_floor = np.std(dct_coeffs) * self.dct_scale
    #     # 将绝对值小于 noise_floor 的系数置为 0
    #     mask = np.abs(dct_coeffs) > noise_floor
    #     return dct_coeffs * mask

    def _apply_dct_noise_gate(self, dct_coeffs: np.ndarray) -> np.ndarray:
        """
        [新增] DCT 系数噪声门控
        借鉴老方案：通过量化去除幅值极小的噪声分量。
        只有当 |coeff * scale| >= 0.5 (即 |coeff| >= 0.05) 时才保留。
        """
        # 量化 (四舍五入)
        quantized = np.around(dct_coeffs * self.dct_scale)
        # 生成掩码：保留非零部分
        mask = (np.abs(quantized) > 1e-9).astype(float)
        # 应用掩码，将微小噪声置零
        return dct_coeffs * mask

    def reset(self):
        """
        [新增] 重置内部历史缓存，通常在环境 reset 时调用。
        """
        self.prev_action_buffer = None

    def compute_bandwidth_index(self, 
                                action_chunk: np.ndarray, 
                                is_gripper: bool = True) -> float:
        """
        计算动作片段的平均频率带宽索引 (k_avg)。
        
        原理：
        1. 对动作序列进行 DCT 变换。
        2. 找出包含 power_threshold (如95%) 能量的频率截止点。
        3. 对所有维度取平均。
        
        Returns:
            float: 代表动作复杂度的频率索引值。值越大，动作越复杂/抖动。
        """
        # 如果不包含夹爪，只取前6维 (XYZ + RPY)
        if not is_gripper:
            data = action_chunk[:, :6]
        else:
            data = action_chunk

        T, D = data.shape
        if T < 2:
            return 0.0

        # 1. 计算 DCT (Type-II, 正交归一化)
        # axis=0: 对时间维度进行变换
        dct_coeffs = dct(data, type=2, norm='ortho', axis=0)
        dct_coeffs = dct_coeffs

        # 2. 应用噪声门限并计算功率谱
        dct_coeffs_clean = self._apply_dct_noise_gate(dct_coeffs)
        power_spectrum = np.square(dct_coeffs_clean)

        cutoff_indices = []

        for d in range(D):
            p_d = power_spectrum[:, d]
            total_energy = np.sum(p_d)

            # 如果该维度几乎没有能量（静止），认为频率复杂度为0
            if total_energy < 1e-9:
                cutoff_indices.append(0)
                continue

            # 3. 计算累积能量分布
            cumsum_energy = np.cumsum(p_d)

            # 4. 寻找越过阈值的截断点 k
            target_energy = total_energy * self.power_threshold # 保存多少比例的该维度能量
            k = np.searchsorted(cumsum_energy, target_energy)
            cutoff_indices.append(k)

        # 返回所有维度频率索引的平均值
        return float(np.mean(cutoff_indices))

    def compute_minimum_effective_horizon(self, action_chunk: np.ndarray) -> int:
        """
        计算动作的'最小有效执行长度' (f_min)。
        
        原理：
        通过分析多维动作的协同趋势累积量，找到动作开始产生实质性变化的时刻。
        防止执行长度过短导致动作还未开始就被截断。
        
        Returns:
            int: 时间步索引
        """
        # 1. 计算协同趋势强度: Sum(dim) -> Square
        # 物理含义：投影到主对角线方向的幅值平方，强调多关节协同运动
        trend_strength = np.square(np.sum(action_chunk, axis=-1))

        # 2. 计算累积趋势
        cumsum_trend = np.cumsum(trend_strength)
        total_trend = cumsum_trend[-1]

        # 边界处理：如果总趋势极小（静止），返回 0
        if total_trend < 1e-6:
            return 0

        # 3. 归一化并查找阈值点
        norm_cumsum = cumsum_trend / total_trend
        
        # 找到第一个累积占比超过 ratio 的索引
        # np.argmax 在布尔数组中会返回第一个 True 的索引
        first_match_idx = np.argmax(norm_cumsum >= self.trend_ratio)

        return int(first_match_idx)

    def decide_execution_length(self, 
                                k_spatial: float, 
                                t_min_temporal: int, 
                                current_full_len: int) -> Tuple[int, str]:
        """
        核心调度算法：根据频域和时域特征决定最终执行长度 N。
        
        公式推导：
            N < L / (lambda * k)
        其中：
            L = current_full_len (预测总长)
            k = k_spatial (频率复杂度)
            lambda = self.safety_lambda (安全系数)
        """
        # 确保分母不为0，且 k 至少有一定大小
        safe_k = max(float(k_spatial), 0.5)

        # === [核心修改] 显式引入 lambda ===
        # 原始公式含义：动作越复杂(k大)，执行步数(target_len)应越短
        # lambda 越大，分母越大 -> target_len 越小 (更保守)
        raw_N = float(current_full_len) / (self.safety_lambda * safe_k)
        
        # 向下取整
        target_len = int(np.floor(raw_N))

        # 截断限制：
        # 下限：不能小于最小有效长度 (t_min_temporal + 1)
        # 上限：不能超过当前预测的总长度
        lower_bound = t_min_temporal + 1
        target_len = int(np.clip(target_len, lower_bound, current_full_len))

        reason = (f"k_avg={k_spatial:.2f}, "
                  f"N_raw={raw_N:.2f}, "
                  f"Limit=[{lower_bound}, {current_full_len}], "
                  f"Lambda={self.safety_lambda}")
        
        return target_len, reason
    

    
    def step(self, 
             current_action: np.ndarray, 
             prev_action: Optional[np.ndarray] = None) -> Tuple[np.ndarray, bool, str]:
        """
        主调用接口：处理输入动作并返回截断后的动作序列。

        Args:
            current_action: 当前模型预测的动作块 (T, D)
            prev_action: 上一步执行的最后动作 (用于保持频率分析的连续性)，可选。

        Returns:
            truncated_action: 截断后的动作序列
            is_truncated: 是否发生了截断 (True/False)
            debug_info: 决策原因描述字符串
        """
        # # 1. 拼接上下文以获得更准确的频率特征
        # if prev_action is not None:
        #     # 假设 prev_action 也是 (T', D)，通常只需拼接最后几帧即可，这里全拼
        #     concat_act = np.concatenate([prev_action, current_action], axis=0)
        # else:
        #     concat_act = np.concatenate([current_action, current_action], axis=0) 

        # 1. [修改] 拼接逻辑：使用内部 self.prev_action_buffer
        if self.prev_action_buffer is not None:
            # 使用上一帧的完整预测作为历史上下文
            concat_act = np.concatenate([self.prev_action_buffer, current_action], axis=0)
        else:
            # 第一帧 (Cold Start): 自我复制以填充上下文
            concat_act = np.concatenate([current_action, current_action], axis=0)


        # 99分位min-max 归一化
        # square 参数
        # q01 = np.array([-0.625, -0.57947, -0.79047, -0.08135263, -0.13670468, -0.44077774, -1])
        # q99 = np.array([1, 0.922, 1, 0.08800747, 0.18450282, 0.46038807, 1])
        # tool hang 参数 
        q01 = array([-0.531     , -1.        , -0.86939   , -0.18211886, -0.24142542,
            -0.38555725, -1.        ])
        q99 = array([0.61439   , 1.        , 1.        , 0.11013618, 0.32159994,
            0.47830307, 1.        ])
        
        concat_act = np.clip(concat_act, q01, q99)
        concat_act = (concat_act - q01) / (q99 - q01)
        concat_act = concat_act * 2 - 1



        # z-score归一化
        # square 参数
        # mean = np.array([ 1.69612456e-01,  6.57085627e-02, -7.06431319e-02,  1.54050118e-04,
        #                 1.04218971e-02, -2.87262560e-05,  5.65762420e-02])
        # std = np.array([0.41852544, 0.27245218, 0.3836991 , 0.03187443, 0.06657575,
        #                 0.15493114, 0.99839828])
        # tool hang 参数
        # mean = array([ 9.31843855e-03, -9.27273296e-03, -3.22315708e-02, -2.01073064e-04,
        #     8.50217240e-03,  2.24776253e-02,  3.97011317e-01])
        # std = array([0.20952107, 0.33178265, 0.36589365, 0.04881134, 0.09701974,
        #     0.14167492, 0.91781371])

        # concat_act = (concat_act - mean) / std


        


        # 2. 计算频域复杂度 (k)
        # 注意：这里保留了原逻辑中的除以 2.0 (freq_smoothing_factor)
        # 这通常是为了将 DCT 索引调整为更符合物理直觉的数值
        k_raw = self.compute_bandwidth_index(concat_act, is_gripper=True)
        k_spatial = k_raw / self.freq_smoothing_factor

        # 3. 计算时域最小有效长度 (min_len)
        # 注意：这里只用 current_action，因为我们关心的是当前预测段的起始趋势
        # t_min_temporal = self.compute_minimum_effective_horizon(current_action)
        t_min_temporal = self.compute_minimum_effective_horizon(concat_act[10:20])

        # 4. 决策
        target_len, reason = self.decide_execution_length(
            k_spatial, t_min_temporal, len(current_action)
        )

        # 5. [新增] 更新内部缓存 (在切片之前，保存完整的 current_action 用于下一帧)
        self.prev_action_buffer = current_action.copy()

        # 6. 执行切片
        final_action = current_action[:target_len]
        is_truncated = (target_len < len(current_action))

        
        return final_action, is_truncated, target_len, reason

# ==========================================
# 使用示例
# ==========================================
if __name__ == "__main__":
    # 1. 初始化调度器，设置 lambda = 1.2 (稍微保守一点)
    scheduler = ActionScheduler(safety_lambda=2.5, power_threshold=0.95)

    # 2. 模拟数据
    # 模拟一个比较平滑的动作 (T=50, D=7)
    t = np.linspace(0, 1, 50)
    smooth_action = np.zeros((50, 7))
    for i in range(7):
        smooth_action[:, i] = np.sin(2 * np.pi * 1 * t) # 1Hz 低频

    # 模拟上一步动作 (用于上下文)
    prev_act = smooth_action[-10:] 

    # 3. 执行调度
    action_chunk, truncated, info = scheduler.step(smooth_action, prev_action=prev_act)

    print(f"原始长度: {len(smooth_action)}")
    print(f"执行长度: {len(action_chunk)}")
    print(f"是否截断: {truncated}")
    print(f"决策详情: {info}")
    
    # 输出示例解释:
    # 如果 lambda=1.2, k_spatial 很小 (例如 1.0)
    # N_raw = 50 / (1.2 * 1.0) = 41.6 -> 41
    # 结果会切片取前 41 帧