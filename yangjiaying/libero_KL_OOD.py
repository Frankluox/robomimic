# 确定执行长度
import numpy as np
from collections import deque 
class PhysicsScheduler:
    def __init__(self, lambda_factor=1.0, window_size=10, z_smooth=0.0, sigmoid_temp=1):
        self.lambda_factor = lambda_factor
        self.window_size = window_size
        self.z_smooth = z_smooth 
        self.sigmoid_temp = sigmoid_temp  
        self.kl_history = deque(maxlen=window_size) 
        print(f"DEBUG: PhysicsScheduler | Win={window_size} | Smooth={z_smooth} | Temp={sigmoid_temp}")

    def reset(self):
        self.kl_history.clear()
        
    def decide(self, f_max_idx, f_min_idx, kl_score, current_full_len):
        self.kl_history.append(float(kl_score))
        
        if len(self.kl_history) < 4:
            kl_z = 0.0
            p_uncertain = 0.25 # 初始设置保守
        else:
            arr = np.array(self.kl_history)
            mean = np.mean(arr)
            std = np.std(arr) 
            
            denominator = std + self.z_smooth
            kl_z = (kl_score - mean) / denominator

            p_uncertain = 1.0 / (1.0 + np.exp(kl_z / self.sigmoid_temp))

        safe_k = max(float(f_max_idx), 0.5)
        raw_N = float(current_full_len) / (self.lambda_factor * safe_k)
        n_freq = int(np.floor(raw_N))
        
        n_trend = int(f_min_idx) + 1
        
        n_kl = 1 + (current_full_len - 1) * p_uncertain
        n_kl = int(np.round(n_kl))
        
        target_len = max(n_freq, n_trend, n_kl)
        target_len = int(np.clip(target_len, 1, current_full_len))
        
        reason = (f"Freq={n_freq}, Trend={n_trend}, KL_N={n_kl} (Z={kl_z:.2f}, P={p_uncertain:.2f})")
        
        return target_len, n_freq, n_trend, n_kl, kl_z, p_uncertain, reason
    
# 计算自回归模型的推理KL
import torch
def predict_action_KL(
        self, input_ids: Optional[torch.LongTensor] = None, unnorm_key: Optional[str] = None, **kwargs: str
    ) -> np.ndarray:
        output = self.generate(input_ids, max_new_tokens=256, 
                                    output_scores=True,
                                    return_dict_in_generate=True,
                                    **kwargs)
        generated_ids = output.sequences
        scores = output.scores # tuple of (batch_size, vocab_size)

        import torch.nn.functional as F
        kl_list = []
        for step_logits in scores:
            probs = F.softmax(step_logits, dim=-1)
            # 计算 Entropy H(P) = - sum(p * log(p))
            step_entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
            # 计算 KL = log(N) - H(P)
            vocab_size = step_logits.shape[-1]
            max_entropy = np.log(vocab_size) # Log of uniform distribution
            step_kl = (max_entropy - step_entropy) / max_entropy 
            kl_list.append(step_kl)

        if len(kl_list) > 0:
            avg_kl = torch.stack(kl_list).mean().item()
        else:
            avg_kl = 0.0

        input_token_len = input_ids.shape[1]
        predicted_ids = generated_ids[0, input_token_len:].cpu().numpy()

        actions = self.action_tokenizer.decode_token_ids_to_actions(predicted_ids)

        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()

        actions = np.clip(actions, -1.0, 1.0)

        action_norm_stats = self.get_action_stats(unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        
        actions_unnorm = np.where(
            mask,
            0.5 * (actions + 1) * (action_high - action_low) + action_low,
            actions,
        )

        return actions_unnorm, avg_kl