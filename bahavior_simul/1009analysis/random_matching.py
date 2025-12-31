# -*- coding: utf-8 -*-
"""
random_matching_experiment.py

功能：
1. 读取原始 Excel 数据 (1-21组)。
2. 对每一组生成【随机匹配】(Random Matching)：
   - 随机打乱男性和女性列表。
   - 强制一一配对，多余的人单身。
3. 计算该随机匹配的期望阻塞对 E[BP]。
4. 重复多次 (N_ITER) 取平均，作为 Random Baseline。
5. 将结果保存为 CSV，方便与 AI 和 Human 结果对比。

评价标准：
- 使用与 AI 分析完全相同的参数 (beta0, lambda, s0)。
- [修改] 阻塞对计算时开启截断 (truncate=True)，即如果新对象分数 < s0，则交换概率为 0。
"""

import os, sys, re, json, random
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple

# ================= 配置路径 =================
# 1. 拟合参数 (beta0, lambda, s0)
PARAMS_JSON = "/home/lsy/match/bahavior_simul/1009analysis/unified_params_gpt4_en_fitting.json"

# 2. 源数据 Excel (包含 iid, pid 和 6维打分)
SOURCE_EXCEL = "/home/lsy/match/dataset/save_merge_select_null_3.xlsx"

# 3. 结果输出路径
OUTPUT_CSV = "/home/lsy/match/bahavior_simul/1009analysis/different_analysis/Ebp_Random_Baseline_21.csv"

# 4. 随机实验次数 (每组跑多少次随机取平均)
N_ITER = 100 
# ===========================================

REPO_ROOT = "/home/lsy/match"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from config import config
    from load_data import load_source_scores
except ImportError:
    # 简单的 fallback，防止找不到模块
    pass

# ---------- 复用核心数学函数 ----------

def _sigmoid(x: float) -> float:
    if x > 30: return 1.0
    if x < -30: return 0.0
    return 1.0 / (1.0 + np.exp(-x))

def _p_switch(s_new: float, s_cur: float,
              beta0: float, lam: float, s0: float, 
              truncate_below_s0: bool = False) -> float:
    if s_new is None: return 0.0
    val_cur = s0 if s_cur is None else s_cur
    
    # [修改] 响应用户要求，开启截断
    if truncate_below_s0 and (s_new < s0):
        return 0.0
    return float(_sigmoid(beta0 + lam * (s_new - val_cur)))

def _expected_numbp_for_matching(matching: Dict[int, Any],
                                 men_ids: List[int], women_ids: List[int],
                                 score_dict: Dict[Tuple[int, int], float],
                                 beta0: float, lam: float, s0: float,
                                 truncate: bool) -> float:
    """
    计算给定匹配的期望阻塞对数量。
    """
    total = 0.0
    for m in men_ids:
        # m 的状态
        m_partner = matching.get(m)
        if m_partner in [None, "rejected", m]:
            m_is_single = True
            val_m_cur = s0
        else:
            m_is_single = False
            # 查分 (0-100)
            val_m_cur = score_dict.get((m, int(m_partner)), s0)

        for w in women_ids:
            # 如果已匹配，跳过
            if not m_is_single and m_partner == w:
                continue
            
            # P(m wants w)
            s_mw = score_dict.get((m, w))
            p_m = _p_switch(s_mw, val_m_cur, beta0, lam, s0, truncate_below_s0=truncate)

            # P(w wants m)
            w_partner = matching.get(w)
            if w_partner in [None, "rejected", w]:
                val_w_cur = s0
            else:
                val_w_cur = score_dict.get((w, int(w_partner)), s0)
            
            s_wm = score_dict.get((w, m))
            p_w = _p_switch(s_wm, val_w_cur, beta0, lam, s0, truncate_below_s0=truncate)

            total += p_m * p_w
            
    return float(total)

# ---------- 随机匹配生成器 ----------

def generate_random_matching(men_ids: List[int], women_ids: List[int]) -> Dict[int, int]:
    """
    生成一个随机的一一配对。
    """
    # 1. 复制列表以免修改原件
    m_pool = list(men_ids)
    w_pool = list(women_ids)
    
    # 2. 打乱
    random.shuffle(m_pool)
    random.shuffle(w_pool)
    
    # 3. 配对
    matching = {}
    
    # 取较短的长度
    n_pairs = min(len(m_pool), len(w_pool))
    
    for i in range(n_pairs):
        m = m_pool[i]
        w = w_pool[i]
        matching[m] = w
        matching[w] = m
        
    # 4. 剩余的人设为单身 (self_id)
    # 处理剩下的男人
    for i in range(n_pairs, len(m_pool)):
        m = m_pool[i]
        matching[m] = m # 显式单身
        
    # 处理剩下的女人
    for i in range(n_pairs, len(w_pool)):
        w = w_pool[i]
        matching[w] = w # 显式单身
        
    return matching

def load_unified_params(json_path: str) -> Tuple[float, float, float]:
    with open(json_path, "r", encoding="utf-8") as f:
        pack = json.load(f)
    return float(pack["beta0_taken"]), float(pack["lambda_taken"]), float(pack["s0"])

# ================= 主流程 =================

def main():
    print("=== Random Matching Baseline Experiment ===")
    
    # 1. 加载参数
    if not os.path.exists(PARAMS_JSON):
        print(f"Error: Params file not found: {PARAMS_JSON}")
        return
    beta0, lam, s0 = load_unified_params(PARAMS_JSON)
    print(f"Params Loaded: beta0={beta0:.4f}, lambda={lam:.4f}, s0={s0:.4f}")
    
    # 2. 加载分数 (加权分 0-100)
    # 注意：这里直接调用 load_source_scores，它应该已经根据你更新后的 load_data.py 返回加权分了
    print(f"Loading scores from: {SOURCE_EXCEL}")
    score_dict = load_source_scores(SOURCE_EXCEL)
    df_source = pd.read_excel(SOURCE_EXCEL)
    
    if score_dict is None:
        print("Error: Failed to load scores.")
        return

    results = []
    
    # 3. 遍历 1-21 组
    for gid in range(1, 22): # 1 to 21
        g_data = df_source[df_source['group'] == gid]
        if g_data.empty:
            print(f"Group {gid}: No data found, skipping.")
            continue
            
        # 提取 ID
        if 'gender' in g_data.columns:
            men_ids = sorted(g_data[g_data['gender'] == 1]['iid'].dropna().unique().astype(int).tolist())
            women_ids = sorted(g_data[g_data['gender'] == 0]['iid'].dropna().unique().astype(int).tolist())
        else:
            # Fallback
            men_ids = sorted(g_data['iid'].dropna().unique().astype(int).tolist())
            women_ids = sorted(g_data['pid'].dropna().unique().astype(int).tolist())
            women_ids = [w for w in women_ids if w not in men_ids]

        if not men_ids or not women_ids:
            continue

        print(f"Processing Group {gid} (N_iter={N_ITER})...", end="", flush=True)
        
        ebp_sum = 0.0
        
        # 4. 重复 N 次实验取平均
        for _ in range(N_ITER):
            # 生成随机匹配
            rand_match = generate_random_matching(men_ids, women_ids)
            
            # 计算 E[BP]
            # [修改] 根据要求开启截断 truncate=True
            ebp = _expected_numbp_for_matching(rand_match, men_ids, women_ids, score_dict, beta0, lam, s0, truncate=True)
            ebp_sum += ebp
            
        avg_ebp = ebp_sum / N_ITER
        print(f" Avg E[BP] = {avg_ebp:.4f}")
        
        results.append({
            "group": gid,
            "Ebp_Random": avg_ebp
        })

    # 5. 保存结果
    df_out = pd.DataFrame(results)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df_out.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    
    print("\n" + "="*40)
    print(f"Random Baseline Calculation Complete.")
    print(f"Results saved to: {OUTPUT_CSV}")
    print(f"Overall Average E[BP] (Random): {df_out['Ebp_Random'].mean():.4f}")
    print("="*40)

if __name__ == "__main__":
    main()