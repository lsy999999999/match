# -*- coding: utf-8 -*-
"""
compute_stability_from_unified_json_customdir_21.py

修改说明 (2024/11 - Weighted Score Update):
1. _build_objective_prefs: 
   - [关键修正] 不再在函数内部手动计算加权分。
   - 改为直接查阅 score_dict，因为 load_data.py 已经按照 (Sum(Score*Weight)/Sum(Weight))*10 算好了 0-100 的分。
   - 这样保证了 Preference List 里的 partner分数 和 s0 (也是0-100) 是完全同量级的。
2. _expected_numbp_for_matching:
   - 保持不变，使用新的 score_dict 和 s0 进行概率计算。
"""

import os, sys, re, json, glob
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ========= 配置路径 (请根据实际情况确认) =========
PARAMS_JSON = "/home/lsy/match/bahavior_simul/analysis1127/unified_params_gpt4_en_fitting.json"
MATCH_DIR   = "/home/lsy/match/bahavior_simul/0627_gpt4_eng"
OUTPUT_CSV  = "/home/lsy/match/bahavior_simul/analysis1127/different_analysis/Ebp_from_unified_json_0627_en_21_weighted.csv"
# =================================================

REPO_ROOT = "/home/lsy/match"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config import config
from load_data import load_source_scores
from gale_shapley_classic import classic_gale_shapley_matcher

# ---------- 统一模型 ----------
def _sigmoid(x: float) -> float:
    # 限制范围防止溢出
    if x > 30: return 1.0
    if x < -30: return 0.0
    return 1.0 / (1.0 + np.exp(-x))

def _p_switch(s_new: Optional[float], s_cur: Optional[float],
              beta0: float, lam: float, s0: float, 
              truncate_below_s0: bool = False) -> float:
    """
    计算切换概率。
    :param truncate_below_s0: 如果为 True (Human模式)，当 s_new < s0 时强制概率为 0。
    """
    if s_new is None:
        return 0.0
    
    # 这里的 s_cur 已经是处理过的数值 (如果是单身，传入的就是 s0)
    # 但为了保险，如果传入 None 还是回退到 s0
    val_cur = s0 if s_cur is None else s_cur
    
    # --- Human GS 的截断逻辑 ---
    # 如果新对象的分数低于单身分数 (s0)，且开启了截断，则不可能交换
    if truncate_below_s0 and (s_new < s0):
        return 0.0
        
    return float(_sigmoid(beta0 + lam * (s_new - val_cur)))

# ---------- 偏好构造（核心修改：适配加权分） ----------
def _build_objective_prefs_with_self(df_group: pd.DataFrame, score_dict: Dict[Tuple[int, int], float], s0: float) -> Tuple[Dict[int, List[int]], Dict[int, List[int]], List[int], List[int]]:
    """
    构建偏好列表，显式插入 self_id。
    
    [修正]: 
    不再使用 df_group 中的列计算 weighted_score。
    而是直接使用 score_dict 中的分数进行排序。
    score_dict 来自 load_data.py，已经是 0-100 分制的加权分。
    s0 来自 fit_params，也是 0-100 分制。
    两者量级一致，可以直接比较排序。
    """
    
    # 提取 ID
    if 'gender' in df_group.columns and df_group['gender'].notna().any():
        men_ids = sorted(df_group[df_group['gender'] == 1]['iid'].dropna().astype(int).unique().tolist())
        women_ids = sorted(df_group[df_group['gender'] == 0]['iid'].dropna().astype(int).unique().tolist())
    else:
        men_ids = sorted(df_group['iid'].dropna().astype(int).unique().tolist())
        women_ids = sorted(df_group['pid'].dropna().astype(int).unique().tolist())
        # 简单去重
        women_ids = [w for w in women_ids if w not in men_ids]

    # --- 构建男性偏好 ---
    men_prefs = {}
    for m in men_ids:
        # 找到 m 在该组中见过的所有人 (candidates)
        candidates_pids = df_group[df_group['iid'] == m]['pid'].dropna().unique().astype(int)
        
        pref_list = []
        for p in candidates_pids:
            p = int(p)
            # 直接查 score_dict (0-100分)
            s = score_dict.get((m, p))
            if s is not None:
                pref_list.append((p, s))
        
        # 插入自己 (单身选项，分数为 s0)
        pref_list.append((m, s0))
        
        # 按分数降序排序
        pref_list.sort(key=lambda x: x[1], reverse=True)
        # 只保留 ID 用于 GS 算法
        men_prefs[m] = [x[0] for x in pref_list]

    # --- 构建女性偏好 ---
    women_prefs = {}
    for w in women_ids:
        # 找到 w 在该组中见过的所有人 (处理双向/单向数据)
        candidates_pids = df_group[df_group['iid'] == w]['pid'].dropna().unique().astype(int).tolist()
        if not candidates_pids:
            candidates_pids = df_group[df_group['pid'] == w]['iid'].dropna().unique().astype(int).tolist()
            
        pref_list = []
        for p in candidates_pids:
            p = int(p)
            # 查 score_dict (w 对 p 的打分)
            s = score_dict.get((w, p))
            if s is not None:
                pref_list.append((p, s))
                
        # 插入自己
        pref_list.append((w, s0))
        
        # 排序
        pref_list.sort(key=lambda x: x[1], reverse=True)
        women_prefs[w] = [x[0] for x in pref_list]
        
    return men_prefs, women_prefs, men_ids, women_ids

# ---------- 期望阻塞对 ----------
def _expected_numbp_for_matching(matching: Dict[int, Any],
                                 men_ids: List[int], women_ids: List[int],
                                 score_dict: Dict[Tuple[int, int], float],
                                 beta0: float, lam: float, s0: float,
                                 truncate: bool) -> float:
    """
    计算 E[BP]。
    :param truncate: 是否截断低于 s0 的交换意愿 (Human=True, AI=False)
    """
    total = 0.0
    for m in men_ids:
        # 获取 m 的对象
        m_partner = matching.get(m)
        
        # 处理单身状态: None, "rejected", 或自己
        if m_partner in [None, "rejected", m]:
            m_is_single = True
            val_m_cur = s0
        else:
            m_is_single = False
            # 查分 (0-100)
            val_m_cur = score_dict.get((m, int(m_partner)), s0)

        for w in women_ids:
            # 如果 m 已经和 w 匹配，跳过
            if not m_is_single and m_partner == w:
                continue
            
            # 1. m -> w 意愿
            s_mw = score_dict.get((m, w))
            p_m = _p_switch(s_mw, val_m_cur, beta0, lam, s0, truncate_below_s0=truncate)

            # 2. w -> m 意愿
            w_partner = matching.get(w)
            if w_partner in [None, "rejected", w]:
                val_w_cur = s0
            else:
                val_w_cur = score_dict.get((w, int(w_partner)), s0)
            
            s_wm = score_dict.get((w, m))
            p_w = _p_switch(s_wm, val_w_cur, beta0, lam, s0, truncate_below_s0=truncate)

            # 3. 累加
            total += p_m * p_w
            
    return float(total)

# ---------- 辅助函数 ----------
def _extract_group_id(fname: str) -> Optional[int]:
    b = os.path.basename(fname)
    m = re.search(r"group\D*(\d+)", b, re.IGNORECASE)
    if m: return int(m.group(1))
    nums = re.findall(r"(\d+)", b)
    return int(nums[-1]) if nums else None

def _symmetrize_matching(m: Dict[Any, Any]) -> Dict[int, Any]:
    out: Dict[int, Any] = {}
    for k, v in m.items():
        try:
            ki = int(k)
        except: continue
        
        if v in ["rejected", None]:
            out[ki] = ki # 统一用 self_id 表示单身
        else:
            try:
                vi = int(v)
                if vi == ki:
                    out[ki] = ki
                else:
                    out[ki] = vi
                    out[vi] = ki
            except:
                out[ki] = ki
    return out

def load_unified_params(json_path: str) -> Tuple[float, float, float]:
    with open(json_path, "r", encoding="utf-8") as f:
        pack = json.load(f)
    return float(pack["beta0_taken"]), float(pack["lambda_taken"]), float(pack["s0"])

def load_ai_matchings_from_dir(directory: str) -> List[Tuple[int, Dict[int, Any]]]:
    files = sorted(glob.glob(os.path.join(directory, "*.json")))
    pairs = []
    for fp in files:
        gid = _extract_group_id(fp)
        if gid is None: continue
        try:
            with open(fp, "r") as f:
                data = json.load(f)
            pairs.append((gid, _symmetrize_matching(data)))
        except: pass
    pairs.sort(key=lambda x: x[0])
    return pairs

# ================= 主流程 =================
def main():
    beta0, lam, s0 = load_unified_params(PARAMS_JSON)
    print(f"[Params] beta0={beta0:.4f}, lambda={lam:.4f}, s0={s0:.4f}")

    # 读取分数 (load_data 已更新为返回 0-100 的加权分)
    src_excel  = config["source_data_path"]
    df_source  = pd.read_excel(src_excel)
    score_dict = load_source_scores(src_excel)
    
    if score_dict is None:
        print("Error: score_dict is None. Check load_data.py.")
        return

    # 读取 AI 匹配
    ai_pairs = load_ai_matchings_from_dir(MATCH_DIR)
    ai_pairs = [p for p in ai_pairs if 1 <= p[0] <= 21]
    
    rows = []
    print(f"Processing {len(ai_pairs)} groups...")
    
    for gid, ai_matching in ai_pairs:
        g = df_source[df_source['group'] == gid].copy()
        if g.empty: 
            print(f"Group {gid}: No source data found.")
            continue

        # --- 1. 构建偏好列表 ---
        # 此时 score_dict 和 s0 都是 0-100 量级，排序正确
        men_prefs, women_prefs, men_ids, women_ids = _build_objective_prefs_with_self(g, score_dict, s0)

        # --- 2. 运行改版 GS (Human) ---
        human_matching = classic_gale_shapley_matcher(men_prefs, women_prefs)
        
        # --- 3. 计算 E[BP] ---
        # Human: 开启 truncate (分数 < s0 则 P=0)
        e_hu = _expected_numbp_for_matching(human_matching, men_ids, women_ids, score_dict, beta0, lam, s0, truncate=True)
        
        # AI: 关闭 truncate (保留非理性/小概率交换)
        e_ai = _expected_numbp_for_matching(ai_matching, men_ids, women_ids, score_dict, beta0, lam, s0, truncate=False)

        rows.append({"group": gid, "Ebp_AI": e_ai, "Ebp_Human": e_hu})

    # 输出
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df_out = pd.DataFrame(rows).sort_values("group").reset_index(drop=True)
    df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    
    print(f"Saved to {OUTPUT_CSV}")
    if not df_out.empty:
        print(df_out.to_string(index=False))
        better = df_out[df_out["Ebp_AI"] < df_out["Ebp_Human"]]["group"].tolist()
        print(f"\nGroups where AI is more stable (Ebp_AI < Ebp_Human): {better}")

if __name__ == "__main__":
    main()