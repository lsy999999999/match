# -*- coding: utf-8 -*-
"""
compute_stability_from_unified_json.py

功能：
  - 读取 unified_params JSON（包含 beta0, lambda, s0）
  - 遍历每一组、每一轮(Run)的 AI 匹配结果
  - 计算 AI 与 Human(客观加权GS) 的期望阻塞对 E[numbp]
  - 输出包含 (Group, Run) 维度的详细 CSV
  - 明确列出 Human E[bp] > AI E[bp] (即 AI 表现更好) 的场次

修改说明：
  - 内置了多轮(Run)文件加载逻辑，不再依赖 load_data.py 中的 load_matchings_from_json。
  - 自动检测 config 模板是否包含 {run_id}，兼容单轮和多轮数据。
"""

import os, sys, json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ======== 按需硬编码 ========
REPO_ROOT   = "/home/lsy/match"
# 请确保 config.py 中有此 key，且 template 包含 {run_id} (如果是多轮数据)
# 例如: "gpt4_en_makeup" 或你之前用的 "gpt4_en_fitting"
MODEL_KEY   = "gpt4_en_makeup" 
PARAMS_JSON = "/home/lsy/match/bahavior_simul/1009analysis/unified_params_gpt4_en_fitting.json"
OUTPUT_CSV  = "/home/lsy/match/bahavior_simul/1009analysis/different_analysis/Ebp_from_unified_json_en_makeup.csv"
# =================================

for p in [REPO_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from config import config
from load_data import load_source_scores
from gale_shapley_classic import classic_gale_shapley_matcher

# ---- 核心计算函数 (保持原逻辑) ----
def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))

def _p_switch(score_new: Optional[float], score_cur: Optional[float],
              beta0: float, lam: float, s0: float) -> float:
    if score_new is None: return 0.0
    if score_cur is None: score_cur = s0
    return float(_sigmoid(beta0 + lam * (score_new - score_cur)))

def _build_objective_prefs(df_group: pd.DataFrame) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    dims = ['attractive', 'sincere', 'intelligence', 'funny', 'ambition', 'shared_interests']
    score_cols = [f'{d}_partner' for d in dims]
    imp_cols   = [f'{d}_important' for d in dims]

    df = df_group.copy()
    ws = np.zeros(len(df))
    for i in range(len(dims)):
        ws += df[score_cols[i]].fillna(0).astype(float) * df[imp_cols[i]].fillna(0).astype(float)
    df['weighted_score'] = ws / 100.0

    men_ids   = sorted(df[df['gender'] == 1]['iid'].astype(int).unique().tolist())
    women_ids = sorted(df[df['gender'] == 0]['iid'].astype(int).unique().tolist())

    men_prefs = {m: df[df['iid'] == m].sort_values('weighted_score', ascending=False)['pid'].astype(int).tolist()
                 for m in men_ids}
    women_prefs = {w: df[df['pid'] == w].sort_values('weighted_score', ascending=False)['iid'].astype(int).tolist()
                   for w in women_ids}
    return men_prefs, women_prefs

def _expected_numbp_for_matching(matching: Dict[int, Any],
                                 men_ids: List[int], women_ids: List[int],
                                 score_dict: Dict[Tuple[int, int], float],
                                 beta0: float, lam: float, s0: float) -> float:
    total = 0.0
    for m in men_ids:
        for w in women_ids:
            if matching.get(m) == w:
                continue
            
            # m 的换伴概率
            m_partner = matching.get(m)
            s_mw   = score_dict.get((m, w))
            s_mcur = None if (m_partner in [None, "rejected"]) else score_dict.get((m, int(m_partner)))
            p1 = _p_switch(s_mw, s_mcur, beta0, lam, s0)

            # w 的换伴概率
            w_partner = matching.get(w)
            s_wm   = score_dict.get((w, m))
            s_wcur = None if (w_partner in [None, "rejected"]) else score_dict.get((w, int(w_partner)))
            p2 = _p_switch(s_wm, s_wcur, beta0, lam, s0)

            total += p1 * p2
    return float(total)

def load_unified_params(json_path: str) -> Tuple[float, float, float]:
    with open(json_path, "r", encoding="utf-8") as f:
        pack = json.load(f)
    b0  = float(pack["beta0_taken"])
    lam = float(pack["lambda_taken"])
    s0  = pack.get("s0", None)
    if s0 is None:
        b0_single = pack.get("beta0_single", None)
        if (b0_single is None) or lam == 0.0:
            raise ValueError("JSON Error: missing s0.")
        s0 = (b0 - float(b0_single)) / lam
    return float(b0), float(lam), float(s0)

# ---- 新增：支持多轮加载的迭代器 ----
def iter_group_run_matchings(model_key: str):
    """
    生成器：yield (group_id, run_id, matching_dict)
    自动处理 {run_id} 模板。
    """
    if model_key not in config:
        print(f"Config key '{model_key}' not found.")
        return

    m_conf = config[model_key]
    base_path = m_conf.get("base_path", "")
    json_tpl  = m_conf.get("json_template", "")
    num_groups = m_conf.get("num_groups", 0)
    
    # 简单的容错：如果没有 run_id，假设只有 run 1
    has_run = "{run_id}" in json_tpl
    
    for gid in range(1, num_groups + 1):
        if not has_run:
            # 单轮模式 (旧数据)
            fpath = os.path.join(base_path, json_tpl.format(group_id=gid))
            if os.path.exists(fpath):
                try:
                    with open(fpath, 'r', encoding='utf-8') as f:
                        yield gid, 1, json.load(f)
                except Exception as e:
                    print(f"Error reading {fpath}: {e}")
            continue
            
        # 多轮模式
        # 尝试读取 run 1 到 100 (或者直到文件不存在)
        # 如果 config 中有 num_runs 可以用，这里用 while exist 更通用
        run = 1
        while True:
            # 防止死循环，设置一个上限，比如 100
            if run > 100: 
                break
                
            fname = json_tpl.format(group_id=gid, run_id=run)
            fpath = os.path.join(base_path, fname)
            
            if not os.path.exists(fpath):
                if run == 1:
                    pass # 第1个就不存在，可能该组数据缺失
                else:
                    pass # 之前的 run 存在，这个不存在，说明跑完了
                break # 该组结束，进入下一组
            
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    yield gid, run, json.load(f)
            except Exception as e:
                print(f"Error reading {fpath}: {e}")
            
            run += 1

def main():
    beta0, lam, s0 = load_unified_params(PARAMS_JSON)
    print(f"Params: beta0={beta0:.4f}, lambda={lam:.4f}, s0={s0:.4f}")

    # 加载源数据分数
    src_excel  = config["source_data_path"]
    df_source  = pd.read_excel(src_excel)
    score_dict = load_source_scores(src_excel)
    if score_dict is None:
        return

    rows = []
    
    print(f"Scanning matchings for model: {MODEL_KEY}...")
    
    # 使用新迭代器遍历所有 (Group, Run)
    for gid, rid, ai_matching in iter_group_run_matchings(MODEL_KEY):
        # 1. 规范化 AI Matching
        ai_matching = {int(k): (int(v) if str(v).isdigit() else v) for k, v in ai_matching.items()}

        # 2. 获取该组源数据
        g = df_source[df_source['group'] == gid].copy()
        if g.empty: continue
        
        men_ids   = sorted(g[g['gender'] == 1]['iid'].astype(int).unique().tolist())
        women_ids = sorted(g[g['gender'] == 0]['iid'].astype(int).unique().tolist())

        # 3. 计算 AI E[bp]
        e_ai = _expected_numbp_for_matching(ai_matching, men_ids, women_ids, score_dict, beta0, lam, s0)

        # 4. 计算 Human E[bp] (每组只算一次，但为了对齐每一行都放进去)
        men_prefs, women_prefs = _build_objective_prefs(g)
        human_matching = classic_gale_shapley_matcher(men_prefs, women_prefs)
        e_hu = _expected_numbp_for_matching(human_matching, men_ids, women_ids, score_dict, beta0, lam, s0)
        
        # 5. 记录
        # Human > AI 意味着 AI 的 E[bp] 更小，AI 更稳定
        is_human_gt_ai = (e_hu > e_ai) 
        
        rows.append({
            "group": gid, 
            "run": rid,
            "Ebp_AI": e_ai, 
            "Ebp_Human": e_hu,
            "Human_gt_AI": is_human_gt_ai, # 这是一个布尔标记
            "Diff_Human_minus_AI": e_hu - e_ai
        })

    # 保存 CSV
    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_CSV)), exist_ok=True)
    df_out = pd.DataFrame(rows)
    if df_out.empty:
        print("No data found. Please check paths and config.")
        return

    df_out = df_out.sort_values(["group", "run"]).reset_index(drop=True)
    df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\nSaved detailed results to -> {OUTPUT_CSV}")

    # ---- 终端输出摘要 ----
    print(f"\n{'='*20} Summary {'='*20}")
    print(f"Total entries (Group x Run): {len(df_out)}")
    print(f"Average Ebp_AI   : {df_out['Ebp_AI'].mean():.4f}")
    print(f"Average Ebp_Human: {df_out['Ebp_Human'].mean():.4f}")
    
    # 筛选出 Human > AI 的情况 (即你要求的“大于 AI 结果”的部分)
    better_cases = df_out[df_out["Human_gt_AI"] == True]
    
    print(f"\n[Cases where Human_Result > AI_Result (meaning AI is MORE stable)]")
    print(f"Count: {len(better_cases)} / {len(df_out)}")
    
    if not better_cases.empty:
        print("\nList of (Group, Run) where Human > AI:")
        # 为了不刷屏，如果是列表太长，可以只打印前20个或者打印特定格式
        for idx, row in better_cases.iterrows():
            print(f"  Group {int(row['group'])}, Run {int(row['run'])}: Human({row['Ebp_Human']:.2f}) > AI({row['Ebp_AI']:.2f}) [Diff: {row['Diff_Human_minus_AI']:.2f}]")
    else:
        print("  None.")

if __name__ == "__main__":
    main()