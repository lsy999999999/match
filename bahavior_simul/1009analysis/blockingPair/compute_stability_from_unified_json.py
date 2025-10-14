# -*- coding: utf-8 -*-
"""
compute_stability_from_unified_json.py

功能：
  - 读取你前面拟合保存的 unified_params JSON（包含 beta0, lambda, s0）
  - 逐组计算 AI 最终匹配 与 Human(客观加权GS) 的期望阻塞对 E[numbp]
  - 输出 CSV + 终端概要（均值/AI更稳的组）

依赖：
  - config.py：源Excel路径和 {model_key} 的JSON路径模板
  - load_data.py：load_matchings_from_json(), load_source_scores()
  - gale_shapley_classic.py：classic_gale_shapley_matcher()
"""

import os, sys, json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ======== 按需硬编码（3处）========
REPO_ROOT   = "/home/lsy/match"
MODEL_KEY   = "gpt4_zh_fitting"
PARAMS_JSON = "/home/lsy/match/bahavior_simul/1009analysis/unified_params_gpt4_zh_fitting.json"
OUTPUT_CSV  = "/home/lsy/match/bahavior_simul/1009analysis/different_analysis/Ebp_from_unified_json_zh.csv"
# =================================

for p in [REPO_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from config import config                             # 源Excel路径/模型路径配置
from load_data import load_matchings_from_json, load_source_scores
from gale_shapley_classic import classic_gale_shapley_matcher

# ---- 统一模型 ----
def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))

def _p_switch(score_new: Optional[float], score_cur: Optional[float],
              beta0: float, lam: float, s0: float) -> float:
    """
    单身：score_cur = s0；有伴侣：score_cur = 当前伴侣得分
    P = σ(β0 + λ*(score_new - score_cur))
    """
    if score_new is None:
        return 0.0
    if score_cur is None:
        score_cur = s0
    return float(_sigmoid(beta0 + lam * (score_new - score_cur)))

def _build_objective_prefs(df_group: pd.DataFrame) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    """
    用客观六维×重要性构造两侧偏好并喂给经典GS（男性求偶版）
    """
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
    """
    E[#bp] = sum_{(m,w) not matched} P_bp(m,w) = sum prob1*prob2
    """
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
    """
    读取你前面保存的 unified_params JSON：
      { "beta0_taken":..., "lambda_taken":..., "s0":... }
    若无 s0 但有 beta0_single，则按 s0 = (beta0_taken - beta0_single)/lambda_taken 推回。
    """
    with open(json_path, "r", encoding="utf-8") as f:
        pack = json.load(f)
    b0  = float(pack["beta0_taken"])
    lam = float(pack["lambda_taken"])
    s0  = pack.get("s0", None)
    if s0 is None:
        b0_single = pack.get("beta0_single", None)
        if (b0_single is None) or lam == 0.0:
            raise ValueError("JSON 中缺少 s0 且无法由 beta0_single/λ 推回。")
        s0 = (b0 - float(b0_single)) / lam
    return float(b0), float(lam), float(s0)

def main():
    beta0, lam, s0 = load_unified_params(PARAMS_JSON)
    print(f"[Unified Params from JSON] beta0={beta0:.6f}, lambda={lam:.6f}, s0={s0:.6f}")

    # 源Excel总分 (iid,pid)->total_score
    src_excel  = config["source_data_path"]
    df_source  = pd.read_excel(src_excel)
    score_dict = load_source_scores(src_excel)
    if score_dict is None:
        raise RuntimeError("无法从 Excel 计算 (iid,pid)->总分。")

    # AI 最终匹配（按 model_key 读取对应目录的 JSON）
    ai_matchings = load_matchings_from_json(MODEL_KEY)

    rows = []
    for gid, ai_matching in enumerate(ai_matchings, start=1):
        # 规范键值：'1'/'rejected' -> int 或 'rejected'
        ai_matching = {int(k): (int(v) if str(v).isdigit() else v) for k, v in ai_matching.items()}

        g = df_source[df_source['group'] == gid].copy()
        if g.empty:
            continue
        men_ids   = sorted(g[g['gender'] == 1]['iid'].astype(int).unique().tolist())
        women_ids = sorted(g[g['gender'] == 0]['iid'].astype(int).unique().tolist())
        if not men_ids or not women_ids:
            continue

        # AI 的 E[#bp]
        e_ai = _expected_numbp_for_matching(ai_matching, men_ids, women_ids, score_dict, beta0, lam, s0)

        # Human 的 E[#bp]（客观加权GS）
        men_prefs, women_prefs = _build_objective_prefs(g)
        human_matching = classic_gale_shapley_matcher(men_prefs, women_prefs)
        e_hu = _expected_numbp_for_matching(human_matching, men_ids, women_ids, score_dict, beta0, lam, s0)

        rows.append({"group": gid, "Ebp_AI": e_ai, "Ebp_Human": e_hu})

    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_CSV)), exist_ok=True)
    df_out = pd.DataFrame(rows).sort_values("group").reset_index(drop=True)
    df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"Saved CSV -> {OUTPUT_CSV}")

    if not df_out.empty:
        mean_ai = df_out["Ebp_AI"].mean()
        mean_hu = df_out["Ebp_Human"].mean()
        better  = df_out[df_out["Ebp_AI"] < df_out["Ebp_Human"]]["group"].tolist()
        print(f"[{MODEL_KEY}] groups={len(df_out)} | Mean_AI={mean_ai:.4f} | Mean_Human={mean_hu:.4f}")
        print(f"Groups where AI E[numbp] < Human: {better}")

if __name__ == "__main__":
    main()
