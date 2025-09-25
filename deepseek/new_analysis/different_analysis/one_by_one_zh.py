# -*- coding: utf-8 -*-
import os, sys
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

REPO_ROOT  = "/home/lsy/match"
MODEL_KEY  = "deepseek_zh_fitting"

BETA0_TAKEN   = -0.8906
LAMBDA_TAKEN  =  0.0646
S0            = -17.41
BETA0_SINGLE  = 0.2335

OUTPUT_CSV = "/home/lsy/match/deepseek/new_analysis/different_analysis/Ebp_by_group_zh.csv"

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config import config
from load_data import load_matchings_from_json, load_source_scores
from gale_shapley_classic import classic_gale_shapley_matcher

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))

def _p_switch(score_new: Optional[float], score_cur: Optional[float],
              beta0: float, lam: float, s0: float) -> float:
    if score_new is None:
        return 0.0
    if score_cur is None:
        score_cur = s0
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

    men_ids   = sorted(df[df['gender'] == 1]['iid'].unique().tolist())
    # 修复点①：女性集合按 gender==0 的 iid
    women_ids = sorted(df[df['gender'] == 0]['iid'].unique().tolist())

    men_prefs = {m: df[df['iid'] == m].sort_values('weighted_score', ascending=False)['pid'].tolist()
                 for m in men_ids}
    women_prefs = {w: df[df['pid'] == w].sort_values('weighted_score', ascending=False)['iid'].tolist()
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
            m_partner = matching.get(m)
            s_mw   = score_dict.get((m, w))
            s_mcur = None if (m_partner in [None, "rejected"]) else score_dict.get((m, int(m_partner)))
            p1 = _p_switch(s_mw, s_mcur, beta0, lam, s0)

            w_partner = matching.get(w)
            s_wm   = score_dict.get((w, m))
            s_wcur = None if (w_partner in [None, "rejected"]) else score_dict.get((w, int(w_partner)))
            p2 = _p_switch(s_wm, s_wcur, beta0, lam, s0)

            total += p1 * p2
    return float(total)

def main():
    if S0 is None:
        if LAMBDA_TAKEN == 0:
            raise RuntimeError("λ 不能为 0；若没填 S0，请提供 β0_single 以反推 S0。")
        s0 = (BETA0_TAKEN - BETA0_SINGLE) / LAMBDA_TAKEN
    else:
        s0 = float(S0)
    beta0 = float(BETA0_TAKEN)
    lam   = float(LAMBDA_TAKEN)
    print(f"[Unified Params] beta0={beta0:.6f}, lambda={lam:.6f}, s0={s0:.6f}")

    src_excel  = config["source_data_path"]
    df_source  = pd.read_excel(src_excel)
    score_dict = load_source_scores(src_excel)
    if score_dict is None:
        raise RuntimeError("无法从 Excel 计算 (iid,pid)->总分。")

    ai_matchings = load_matchings_from_json(MODEL_KEY)

    rows = []
    for gid, ai_matching in enumerate(ai_matchings, start=1):
        # 修复点②：AI 匹配的键值做一次 int 规范，避免 '1'/'rejected' 混用
        ai_matching = {int(k): (int(v) if str(v).isdigit() else v) for k, v in ai_matching.items()}

        g = df_source[df_source['group'] == gid].copy()
        if g.empty:
            continue
        men_ids   = sorted(g[g['gender'] == 1]['iid'].unique().tolist())
        # 修复点③：女性集合按 gender==0 的 iid
        women_ids = sorted(g[g['gender'] == 0]['iid'].unique().tolist())
        if not men_ids or not women_ids:
            continue

        e_ai = _expected_numbp_for_matching(ai_matching, men_ids, women_ids, score_dict, beta0, lam, s0)

        men_prefs, women_prefs = _build_objective_prefs(g)
        human_matching = classic_gale_shapley_matcher(men_prefs, women_prefs)
        e_hu = _expected_numbp_for_matching(human_matching, men_ids, women_ids, score_dict, beta0, lam, s0)

        rows.append({"group": gid, "Ebp_AI": e_ai, "Ebp_Human": e_hu})

    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_CSV)), exist_ok=True)
    df_out = pd.DataFrame(rows).sort_values("group")
    df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    if not df_out.empty:
        m_ai = df_out["Ebp_AI"].mean()
        m_hu = df_out["Ebp_Human"].mean()
        print(f"[{MODEL_KEY}] groups={len(df_out)} | Mean_AI={m_ai:.4f} | Mean_Human={m_hu:.4f}")
        print(f"Saved: {OUTPUT_CSV}")
    else:
        print("没有任何组结果，请检查参数与数据路径。")

    
    # 列出 AI 更稳的组
    better = df_out[df_out["Ebp_AI"] < df_out["Ebp_Human"]]["group"].tolist()
    print(f"Groups where AI E[numbp] < Human: {better}")
if __name__ == "__main__":
    main()
