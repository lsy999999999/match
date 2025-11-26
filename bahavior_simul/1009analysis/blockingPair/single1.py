# -*- coding: utf-8 -*-
"""
compute_expected_blocking_pairs_with_single.py

核心：
- P_switch = sigmoid(beta0 + lambda * (S_new - S_cur))
- 单身时：S_cur := S0
- E[#bp] = sum_{(m,w) not matched} P_switch(m->w) * P_switch(w->m)

注意：
- 这里的 score_dict 必须能查到 (agent, candidate)->score
- matching 会被补全：每个 men_ids / women_ids 都有键（单身为 None）
"""

import os, sys, re, json, glob
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ================= 配置路径 =================
# 参数文件（包含 beta0_taken, lambda_taken, s0）
PARAMS_JSON = "/home/lsy/match/bahavior_simul/1009analysis/unified_params_gpt4_en_fitting.json"
# AI 匹配结果文件夹
MATCH_DIR   = "/home/lsy/match/bahavior_simul/0627_gpt4_eng"
# 结果输出路径
OUTPUT_CSV  = "/home/lsy/match/bahavior_simul/1009analysis/single_add/Ebp_with_S0_logic_0627_gpt_en.csv"
# ===========================================

REPO_ROOT = "/home/lsy/match"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config import config
from load_data import load_source_scores
from gale_shapley_classic import classic_gale_shapley_matcher

# ---------- 数值稳定 sigmoid ----------
def sigmoid(x: float) -> float:
    # 标量版，够用
    if x >= 0:
        return float(1.0 / (1.0 + np.exp(-x)))
    ex = float(np.exp(x))
    return float(ex / (1.0 + ex))

# ---------- 读取拟合参数 ----------
def load_unified_params(json_path: str) -> Tuple[float, float, float]:
    with open(json_path, "r", encoding="utf-8") as f:
        pack = json.load(f)

    beta0 = float(pack["beta0_taken"])
    lam   = float(pack["lambda_taken"])

    # 你已经在拟合脚本里把 s0 存进 json 了：直接读即可
    if "s0" not in pack:
        raise ValueError("unified params JSON 缺少 s0（单身基准分）。")
    s0 = float(pack["s0"])
    return beta0, lam, s0

# ---------- 从目录读 AI matching ----------
def _extract_group_id(fname: str) -> Optional[int]:
    b = os.path.basename(fname)
    m = re.search(r"group\s*([0-9]+)", b, re.IGNORECASE)
    if m:
        return int(m.group(1))
    nums = re.findall(r"([0-9]+)", b)
    return int(nums[-1]) if nums else None

def symmetrize_matching(raw: Dict[Any, Any]) -> Dict[int, Any]:
    """
    把 {a:b} 补成 {a:b, b:a}。
    若 value == "rejected" 视为单身标记之一。
    """
    out: Dict[int, Any] = {}
    for k, v in raw.items():
        try:
            a = int(k)
        except Exception:
            continue

        if isinstance(v, str) and v == "rejected":
            out[a] = None
            continue

        try:
            b = int(v)
        except Exception:
            out[a] = None
            continue

        out[a] = b
        out[b] = a
    return out

def load_ai_matchings_from_dir(directory: str) -> List[Tuple[int, Dict[int, Any]]]:
    files = sorted(glob.glob(os.path.join(directory, "*.json")))
    pairs: List[Tuple[int, Dict[int, Any]]] = []
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            gid = _extract_group_id(fp)
            if gid is None:
                continue

            raw = {k: v for k, v in data.items()}
            pairs.append((gid, symmetrize_matching(raw)))
        except Exception as e:
            print(f"[Warn] skip {fp}: {e}")
    pairs.sort(key=lambda x: x[0])
    return pairs

# ---------- 偏好构造（Human：客观六维×重要性） ----------
def build_objective_prefs(df_group: pd.DataFrame) -> Tuple[Dict[int, List[int]], Dict[int, List[int]], List[int], List[int]]:
    dims = ['attractive','sincere','intelligence','funny','ambition','shared_interests']
    score_cols = [f'{d}_partner' for d in dims]
    imp_cols   = [f'{d}_important' for d in dims]

    df = df_group.copy()
    ws = np.zeros(len(df))
    for i in range(len(dims)):
        ws += df[score_cols[i]].fillna(0).astype(float) * df[imp_cols[i]].fillna(0).astype(float)
    df['weighted_score'] = ws / 100.0

    if 'gender' not in df.columns or not df['gender'].notna().any():
        # 没 gender 的情况下你原脚本用 pid 当 women_ids，会导致 (w,m) 评分查不到而低估 bp；
        # 这里我们仍然退化，但会打印提醒。
        men_ids = sorted(df['iid'].dropna().astype(int).unique().tolist())
        women_ids = sorted(df['pid'].dropna().astype(int).unique().tolist())
        print("[Warn] gender 缺失：women_ids 用 pid 退化，w->m 分数可能查不到，E[bp] 可能被低估。")
    else:
        men_ids = sorted(df[df['gender'] == 1]['iid'].dropna().astype(int).unique().tolist())
        women_ids = sorted(df[df['gender'] == 0]['iid'].dropna().astype(int).unique().tolist())

    men_prefs = {
        m: df[df['iid'] == m].sort_values('weighted_score', ascending=False)['pid'].astype(int).tolist()
        for m in men_ids
    }
    women_prefs = {
        w: df[df['pid'] == w].sort_values('weighted_score', ascending=False)['iid'].astype(int).tolist()
        for w in women_ids
    }
    return men_prefs, women_prefs, men_ids, women_ids

# ---------- 单身纳入：当前分数 fallback 到 S0 ----------
def p_switch(score_new: Optional[float], score_cur: Optional[float],
             beta0: float, lam: float, s0: float) -> float:
    """
    score_cur=None 表示单身/未匹配 -> 用 s0
    score_new=None 表示这条边没有分数 -> 认为不会换（0）
    """
    if score_new is None:
        return 0.0
    if score_cur is None:
        score_cur = s0
    return sigmoid(beta0 + lam * (float(score_new) - float(score_cur)))

def complete_matching(matching: Dict[int, Any], men_ids: List[int], women_ids: List[int]) -> Dict[int, Optional[int]]:
    """
    保证每个人都有键：单身为 None
    """
    out: Dict[int, Optional[int]] = {}
    for x in men_ids + women_ids:
        v = matching.get(x, None)
        if v is None:
            out[x] = None
        else:
            try:
                out[x] = int(v)
            except Exception:
                out[x] = None
    return out

def expected_blocking_pairs(matching: Dict[int, Optional[int]],
                            men_ids: List[int], women_ids: List[int],
                            score_dict: Dict[Tuple[int, int], float],
                            beta0: float, lam: float, s0: float) -> float:
    total = 0.0
    for m in men_ids:
        m_partner = matching.get(m, None)
        s_mcur = score_dict.get((m, m_partner)) if m_partner is not None else None

        for w in women_ids:
            if matching.get(m, None) == w:
                continue

            # m 侧：m 用 (m,w) 与 (m,m_partner/单身) 比
            s_mw = score_dict.get((m, w))
            p_m = p_switch(s_mw, s_mcur, beta0, lam, s0)

            # w 侧：w 用 (w,m) 与 (w,w_partner/单身) 比
            w_partner = matching.get(w, None)
            s_wcur = score_dict.get((w, w_partner)) if w_partner is not None else None
            s_wm = score_dict.get((w, m))
            p_w = p_switch(s_wm, s_wcur, beta0, lam, s0)

            total += p_m * p_w

    return float(total)

def main():
    beta0, lam, s0 = load_unified_params(PARAMS_JSON)
    print(f"[Unified Params] beta0={beta0:.6f}, lambda={lam:.6f}, S0={s0:.6f}")

    # Excel -> (iid,pid)->score
    src_excel = config["source_data_path"]
    df_source = pd.read_excel(src_excel)
    score_dict = load_source_scores(src_excel)
    if score_dict is None:
        raise RuntimeError("无法从 Excel 计算 (iid,pid)->total_score。")

    ai_pairs = load_ai_matchings_from_dir(MATCH_DIR)
    ai_pairs = [p for p in ai_pairs if 1 <= p[0] <= 21]
    ai_pairs.sort(key=lambda x: x[0])
    print(f"[AI matchings] {MATCH_DIR} -> {len(ai_pairs)} files (1..21)")

    rows = []
    for gid, ai_m in ai_pairs:
        g = df_source[df_source["group"] == gid].copy()
        if g.empty:
            print(f"[Warn] group {gid} not in Excel; skip.")
            continue

        men_prefs, women_prefs, men_ids, women_ids = build_objective_prefs(g)
        if not men_ids or not women_ids:
            print(f"[Warn] group {gid} men/women empty; skip.")
            continue

        # Human baseline (GS)
        human_raw = classic_gale_shapley_matcher(men_prefs, women_prefs)
        human_m = complete_matching(symmetrize_matching(human_raw), men_ids, women_ids)

        # AI matching（补全到所有人）
        ai_m = complete_matching(ai_m, men_ids, women_ids)

        e_ai = expected_blocking_pairs(ai_m, men_ids, women_ids, score_dict, beta0, lam, s0)
        e_hu = expected_blocking_pairs(human_m, men_ids, women_ids, score_dict, beta0, lam, s0)

        rows.append({"group": gid, "Ebp_AI": e_ai, "Ebp_Human": e_hu})

    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_CSV)), exist_ok=True)
    out = pd.DataFrame(rows).sort_values("group").reset_index(drop=True)
    out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"Saved CSV -> {OUTPUT_CSV}")

    if not out.empty:
        print(f"[Summary] groups={len(out)} | Mean_AI={out.Ebp_AI.mean():.4f} | Mean_Human={out.Ebp_Human.mean():.4f}")
        better = out[out["Ebp_AI"] < out["Ebp_Human"]]["group"].tolist()
        print(f"Groups where AI E[bp] < Human: {better}")

if __name__ == "__main__":
    main()
