# -*- coding: utf-8 -*-
"""
compute_stability_from_unified_json_customdir_21.py

用途：
  - 读取 unified_params JSON（beta0, lambda, s0）
  - 只计算 1..21 组：从指定目录(eg. /home/lsy/match/bahavior_simul/0921_gpt4_eng) 读取 AI 匹配 JSON
  - 与 Human(客观加权 GS) 比较，输出每组 E[numbp] + 概览

注意：
  - 源 Excel 仍来自 config["source_data_path"]（提供 iid/pid 的原始六维分）
  - 匹配（AI/GS）都做“对称化”（m->w 同时写入 w->m），避免将一侧误判为单身
  - women_ids 若性别列不可用，则回退为“该组出现过的 pid 去重”
"""

import os, sys, re, json, glob
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ========= 仅改这 3 行 =========
PARAMS_JSON = "/home/lsy/match/gemini/analysis/unified_params_gemini_zh_fitting.json"
MATCH_DIR   = "/home/lsy/match/gemini/0713_gemini_Chinese"
OUTPUT_CSV  = "/home/lsy/match/gemini/analysis/blockingPair/Ebp_from_unified_json_0713_zh_21.csv"
# =================================

REPO_ROOT = "/home/lsy/match"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config import config
from load_data import load_source_scores
from gale_shapley_classic import classic_gale_shapley_matcher

# ---------- 统一模型 ----------
def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))

def _p_switch(s_new: Optional[float], s_cur: Optional[float],
              beta0: float, lam: float, s0: float) -> float:
    if s_new is None:
        return 0.0
    if s_cur is None:
        s_cur = s0
    return float(_sigmoid(beta0 + lam * (s_new - s_cur)))

# ---------- 偏好构造（客观六维×重要性） ----------
def _build_objective_prefs(df_group: pd.DataFrame) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    dims = ['attractive','sincere','intelligence','funny','ambition','shared_interests']
    score_cols = [f'{d}_partner' for d in dims]
    imp_cols   = [f'{d}_important' for d in dims]

    df = df_group.copy()
    ws = np.zeros(len(df))
    for i in range(len(dims)):
        ws += df[score_cols[i]].fillna(0).astype(float) * df[imp_cols[i]].fillna(0).astype(float)
    df['weighted_score'] = ws / 100.0

    # men：优先 gender==1 的 iid；否则回退为该组 iid 去重
    if 'gender' in df.columns and df['gender'].notna().any():
        men_ids = sorted(df[df['gender'] == 1]['iid'].dropna().astype(int).unique().tolist())
        if not men_ids:
            men_ids = sorted(df['iid'].dropna().astype(int).unique().tolist())
    else:
        men_ids = sorted(df['iid'].dropna().astype(int).unique().tolist())

    # women：优先 gender==0 的 iid；若取不到，回退为该组出现过的 pid 去重
    if 'gender' in df.columns and df['gender'].notna().any():
        women_ids = sorted(df[df['gender'] == 0]['iid'].dropna().astype(int).unique().tolist())
    else:
        women_ids = []
    if not women_ids:
        women_ids = sorted(df['pid'].dropna().astype(int).unique().tolist())

    men_prefs = {
        m: df[df['iid'] == m].sort_values('weighted_score', ascending=False)['pid'].astype(int).tolist()
        for m in men_ids
    }
    women_prefs = {
        w: df[df['pid'] == w].sort_values('weighted_score', ascending=False)['iid'].astype(int).tolist()
        for w in women_ids
    }
    return men_prefs, women_prefs

# ---------- 期望阻塞对 ----------
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

# ---------- 匹配对称化 ----------
def _symmetrize_matching(m: Dict[Any, Any]) -> Dict[int, Any]:
    out: Dict[int, Any] = {}
    for k, v in m.items():
        try:
            ki = int(k)
        except Exception:
            continue
        if isinstance(v, str) and v == "rejected":
            out[ki] = "rejected"
        else:
            try:
                vi = int(v)
            except Exception:
                out[ki] = v
                continue
            out[ki] = vi
            out[vi] = ki
    return out

# ---------- 目录读取 ----------
def _extract_group_id(fname: str) -> Optional[int]:
    b = os.path.basename(fname)
    m = re.search(r"group\s*([0-9]+)", b, re.IGNORECASE)
    if m: return int(m.group(1))
    nums = re.findall(r"([0-9]+)", b)
    return int(nums[-1]) if nums else None

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
            # 规范 + 对称化
            raw = {int(k): (int(v) if str(v).isdigit() else v) for k, v in data.items()}
            pairs.append((gid, _symmetrize_matching(raw)))
        except Exception as e:
            print(f"[Warn] skip {fp}: {e}")
    pairs.sort(key=lambda x: x[0])
    return pairs

# ---------- 参数读取 ----------
def load_unified_params(json_path: str) -> Tuple[float, float, float]:
    with open(json_path, "r", encoding="utf-8") as f:
        pack = json.load(f)
    b0  = float(pack["beta0_taken"])
    lam = float(pack["lambda_taken"])
    s0  = pack.get("s0", None)
    if s0 is None:
        b0_single = pack.get("beta0_single", None)
        if (b0_single is None) or lam == 0.0:
            raise ValueError("JSON 缺少 s0 且无法由 beta0_single/λ 推回。")
        s0 = (b0 - float(b0_single)) / lam
    return float(b0), float(lam), float(s0)

# ================= 主流程 =================
def main():
    beta0, lam, s0 = load_unified_params(PARAMS_JSON)
    print(f"[Unified Params] beta0={beta0:.6f}, lambda={lam:.6f}, s0={s0:.6f}")

    # 源 Excel
    src_excel  = config["source_data_path"]
    df_source  = pd.read_excel(src_excel)
    score_dict = load_source_scores(src_excel)
    if score_dict is None:
        raise RuntimeError("无法从 Excel 计算 (iid,pid)->total_score。")

    # 目录里的 AI 匹配
    ai_pairs = load_ai_matchings_from_dir(MATCH_DIR)
    # 只保留 1..21
    ai_pairs = [p for p in ai_pairs if 1 <= p[0] <= 21]
    ai_pairs.sort(key=lambda x: x[0])
    print(f"[AI matchings from dir] {MATCH_DIR} -> {len(ai_pairs)} files (1..21)")

    rows = []
    for gid, ai_matching in ai_pairs:
        g = df_source[df_source['group'] == gid].copy()
        if g.empty:
            print(f"[Warn] group {gid} not in Excel; skip.")
            continue

        # 偏好与人群
        men_prefs, women_prefs = _build_objective_prefs(g)

        if 'gender' in g.columns and g['gender'].notna().any():
            men_ids = sorted(g[g['gender'] == 1]['iid'].dropna().astype(int).unique().tolist())
            if not men_ids:
                men_ids = sorted(g['iid'].dropna().astype(int).unique().tolist())
            women_ids = sorted(g[g['gender'] == 0]['iid'].dropna().astype(int).unique().tolist())
        else:
            men_ids = sorted(g['iid'].dropna().astype(int).unique().tolist())
            women_ids = []
        if not women_ids:
            women_ids = sorted(g['pid'].dropna().astype(int).unique().tolist())

        if not men_ids or not women_ids:
            print(f"[Warn] group {gid} men/women empty; skip.")
            continue

        # Human 匹配（对称化）
        human_matching_raw = classic_gale_shapley_matcher(men_prefs, women_prefs)
        human_matching = _symmetrize_matching(human_matching_raw)

        # 计算 E[#bp]
        e_ai = _expected_numbp_for_matching(ai_matching, men_ids, women_ids, score_dict, beta0, lam, s0)
        e_hu = _expected_numbp_for_matching(human_matching, men_ids, women_ids, score_dict, beta0, lam, s0)

        rows.append({"group": gid, "Ebp_AI": e_ai, "Ebp_Human": e_hu})

    # 导出
    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_CSV)), exist_ok=True)
    df_out = pd.DataFrame(rows).sort_values("group").reset_index(drop=True)
    df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"Saved CSV -> {OUTPUT_CSV}")

    if not df_out.empty:
        mean_ai = df_out["Ebp_AI"].mean()
        mean_hu = df_out["Ebp_Human"].mean()
        better  = df_out[df_out["Ebp_AI"] < df_out["Ebp_Human"]]["group"].tolist()
        print(f"[Summary] groups={len(df_out)} | Mean_AI={mean_ai:.4f} | Mean_Human={mean_hu:.4f}")
        print(f"Groups where AI E[numbp] < Human: {better}")

if __name__ == "__main__":
    main()
