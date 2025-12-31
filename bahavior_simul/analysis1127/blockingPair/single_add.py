# -*- coding: utf-8 -*-
"""
compute_stability_from_unified_json_customdir_with_S0_filter.py

改动点（只有一个核心改动）：
- 在计算未匹配对 (m,w) 的阻塞对期望贡献前，先做“可接受性过滤”：
    若 score(m,w) < S0 或 score(w,m) < S0，则这对直接跳过，不计入 Ebp。
- 其他逻辑保持不变：p_switch 仍为 sigmoid(beta0 + lambda*(S_new - S_cur)),
  且 S_cur 为单身时用 S0。
"""

import os, sys, re, json, glob
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ================= 配置路径 =================
# 参数文件（包含 beta0_taken, lambda_taken, s0）
PARAMS_JSON = "/home/lsy/match/bahavior_simul/1009analysis/unified_params_gpt4_zh_fitting.json"
# AI 匹配结果文件夹
MATCH_DIR   = "/home/lsy/match/bahavior_simul/1022_gpt_Chinese"
# 结果输出路径
OUTPUT_CSV  = "/home/lsy/match/bahavior_simul/1009analysis/single_choose/Ebp_with_S0_logic_1022_gpt_zh.csv"
# ===========================================


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


# ---------- 期望阻塞对（加入 S0 可接受性过滤） ----------
def _expected_numbp_for_matching(matching: Dict[int, Any],
                                 men_ids: List[int], women_ids: List[int],
                                 score_dict: Dict[Tuple[int, int], float],
                                 beta0: float, lam: float, s0: float) -> float:
    """
    你的新要求：
    - 对未匹配对 (m,w)，如果 score(m,w) < s0 或 score(w,m) < s0，则该对不计入 Ebp。
      （等价于：blocking pair 必须满足双方都“可接受”，不低于单身阈值 S0。）
    """
    total = 0.0
    for m in men_ids:
        for w in women_ids:
            if matching.get(m) == w:
                continue

            # 先取双方互评的分数
            s_mw = score_dict.get((m, w))
            s_wm = score_dict.get((w, m))

            # 若任一方向缺分数，或者任一方向低于 S0：直接跳过（不计入 blocking-pair 期望）
            if (s_mw is None) or (s_wm is None):
                continue
            if (float(s_mw) < float(s0)) or (float(s_wm) < float(s0)):
                continue

            # 通过过滤后，再按原逻辑算 p1*p2
            m_partner = matching.get(m)
            s_mcur = None if (m_partner in [None, "rejected"]) else score_dict.get((m, int(m_partner)))
            p1 = _p_switch(s_mw, s_mcur, beta0, lam, s0)

            w_partner = matching.get(w)
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
    ai_pairs = [p for p in ai_pairs if 1 <= p[0] <= 21]
    ai_pairs.sort(key=lambda x: x[0])
    print(f"[AI matchings from dir] {MATCH_DIR} -> {len(ai_pairs)} files (1..21)")

    rows = []
    for gid, ai_matching in ai_pairs:
        g = df_source[df_source['group'] == gid].copy()
        if g.empty:
            print(f"[Warn] group {gid} not in Excel; skip.")
            continue

        men_prefs, women_prefs = _build_objective_prefs(g)

        # men_ids / women_ids（保持与你原脚本一致的写法）
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

        # 计算 E[#bp]（带 S0 过滤）
        e_ai = _expected_numbp_for_matching(ai_matching, men_ids, women_ids, score_dict, beta0, lam, s0)
        e_hu = _expected_numbp_for_matching(human_matching, men_ids, women_ids, score_dict, beta0, lam, s0)

        rows.append({"group": gid, "Ebp_AI": e_ai, "Ebp_Human": e_hu})

    # 导出
    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_CSV)), exist_ok=True)
    df_out = pd.DataFrame(rows).sort_values("group").reset_index(drop=True)
    df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"Saved CSV -> {OUTPUT_CSV}")

    if not df_out.empty:
        print(f"[Summary] groups={len(df_out)} | Mean_AI={df_out.Ebp_AI.mean():.4f} | Mean_Human={df_out.Ebp_Human.mean():.4f}")

     # === 输出 AI < Human 的组 ===
    better = df_out[df_out["Ebp_AI"] < df_out["Ebp_Human"]].copy()
    if better.empty:
        print("[AI < Human] none")
    else:
        better["gap(H-A)"] = better["Ebp_Human"] - better["Ebp_AI"]
        better = better.sort_values("gap(H-A)", ascending=False)

        groups = better["group"].astype(int).tolist()
        print(f"[AI < Human] count={len(groups)} groups={groups}")

        # 可选：打印每组具体数值（方便你在论文里引用/截图）
        for _, r in better.iterrows():
            print(f"  group {int(r['group'])}: AI={r['Ebp_AI']:.6f}, Human={r['Ebp_Human']:.6f}, gap={r['gap(H-A)']:.6f}")
if __name__ == "__main__":
    main()
