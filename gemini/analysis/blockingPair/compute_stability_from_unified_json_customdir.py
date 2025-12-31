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
PARAMS_JSON = "/home/lsy/match/gemini/analysis/unified_params_gemini_en_fitting.json"
MATCH_DIR   = "/home/lsy/match/gemini/0725_gemini_eng"
OUTPUT_CSV  = "/home/lsy/match/gemini/analysis/different_analysis/Ebp_from_unified_json_0725_en_21.csv"
# =================================

REPO_ROOT = "/home/lsy/match"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config import config
from load_data import load_source_scores
from gale_shapley_classic import classic_gale_shapley_matcher

# 单身状态对应的客观分数（由 unified_params 的 s0 设置）
SINGLE_SCORE: Optional[float] = None

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
    """
    构造 men_prefs / women_prefs：
    - 先按六维客观打分 × 重要性计算 weighted_score；
    - 再为每个人在自己的偏好表中插入一个 “single/self_id” 选项，分数视为 SINGLE_SCORE，
      并根据 SINGLE_SCORE 在该人的客观分数序列中的位置插入。
    这样 Gale-Shapley 在运行时就可以把 self_id 视作一个普通候选，且位置由 S 决定。
    """
    global SINGLE_SCORE
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

    def _insert_self_option(sorted_ids: List[int], sorted_scores: List[float], self_id: int) -> List[int]:
        """
        给定该人的候选列表及对应分数（按分数降序），
        在 SINGLE_SCORE 对应的位置插入 self_id。
        若 SINGLE_SCORE 为 None，则保持原顺序。
        """
        if SINGLE_SCORE is None:
            return list(sorted_ids)

        s0 = float(SINGLE_SCORE)
        # 找到第一个 score < s0 的位置，在此处插入 self_id
        pos = 0
        while pos < len(sorted_scores) and float(sorted_scores[pos]) >= s0:
            pos += 1
        return list(sorted_ids[:pos]) + [self_id] + list(sorted_ids[pos:])

    # men 的偏好：对每个 m，用 df 中 (iid==m) 的记录构造他的候选 woman 列表
    men_prefs: Dict[int, List[int]] = {}
    for m in men_ids:
        sub = df[df['iid'] == m].copy()
        if sub.empty:
            # 没有任何记录：如果有 SINGLE_SCORE，就至少把自己放进去
            men_prefs[m] = [m] if SINGLE_SCORE is not None else []
            continue
        sub = sub.sort_values('weighted_score', ascending=False)
        cand_ids    = sub['pid'].astype(int).tolist()
        cand_scores = sub['weighted_score'].astype(float).tolist()
        men_prefs[m] = _insert_self_option(cand_ids, cand_scores, m)

    # women 的偏好：对每个 w，用 df 中 (pid==w) 的记录构造她的候选 man 列表
    women_prefs: Dict[int, List[int]] = {}
    for w in women_ids:
        sub = df[df['pid'] == w].copy()
        if sub.empty:
            women_prefs[w] = [w] if SINGLE_SCORE is not None else []
            continue
        sub = sub.sort_values('weighted_score', ascending=False)
        cand_ids    = sub['iid'].astype(int).tolist()
        cand_scores = sub['weighted_score'].astype(float).tolist()
        women_prefs[w] = _insert_self_option(cand_ids, cand_scores, w)

    return men_prefs, women_prefs

# ---------- 期望阻塞对 ----------
def _expected_numbp_for_matching(matching: Dict[int, Any],
                                 men_ids: List[int], women_ids: List[int],
                                 score_dict: Dict[Tuple[int, int], float],
                                 beta0: float, lam: float, s0: float,
                                 truncate_by_s0: bool = False) -> float:
    """
    计算给定 matching 的期望阻塞对数量 E[#bp]。

    参数：
    - truncate_by_s0=True 时（用于 Human/GS）：
        若某一对 (m,w) 中，任一方向的客观分数 score(m,w) 或 score(w,m) < s0，
        则这对直接跳过，不计入 E[#bp]（“低于单身分数的候选不视为真正的 blocking pair”）。
      truncate_by_s0=False 时（用于 AI），保留所有分数（不截断）。
    - 对于匹配到 self_id 的人（matching[x] == x），视为单身，其当前分数 s_cur = s0
      （在 _p_switch 中通过 s_cur=None→s0 实现）。
    """
    total = 0.0
    for m in men_ids:
        for w in women_ids:
            # 已经匹配在一起的 (m,w) 不构成阻塞对
            if matching.get(m) == w:
                continue

            # 互评分数
            s_mw = score_dict.get((m, w))
            s_wm = score_dict.get((w, m))

            # Human 端截断：任一方向 score < s0，则该对直接跳过
            if truncate_by_s0:
                if s_mw is None or s_wm is None:
                    continue
                if float(s_mw) < float(s0) or float(s_wm) < float(s0):
                    continue

            # m 的当前对象及分数（self_id 或单身都视作 s0）
            m_partner = matching.get(m)
            if m_partner in [None, "rejected"] or m_partner == m:
                s_mcur = None  # 在 _p_switch 中会被替换为 s0
            else:
                s_mcur = score_dict.get((m, int(m_partner)))

            p1 = _p_switch(s_mw, s_mcur, beta0, lam, s0)

            # w 的当前对象及分数（self_id 或单身都视作 s0）
            w_partner = matching.get(w)
            if w_partner in [None, "rejected"] or w_partner == w:
                s_wcur = None
            else:
                s_wcur = score_dict.get((w, int(w_partner)))

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
    global SINGLE_SCORE
    beta0, lam, s0 = load_unified_params(PARAMS_JSON)
    SINGLE_SCORE = s0  # 把拟合出来的 S 存成全局单身分数
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
        # AI：不过滤（不截断），Human(GS)：去除低于单身分数 s0 的候选
        e_ai = _expected_numbp_for_matching(
            ai_matching, men_ids, women_ids, score_dict, beta0, lam, s0,
            truncate_by_s0=False
        )
        e_hu = _expected_numbp_for_matching(
            human_matching, men_ids, women_ids, score_dict, beta0, lam, s0,
            truncate_by_s0=True
        )

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
