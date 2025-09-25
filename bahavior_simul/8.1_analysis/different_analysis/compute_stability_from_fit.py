# -*- coding: utf-8 -*-
"""
compute_stability_from_fit.py

用途：
  - 通过 --fit-module 指向类似 /match/bahavior_simul/8.1_analysis/S/fit_scalar_logit.py
    的路径，动态加载/读取 (β0_taken, λ_taken, S0)，然后对指定模型键 --model-key
    的每一组匹配分别计算期望阻塞对 E[numbp]。
  - 仅需“改路径”，无需改代码，就能切换不同语言/不同组的 λ/β 计算方式。

兼容策略（按优先级）：
  1) 如果 fit_scalar_logit.py 暴露 get_unified_model_params(model_key)：
       直接调用获取 (β0_taken, λ_taken, S0)；
  2) 若模块内存在常量/变量：
       BETA0_TAKEN / LAMBDA_TAKEN / (S0 或 BETA0_SINGLE) -> 推回 S0；
  3) 若同目录存在 'unified_params.json'：
       读取 {"beta0_taken":..., "lambda_taken":..., "beta0_single":..., "s0":...}；

依赖你的工程：
  - config.py: 提供 source_data_path、{model_key}.base_path / .json_template / .num_groups
  - load_data.py: load_matchings_from_json(), load_source_scores()
    （总分用六项 *_partner 求和，键为 (iid, pid)）.
"""

import os, sys, json, argparse, importlib.util, inspect
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import pandas as pd

# 将仓库根目录加入 sys.path（保证可 import 本仓库模块）
# 你可以按实际情况修改 repo_root；默认尝试递归向上找到包含 config.py 的目录
def _find_repo_root(start: str) -> str:
    cur = os.path.abspath(start)
    for _ in range(8):
        if os.path.exists(os.path.join(cur, "config.py")):
            return cur
        nxt = os.path.dirname(cur)
        if nxt == cur: break
        cur = nxt
    return os.path.abspath(start)

REPO_ROOT = _find_repo_root(os.getcwd())
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config import config  # 提供 source_data_path / 模型键配置 :contentReference[oaicite:4]{index=4}
from load_data import load_matchings_from_json, load_source_scores  # :contentReference[oaicite:5]{index=5}

# ---------- 统一模型：概率 ----------
def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))

def p_switch(score_new: Optional[float],
             score_cur: Optional[float],
             beta0: float,
             lam: float,
             s0: float) -> float:
    """
    统一模型概率：
      单身：score_cur = s0
      否则：score_cur = 当前伴侣得分
      P = σ(β0 + λ * (score_new - score_cur))
    见 8.1 文档统一模型推导与实现。:contentReference[oaicite:6]{index=6}
    """
    if score_new is None:
        return 0.0
    if score_cur is None:
        score_cur = s0
    z = beta0 + lam * (score_new - score_cur)
    return float(sigmoid(z))

# ---------- 从 fit_module 路径提取 (β0, λ, S0) ----------
def _import_module_from_path(py_path: str):
    spec = importlib.util.spec_from_file_location("fit_module_dyn", py_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块：{py_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod

def _try_read_json_nearby(py_path: str) -> Dict[str, Any]:
    """兼容方案3：同目录 unified_params.json"""
    folder = os.path.dirname(os.path.abspath(py_path))
    cand = os.path.join(folder, "unified_params.json")
    if os.path.exists(cand):
        with open(cand, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def load_unified_params_from_fit(py_path: str,
                                 model_key: Optional[str]) -> Tuple[float, float, float]:
    """
    返回 (beta0_taken, lambda_taken, s0)
    允许 fit 模块以三种方式暴露结果（函数/变量/旁文件），见文件头的“兼容策略”。
    """
    mod = _import_module_from_path(py_path)

    # 1) 函数优先
    if hasattr(mod, "get_unified_model_params"):
        fn = getattr(mod, "get_unified_model_params")
        try:
            if model_key and len(inspect.signature(fn).parameters) == 1:
                b0, lam, s0 = fn(model_key)
            else:
                b0, lam, s0 = fn()
            if b0 is not None and lam is not None and s0 is not None:
                return float(b0), float(lam), float(s0)
        except Exception:
            pass  # 进入下一种方式

    # 2) 直接变量
    b0 = getattr(mod, "BETA0_TAKEN", None)
    lam = getattr(mod, "LAMBDA_TAKEN", None)
    s0 = getattr(mod, "S0", None)
    b0_single = getattr(mod, "BETA0_SINGLE", None)
    if (b0 is not None) and (lam is not None):
        if s0 is None and (b0_single is not None) and float(lam) != 0.0:
            s0 = (float(b0) - float(b0_single)) / float(lam)
        if s0 is not None:
            return float(b0), float(lam), float(s0)

    # 3) 旁文件 JSON
    pack = _try_read_json_nearby(py_path)
    if pack:
        b0 = pack.get("beta0_taken")
        lam = pack.get("lambda_taken")
        s0 = pack.get("s0")
        b0_single = pack.get("beta0_single")
        if (b0 is not None) and (lam is not None):
            if s0 is None and (b0_single is not None) and float(lam) != 0.0:
                s0 = (float(b0) - float(b0_single)) / float(lam)
            if s0 is not None:
                return float(b0), float(lam), float(s0)

    raise RuntimeError("未能从 fit 模块或同目录 JSON 成功取得 (β0, λ, S0)")

# ---------- 逐组 E[#bp] ----------
def compute_group_ids(df_all: pd.DataFrame) -> List[int]:
    if "group" not in df_all.columns:
        raise ValueError("源 Excel 缺少 'group' 列")
    groups = sorted([int(g) for g in df_all["group"].dropna().unique()])
    return groups

def iter_group_people(df_all: pd.DataFrame, gid: int) -> Tuple[List[int], List[int]]:
    """
    返回 (men_ids, women_ids)
      men: group==gid 且 gender==1 的 iid
      women: 该组出现过的 pid 去重（更稳健地覆盖女性 id）
    """
    gdf = df_all[df_all["group"] == gid]
    men_ids = sorted(gdf[gdf.get("gender", 1) == 1]["iid"].dropna().astype(int).unique().tolist())
    women_ids = sorted(gdf["pid"].dropna().astype(int).unique().tolist())
    return men_ids, women_ids

def expected_numbp_for_matching(matching: Dict[int, Any],
                                men_ids: List[int],
                                women_ids: List[int],
                                score_dict: Dict[Tuple[int, int], float],
                                beta0: float, lam: float, s0: float) -> float:
    total = 0.0
    for m in men_ids:
        for w in women_ids:
            if matching.get(m) == w:
                continue
            # m 的换伴概率
            m_partner = matching.get(m)
            s_mw = score_dict.get((m, w))
            s_mcur = None if (m_partner in [None, "rejected"]) else score_dict.get((m, int(m_partner)))
            p1 = p_switch(s_mw, s_mcur, beta0, lam, s0)

            # w 的换伴概率
            w_partner = matching.get(w)
            s_wm = score_dict.get((w, m))
            s_wcur = None if (w_partner in [None, "rejected"]) else score_dict.get((w, int(w_partner)))
            p2 = p_switch(s_wm, s_wcur, beta0, lam, s0)

            total += p1 * p2
    return float(total)

def compute_all_groups(fit_module_path: str,
                       model_key: str,
                       output_csv: Optional[str] = None) -> pd.DataFrame:
    # 统一模型参数
    beta0, lam, s0 = load_unified_params_from_fit(fit_module_path, model_key)

    # 源分数与全表（供 group/gender）
    src_excel = config["source_data_path"]  # :contentReference[oaicite:7]{index=7}
    score_dict = load_source_scores(src_excel)          # (iid,pid)->总分  :contentReference[oaicite:8]{index=8}
    df_all = pd.read_excel(src_excel)

    # 模型对应的各组最终匹配
    matchings = load_matchings_from_json(model_key)     # 读取 base_path/json_template/num_groups :contentReference[oaicite:9]{index=9}

    # 逐组计算
    rows = []
    groups = compute_group_ids(df_all)
    for idx, matching in enumerate(matchings, start=1):
        gid = idx if idx in groups else (groups[idx-1] if idx-1 < len(groups) else idx)
        men_ids, women_ids = iter_group_people(df_all, gid)
        if not men_ids or not women_ids:
            continue
        e_bp = expected_numbp_for_matching(matching, men_ids, women_ids, score_dict, beta0, lam, s0)
        rows.append({"group": gid, "E[numbp]": e_bp})

    df_res = pd.DataFrame(rows).sort_values("group")

    if output_csv:
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
        df_res.to_csv(output_csv, index=False, encoding="utf-8-sig")

    # 打印摘要
    if not df_res.empty:
        mean_v = df_res["E[numbp]"].mean()
        std_v = df_res["E[numbp]"].std(ddof=0)
        print(f"[Unified Params] beta0={beta0:.4f}, lambda={lam:.4f}, S0={s0:.4f}")
        print(f"[{model_key}] N_groups={len(df_res)} | Mean={mean_v:.4f} | Std={std_v:.4f}")
    else:
        print("未得到任何组的稳定性结果，请检查路径与数据。")
    return df_res

def main():
    ap = argparse.ArgumentParser(
        description="按给定 fit 模块路径提取 λ/β/S0，并逐组计算期望阻塞对 E[numbp]"
    )
    ap.add_argument("--fit-module", required=True,
                    help="指向 fit_scalar_logit.py 的绝对路径（或同目录 unified_params.json）")
    ap.add_argument("--model-key", required=True,
                    help="与 config.py 中键名一致，例如 'claude_en_fitting' 或 'claude_zh_fitting'")
    ap.add_argument("--output-csv", default=None,
                    help="可选：结果导出路径（CSV）")
    args = ap.parse_args()

    compute_all_groups(args.fit_module, args.model_key, args.output_csv)

if __name__ == "__main__":
    main()
