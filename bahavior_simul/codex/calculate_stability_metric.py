from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm

from config import config
from load_data import load_all_group_data_for_model, load_source_scores


@dataclass
class StabilityStats:
    expected_bp: float
    evaluated_pairs: int
    skipped_pairs: int
    total_pairs: int


def fit_unified_logit_parameters(model_key: str) -> Tuple[float, float, float]:
    """Fit the unified logit parameters (beta0, lambda, s0) for a model."""

    df_log, _ = load_all_group_data_for_model(model_key)
    if df_log.empty:
        raise RuntimeError("未能加载到有效的决策日志数据，无法拟合Logit模型。")

    source_scores = load_source_scores(config["source_data_path"])
    if source_scores is None:
        raise RuntimeError("评分数据加载失败，无法拟合Logit模型。")

    df_taken = df_log[df_log["current_partner"].notna()].copy()

    def _extract_taken_scores(row: pd.Series) -> Tuple[Optional[float], Optional[float]]:
        try:
            decision_maker = int(row["target"])
            new_suitor = int(row["proposer"])
            current_partner = int(row["current_partner"])
        except (TypeError, ValueError):
            return None, None
        Sa = source_scores.get((decision_maker, new_suitor))
        Sb = source_scores.get((decision_maker, current_partner))
        return Sa, Sb

    df_taken[["Sa", "Sb"]] = df_taken.apply(_extract_taken_scores, axis=1, result_type="expand")
    df_taken.dropna(subset=["Sa", "Sb"], inplace=True)
    if df_taken.empty:
        raise RuntimeError("缺少包含现任伴侣比较的决策记录，无法拟合 beta0 与 lambda。")

    df_taken["score_diff"] = df_taken["Sa"] - df_taken["Sb"]
    X_taken = sm.add_constant(df_taken["score_diff"].astype(float))
    y_taken = df_taken["result"].astype(float)
    taken_fit = sm.Logit(y_taken, X_taken).fit(disp=0)
    beta0 = float(taken_fit.params["const"])
    lambda_param = float(taken_fit.params["score_diff"])

    df_single = df_log[df_log["current_partner"].isna()].copy()

    def _extract_single_score(row: pd.Series) -> Optional[float]:
        try:
            decision_maker = int(row["target"])
            proposer = int(row["proposer"])
        except (TypeError, ValueError):
            return None
        return source_scores.get((decision_maker, proposer))

    df_single["S"] = df_single.apply(_extract_single_score, axis=1)
    df_single.dropna(subset=["S"], inplace=True)
    if df_single.empty:
        raise RuntimeError("缺少单身状态下的决策记录，无法推导 S0。")

    X_single = sm.add_constant(df_single["S"].astype(float))
    y_single = df_single["result"].astype(float)
    single_fit = sm.Logit(y_single, X_single).fit(disp=0)
    beta0_single = float(single_fit.params["const"])

    if lambda_param == 0:
        raise RuntimeError("拟合得到的 lambda 等于0，无法计算 S0。")

    s0 = (beta0 - beta0_single) / lambda_param
    return beta0, lambda_param, float(s0)


def prepare_source_dataframe(source_path: Path) -> pd.DataFrame:
    """Load and preprocess the source Excel file used for stability evaluation."""

    df = pd.read_excel(source_path)
    essential_cols = ["group", "iid", "pid", "gender"]
    for col in essential_cols:
        if col not in df.columns:
            raise RuntimeError(f"源数据缺少必要列: {col}")

    df = df.dropna(subset=essential_cols).copy()
    df["group"] = df["group"].astype(int)
    df["iid"] = df["iid"].astype(int)
    df["pid"] = df["pid"].astype(int)
    df["gender"] = df["gender"].astype(int)

    score_dims = ["attractive", "sincere", "intelligence", "funny", "ambition", "shared_interests"]
    score_cols = [f"{dim}_partner" for dim in score_dims]
    importance_cols = [f"{dim}_important" for dim in score_dims]

    for col in score_cols + importance_cols + ["gpt_score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan

    weighted_score = np.zeros(len(df), dtype=float)
    for s_col, w_col in zip(score_cols, importance_cols):
        weighted_score += df[s_col].fillna(0).to_numpy(dtype=float) * df[w_col].fillna(0).to_numpy(dtype=float)
    df["weighted_score"] = weighted_score / 100.0

    return df


def get_group_participants(df_source: pd.DataFrame) -> Dict[int, Tuple[List[int], List[int]]]:
    participants: Dict[int, Tuple[List[int], List[int]]] = {}
    for group_id in sorted(df_source["group"].unique().astype(int).tolist()):
        group_df = df_source[df_source["group"] == group_id]
        men_ids = sorted({int(v) for v in group_df[group_df["gender"] == 1]["iid"].tolist()})
        women_ids = sorted({int(v) for v in group_df[group_df["gender"] == 0]["iid"].tolist()})
        participants[group_id] = (men_ids, women_ids)
    return participants


def build_preference_dict(
    group_df: pd.DataFrame,
    actor_ids: Sequence[int],
    partner_pool: Iterable[int],
    gender_value: int,
    score_col: str,
) -> Dict[int, List[int]]:
    partner_set = {int(pid) for pid in partner_pool}
    preferences: Dict[int, List[int]] = {}
    for actor_id in actor_ids:
        subset = group_df[(group_df["gender"] == gender_value) & (group_df["iid"] == actor_id)].copy()
        subset = subset.dropna(subset=["pid", score_col])
        if subset.empty:
            preferences[int(actor_id)] = []
            continue
        subset["pid"] = subset["pid"].astype(int)
        subset = subset[subset["pid"].isin(partner_set)]
        subset = subset.sort_values(by=score_col, ascending=False)
        preferences[int(actor_id)] = [int(pid) for pid in subset["pid"].tolist()]
    return preferences


def gale_shapley_match(men_prefs: Dict[int, List[int]], women_prefs: Dict[int, List[int]]) -> Dict[int, Optional[int]]:
    free_men = [int(m) for m in men_prefs.keys()]
    proposal_index = {int(m): 0 for m in men_prefs.keys()}
    women_current: Dict[int, Optional[int]] = {int(w): None for w in women_prefs.keys()}
    women_rank: Dict[int, Dict[int, int]] = {
        int(w): {int(m): idx for idx, m in enumerate(pref_list)} for w, pref_list in women_prefs.items()
    }
    matches: Dict[int, Optional[int]] = {int(m): None for m in men_prefs.keys()}

    while free_men:
        man = free_men.pop(0)
        pref_list = men_prefs.get(man, [])
        idx = proposal_index[man]
        if idx >= len(pref_list):
            matches[man] = None
            continue
        woman = pref_list[idx]
        proposal_index[man] += 1
        if woman not in women_prefs:
            free_men.append(man)
            continue
        current_partner = women_current[woman]
        if current_partner is None:
            women_current[woman] = man
            matches[man] = woman
        else:
            ranking = women_rank.get(woman, {})
            new_rank = ranking.get(man)
            current_rank = ranking.get(current_partner)
            if new_rank is not None and (current_rank is None or new_rank < current_rank):
                women_current[woman] = man
                matches[man] = woman
                matches[current_partner] = None
                free_men.append(current_partner)
            else:
                free_men.append(man)

    final_matching: Dict[int, Optional[int]] = {**matches}
    for woman, partner in women_current.items():
        final_matching[woman] = partner
    return final_matching


def normalize_matching(raw: Optional[Dict[int, object]]) -> Dict[int, Optional[int]]:
    normalized: Dict[int, Optional[int]] = {}
    if not raw:
        return normalized
    for key, value in raw.items():
        try:
            key_int = int(key)
        except (TypeError, ValueError):
            continue
        normalized[key_int] = _normalize_partner(value)
    return normalized


def _normalize_partner(value: object) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float):
        if math.isnan(value):
            return None
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"none", "nan", "rejected", "null"}:
            return None
        try:
            return int(float(text))
        except ValueError:
            return None
    return None


def logistic_switch_probability(
    score_new: Optional[float],
    score_current: Optional[float],
    beta0: float,
    lambda_param: float,
) -> Optional[float]:
    if score_new is None or score_current is None:
        return None
    diff = float(score_new) - float(score_current)
    z = beta0 + lambda_param * diff
    return float(1.0 / (1.0 + math.exp(-z)))


def evaluate_matching(
    matching: Dict[int, Optional[int]],
    men: Sequence[int],
    women: Sequence[int],
    source_scores: Dict[Tuple[int, int], float],
    beta0: float,
    lambda_param: float,
    s0: float,
) -> StabilityStats:
    total_pairs = len(men) * len(women)
    if total_pairs == 0:
        return StabilityStats(expected_bp=float("nan"), evaluated_pairs=0, skipped_pairs=0, total_pairs=0)

    total_prob = 0.0
    evaluated = 0
    skipped = 0

    for man in men:
        man_partner = _normalize_partner(matching.get(man))
        man_partner_score = None
        if man_partner is not None:
            man_partner_score = source_scores.get((man, man_partner))
        for woman in women:
            if man_partner == woman:
                continue
            woman_partner = _normalize_partner(matching.get(woman))
            score_m_w = source_scores.get((man, woman))
            if man_partner is None:
                base_man_score = s0
            else:
                base_man_score = man_partner_score
            prob1 = logistic_switch_probability(score_m_w, base_man_score, beta0, lambda_param)
            if prob1 is None:
                skipped += 1
                continue

            score_w_m = source_scores.get((woman, man))
            if woman_partner is None:
                base_woman_score = s0
            else:
                base_woman_score = source_scores.get((woman, woman_partner))
            prob2 = logistic_switch_probability(score_w_m, base_woman_score, beta0, lambda_param)
            if prob2 is None:
                skipped += 1
                continue

            total_prob += prob1 * prob2
            evaluated += 1

    expected_bp = float("nan") if evaluated == 0 else total_prob
    return StabilityStats(expected_bp=expected_bp, evaluated_pairs=evaluated, skipped_pairs=skipped, total_pairs=total_pairs)


def compute_human_matchings(
    df_source: pd.DataFrame,
    participants: Dict[int, Tuple[List[int], List[int]]],
    score_col: str,
) -> Dict[int, Dict[int, Optional[int]]]:
    human_matchings: Dict[int, Dict[int, Optional[int]]] = {}
    for group_id, (men_ids, women_ids) in participants.items():
        if not men_ids or not women_ids:
            continue
        group_df = df_source[df_source["group"] == group_id]
        men_prefs = build_preference_dict(group_df, men_ids, women_ids, gender_value=1, score_col=score_col)
        women_prefs = build_preference_dict(group_df, women_ids, men_ids, gender_value=0, score_col=score_col)
        if not men_prefs or not women_prefs:
            continue
        human_matchings[group_id] = gale_shapley_match(men_prefs, women_prefs)
    return human_matchings


def build_participant_signatures(
    participants: Dict[int, Tuple[List[int], List[int]]]
) -> Tuple[Dict[Tuple[int, ...], int], Dict[int, set[int]]]:
    signature_map: Dict[Tuple[int, ...], int] = {}
    participant_sets: Dict[int, set[int]] = {}
    for group_id, (men_ids, women_ids) in participants.items():
        combined = list(men_ids) + list(women_ids)
        signature = tuple(sorted(int(pid) for pid in combined))
        signature_map[signature] = group_id
        participant_sets[group_id] = {int(pid) for pid in combined}
    return signature_map, participant_sets


def identify_source_group(
    ai_matching: Dict[int, Optional[int]],
    signature_map: Dict[Tuple[int, ...], int],
    participant_sets: Dict[int, set[int]],
    fallback_group: int,
) -> int:
    if not ai_matching:
        return fallback_group

    signature = tuple(sorted(int(pid) for pid in ai_matching.keys()))
    mapped_group = signature_map.get(signature)
    if mapped_group is not None:
        return mapped_group

    key_set = {int(pid) for pid in ai_matching.keys()}
    best_group = fallback_group
    best_overlap = -1
    for group_id, participant_set in participant_sets.items():
        overlap = len(key_set & participant_set)
        if overlap > best_overlap:
            best_overlap = overlap
            best_group = group_id
    return best_group


def run_group_stability_analysis(
    model_key: str,
    human_score_column: str,
    output_csv: Optional[Path] = None,
) -> pd.DataFrame:
    if model_key not in config:
        raise RuntimeError(f"配置中不存在模型键: {model_key}")

    beta0, lambda_param, s0 = fit_unified_logit_parameters(model_key)
    print(f"拟合参数 -> beta0: {beta0:.4f}, lambda: {lambda_param:.4f}, s0: {s0:.4f}")

    df_source = prepare_source_dataframe(config["source_data_path"])
    participants = get_group_participants(df_source)
    human_matchings = compute_human_matchings(df_source, participants, human_score_column)
    signature_map, participant_sets = build_participant_signatures(participants)

    source_scores = load_source_scores(config["source_data_path"])
    if source_scores is None:
        raise RuntimeError("无法加载评分数据。")

    _, ai_matchings_raw = load_all_group_data_for_model(model_key)

    model_cfg = config[model_key]
    num_groups = int(model_cfg.get("num_groups", len(ai_matchings_raw)))
    source_group_ids = sorted(participants.keys())
    if not source_group_ids:
        raise RuntimeError("源数据中没有可用的 group 信息。")

    rows: List[Dict[str, object]] = []
    for idx in range(num_groups):
        group_index = idx + 1
        default_group = source_group_ids[idx % len(source_group_ids)]

        ai_matching = normalize_matching(ai_matchings_raw[idx] if idx < len(ai_matchings_raw) else None)
        source_group = identify_source_group(ai_matching, signature_map, participant_sets, default_group)
        men_ids, women_ids = participants.get(source_group, ([], []))

        ai_stats = evaluate_matching(ai_matching, men_ids, women_ids, source_scores, beta0, lambda_param, s0)

        human_matching = normalize_matching(human_matchings.get(source_group))
        human_stats = evaluate_matching(human_matching, men_ids, women_ids, source_scores, beta0, lambda_param, s0)

        diff = np.nan
        if not math.isnan(ai_stats.expected_bp) and not math.isnan(human_stats.expected_bp):
            diff = ai_stats.expected_bp - human_stats.expected_bp

        rows.append(
            {
                "group": group_index,
                "source_group": source_group,
                "ai_expected_bp": ai_stats.expected_bp,
                "human_expected_bp": human_stats.expected_bp,
                "ai_evaluated_pairs": ai_stats.evaluated_pairs,
                "human_evaluated_pairs": human_stats.evaluated_pairs,
                "ai_skipped_pairs": ai_stats.skipped_pairs,
                "human_skipped_pairs": human_stats.skipped_pairs,
                "total_pairs": ai_stats.total_pairs,
                "ai_minus_human": diff,
                "ai_better": (not math.isnan(diff)) and diff < 0,
            }
        )

    result_df = pd.DataFrame(rows)
    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(output_csv, index=False)
        print(f"结果已保存到 {output_csv}")

    display_cols = [
        "group",
        "source_group",
        "ai_expected_bp",
        "human_expected_bp",
        "ai_minus_human",
        "ai_evaluated_pairs",
        "human_evaluated_pairs",
        "total_pairs",
        "ai_better",
    ]
    pd.set_option("display.max_rows", None)
    print("\n--- 分组稳定性结果 ---")
    print(result_df[display_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    mask_ai = result_df["ai_expected_bp"].notna()
    mask_human = result_df["human_expected_bp"].notna()
    if mask_ai.any():
        print(
            f"\nAI 平均 E[numbp]: {result_df.loc[mask_ai, 'ai_expected_bp'].mean():.4f}"
        )
    if mask_human.any():
        print(
            f"人类 GS 平均 E[numbp]: {result_df.loc[mask_human, 'human_expected_bp'].mean():.4f}"
        )
    if (mask_ai & mask_human).any():
        better_groups = result_df.loc[(mask_ai & mask_human) & (result_df["ai_better"]), "group"].tolist()
        print(f"AI 稳定性优于人的组数: {len(better_groups)} -> {better_groups}")

    return result_df


def main() -> None:
    parser = argparse.ArgumentParser(description="逐组计算AI与人类GS的稳定性指标")
    model_keys = [k for k, v in config.items() if isinstance(v, dict)]
    parser.add_argument(
        "--model",
        default="gpt4_en_fitting",
        choices=model_keys,
        help="需要分析的模型键",
    )
    parser.add_argument(
        "--human-score-column",
        default="weighted_score",
        help="用于构造人类GS偏好列表的得分列名 (默认 weighted_score)",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        help="可选：将结果保存为CSV",
    )

    args = parser.parse_args()
    run_group_stability_analysis(args.model, args.human_score_column, args.output_csv)


if __name__ == "__main__":
    main()
