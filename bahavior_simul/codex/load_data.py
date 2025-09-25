from __future__ import annotations

import csv
import json
from json import JSONDecodeError
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from config import config


def load_all_group_data_for_model(model_key: str) -> Tuple[pd.DataFrame, List[Optional[Dict[int, Optional[int]]]]]:
    """加载指定模型的所有组数据和最终匹配结果。

    返回的 ``DataFrame`` 聚合了所有有效的决策日志记录，
    ``list`` 中的元素则是每个实验组的最终匹配字典（若解析失败为 ``None``）。
    """

    if model_key not in config:
        print(f"错误: 模型键 '{model_key}' 在 config.py 中未找到。")
        return pd.DataFrame(), []

    model_config = config[model_key]
    base_path = Path(model_config.get("base_path", ""))
    csv_template = model_config.get("csv_template", "")
    json_template = model_config.get("json_template")
    num_groups = int(model_config.get("num_groups", 0))

    all_dfs: List[pd.DataFrame] = []
    all_matchings: List[Optional[Dict[int, Optional[int]]]] = []

    for group_id in range(1, num_groups + 1):
        csv_path = base_path / csv_template.format(group_id=group_id)
        group_rows: List[Dict[str, object]] = []

        if csv_path.exists():
            try:
                with csv_path.open("r", encoding="utf-8") as f_csv:
                    reader = csv.reader(f_csv)
                    for row in reader:
                        if len(row) < 4:
                            continue

                        prompt = row[0]
                        reason = row[1] if len(row) > 1 else ""
                        try:
                            target = _safe_int(row[-4])
                            proposer = _safe_int(row[-3])
                            result = _safe_int(row[-1])
                        except ValueError:
                            continue

                        if target is None or proposer is None or result is None:
                            continue

                        current_partner = _parse_partner_value(row[-2] if len(row) >= 2 else None)

                        group_rows.append(
                            {
                                "prompt": str(prompt),
                                "reason": str(reason),
                                "target": target,
                                "proposer": proposer,
                                "current_partner": current_partner,
                                "result": result,
                                "group": group_id,
                            }
                        )
            except Exception as exc:  # pragma: no cover - I/O safety net
                print(f"  错误: 处理CSV文件 {csv_path} 时发生未知错误: {exc}")

        if group_rows:
            all_dfs.append(pd.DataFrame(group_rows))

        matching: Optional[Dict[int, Optional[int]]] = None
        if json_template:
            json_path = base_path / json_template.format(group_id=group_id)
            matching = _load_final_matching(json_path)
        all_matchings.append(matching)

    combined_df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    return combined_df, all_matchings


def _safe_int(value: object) -> Optional[int]:
    """尽可能将值转换为整数；若无法转换则返回 ``None``。"""

    if value is None:
        return None
    if isinstance(value, int):
        return value
    try:
        text = str(value).strip()
        if not text:
            return None
        return int(float(text))
    except (TypeError, ValueError):
        return None


def _parse_partner_value(value: object) -> Optional[int]:
    """解析伴侣ID，将 'rejected'/空字符串 统一转换为 ``None``。"""

    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan", "rejected"}:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _load_final_matching(json_path: Path) -> Optional[Dict[int, Optional[int]]]:
    """从JSON日志中提取最后一次匹配结果。"""

    if not json_path.exists():
        return None

    try:
        content = json_path.read_text(encoding="utf-8").strip()
    except OSError:
        return None

    if not content:
        return None

    decoder = json.JSONDecoder()
    idx = 0
    last_obj = None
    while idx < len(content):
        while idx < len(content) and content[idx].isspace():
            idx += 1
        if idx >= len(content):
            break
        try:
            obj, next_idx = decoder.raw_decode(content, idx)
        except JSONDecodeError:
            idx += 1
            continue
        last_obj = obj
        idx = next_idx

    if not isinstance(last_obj, dict):
        return None

    cleaned: Dict[int, Optional[int]] = {}
    for raw_key, raw_value in last_obj.items():
        key = _safe_int(raw_key)
        if key is None:
            continue
        cleaned[key] = _parse_partner_value(raw_value)
    return cleaned


def load_source_scores(path: Path) -> Optional[Dict[Tuple[int, int], float]]:
    """加载评分数据，并返回 {(评价者, 被评价者): 总分} 的映射。"""

    try:
        df = pd.read_excel(path)
    except FileNotFoundError:
        print(f"错误: 源数据文件未找到: {path}")
        return None
    except Exception as exc:  # pragma: no cover - I/O safety net
        print(f"加载源数据Excel文件时发生未知错误: {exc}")
        return None

    score_cols = [
        "attractive_partner",
        "sincere_partner",
        "intelligence_partner",
        "funny_partner",
        "ambition_partner",
        "shared_interests_partner",
    ]

    if "intelliger" in df.columns and "intelligence_partner" not in df.columns:
        df.rename(columns={"intelliger": "intelligence_partner"}, inplace=True)

    required_cols = ["iid", "pid"] + score_cols
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"错误: 源数据Excel文件中缺少必需的列: {missing}")
        return None

    for col in score_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["total_score"] = df[score_cols].sum(axis=1)
    score_dict = df.set_index(["iid", "pid"])["total_score"].to_dict()
    print("源数据Excel评分已成功加载并处理。")
    return score_dict


def load_gender_map(path: Path) -> Dict[int, int]:
    """加载参与者的性别映射，gender=1表示男性，0表示女性。"""

    try:
        df = pd.read_excel(path)
    except FileNotFoundError:
        print(f"错误: 源数据文件未找到: {path}")
        return {}
    except Exception as exc:  # pragma: no cover - I/O safety net
        print(f"加载性别映射时发生未知错误: {exc}")
        return {}

    if "iid" not in df.columns or "gender" not in df.columns:
        print("警告: 源数据中缺少 'iid' 或 'gender' 列，无法构建性别映射。")
        return {}

    gender_df = df[["iid", "gender"]].dropna(subset=["iid", "gender"]).copy()
    gender_df["iid"] = pd.to_numeric(gender_df["iid"], errors="coerce")
    gender_df["gender"] = pd.to_numeric(gender_df["gender"], errors="coerce")
    gender_df.dropna(subset=["iid", "gender"], inplace=True)

    gender_df["iid"] = gender_df["iid"].astype(int)
    gender_df["gender"] = gender_df["gender"].astype(int)

    # 同一 iid 可能出现多次，保留第一次出现的性别值
    gender_map = gender_df.drop_duplicates(subset=["iid"]).set_index("iid")["gender"].to_dict()
    print("参与者性别映射已成功加载。")
    return gender_map
