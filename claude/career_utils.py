import json, unicodedata, re
from pathlib import Path
import pandas as pd
import math

EXCEL_PATH = Path("/home/lsy/match/dataset/save_merge_select_null_3.xlsx")  # 职业来源表
MAP_PATH    = Path("career_map.json")                         # 存储翻译映射

def load_career_set() -> set[str]:
    """读取 Excel 第 I 列（索引 8），返回职业集合（去重、去空）"""
    df = pd.read_excel(EXCEL_PATH, header=None, usecols=[8])
    s  = (df.iloc[:, 0]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
         )
    return set(s)

def load_map() -> dict[str, str]:
    """载入已有映射，没有则返回空 dict"""
    if MAP_PATH.exists():
        with MAP_PATH.open(encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_map(map_dict: dict[str, str]) -> None:
    MAP_PATH.write_text(json.dumps(map_dict, ensure_ascii=False, indent=2),
                        encoding="utf-8")

def build_or_update_map() -> None:
    """比对现有职业与映射，提示缺失，并保存更新后的映射文件"""
    careers = load_career_set()
    mapping = load_map()

    missing = sorted(c for c in careers if c not in mapping)
    if missing:
        print("⚠ 发现未翻译职业共", len(missing), "项：")
        for c in missing:
            print("  -", c)
            mapping[c] = c  # 先用原文占位，待人工翻译后改写

    save_map(mapping)
    print("\n✅ career_map.json 已更新／生成，缺失翻译请在文件中补充。")

def translate_career(career_raw: str) -> str:
    """将英文职业翻译成中文；若无翻译则返回原值"""
    if career_raw is None:
        return "未知"

    # 若为 float 且是 nan
    if isinstance(career_raw, float) and math.isnan(career_raw):
        return "未知"

    key = str(career_raw).strip()
    if key.lower() in ("", "nan", "none"):
        return "未知"

    mapping = load_map()
    return mapping.get(key, key)
if __name__ == "__main__":
    build_or_update_map()