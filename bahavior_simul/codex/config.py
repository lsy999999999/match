from pathlib import Path

# 以当前文件位置为基准，推断仓库中的数据位置，避免硬编码的绝对路径
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DATA_DIR = _REPO_ROOT / "dataset"
_SIMULATION_DIR = _REPO_ROOT / "bahavior_simul"


config = {
    "source_data_path": _DATA_DIR / "/home/lsy/match/dataset/save_merge_select_null_3.xlsx",
    "gpt4_en_fitting": {
        "label": "GPT-4 (Eng)",
        "num_groups": 42,
        "base_path": _SIMULATION_DIR / "/home/lsy/match/bahavior_simul/0627_gpt4_eng",
        "csv_template": "0618_gpt4_turbo_random_group{group_id}.csv",
        "json_template": "0618_gpt4_turbo_random_group{group_id}.json",
    },
    "gpt4_zh_fitting": {
        "label": "GPT-4 (Chinese)",
        "num_groups": 50,
        "base_path": _SIMULATION_DIR / "/home/lsy/match/bahavior_simul/0629_gpt_Chinese",
        "csv_template": "0629_gpt4_Chinese_group{group_id}.csv",
        "json_template": "0629_gpt4_Chinese_group{group_id}.json",
    },
}
