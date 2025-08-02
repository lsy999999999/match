config = {
    
    "source_data_path": "/home/lsy/match/dataset/save_merge_select_null_3.xlsx",
    "claude_en_fitting": {
        "label": "claude (Eng)",
        "num_groups": 42,
        "base_path": "/home/lsy/match/claude/0725_claude_eng",
        "csv_template": "0725_claude_eng_group{group_id}.csv",
        "json_template": "0725_claude_eng_group{group_id}.json",
    },

    "claude_zh_fitting": {
        "label": "claude (Chinese)",       # 图表中的英文标签
        "num_groups": 50,                 # 请根据你的实际有效组数修改
        "base_path": "/home/lsy/match/claude/0726_claude_Chinese",
        "csv_template": "0726_claude_Chinese_group{group_id}.csv",
        "json_template": "0726_claude_Chinese_group{group_id}.json",
    },


}