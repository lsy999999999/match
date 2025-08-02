config = {
    
    "source_data_path": "/home/lsy/match/dataset/save_merge_select_null_3.xlsx",
    "qwen_en_fitting": {
        "label": "qwen (Eng)",
        "num_groups": 42,
        "base_path": "/home/lsy/match/qwen/0707_qw_eng",
        "csv_template": "0707_qw_eng_group{group_id}.csv",
        "json_template": "0707_qw_eng_group{group_id}.json",
    },

    "qwen_zh_fitting": {
        "label": "qwen (Chinese)",       # 图表中的英文标签
        "num_groups": 42,                 # 请根据你的实际有效组数修改
        "base_path": "/home/lsy/match/qwen/0707_qw_Chinese",
        "csv_template": "0707_qw_Chinese_group{group_id}.csv",
        "json_template": "0707_qw_Chinese_group{group_id}.json",
    },


}