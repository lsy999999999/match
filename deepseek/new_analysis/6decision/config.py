config = {
    
    "source_data_path": "/home/lsy/match/dataset/save_merge_select_null_3.xlsx",
    "deepseek_en_fitting": {
        "label": "deeepseek (Eng)",
        "num_groups": 21,
        "base_path": "/home/lsy/match/deepseek/0703_ds_eng",
        "csv_template": "0703_ds_eng_group{group_id}.csv",
        "json_template": "0703_ds_eng_group{group_id}.json",
    },

    "deepseek_zh_fitting": {
        "label": "deepseek (Chinese)",       # 图表中的英文标签
        "num_groups": 21,                 # 请根据你的实际有效组数修改
        "base_path": "/home/lsy/match/deepseek/0704_ds_Chinese",
        "csv_template": "0704_ds_Chinese_group{group_id}.csv",
        "json_template": "0704_ds_Chinese_group{group_id}.json",
    },


}