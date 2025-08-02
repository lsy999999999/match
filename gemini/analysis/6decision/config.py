config = {
    
    "source_data_path": "/home/lsy/match/dataset/save_merge_select_null_3.xlsx",
    "gemini_en_fitting": {
        "label": "GPT-4 (Eng)",
        "num_groups": 42,
        "base_path": "/home/lsy/match/gemini/0713_gemini_eng",
        "csv_template": "0713_gemini_eng_group{group_id}.csv",
        "json_template": "0713_gemini_eng_group{group_id}.json",
    },

    "gemini_zh_fitting": {
        "label": "GPT-4 (Chinese)",       # 图表中的英文标签
        "num_groups": 50,                 # 请根据你的实际有效组数修改
        "base_path": "/home/lsy/match/gemini/0713_gemini_Chinese",
        "csv_template": "0713_gemini_Chinese_group{group_id}.csv",
        "json_template": "0713_gemini_Chinese_group{group_id}.json",
    },


}