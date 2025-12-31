config = {
    
    "source_data_path": "/home/lsy/match/dataset/save_merge_select_null_3.xlsx",
    "gpt4_en_fitting": {
        "label": "GPT-4 (Eng)",
        "num_groups": 42,
        "base_path": "/home/lsy/match/bahavior_simul/0627_gpt4_eng",
        "csv_template": "0618_gpt4_turbo_random_group{group_id}.csv",
        "json_template": "0618_gpt4_turbo_random_group{group_id}.json",
    },

    "gpt4_zh_fitting": {
        "label": "GPT-4 (Chinese)",       # 图表中的英文标签
        "num_groups": 50,                 # 请根据你的实际有效组数修改
        "base_path": "/home/lsy/match/bahavior_simul/1021_gpt_Chinese",
        "csv_template": "1021_gpt4_Chinese_group{group_id}.csv",
        "json_template": "1021_gpt4_Chinese_group{group_id}.json",
    },
    
    # 新增一个配置项，对应上面的 python 脚本生成的数据
    "gpt4_en_makeup": {
        "label": "GPT-4 (Eng Makeup)",
        "num_groups": 15, # 根据你的 range_configuration (5-15) 设定，或者设大一点
        "num_runs": 20,   # 【新增】告诉 loader 每个组有多少个 run
        "base_path": "/home/lsy/match/bahavior_simul/1022_gpt_eng_makeup",
        # 模板中必须包含 {run_id} 和 {group_id}
        "csv_template": "1022_gpt_eng_run{run_id}_group{group_id}.csv",
        "json_template": "1022_gpt_eng_run{run_id}_group{group_id}.json",
    },

}