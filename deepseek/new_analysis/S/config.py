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
    
    # 新增一个配置项，对应上面的 python 脚本生成的数据
    "deepaseek_en_makeup": {
        "label": "depseek (Eng Makeup)",
        "num_groups": 21, # 根据你的 range_configuration (5-15) 设定，或者设大一点
        "num_runs": 20,   # 【新增】告诉 loader 每个组有多少个 run
        "base_path": "/home/lsy/match/deepseek/1125_ds_eng_makeup",
        # 模板中必须包含 {run_id} 和 {group_id}
        "csv_template": "1024_ds_eng_run{run_id}_group{group_id}.csv",
        "json_template": "1024_ds_eng_run{run_id}_group{group_id}.json",
    },


}