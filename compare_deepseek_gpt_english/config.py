# 文件名: config.py

config = {
    # --- GPT-4 (英文) 配置 ---
    "gpt4_en": {
        "name": "GPT-4 (English)",
        "base_path": '/home/lsy/match/bahavior_simul/0618_gpt4_turbo_random',
        "num_groups": 49,
        "csv_template": "0618_gpt4_turbo_random_group{group_id}.csv",
        "json_template": "0618_gpt4_turbo_random_group{group_id}.json",
        "label": "GPT-4 (Eng, N=50)" # 用于图表标签
    },
    
    # --- DeepSeek (英文) 配置 ---
    "deepseek_en": {
        "name": "DeepSeek (English)",
        "base_path": '/home/lsy/match/have_deepseek/0620_deepseek_eng',
        "num_groups": 20,
        "csv_template": "0620_deepseek_ai_DeepSeek-R1-0528_res{group_id}.csv",
        "json_template": "0620_deepseek_ai_DeepSeek-R1-0528_res{group_id}.json",
        "label": "DeepSeek (Eng, N=20)" # 用于图表标签
    },
    
    # --- 可视化配置 ---
    "plot_style": "whitegrid",
    "plot_context": "talk", # 更大的字体，适合报告
    "plot_palette_models": ["#1f77b4", "#ff7f0e"], # 为两个模型指定不同颜色 (蓝色, 橙色)
    "figure_size_large": (14, 9),
    "figure_size_medium": (12, 8),
    "figure_size_small": (8, 8),
    "title_fontsize": 18,
    "label_fontsize": 14,
    
    # --- 统计分析配置 ---
    "bootstrap_iterations": 100, # 如果需要做不一致性分析
}