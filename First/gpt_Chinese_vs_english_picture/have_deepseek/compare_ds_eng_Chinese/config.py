config = {
    # --- 文件路径配置 ---
    "base_path_en": '/home/lsy/match/have_deepseek/0620_deepseek_eng',
    "base_path_zh": '/home/lsy/match/have_deepseek/0620_deepseek_Chinese',
    
    # --- 实验参数配置 ---
    "num_groups_en": 20,
    "num_groups_zh": 20,
    
    # --- 文件名格式配置 ---
    "csv_template_en": "0620_deepseek_ai_DeepSeek-R1-0528_res{group_id}.csv",
    "json_template_en": "0620_deepseek_ai_DeepSeek-R1-0528_res{group_id}.json",
    "csv_template_zh": "0620_deepseek-ai_DeepSeek-R1-0528_Chinese_res{group_id}.csv",
    "json_template_zh": "0620_deepseek-ai_DeepSeek-R1-0528_Chinese_res{group_id}.json",
    
    
    # --- 可视化配置 ---
    "plot_style": "whitegrid",
    "plot_context": "talk",
    "plot_palette_pastel": "pastel",
    "plot_palette_muted": "muted",
    "plot_palette_viridis": "viridis",

    "figure_size_large": (14, 9),
    "figure_size_medium": (12, 8),
    "figure_size_small": (8, 8),
    

    "title_fontsize": 18,
    "label_fontsize": 14,
    
    # --- 统计分析配置 ---
    "bootstrap_iterations": 100,
}