# 文件名: config.py

config = {
    # --- 通用可视化配置 ---
    "plot_style": "whitegrid",
    "plot_context": "talk", # "notebook", "paper", "talk", "poster"
    "plot_palette_pastel": "pastel",
    "plot_palette_muted": "muted",
    "plot_palette_viridis": "viridis",
    "plot_palette_models_lang": "tab10", # 一个适合4个或更多类别的调色板
    "figure_size_large": (14, 10), # 稍微调大一点，容纳4组对比
    "figure_size_medium": (12, 8),
    "figure_size_small": (8, 8),
    "title_fontsize": 20, # 稍微调大一点
    "label_fontsize": 16, # 稍微调大一点
    "legend_fontsize": 14,

    # --- GPT-4 英文实验配置 ---
    "gpt4_en": {
        "label": "GPT-4 (Eng)",         # 图例中显示的标签
        "num_groups": 50,               # 实际有效的组数
        "base_path": "/home/lsy/match/bahavior_simul/0618_gpt4_turbo_random",
        "csv_template": "0618_gpt4_turbo_random_group{group_id}.csv",
        "json_template": "0618_gpt4_turbo_random_group{group_id}.json",
        "color": "blue" # 可以为每个系列指定颜色
    },

    # --- GPT-4 中文实验配置 ---
    "gpt4_zh": {
        "label": "GPT-4 (Chinese)",
        "num_groups": 50,               # 实际有效的组数
        "base_path": "/home/lsy/match/bahavior_simul/0619_gpt4_turbo_Chinese",
        "csv_template": "0619_gpt4_turbo_random_group{group_id}.csv",
        # 注意：这里JSON文件名模板可能与其他不同
        "json_template": "0619_gpt4_turbo_random_group_Chinese{group_id}.json",
        "color": "cornflowerblue"
    },

    # --- DeepSeek 英文实验配置 ---
    "deepseek_en": {
        "label": "DeepSeek (Eng)",
        "num_groups": 20,               # 实际有效的组数
        "base_path": "/home/lsy/match/have_deepseek/0620_deepseek_eng",
        "csv_template": "0620_deepseek_ai_DeepSeek-R1-0528_res{group_id}.csv",
        "json_template": "0620_deepseek_ai_DeepSeek-R1-0528_res{group_id}.json",
        "color": "darkorange"
    },

    # --- DeepSeek 中文实验配置 ---
    "deepseek_zh": {
        "label": "DeepSeek (中文)",
        "num_groups": 20,               # 实际有效的组数
        "base_path": "/home/lsy/match/have_deepseek/0620_deepseek_Chinese",
        "csv_template": "0620_deepseek-ai_DeepSeek-R1-0528_Chinese_res{group_id}.csv",
        "json_template": "0620_deepseek-ai_DeepSeek-R1-0528_Chinese_res{group_id}.json",
        "color": "sandybrown"
    },
    
    # --- 文本分析关键词配置 (如果进行关键词分析) ---
    "keyword_categories_en": { # 英文关键词
        'Attractiveness': ['attractive', 'attraction', 'appearance', 'look', 'physical', 'beauty', 'handsome', 'pretty'],
        'Sincerity': ['sincere', 'sincerity', 'honest', 'honesty', 'genuine', 'trust', 'truthful'],
        'Intelligence': ['intelligence', 'intelligent', 'smart', 'intellectual', 'knowledgeable', 'bright', 'clever'],
        'Funny': ['funny', 'humor', 'humour', 'laugh', 'entertaining', 'jovial', 'witty'],
        'Ambition': ['ambition', 'ambitious', 'career', 'driven', 'goal', 'aspiring', 'success-oriented'],
        'Shared Interests': ['shared interest', 'common interest', 'hobby', 'activity', 'compatible', 'similarity', 'connect']
    },
    "keyword_categories_zh": { # 中文关键词 (示例，你需要根据你的Prompt和预期调整)
        'Attractiveness': ['吸引力', '颜值', '外貌', '长相', '好看', '漂亮', '帅气'],
        'Sincerity': ['真诚', '诚实', '真心', '可靠', '信任'],
        'Intelligence': ['智力', '聪明', '智慧', '头脑', '学识', '才华'],
        'Funny': ['幽默', '有趣', '搞笑', '风趣', '开心', '乐子'],
        'Ambition': ['抱负', '上进心', '事业心', '目标', '追求', '有作为'],
        'Shared Interests': ['共同兴趣', '共同爱好', '共同话题', '合得来', '相似', '默契']
    }
}