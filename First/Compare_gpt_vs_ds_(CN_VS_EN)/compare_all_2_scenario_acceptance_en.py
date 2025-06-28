# 分场景jieshoulv
# 目标: 比较四组实验在“向单身求偶”和“挖墙脚”两种场景下的接受率。
# 图表: 分组箱形图 (Grouped Boxplot)。
# 图表元素:
# Title: "Proposal Acceptance Rate by Scenario"
# Y-axis: "Acceptance Rate (%)"
# X-axis: "Scenario" (Labels: "Proposing to Single", "Proposing to Taken")
# Hue/Legend: "Experiment Group" (e.g., "GPT-4 (Eng, N=50)")
# 含义: 揭示不同模型和语言在不同社交情境下的决策倾向，特别是对于“挖墙脚”这种敏感行为的态度。





import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from config import config
from load_data import load_all_group_data_for_model

# 【重要】设置支持中文的字体，确保Matplotlib在处理含中文的标签时不会出错
# 以下列表的顺序很重要，它会依次尝试这些字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans'] 
plt.rcParams['axes.unicode_minus'] = False # 正确显示负号

sns.set_style(config.get("plot_style", "whitegrid"))
sns.set_context(config.get("plot_context", "talk"))
# ... (matplotlib rcParams setup as in script 1) ...
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS'] # Ensure a font that supports your labels
plt.rcParams['axes.unicode_minus'] = False
sns.set_style(config.get("plot_style", "whitegrid"))
sns.set_context(config.get("plot_context", "talk"))


def analyze_and_plot():
    print("--- Analysis 2: Proposal Acceptance Rate by Scenario ---")
    model_keys = ["gpt4_en", "gpt4_zh", "deepseek_en", "deepseek_zh"]
    all_scenario_data = []

    for key in model_keys:
        if key not in config:
            print(f"Warning: Configuration for '{key}' not found. Skipping.")
            continue
        
        df, _ = load_all_group_data_for_model(key)
        model_config = config[key]
        model_label = model_config.get('label', key)
        num_groups = model_config.get('num_groups', 0)

        if df.empty or 'result' not in df.columns:
            print(f"Warning: No data or 'result' column for {model_label}. Skipping.")
            continue

        for i in range(1, num_groups + 1):
            group_df = df[df['group'] == i]
            if group_df.empty: continue

            df_single = group_df[group_df['current_partner'].isna()]
            if not df_single.empty:
                rate_single = df_single['result'].sum() / len(df_single) * 100
                all_scenario_data.append({
                    'Acceptance Rate (%)': rate_single,
                    'Scenario': 'Proposing to Single',
                    'Experiment Group': f"{model_label} (N={num_groups})" # Use total N for the group label
                })

            elif key == "deepseek_zh": # 【新增调试】如果deepseek_zh的df_single为空
                print(f"  调试: For {model_label}, group {i}, 'Proposing to Single' (df_single) is EMPTY.")
                print(f"    group_df['current_partner'].isna().sum() for this group: {group_df['current_partner'].isna().sum()}")




            df_taken = group_df[group_df['current_partner'].notna()]
            if not df_taken.empty:
                rate_taken = df_taken['result'].sum() / len(df_taken) * 100
                all_scenario_data.append({
                    'Acceptance Rate (%)': rate_taken,
                    'Scenario': 'Proposing to Taken',
                    'Experiment Group': f"{model_label} (N={num_groups})"
                })
    
    if not all_scenario_data:
        print("No scenario data to plot.")
        return

    plot_df = pd.DataFrame(all_scenario_data)


    print("\n--- 调试: Data for 'DeepSeek (中文)' in 'Proposing to Single' ---")
    ds_zh_single_data = plot_df[
        (plot_df['Experiment Group'] == f"{config['deepseek_zh']['label']} (N={config['deepseek_zh']['num_groups']})") &
        (plot_df['Scenario'] == 'Proposing to Single')
    ]
    print(ds_zh_single_data)
    if ds_zh_single_data.empty:
        print("错误：DeepSeek (中文) 在 Proposing to Single 场景的数据在最终plot_df中为空！")
    elif len(ds_zh_single_data) != config['deepseek_zh']['num_groups']:
        print(f"警告：DeepSeek (中文) 在 Proposing to Single 场景的数据点数量 ({len(ds_zh_single_data)}) 与期望的组数 ({config['deepseek_zh']['num_groups']}) 不符。")


    # 【临时测试】
    # 只绘制DeepSeek中文的数据
    temp_plot_df_ds_zh = plot_df[plot_df['Experiment Group'] == f"{config['deepseek_zh']['label']} (N={config['deepseek_zh']['num_groups']})"]
    if not temp_plot_df_ds_zh.empty:
        plt.figure(figsize=config.get("figure_size_medium"))
        sns.boxplot(x='Scenario', y='Acceptance Rate (%)', data=temp_plot_df_ds_zh, color='red') # 强制红色
        plt.title(f"DEBUG: {config['deepseek_zh']['label']} Data Only")
        plt.savefig("debug_deepseek_zh_scenario_acceptance.png", dpi=300)
        plt.close()
    else:
        print("DEBUG: temp_plot_df_ds_zh is empty.")
    # 【临时测试结束】


    plt.figure(figsize=config.get("figure_size_large", (14,9)))
    sns.boxplot(x='Scenario', y='Acceptance Rate (%)', hue='Experiment Group', data=plot_df, 
                palette=config.get("plot_palette_models_lang", "tab10"))
    plt.title('Proposal Acceptance Rate by Scenario', fontsize=config.get("title_fontsize", 18))
    plt.ylabel('Acceptance Rate (%)', fontsize=config.get("label_fontsize", 14))
    plt.xlabel('Scenario', fontsize=config.get("label_fontsize", 14))
    plt.legend(title='Experiment Group', fontsize=config.get("legend_fontsize", 12), bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend

    output_filename = "compare_all_2_scenario_acceptance_en.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Chart saved as: {output_filename}")
    plt.close()

if __name__ == '__main__':
    analyze_and_plot()