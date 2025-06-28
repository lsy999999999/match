#最终配对率对比 
# 目标: 比较四组实验最终的平均配对成功率。
# 图表: 箱形图 (Boxplot)。
# 图表元素:
# Title: "Final Matching Rate Distribution (GPT-4 vs DeepSeek, Eng/中文)"
# Y-axis: "Matching Rate (%)"
# X-axis: "Experiment Group" (标签如 "GPT-4 (Eng, N=50)")
# Legend: (如果颜色代表模型，形状代表语言，或者直接用X轴区分)
# 含义: 展示了不同模型和语言组合下，配对成功的整体情况和稳定性。P-value（如果添加）可以指示组间平均配对率差异是否显著。


# compare_all_1_matching_rate_en.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
from config import config
from load_data import load_all_group_data_for_model # Use this

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS'] # Ensure a font that supports your labels
plt.rcParams['axes.unicode_minus'] = False
sns.set_style(config.get("plot_style", "whitegrid"))
sns.set_context(config.get("plot_context", "talk"))

def analyze_and_plot():
    print("--- Analysis 1: Final Matching Rate Comparison ---")
    model_keys = ["gpt4_en", "gpt4_zh", "deepseek_en", "deepseek_zh"]
    all_rates_data = []

    for key in model_keys:
        if key not in config:
            print(f"Warning: Configuration for '{key}' not found in config.py. Skipping.")
            continue
        _, matchings = load_all_group_data_for_model(key)
        model_label = config[key].get('label', key)
        num_valid_groups = 0
        rates = []
        for m_idx, m in enumerate(matchings):
            if not m: 
                # print(f"Warning: Empty matching result for {model_label}, group {m_idx+1}")
                continue
            total = len(m)
            if total == 0: continue
            matched = len({p for p, partner in m.items() if partner is not None and str(partner).lower() != 'rejected'})
            rates.append(matched / total * 100)
            num_valid_groups +=1
        
        for rate in rates:
            all_rates_data.append({'Matching Rate (%)': rate, 'Experiment Group': f"{model_label} (N={num_valid_groups})"})

    if not all_rates_data:
        print("No data to plot for matching rates.")
        return

    plot_df = pd.DataFrame(all_rates_data)
    
    plt.figure(figsize=config.get("figure_size_medium", (12, 8)))
    sns.boxplot(x='Experiment Group', y='Matching Rate (%)', data=plot_df, palette=config.get("plot_palette_models_lang", "tab10"))
    sns.stripplot(x='Experiment Group', y='Matching Rate (%)', data=plot_df, color=".25", size=4, jitter=True)
    
    plt.title('Final Matching Rate Distribution', fontsize=config.get("title_fontsize", 18))
    plt.ylabel('Matching Rate (%)', fontsize=config.get("label_fontsize", 14))
    plt.xlabel('Experiment Group (Model, Language, Valid N)', fontsize=config.get("label_fontsize", 14))
    plt.xticks(rotation=15, ha="right") # Rotate x-axis labels if they overlap
    plt.tight_layout()
    
    output_filename = "compare_all_1_matching_rate_en.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Chart saved as: {output_filename}")
    plt.close()

if __name__ == '__main__':
    analyze_and_plot()