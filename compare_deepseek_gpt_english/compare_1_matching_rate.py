# 文件名: compare_1_matching_rate.py

# 核心问题：“DeepSeek模型在英文语境下产生的最终配对成功率，与GPT-4相比，是否存在显著差异？”
# 图表类型：箱形图（Boxplot）叠加散点图（Stripplot）。
# Y轴：最终配对率（%）。
# X轴：两个模型，标签中注明了各自的有效实验组数（例如 "GPT-4 (Eng, N=50)"）。
# 解读：
# 比较两个箱体的位置和大小：哪个模型的中位数配对率更高？哪个模型的配对率更稳定（箱体更窄）？
# 观察P-value：如果P < 0.05，说明两个模型在平均配对率上的差异具有统计学显著性。由于样本量不同，这里使用了Welch's t-test，它对样本量不均和方差不齐的情况更鲁棒。


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
from config import config
from load_data import load_all_group_data_for_model

# 设置绘图环境
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style(config["plot_style"])
sns.set_context(config["plot_context"])

def analyze_and_plot():
    print("--- 对比分析1: 最终配对率 ---")
    
    # 加载数据
    _, matchings_gpt4 = load_all_group_data_for_model('gpt4_en')
    _, matchings_deepseek = load_all_group_data_for_model('deepseek_en')
    
    # 计算配对率
    def get_rates(matchings):
        rates = []
        for m in matchings:
            if not m or len(m) == 0: continue
            total = len(m)
            matched = len({p for p, partner in m.items() if partner is not None and partner != 'rejected'})
            rates.append(matched / total * 100)
        return rates

    rates_gpt4 = get_rates(matchings_gpt4)
    rates_deepseek = get_rates(matchings_deepseek)
    
    # 创建DataFrame并绘图
    plot_df = pd.DataFrame({
        '配对率 (%)': rates_gpt4 + rates_deepseek,
        '模型': [config['gpt4_en']['label']] * len(rates_gpt4) + \
                [config['deepseek_en']['label']] * len(rates_deepseek)
    })
    
    plt.figure(figsize=config["figure_size_medium"])
    sns.boxplot(x='模型', y='配对率 (%)', data=plot_df, palette=config["plot_palette_models"])
    sns.stripplot(x='模型', y='配对率 (%)', data=plot_df, color=".25", size=4)
    plt.title('GPT-4 vs DeepSeek: Final simulation rate distribution', fontsize=config["title_fontsize"])
    plt.ylabel('Display matching rate (%)', fontsize=config["label_fontsize"])
    plt.xlabel('Model and number of effective experimental groups', fontsize=config["label_fontsize"])
    
    # 添加统计检验 (Welch's t-test for unequal sample sizes)
    if rates_gpt4 and rates_deepseek:
        t_stat, p_value = stats.ttest_ind(rates_gpt4, rates_deepseek, equal_var=False)
        y_max = plt.ylim()[1]
        plt.text(0.5, y_max * 0.95, f'P-value (Welch\'s t-test): {p_value:.4f}', 
                 ha='center', fontsize=12, color='red')
    
    output_filename = "compare_1_matching_rate.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"图表已保存为: {output_filename}")
    plt.close() # 关闭图形

if __name__ == '__main__':
    analyze_and_plot()