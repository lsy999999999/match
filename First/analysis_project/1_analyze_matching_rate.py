



#分析，绘制最终配对率
#在所有的求偶尝试中，GPT的同意率分别是多少？
#这反映了模型在不同语言下的“严格”或“宽松”程度。例如，也许中文语境下的模型更“保守”，接受率更低。

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
from config import config
from load_data import load_all_group_data

# 设置绘图环境
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style(config["plot_style"])
sns.set_context(config["plot_context"])

def analyze_and_plot():
    # 1. 加载数据
    _, matchings_en = load_all_group_data('en')
    _, matchings_zh = load_all_group_data('zh')
    
    # 2. 计算配对率
    rates_en = [len({p for p, partner in m.items() if partner is not None and partner != 'rejected'}) / len(m) * 100 for m in matchings_en if m]
    rates_zh = [len({p for p, partner in m.items() if partner is not None and partner != 'rejected'}) / len(m) * 100 for m in matchings_zh if m]
    
    # 3. 创建DataFrame并绘图
    plot_df = pd.DataFrame({
        '配对率 (%)': rates_en + rates_zh,
        'Prompt 语言': [f'English (N={len(rates_en)})'] * len(rates_en) + \
                       [f'Chinese (N={len(rates_zh)})'] * len(rates_zh)
    })
    
    plt.figure(figsize=config["figure_size_medium"])
    sns.boxplot(x='Prompt 语言', y='配对率 (%)', data=plot_df, palette=config["plot_palette_pastel"])
    sns.stripplot(x='Prompt 语言', y='配对率 (%)', data=plot_df, color=".25", size=4)
    plt.title('Distribution of final matching rates under Chinese and English prompts', fontsize=config["title_fontsize"])
    plt.ylabel('Matching rate (%)', fontsize=config["label_fontsize"])
    plt.xlabel('Prompt language and number of samples', fontsize=config["label_fontsize"])
    #  plt.title('中英文Prompt下最终配对率分布', fontsize=config["title_fontsize"])
    # plt.ylabel('配对率 (%)', fontsize=config["label_fontsize"])
    # plt.xlabel('Prompt 语言及样本数', fontsize=config["label_fontsize"])
    
    # 4. 添加统计检验
    t_stat, p_value = stats.ttest_ind(rates_en, rates_zh)
    y_max = plt.ylim()[1]
    plt.text(0.5, y_max * 0.95, f'T-test P-value: {p_value:.4f}', ha='center', fontsize=12, color='red')
    

    output_filename = "1_matching_rate_analysis.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"图表已保存为: {output_filename}")

    print("最终配对率分析完成，图表已生成。")
    

if __name__ == '__main__':
    analyze_and_plot()



# Y轴: 配对率（成功找到伴侣的人数 / 总人数 * 100%）。
# X轴: 两个类别，“英文Prompt”和“中文Prompt”。
# 箱体: 代表了中间50%的实验结果。箱子的上下边缘分别是上四分位数（75%）和下四分位数（25%），中间的横线是中位数（50%）。
# 小黑点: 代表每一次独立实验的最终配对率结果。
# P-value: 告诉你两种语言下的平均配对率差异是否显著。如果 P < 0.05，说明这种差异不太可能是偶然发生的。
# 核心问题: “使用不同语言的提示词，会影响最终的配对成功率吗？”