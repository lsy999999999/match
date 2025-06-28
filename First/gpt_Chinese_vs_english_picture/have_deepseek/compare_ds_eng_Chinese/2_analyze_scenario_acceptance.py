# 分场景接受率 (Scenario-based Acceptance Rate)
# 我们需要区分两种求偶场景：
# 场景A: 向单身人士求偶。
# 场景B: 向已有伴侣的人士求偶（挖墙脚）。
# 模型在这两种场景下的决策差异，可能揭示深层的文化偏见。
# 思路：根据 current_partner 列是否为空来区分场景。



# 也许中文Prompt下，“挖墙脚”的成功率显著低于英文Prompt，反映出一种更强的“忠诚”或“稳定”偏好。
# 或者，两种语言在“向单身求偶”时接受率相近，但在“挖墙脚”时差异巨大。这能非常有力地证明语言和文化背景对AI决策的影响。

# 文件名: 2_analyze_scenario_acceptance.py

import matplotlib.pyplot as plt
import seaborn as sns
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
    # 【错误点修正】之前是 df_en, df_zh = ... 这是错误的。
    # 应该分别加载，并用两个变量接收返回值。
    # 我们只需要DataFrame，所以用下划线 _ 忽略掉不需要的matchings列表。
    df_en, _ = load_all_group_data('en')
    df_zh, _ = load_all_group_data('zh')

    # 2. 计算分场景接受率
    def get_rates(df, num_groups):
        single, taken = [], []
        for i in range(1, num_groups + 1):
            group_df = df[df['group'] == i]
            # 场景A: 向单身求偶
            df_single = group_df[group_df['current_partner'].isna()]
            if not df_single.empty: 
                single.append(df_single['result'].sum() / len(df_single) * 100)
            # 场景B: 向非单身求偶
            df_taken = group_df[group_df['current_partner'].notna()]
            if not df_taken.empty: 
                taken.append(df_taken['result'].sum() / len(df_taken) * 100)
        return single, taken

    single_en, taken_en = get_rates(df_en, config["num_groups_en"])
    single_zh, taken_zh = get_rates(df_zh, config["num_groups_zh"])

    # 3. 创建DataFrame并绘图
    plot_df = pd.DataFrame({
        '接受率 (%)': single_en + taken_en + single_zh + taken_zh,
        '场景': ['Seeking a mate from a single person'] * len(single_en) + ['Seeking a mate from a non-single person'] * len(taken_en) + \
               ['Seeking a mate from a single person'] * len(single_zh) + ['Seeking a mate from a non-single person'] * len(taken_zh),
        'Prompt 语言': [f'English (N={config["num_groups_en"]})'] * (len(single_en) + len(taken_en)) + \
                      [f'Chinese (N={config["num_groups_zh"]})'] * (len(single_zh) + len(taken_zh))
    })

    plt.figure(figsize=config["figure_size_large"])
    sns.boxplot(x='场景', y='接受率 (%)', hue='Prompt 语言', data=plot_df, palette=config["plot_palette_muted"])
    plt.title('Distribution of courtship acceptance rate in different scenarios under Chinese and English prompts', fontsize=config["title_fontsize"])
    plt.ylabel('Acceptance rate (%)', fontsize=config["label_fontsize"])
    plt.xlabel('Courtship scenario', fontsize=config["label_fontsize"])
    plt.legend(title='Prompt language and sample number')
    plt.tight_layout()

    # (可选) 添加统计检验标注
    try:
        y_max = plt.ylim()[1]
        t_stat_single, p_value_single = stats.ttest_ind(single_en, single_zh)
        plt.text(-0.2, y_max * 0.95, f'P-value: {p_value_single:.3f}', ha='center', fontsize=11, color='purple')

        t_stat_taken, p_value_taken = stats.ttest_ind(taken_en, taken_zh)
        plt.text(0.8, y_max * 0.95, f'P-value: {p_value_taken:.3f}', ha='center', fontsize=11, color='purple')
        if p_value_taken < 0.05:
             plt.text(0.8, y_max * 1.0, 'big different', ha='center', fontsize=11, color='purple', weight='bold')

    except Exception as e:
        print(f"无法执行统计检验，可能因为某组数据为空: {e}")


    output_filename = "2_Scenario-based_Acceptance_ds_Rate.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"图表已保存为: {output_filename}")


    print("分场景接受率分析完成，图表已生成。")
    

if __name__ == '__main__':
    analyze_and_plot()



# Y轴: 接受率（求偶成功的次数 / 总求偶次数 * 100%）。
# X轴: 两种决策场景，“向单身求偶”和“向非单身求偶”（即“挖墙脚”）。
# 图例 (Hue): 按“英文Prompt”和“中文Prompt”对每个场景的数据进行再分组。
# 核心问题:
# “挖墙脚”的成功率是否普遍低于向单身人士求偶？（比较同一颜色下，左右两个箱体的高低）
# 在“挖墙脚”这个特定场景下，中文Prompt的成功率是否显著低于英文？（比较右边两个不同颜色的箱体）
# 语言的影响是否在不同场景下有差异？（例如，可能向单身求偶时中英文差不多，但挖墙脚时差异巨大）