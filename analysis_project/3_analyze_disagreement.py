# 量化了语言变化到底造成了多大比例的“决策反转”。

# 3a_overall_disagreement.png (饼图):
# 核心问题: “语言变化对决策结果的‘扰动’有多大？”
# 它显示了在所有完全相同的决策情境下，有多少比例的决策因为语言从英文换成中文而发生了反转（比如英文说YES，中文说NO）。这个比例越高，说明模型对语言越敏感。
# 3b_scenario_disagreement_rate.png (条形图):
# 核心问题: “哪种场景下的决策更容易被语言影响？”
# 它分别计算了“向单身求偶”和“向非单身求偶”这两种场景下的不一致率。你可能会发现，在“挖墙脚”这种更具道德模糊性的场景下，语言和文化暗示的影响力更大，导致更高的决策不一致率。这是一个非常有价值的发现。

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from config import config
from load_data import load_all_group_data

# 设置绘图环境
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style(config["plot_style"])
sns.set_context(config["plot_context"])

def analyze_and_plot():
    print("--- 任务3: 分析决策不一致性 ---")
    df_en, _ = load_all_group_data('en')
    df_zh, _ = load_all_group_data('zh')

    # 1. 创建唯一情境ID
    df_en['context_id'] = df_en.apply(lambda r: f"{r['group']}-{r['proposer']}-{r['target']}-{r.get('current_partner', 'None')}", axis=1)
    df_zh['context_id'] = df_zh.apply(lambda r: f"{r['group']}-{r['proposer']}-{r['target']}-{r.get('current_partner', 'None')}", axis=1)

    # 2. 合并，找到可比较的决策
    merged_df = pd.merge(
        df_en[['context_id', 'result', 'current_partner']],
        df_zh[['context_id', 'result']],
        on='context_id',
        suffixes=('_en', '_zh')
    )

    # 【健壮性修复】检查merged_df是否为空
    if merged_df.empty:
        print("错误: 英文和中文数据中没有找到任何可供比较的共同决策情境。")
        print("这可能是因为大量数据文件缺失或为空。请检查你的数据文件和config.py中的组数设置。")
        print("无法生成不一致性分析图表。")
        return # 提前退出函数

    # 3. 找出不一致的决策
    disagreements = merged_df[merged_df['result_en'] != merged_df['result_zh']]
    total_comparable = len(merged_df)
    disagreement_count = len(disagreements)
    
    print(f"在 {total_comparable} 次可比较的决策中，发现了 {disagreement_count} 次不一致。")

    # --- 图表1: 总体不一致率 ---
    # 【KeyError修复】确保config.py中有 'figure_size_small'
    plt.figure(figsize=config["figure_size_small"])
    plt.pie([disagreement_count, total_comparable - disagreement_count],
            labels=[f'diffenrent decision\n({disagreement_count})', f'same decision\n({total_comparable - disagreement_count})'],
            autopct='%1.1f%%', startangle=90, colors=['#FF6347', '#90EE90'])
    plt.title('Analysis of consistency of decision in Chinese and English prompts', fontsize=config["title_fontsize"])
    output_filename_1 = "3a_overall_disagreement.png"
    plt.savefig(output_filename_1, dpi=300, bbox_inches='tight')
    print(f"Chart has been saved as: {output_filename_1}")
    plt.close() # 关闭当前图形，防止后续的plt.show()显示它

    # --- 图表2: 分场景的不一致率 ---
    # 【健壮性修复】检查是否有不一致的案例
    if disagreement_count > 0:
        disagreements = disagreements.copy() # 避免SettingWithCopyWarning
        disagreements['is_taken'] = disagreements['current_partner'].notna()
        merged_df['is_taken'] = merged_df['current_partner'].notna()
        
        scene_disagreement_counts = disagreements.groupby('is_taken')['context_id'].count()
        scene_total_counts = merged_df.groupby('is_taken')['context_id'].count()
        
        scene_disagreement_rate = (scene_disagreement_counts / scene_total_counts * 100).fillna(0)
        scene_disagreement_rate.index = scene_disagreement_rate.index.map({True: '向非单身求偶', False: '向单身求偶'})
        
        # 确保两个场景都存在，即使其中一个没有不一致
        if '向非单身求偶' not in scene_disagreement_rate.index:
            scene_disagreement_rate['向非单身求偶'] = 0
        if '向单身求偶' not in scene_disagreement_rate.index:
            scene_disagreement_rate['向单身求偶'] = 0

        plt.figure(figsize=config["figure_size_medium"])
        # 【KeyError修复】确保config.py中有 'plot_palette_viridis'
        sns.barplot(x=scene_disagreement_rate.index, y=scene_disagreement_rate.values, palette=config.get("plot_palette_viridis", "viridis"))
        plt.title('Decision inconsistency rate by scene', fontsize=config["title_fontsize"])
        plt.ylabel('Inconsistency rate (%)', fontsize=config["label_fontsize"])
        plt.xlabel('Courtship scene', fontsize=config["label_fontsize"])
        for index, value in enumerate(scene_disagreement_rate.values):
            plt.text(index, value + 0.5, f'{value:.2f}%', ha='center')
        
        output_filename_2 = "3b_scenario_disagreement_rate.png"
        plt.savefig(output_filename_2, dpi=300, bbox_inches='tight')
        print(f"图表已保存为: {output_filename_2}")
        plt.close() # 关闭当前图形
    else:
        print("没有发现任何不一致的决策，因此不生成分场景不一致率图表。")

if __name__ == '__main__':
    analyze_and_plot()