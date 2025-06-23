# 文件名: compare_2_scenario_acceptance.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
from config import config
from load_data import load_all_group_data_for_model

# 设置绘图环境... (同上)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style(config["plot_style"])
sns.set_context(config["plot_context"])

def analyze_and_plot():
    print("--- 对比分析2: 分场景接受率 ---")
    df_gpt4, _ = load_all_group_data_for_model('gpt4_en')
    df_deepseek, _ = load_all_group_data_for_model('deepseek_en')

    def get_scenario_rates(df, num_groups_for_this_model):
        single, taken = [], []
        for i in range(1, num_groups_for_this_model + 1): # 使用对应模型的组数
            group_df = df[df['group'] == i]
            if group_df.empty: continue
            df_single = group_df[group_df['current_partner'].isna()]
            if not df_single.empty: single.append(df_single['result'].sum() / len(df_single) * 100)
            df_taken = group_df[group_df['current_partner'].notna()]
            if not df_taken.empty: taken.append(df_taken['result'].sum() / len(df_taken) * 100)
        return single, taken

    single_gpt4, taken_gpt4 = get_scenario_rates(df_gpt4, config['gpt4_en']['num_groups'])
    single_deepseek, taken_deepseek = get_scenario_rates(df_deepseek, config['deepseek_en']['num_groups'])

    plot_df = pd.DataFrame({
        '接受率 (%)': single_gpt4 + taken_gpt4 + single_deepseek + taken_deepseek,
        '场景': ['Seeking a mate from a single person'] * len(single_gpt4) + ['Seeking a mate from a non-single person'] * len(taken_gpt4) + \
               ['Seeking a mate from a single person'] * len(single_deepseek) + ['Seeking a mate from a non-single person'] * len(taken_deepseek),
        '模型': [config['gpt4_en']['label']] * (len(single_gpt4) + len(taken_gpt4)) + \
                [config['deepseek_en']['label']] * (len(single_deepseek) + len(taken_deepseek))
    })

    plt.figure(figsize=config["figure_size_large"])
    sns.boxplot(x='场景', y='接受率 (%)', hue='模型', data=plot_df, palette=config["plot_palette_models"])
    plt.title('GPT-4 vs DeepSeek: Distribution of acceptance rate of courtship by scene', fontsize=config["title_fontsize"])
    plt.ylabel('Acceptance rate (%)', fontsize=config["label_fontsize"])
    plt.xlabel('Courtship scene', fontsize=config["label_fontsize"])
    plt.legend(title='Model and number of effective experimental groups')
    plt.tight_layout()
    
    output_filename = "compare_2_scenario_acceptance.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"图表已保存为: {output_filename}")
    plt.close()

if __name__ == '__main__':
    analyze_and_plot()


# 核心问题：“在不同的求偶场景下（向单身求偶 vs. 挖墙脚），DeepSeek和GPT-4的决策偏好（接受率）有何不同？”
# 图表类型：分组箱形图。
# Y轴：接受率（%）。
# X轴：两种场景，“向单身求偶”和“向非单身求偶”。
# 图例 (Hue)：按两个模型（GPT-4 vs. DeepSeek）对每个场景的数据进行再分组。
# 解读：
# 模型内比较：对于GPT-4（或DeepSeek），“挖墙脚”的接受率是否显著低于“向单身求偶”？
# 模型间比较：
# 在“向单身求偶”场景下，DeepSeek的接受率与GPT-4相比如何？
# 在“向非单身求偶”场景下，DeepSeek的接受率与GPT-4相比如何？（这可能是最有意思的发现点）
# 观察DeepSeek的箱体是否比GPT-4的更宽或更窄，这可以反映其决策的稳定性。
