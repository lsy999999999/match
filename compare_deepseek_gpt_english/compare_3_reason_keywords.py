# 文件名: compare_3_reason_keywords.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from config import config
from load_data import load_all_group_data_for_model
from collections import Counter
import re # 用于简单文本清理

# 设置绘图环境... (同上)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style(config["plot_style"])
sns.set_context(config["plot_context"])

# 定义英文关键词 (你可以根据实际观察到的模型用词进行调整和扩展)
# 将相似的词归为一类
KEYWORD_CATEGORIES_EN = {
    'Attractiveness': ['attractive', 'attraction', 'appearance', 'looks', 'physical'],
    'Sincerity': ['sincere', 'sincerity', 'honest', 'honesty', 'genuine', 'trust'],
    'Intelligence': ['intelligence', 'intelligent', 'smart', 'intellectual', 'knowledge'],
    'Funny': ['funny', 'humor', 'humour', 'laugh', 'entertaining'],
    'Ambition': ['ambition', 'ambitious', 'career', 'driven', 'goals', 'aspiring'],
    'Shared Interests': ['shared interest', 'shared interests', 'common interest', 'common interests', 'hobbies', 'activities', 'compatible']
}

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text) # 移除标点
    return text

def count_keywords_in_reasons(df, keyword_categories):
    all_reasons_text = " ".join(df['reason'].dropna().astype(str).apply(clean_text).tolist())
    
    category_counts = Counter()
    for category, keywords in keyword_categories.items():
        for keyword in keywords:
            category_counts[category] += all_reasons_text.count(keyword)
    return category_counts

def analyze_and_plot():
    print("--- 对比分析3: 决策理由关键词频率 ---")
    df_gpt4, _ = load_all_group_data_for_model('gpt4_en')
    df_deepseek, _ = load_all_group_data_for_model('deepseek_en')

    if df_gpt4.empty or df_deepseek.empty:
        print("警告: 至少一个模型的数据为空，无法进行关键词分析。")
        return

    counts_gpt4 = count_keywords_in_reasons(df_gpt4, KEYWORD_CATEGORIES_EN)
    counts_deepseek = count_keywords_in_reasons(df_deepseek, KEYWORD_CATEGORIES_EN)

    # 创建DataFrame用于绘图
    df_plot_gpt4 = pd.DataFrame.from_dict(counts_gpt4, orient='index', columns=['Frequency'])
    df_plot_gpt4['Model'] = config['gpt4_en']['label']
    df_plot_gpt4['Dimension'] = df_plot_gpt4.index

    df_plot_deepseek = pd.DataFrame.from_dict(counts_deepseek, orient='index', columns=['Frequency'])
    df_plot_deepseek['Model'] = config['deepseek_en']['label']
    df_plot_deepseek['Dimension'] = df_plot_deepseek.index
    
    combined_plot_df = pd.concat([df_plot_gpt4, df_plot_deepseek]).reset_index(drop=True)

    # 绘图
    plt.figure(figsize=config["figure_size_large"])
    sns.barplot(x='Frequency', y='Dimension', hue='Model', data=combined_plot_df, 
                palette=config["plot_palette_models"], orient='h')
    plt.title('GPT-4 vs DeepSeek: Frequency of keywords in each dimension in decision reasons', fontsize=config["title_fontsize"])
    plt.xlabel('Cumulative frequency of keywords', fontsize=config["label_fontsize"])
    plt.ylabel('Evaluation dimension', fontsize=config["label_fontsize"])
    plt.legend(title='Model')
    plt.tight_layout()

    output_filename = "compare_3_reason_keywords.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"图表已保存为: {output_filename}")
    plt.close()

if __name__ == '__main__':
    analyze_and_plot()