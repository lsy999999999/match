# 文件名: compare_3_reason_keywords_nltk_en.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# 导入你的项目配置和数据加载函数
from config import config  # 假设config.py中包含英文标签和路径
from load_data import load_all_group_data_for_model # 确保此函数能正确加载数据

# --- 配置 ---
# 设置绘图环境 (如果需要全局英文，可以在这里设置locale，但通常Matplotlib会根据标签语言自动调整)
sns.set_style(config.get("plot_style", "whitegrid"))
sns.set_context(config.get("plot_context", "talk"))

# 英文关键词类别 (保持不变或微调)
KEYWORD_CATEGORIES_EN = {
    'Attractiveness': ['attractive', 'attraction', 'appearance', 'look', 'physical', 'beauty', 'handsome', 'pretty'],
    'Sincerity': ['sincere', 'sincerity', 'honest', 'honesty', 'genuine', 'trust', 'truthful'],
    'Intelligence': ['intelligence', 'intelligent', 'smart', 'intellectual', 'knowledgeable', 'bright', 'clever'],
    'Funny': ['funny', 'humor', 'humour', 'laugh', 'entertaining', 'jovial', 'witty'],
    'Ambition': ['ambition', 'ambitious', 'career', 'driven', 'goal', 'aspiring', 'success-oriented'],
    'Shared Interests': ['shared interest', 'common interest', 'hobby', 'activity', 'compatible', 'similarity', 'connect']
}

# 初始化NLTK工具
lemmatizer = WordNetLemmatizer()
stop_words_en = set(stopwords.words('english'))

def preprocess_text_nltk(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text) # 移除标点
    tokens = word_tokenize(text)        # 分词
    # 词形还原并移除停用词
    processed_tokens = [
        lemmatizer.lemmatize(token) for token in tokens if token not in stop_words_en and len(token) > 1
    ]
    return processed_tokens

def calculate_keyword_proportions_nltk(df, keyword_categories):
    all_processed_tokens_flat = []
    
    if 'reason' not in df.columns or df['reason'].dropna().empty:
        return pd.Series({category: 0.0 for category in keyword_categories.keys()}, dtype=float), 0

    for reason_text in df['reason'].dropna().astype(str):
        all_processed_tokens_flat.extend(preprocess_text_nltk(reason_text))

    total_meaningful_words = len(all_processed_tokens_flat)
    if total_meaningful_words == 0:
        return pd.Series({category: 0.0 for category in keyword_categories.keys()}, dtype=float), 0

    word_counts = Counter(all_processed_tokens_flat)
    
    category_proportions = {}
    for category, keywords in keyword_categories.items():
        category_token_count = 0
        # 对关键词也进行词形还原，以匹配处理后的文本
        lemmatized_keywords = {lemmatizer.lemmatize(kw.lower()) for kw in keywords}
        
        for token, count in word_counts.items():
            if token in lemmatized_keywords:
                category_token_count += count
        
        category_proportions[category] = (category_token_count / total_meaningful_words) * 100 if total_meaningful_words > 0 else 0
        
    return pd.Series(category_proportions), total_meaningful_words

def analyze_and_plot():
    print("--- Comparative Analysis 3: Keyword Proportion in Decision Reasons (NLTK Enhanced, English Output) ---")
    
    # 从config获取模型标识符和标签
    model1_key = 'gpt4_en' # 示例，请确保config.py中有这些键
    model2_key = 'deepseek_en' # 示例

    if model1_key not in config or model2_key not in config:
        print(f"Error: Configuration for '{model1_key}' or '{model2_key}' not found in config.py.")
        return

    df_model1, _ = load_all_group_data_for_model(model1_key)
    df_model2, _ = load_all_group_data_for_model(model2_key)

    label_model1 = config[model1_key].get('label', 'Model 1') + f" (N={config[model1_key].get('num_groups', 0)})"
    label_model2 = config[model2_key].get('label', 'Model 2') + f" (N={config[model2_key].get('num_groups', 0)})"


    proportions_model1, total_words_model1 = calculate_keyword_proportions_nltk(df_model1, KEYWORD_CATEGORIES_EN)
    proportions_model2, total_words_model2 = calculate_keyword_proportions_nltk(df_model2, KEYWORD_CATEGORIES_EN)

    print(f"{label_model1} - Total Meaningful Words in Reasons: {total_words_model1}")
    print(f"{label_model2} - Total Meaningful Words in Reasons: {total_words_model2}")

    # 创建DataFrame用于绘图
    df_plot_model1 = proportions_model1.reset_index()
    df_plot_model1.columns = ['Dimension', 'Proportion (%)']
    df_plot_model1['Model'] = label_model1

    df_plot_model2 = proportions_model2.reset_index()
    df_plot_model2.columns = ['Dimension', 'Proportion (%)']
    df_plot_model2['Model'] = label_model2
    
    combined_plot_df = pd.concat([df_plot_model1, df_plot_model2])

    # 绘图
    plt.figure(figsize=config.get("figure_size_large", (14, 9)))
    sns.barplot(x='Proportion (%)', y='Dimension', hue='Model', data=combined_plot_df, 
                palette=config.get("plot_palette_models", "viridis"), orient='h')
    
    # 更新标题和标签为英文
    plt.title('GPT-4 vs DeepSeek: Keyword Proportion by Dimension in Decision Reasons', fontsize=config.get("title_fontsize", 18))
    plt.xlabel('Proportion of Keywords in Total Reason Tokens (%)', fontsize=config.get("label_fontsize", 14))
    plt.ylabel('Evaluation Dimension', fontsize=config.get("label_fontsize", 14))
    plt.legend(title='Model') # 图例标题也应该是英文
    plt.tight_layout()

    output_filename = "compare_3_reason_keywords_proportion_nltk_en.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Chart saved as: {output_filename}")
    plt.close()

if __name__ == '__main__':
    # 确保config.py中有正确的模型配置键，例如 'gpt4_en' 和 'deepseek_en'
    # 例如:
    # config['gpt4_en'] = {'label': 'GPT-4 (Eng)', 'num_groups': 50, 'base_path': '...', ...}
    # config['deepseek_en'] = {'label': 'DeepSeek (Eng)', 'num_groups': 20, 'base_path': '...', ...}
    # 以及 config['plot_palette_models']
    analyze_and_plot()