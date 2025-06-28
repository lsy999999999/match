import json
import numpy as np
from openai import OpenAI
import json
import re


# 保存对话历史到本地文件
def save_conversation_history(conversation_history, filename):
    with open(filename, 'w') as file:
        json.dump(conversation_history, file)


# 从本地文件加载对话历史
def load_conversation_history(filename):
    try:
        with open(filename, 'r') as file:
            conversation_history = json.load(file)
        return conversation_history
    except FileNotFoundError:
        return []
# 处理单身男生提议的场景
def ask_gpt_single_man_propose(man_score):
    words = [np.array(man_score[:3]), np.array(man_score[4:10])]
    words = [item for sublist in words for item in sublist]
    assert len(words) == 9
    prompt = (
        r'Assume you are a {}-year-old single woman, and there is a {}-year-old guy who is courting you. '
        r'Your career type is {}. You evaluated him on six dimensions. These are attractiveness: {}, sincerity: {}, '
        r'intelligence: {}, being funny: {}, ambition: {}, shared interests: {}, would you be willing to date this guy? '
        r'And also tell me the reason. Your answer must be (YES) or (NO) followed by your reason. You must FOLLOW THIS RULE! '
        r'You MUST NOT SAY other words.'.format(*words)
    )
    
    return ask_gpt_api_call(prompt)

# 处理单身女生提议的场景
def ask_gpt_single_woman_propose(woman_score):
    words = [np.array(woman_score[:3]), np.array(woman_score[4:10])]
    words = [item for sublist in words for item in sublist]
    assert len(words) == 9
    prompt = (
        r'Assume you are a {}-year-old single man, and there is a {}-year-old woman who is courting you. '
        r'Your career type is {}. You evaluated her on six dimensions. These are attractiveness: {}, sincerity: {}, '
        r'intelligence: {}, being funny: {}, ambition: {}, shared interests: {}, would you be willing to date this woman? '
        r'And also tell me the reason. Your answer must be (YES) or (NO) followed by your reason. You must FOLLOW THIS RULE! '
        r'You MUST NOT SAY other words.'.format(*words)
    )
    
    return ask_gpt_api_call(prompt)

# 处理有伴男生提议的场景
def ask_gpt_non_single_man_propose(man_score, current_man_score):
    words = [np.array(current_man_score[:3]), np.array(current_man_score[4:10])]
    words = [item for sublist in words for item in sublist]
    words.append(man_score[1])  # new man's age
    for i in range(4, 10):
        words.append(man_score[i])
    
    assert len(words) == 16
    prompt = (
        r'Assume you are a {}-year-old woman, and you have been in a relationship with your {}-year-old boyfriend recently. '
        r'Your career type is {}. Here is how you rated your boyfriend on the six dimensions: attractiveness: {}, sincerity: {}, '
        r'intelligence: {}, being funny: {}, ambition: {}, shared interests: {}. Now, another {}-year-old man is chasing you. '
        r'You had a date with him, and you also scored him on the six dimensions, considering the same varying levels of importance: '
        r'attractiveness: {}, sincerity: {}, intelligence: {}, being funny: {}, ambition: {}, shared interests: {}. If you believe '
        r'the person chasing you excels in dimensions that you consider more important than your current boyfriend, you might consider '
        r'ending the current relationship to explore a new one. Would you be willing to end your current relationship to start a new one with the man chasing you? '
        r'And also tell me the reason. Your answer must be (YES) or (NO) followed by your reason. You must FOLLOW THIS RULE! '
        r'You MUST NOT say other words! '.format(*words)
    )
    
    return ask_gpt_api_call(prompt)

# 处理有伴女生提议的场景
def ask_gpt_non_single_woman_propose(woman_score, current_woman_score):
    words = [np.array(current_woman_score[:3]), np.array(current_woman_score[4:10])]
    words = [item for sublist in words for item in sublist]
    words.append(woman_score[1])  # new woman's age
    for i in range(4, 10):
        words.append(woman_score[i])
    
    assert len(words) == 16
    prompt = (
        r'Assume you are a {}-year-old man, and you have been in a relationship with your {}-year-old girlfriend recently. '
        r'Your career type is {}. Here is how you rated your girlfriend on the six dimensions: attractiveness: {}, sincerity: {}, '
        r'intelligence: {}, being funny: {}, ambition: {}, shared interests: {}. Now, another {}-year-old woman is chasing you. '
        r'You had a date with her, and you also scored her on the six dimensions, considering the same varying levels of importance: '
        r'attractiveness: {}, sincerity: {}, intelligence: {}, being funny: {}, ambition: {}, shared interests: {}. If you believe '
        r'the person chasing you excels in dimensions that you consider more important than your current girlfriend, you might consider '
        r'ending the current relationship to explore a new one. Would you be willing to end your current relationship to start a new one with the woman chasing you? '
        r'And also tell me the reason. Your answer must be (YES) or (NO) followed by your reason. You must FOLLOW THIS RULE! '
        r'You MUST NOT say other words! '.format(*words)
    )
    
    return ask_gpt_api_call(prompt)


# 统一的API调用函数，负责调用 GPT 接口
def ask_gpt_api_call(prompt):
    history_filename = 'conversation_history.json'
    
    # 模拟调用 GPT API 的逻辑
    api_key = "sk-DG9ZYLfnjSbUPt3TE6OTpN3Ge9Mr9hLAgg4UBWxCba3ryvYT"
    client = OpenAI(api_key=api_key, base_url="https://aigcbest.top/v1")
    model = "deepseek-ai/DeepSeek-R1-0528"
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt}
        ]
    )
    
    result = response.choices[0].message.content
    # print(prompt, result)

    log = []
    log.append(prompt)
    log.append(result)

    if 'yes' in result.lower():
        flag = True
    elif 'no' in result.lower():
        flag = False
    else:
        flag = -1

    return flag, log



if __name__=='__main__':
    # 示例使用
    result, log = ask_deepseek_eng(man_score=[25, 30, 'Programmer', 70, 80, 90, 60, 70, 80, 15, 15, 20, 30, 10, 10])
    # print(result)
