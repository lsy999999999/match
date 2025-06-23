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
        r'假设你是一位{}岁的单身女性，而一位{}岁的男士正在追求你。'
        r'你的职业类型是{}。你从六个维度对他进行了评估。这六个维度分别是：吸引力：{}，真诚度：{}, '
        r'智力：{}，幽默感：{}，抱负：{}，共同兴趣：{}，你愿意和这位男士约会吗？ '
        r'请告诉我原因。你的答案必须是（是）或（否），然后加上你的理由。你必须遵守这条规则！ '
        r'你不能说其他话。'.format(*words)
    )
    
    return ask_gpt_api_call(prompt)

# 处理单身女生提议的场景
def ask_gpt_single_woman_propose(woman_score):
    words = [np.array(woman_score[:3]), np.array(woman_score[4:10])]
    words = [item for sublist in words for item in sublist]
    assert len(words) == 9
    prompt = (
        r'假设你是一位 {} 岁的单身男士，有一位 {} 岁的女性正在追求你。 '
        r'你的职业类型是 {}。你从六个维度评估了她。这六个维度分别是：吸引力：{}，真诚度：{},'
        r'智力：{}，幽默感：{}，抱负：{}，共同兴趣：{}，你愿意和这位女士约会吗？ '
        r'请告诉我原因。你的答案必须是（是）或（否），然后加上你的理由。你必须遵守这条规则！ '
        r'你不能说其他话。'.format(*words)
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
        r'假设你是一位 {} 岁的女性，最近和你 {} 岁的男朋友交往。 '
        r'你的职业类型是 {}。你对男朋友在六个维度上的评分如下：吸引力：{}，真诚：{}, '
        r'智力：{}，幽默感：{}，抱负：{}，共同兴趣：{}。现在，另一位 {} 岁的男人正在追求你。 '
        r'你和他约会过，你也根据同样不同的重要性，在六个维度上给他打了分： '
        r'吸引力：{}，真诚：{}，智力：{}，幽默感：{}，抱负：{}，共同兴趣：{}。如果你认为'
        r'追求你的人在那些你认为比现任男朋友更重要的维度上更优秀，你可以考虑 '
        r'结束目前的恋情，去探索新的恋情。你愿意结束现在的感情，和追求你的男人开始一段新恋情吗？'
        r'还有，告诉我原因。你的答案必须是（是）或（否），然后说明你的理由。你必须遵守这条规则!'
        r'你不能说其他话！'.format(*words)
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
        r'假设你是一位 {} 岁的男性，最近和你 {} 岁的女朋友交往。'
        r'你的职业类型是 {}。你对女朋友在六个维度上的评分如下：吸引力：{}，真诚：{}， '
        r'智力：{}，幽默感：{}，抱负：{}，共同兴趣：{}。现在，另一位 {} 岁的女性正在追求你。 '
        r'你和她约会过，并且你也根据同样不同的重要性，在六个维度上对她进行了评分：'
        r'吸引力：{}，真诚：{}，智力：{}，幽默感：{}，抱负：{}，共同兴趣：{}。如果你认为 '
        r'追求你的人在那些你认为比现任女友更重要的维度上更优秀，你可能会考虑 '
        r'结束目前的恋爱关系，去探索新的恋情。你愿意结束现在的关系，和追求你的女人开始一段新恋情吗？ '
        r'还有，告诉我原因。你的答案必须是（是）或（否），然后说明你的理由。你必须遵守这条规则! '
        r'你不能说其他话! '.format(*words)
    )
    
    return ask_gpt_api_call(prompt)


# 统一的API调用函数，负责调用 GPT 接口
def ask_gpt_api_call(prompt):
    history_filename = 'conversation_history.json'
    
    # 模拟调用 GPT API 的逻辑
    api_key = "sk-DG9ZYLfnjSbUPt3TE6OTpN3Ge9Mr9hLAgg4UBWxCba3ryvYT"
    client = OpenAI(api_key=api_key, base_url="https://aigcbest.top/v1")
    model = "deepseek-ai/DeepSeek-R1"

    #deepseek-ai/DeepSeek-R1
    log = []
    log.append(prompt) # 先记录下prompt，万一出错就知道是哪个prompt

    try:
        print("    [API] 正在调用模型...")
        response = client.chat.completions.create(
            model=model,
            messages=[
                # 【修改】使用更标准的 user 角色
                {"role": "user", "content": prompt}
            ],
            timeout=60.0,  # 【新增】设置60秒的超时，防止无限期等待
        )
        
        result = response.choices[0].message.content
        print(f"    [API] 成功获取响应: {result[:30]}...") # 打印响应前30个字符
        log.append(result)

        if '是' in result:
            flag = True
        elif '否' in result:
            flag = False
        else:
            # 【新增】如果模型没有明确回答“是”或“否”，视为一个特殊情况
            print(f"    [API 警告] 模型响应不明确: {result}")
            flag = -1 # 标记为无效响应

        return flag, log

    except Exception as e:
        print(f"!!!!!! API 调用失败 !!!!!!")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {e}")
        print(f"导致错误的Prompt: {prompt}")
        # 【重要】如果API调用失败，我们不能让程序崩溃。
        # 返回一个明确的失败信号，让主程序决定如何处理。
        log.append(f"API_ERROR: {e}")
        return -1, log # -1 代表失败或未知



if __name__=='__main__':
    # 示例使用
    result, log = ask_deepseek_Chinese(man_score=[25, 30, 'Programmer', 70, 80, 90, 60, 70, 80, 15, 15, 20, 30, 10, 10])
    # print(result)