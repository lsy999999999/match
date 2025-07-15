from openai import OpenAI
import json
import time
import numpy as np
from career_utils import translate_career

# Initialize OpenAI client
api_key = "sk-DG9ZYLfnjSbUPt3TE6OTpN3Ge9Mr9hLAgg4UBWxCba3ryvYT"
client = OpenAI(api_key=api_key, base_url="https://aigcbest.top/v1")
model = "gemini-2.5-flash-preview-05-20-thinking"



def ask_gpt_single_man_propose(score_details):
    """
    Ask GPT if a single man should accept a proposal based on detailed numeric scores.
    
    Args:
        score_details (dict): Dictionary containing all scores and importance weights
    
    Returns:
        tuple: (decision (bool), log_entry (list))
    """
    # Get age and career from the score_details
    career = translate_career(score_details.get('career', 'lawyer/policy work'))
    age    = int(score_details.get('age', 23))
    age_o  = int(score_details.get('age_o', 23))
    
    prompt = f"""假设你是一位 {age} 岁的单身男士，有一位 {age_o} 岁的女性正在追求你。你的职业类型是 {career}。你从六个维度对她进行了评估，满分为 10 分。这六个维度分别是：吸引力：{score_details['attractive']}/10，真诚：{score_details['sincere']}/10，智力：{score_details['intelligence']}/10，幽默感：{score_details['funny']}/10，抱负：{score_details['ambition']}/10，共同兴趣：{score_details['shared_interests']}/10。这些属性的重要性权重如下：吸引力：{score_details['attractive_importance']}，真诚：{score_details['sincere_importance']}，智力：{score_details['intelligence_importance']}，幽默感：{score_details['funny_importance']}，抱负：{score_details['ambition_importance']}，共同兴趣：{score_details['shared_interests_importance']}。你愿意和这位女士约会吗？请告诉我原因。你的答案必须是（是）或（否），然后说明你的原因。你必须遵守这条规则！你不能说其他话。"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "您是约会顾问。请回答“是”或“否”，并附上原因。"},
                {"role": "user", "content": prompt}
            ]
        )
        
        answer = response.choices[0].message.content.strip()
        
        # Extract decision
        if '是' in answer:
            decision = True
        elif '否' in answer:
            decision = False
        else:
            # Fallback if neither YES nor NO is found
            decision = score_details['overall_score'] >= 6
        
        # Clean up for CSV (replace commas with semicolons)
        clean_prompt = prompt.replace(",", ";").replace("\n", " ").strip()
        clean_answer = answer.replace(",", ";").replace("\n", " ").strip()
        
        # Create log entry with full prompt and response
        log_entry = [
            clean_prompt,  # Full prompt with numeric scores
            clean_answer   # Full response
        ]
        
        return decision, log_entry
        
    except Exception as e:
        print(f"Error calling GPT: {e}")
        # Default behavior
        decision = score_details['overall_score'] >= 6
        
        # Clean prompt for CSV
        clean_prompt = prompt.replace(",", ";").replace("\n", " ").strip()
        
        # Create default response
        if decision:
            default_answer = f"是，{score_details['overall_score']} 的总体兼容性得分达到可接受的阈值。"
        else:
            default_answer = f"否，{score_details['overall_score']} 的总体兼容性得分低于可接受的阈值。"
        
        log_entry = [
            clean_prompt,
            f"API_ERROR: {default_answer}"
        ]
        
        return decision, log_entry


def ask_gpt_single_woman_propose(score_details):
    """
    Ask GPT if a single woman should accept a proposal based on detailed numeric scores.
    
    Args:
        score_details (dict): Dictionary containing detailed scores and importance weights
    
    Returns:
        tuple: (decision (bool), log_entry (list))
    """
    # Get age and career from the score_details
    career = translate_career(score_details.get('career', 'lawyer/policy work'))
    age    = int(score_details.get('age', 23))
    age_o  = int(score_details.get('age_o', 23))
    
    prompt = f"""假设您是一位 {age} 岁的单身男士，有一位 {age_o} 岁的女性正在追求您。您的职业类型是 {career}。您对她进行了六个维度的评估，满分为 10 分。这六个维度分别是：吸引力：{score_details['attractive']}/10，真诚：{score_details['sincere']}/10，智力：{score_details['intelligence']}/10，幽默感：{score_details['funny']}/10，抱负：{score_details['ambition']}/10，共同兴趣：{score_details['shared_interests']}/10。您对这些属性的重要性权重为：吸引力：{score_details['attractive_importance']}，真诚： {score_details['sincere_importance']}，智力：{score_details['intelligence_importance']}，幽默感：{score_details['funny_importance']}，野心：{score_details['ambition_importance']}，共同兴趣：{score_details['shared_interests_importance']}。你愿意和这位女士约会吗？请告诉我原因。你的答案必须是(是）或（否），然后说明你的理由。你必须遵守这条规则！你不能说其他话。"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "您是约会顾问。请回答“是”或“否”，并附上原因。"},
                {"role": "user", "content": prompt}
            ]
        )
        
        answer = response.choices[0].message.content.strip()
        
        # Extract decision
        if '是' in answer.upper():
            decision = True
        elif '否' in answer.upper():
            decision = False
        else:
            # Fallback if neither YES nor NO is found
            decision = score_details['overall_score'] >= 6
        
        # Clean up for CSV (replace commas with semicolons)
        clean_prompt = prompt.replace(",", ";").replace("\n", " ").strip()
        clean_answer = answer.replace(",", ";").replace("\n", " ").strip()
        
        # Create log entry with full prompt and response
        log_entry = [
            clean_prompt,  # Full prompt with numeric scores
            clean_answer   # Full response
        ]
        
        return decision, log_entry
        
    except Exception as e:
        print(f"Error calling GPT: {e}")
        # Default behavior
        decision = score_details['overall_score'] >= 6
        
        # Clean prompt for CSV
        clean_prompt = prompt.replace(",", ";").replace("\n", " ").strip()
        
        # Create default response
        if decision:
            default_answer =f"是，{score_details['overall_score']} 的总体兼容性得分符合可接受的阈值。"
        else:
            default_answer = f"否，{score_details['overall_score']} 的整体兼容性得分低于可接受的阈值。"
        
        log_entry = [
            clean_prompt,
            f"API_ERROR: {default_answer}"
        ]
        
        return decision, log_entry


def ask_gpt_non_single_man_propose(proposer_score_details, current_partner_score_details):
    """
    Ask GPT if a non-single man should switch partners based on detailed numeric scores.
    
    Args:
        proposer_score_details (dict): Dictionary containing proposer's detailed scores and importance weights
        current_partner_score_details (dict): Dictionary containing current partner's detailed scores and importance weights
    
    Returns:
        tuple: (decision (bool), log_entry (list))
    """
    career = translate_career(proposer_score_details.get('career', 'lawyer/policy work'))
    age    = int(proposer_score_details.get('age', 23))
    
    prompt = f"""假设您是一位 {age} 岁的男性，职业类型为 {career}，目前已匹配到一位伴侣。您当前伴侣的得分（满分 10 分）：吸引力：{current_partner_score_details['attractive']}/10，真诚度：{current_partner_score_details['sincere']}/10，智力：{current_partner_score_details['intelligence']}/10，幽默感：{current_partner_score_details['funny']}/10，野心：{current_partner_score_details['ambition']}/10，共同兴趣：{current_partner_score_details['shared_interests']}/10。一位新女性正在向您求爱，其得分为：吸引力：{proposer_score_details['attractive']}/10，真诚度：{proposer_score_details['sincere']}/10，智力：{proposer_score_details['intelligence']}/10，幽默度：{proposer_score_details['funny']}/10，抱负：{proposer_score_details['ambition']}/10，共同兴趣：{proposer_score_details['shared_interests']}/10。您的重要性权重为：吸引力：{proposer_score_details['attractive_importance']}，真诚度：{proposer_score_details['sincere_importance']}，智力：{proposer_score_details['intelligence_importance']}，幽默度：{proposer_score_details['funny_importance']}，抱负： {proposer_score_details['ambition_importance']}，共同兴趣：{proposer_score_details['shared_interests_importance']}。你会为了新女友离开现在的伴侣吗？你的答案必须是或否，然后说明你的理由。你必须遵守这条规则！你不能说其他话。"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "您是约会顾问。请回答“是”或“否”，并附上原因。"},
                {"role": "user", "content": prompt}
            ]
        )
        
        answer = response.choices[0].message.content.strip()
        
        # Extract decision
        if '是' in answer.upper():
            decision = True
        elif '否' in answer.upper():
            decision = False
        else:
            # Fallback based on score comparison
            decision = proposer_score_details['overall_score'] > current_partner_score_details['overall_score']
        
        # Clean up for CSV
        clean_prompt = prompt.replace(",", ";").replace("\n", " ").strip()
        clean_answer = answer.replace(",", ";").replace("\n", " ").strip()
        
        log_entry = [
            clean_prompt,
            clean_answer
        ]
        
        return decision, log_entry
        
    except Exception as e:
        print(f"Error calling GPT: {e}")
        # Default behavior
        decision = proposer_score_details['overall_score'] > current_partner_score_details['overall_score']
        
        clean_prompt = prompt.replace(",", ";").replace("\n", " ").strip()
        
        if decision:
            default_answer = f"是，新合作伙伴的总体得分更高({proposer_score_details['overall_score']} vs {current_partner_score_details['overall_score']})."
        else:
            default_answer = f"否，当前合作伙伴的总体得分相同或更高({current_partner_score_details['overall_score']} vs {proposer_score_details['overall_score']})."
        
        log_entry = [
            clean_prompt,
            f"API_ERROR: {default_answer}"
        ]
        
        return decision, log_entry


def ask_gpt_non_single_woman_propose(proposer_score_details, current_partner_score_details):
    """
    Ask GPT if a non-single woman should switch partners based on detailed numeric scores.
    
    Args:
        proposer_score_details (dict): Dictionary containing proposer's detailed scores and importance weights
        current_partner_score_details (dict): Dictionary containing current partner's detailed scores and importance weights
    
    Returns:
        tuple: (decision (bool), log_entry (list))
    """
    career = translate_career(proposer_score_details.get('career', 'lawyer/policy work'))
    age    = int(proposer_score_details.get('age', 23))


    prompt = f"""假设您是一位 {age} 岁的女性，职业类型为 {career}，目前已匹配到一位伴侣。您当前伴侣的得分（满分 10 分）：吸引力：{current_partner_score_details['attractive']}/10，真诚度：{current_partner_score_details['sincere']}/10，智力：{current_partner_score_details['intelligence']}/10，幽默度：{current_partner_score_details['funny']}/10，野心：{current_partner_score_details['ambition']}/10，共同兴趣：{current_partner_score_details['shared_interests']}/10。一位新男士正在向您求爱，他的得分为：吸引力：{proposer_score_details['attractive']}/10，真诚度： {proposer_score_details['sincere']}/10，智力：{proposer_score_details['intelligence']}/10，幽默感：{proposer_score_details['funny']}/10，抱负：{proposer_score_details['ambition']}/10，共同兴趣：{proposer_score_details['shared_interests']}/10。您的重要性权重为：吸引力：{proposer_score_details['attractive_importance']}，真诚：{proposer_score_details['sincere_importance']}，智力：{proposer_score_details['intelligence_importance']}，幽默感：{proposer_score_details['funny_importance']}，抱负： {proposer_score_details['ambition_importance']}，共同兴趣：{proposer_score_details['shared_interests_importance']}。你会为了新男友离开现在的伴侣吗？你的答案必须是（是）或（否），然后说明你的理由。你必须遵守这条规则！你绝对不能说其他话。"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "您是约会顾问。请回答“是”或“否”，并附上原因。"},
                {"role": "user", "content": prompt}
            ]
        )
        
        answer = response.choices[0].message.content.strip()
        
        # Extract decision
        if '是' in answer.upper():
            decision = True
        elif '否' in answer.upper():
            decision = False
        else:
            # Fallback based on score comparison
            decision = proposer_score_details['overall_score'] > current_partner_score_details['overall_score']
        
        # Clean up for CSV
        clean_prompt = prompt.replace(",", ";").replace("\n", " ").strip()
        clean_answer = answer.replace(",", ";").replace("\n", " ").strip()
        
        log_entry = [
            clean_prompt,
            clean_answer
        ]
        
        return decision, log_entry
        
    except Exception as e:
        print(f"Error calling GPT: {e}")
        # Default behavior
        decision = proposer_score_details['overall_score'] > current_partner_score_details['overall_score']
        
        clean_prompt = prompt.replace(",", ";").replace("\n", " ").strip()
        
        if decision:
            default_answer = f"是，新伴侣的总体得分更高 ({current_partner_score_details['overall_score']} vs {proposer_score_details['overall_score']})。"
        else:
            default_answer = f"否，当前伴侣的总体得分相等或更高 ({current_partner_score_details['overall_score']} vs {proposer_score_details['overall_score']})。"
        
        log_entry = [
            clean_prompt,
            f"API_ERROR: {default_answer}"
        ]
        
        return decision, log_entry


# Optional: Test function
# def test_functions():
#     """Test the functions with sample data"""
#     test_scores = {
#         'age': 25,
#         'age_o': 23,
#         'career': 'engineer',
#         'overall_score': 7,
#         'attractive': 6,
#         'sincere': 8,
#         'intelligence': 7,
#         'funny': 8,
#         'ambition': 6,
#         'shared_interests': 7,
#         'attractive_importance': 20,
#         'sincere_importance': 25,
#         'intelligence_importance': 20,
#         'funny_importance': 15,
#         'ambition_importance': 10,
#         'shared_interests_importance': 10
#     }
    
#     # Test single man propose
#     print("Testing single man propose...")
#     decision, log = ask_gpt_single_man_propose(test_scores)
#     print(f"Decision: {decision}")
#     print(f"Log: {log}")
#     print()
    
    # # Test non-single scenario
    # current_scores = test_scores.copy()
    # current_scores['overall_score'] = 6
    # new_scores = test_scores.copy()
    # new_scores['overall_score'] = 8
    
    # print("Testing non-single man propose...")
    # decision, log = ask_gpt_non_single_man_propose(new_scores, current_scores)
    # print(f"Decision: {decision}")
    # print(f"Log: {log}")


if __name__ == "__main__":
    # You can run this to test the functions
    test_functions()