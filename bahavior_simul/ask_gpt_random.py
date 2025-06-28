from openai import OpenAI
import json
import time
import numpy as np

# Initialize OpenAI client
api_key = "sk-DG9ZYLfnjSbUPt3TE6OTpN3Ge9Mr9hLAgg4UBWxCba3ryvYT"
client = OpenAI(api_key=api_key, base_url="https://aigcbest.top/v1")
model = "gpt-4-turbo"

def ask_gpt_single_man_propose(score_details):
    """
    Ask GPT if a single man should accept a proposal based on detailed numeric scores.
    
    Args:
        score_details (dict): Dictionary containing all scores and importance weights
    
    Returns:
        tuple: (decision (bool), log_entry (list))
    """
    # Get age and career from the score_details
    age = int(score_details.get('age', 23))
    age_o = int(score_details.get('age_o', 23))
    career = score_details.get('career', 'lawyer/policy work')
    
    prompt = f"""Assume you are a {age}-year-old single man, and there is a {age_o}-year-old woman who is courting you. Your career type is {career}. You evaluated her on six dimensions with scores out of 10. These are attractiveness: {score_details['attractive']}/10, sincerity: {score_details['sincere']}/10, intelligence: {score_details['intelligence']}/10, being funny: {score_details['funny']}/10, ambition: {score_details['ambition']}/10, shared interests: {score_details['shared_interests']}/10. Your importance weights for these attributes are: attractiveness: {score_details['attractive_importance']}, sincerity: {score_details['sincere_importance']}, intelligence: {score_details['intelligence_importance']}, being funny: {score_details['funny_importance']}, ambition: {score_details['ambition_importance']}, shared interests: {score_details['shared_interests_importance']}. Would you be willing to date this woman? And also tell me the reason. Your answer must be (YES) or (NO) followed by your reason. You must FOLLOW THIS RULE! You MUST NOT SAY other words."""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a dating advisor. Please answer with YES or NO followed by a reason."},
                {"role": "user", "content": prompt}
            ]
        )
        
        answer = response.choices[0].message.content.strip()
        
        # Extract decision
        if 'YES' in answer.upper():
            decision = True
        elif 'NO' in answer.upper():
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
            default_answer = f"YES, overall compatibility score of {score_details['overall_score']} meets acceptable threshold."
        else:
            default_answer = f"NO, overall compatibility score of {score_details['overall_score']} is below acceptable threshold."
        
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
    age = int(score_details.get('age', 23))
    age_o = int(score_details.get('age_o', 23))
    career = score_details.get('career', 'lawyer/policy work')
    
    prompt = f"""Assume you are a {age}-year-old single woman, and there is a {age_o}-year-old man who is courting you. Your career type is {career}. You evaluated him on six dimensions with scores out of 10. These are attractiveness: {score_details['attractive']}/10, sincerity: {score_details['sincere']}/10, intelligence: {score_details['intelligence']}/10, being funny: {score_details['funny']}/10, ambition: {score_details['ambition']}/10, shared interests: {score_details['shared_interests']}/10. Your importance weights for these attributes are: attractiveness: {score_details['attractive_importance']}, sincerity: {score_details['sincere_importance']}, intelligence: {score_details['intelligence_importance']}, being funny: {score_details['funny_importance']}, ambition: {score_details['ambition_importance']}, shared interests: {score_details['shared_interests_importance']}. Would you be willing to date this man? And also tell me the reason. Your answer must be (YES) or (NO) followed by your reason. You must FOLLOW THIS RULE! You MUST NOT SAY other words."""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a dating advisor. Please answer with YES or NO followed by a reason."},
                {"role": "user", "content": prompt}
            ]
        )
        
        answer = response.choices[0].message.content.strip()
        
        # Extract decision
        if 'YES' in answer.upper():
            decision = True
        elif 'NO' in answer.upper():
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
            default_answer = f"YES, overall compatibility score of {score_details['overall_score']} meets acceptable threshold."
        else:
            default_answer = f"NO, overall compatibility score of {score_details['overall_score']} is below acceptable threshold."
        
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
    age = int(proposer_score_details.get('age', 23))
    career = proposer_score_details.get('career', 'lawyer/policy work')
    
    prompt = f"""Assume you are a {age}-year-old man with career type {career}, currently matched with a partner. Your current partner's scores (out of 10): attractiveness: {current_partner_score_details['attractive']}/10, sincerity: {current_partner_score_details['sincere']}/10, intelligence: {current_partner_score_details['intelligence']}/10, being funny: {current_partner_score_details['funny']}/10, ambition: {current_partner_score_details['ambition']}/10, shared interests: {current_partner_score_details['shared_interests']}/10. A new woman is courting you with scores: attractiveness: {proposer_score_details['attractive']}/10, sincerity: {proposer_score_details['sincere']}/10, intelligence: {proposer_score_details['intelligence']}/10, being funny: {proposer_score_details['funny']}/10, ambition: {proposer_score_details['ambition']}/10, shared interests: {proposer_score_details['shared_interests']}/10. Your importance weights are: attractiveness: {proposer_score_details['attractive_importance']}, sincerity: {proposer_score_details['sincere_importance']}, intelligence: {proposer_score_details['intelligence_importance']}, being funny: {proposer_score_details['funny_importance']}, ambition: {proposer_score_details['ambition_importance']}, shared interests: {proposer_score_details['shared_interests_importance']}. Would you leave your current partner for the new woman? Your answer must be (YES) or (NO) followed by your reason. You must FOLLOW THIS RULE! You MUST NOT SAY other words."""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a dating advisor. Please answer with YES or NO followed by a reason."},
                {"role": "user", "content": prompt}
            ]
        )
        
        answer = response.choices[0].message.content.strip()
        
        # Extract decision
        if 'YES' in answer.upper():
            decision = True
        elif 'NO' in answer.upper():
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
            default_answer = f"YES, new partner has higher overall score ({proposer_score_details['overall_score']} vs {current_partner_score_details['overall_score']})."
        else:
            default_answer = f"NO, current partner has equal or higher overall score ({current_partner_score_details['overall_score']} vs {proposer_score_details['overall_score']})."
        
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
    age = int(proposer_score_details.get('age', 23))
    career = proposer_score_details.get('career', 'lawyer/policy work')
    
    prompt = f"""Assume you are a {age}-year-old woman with career type {career}, currently matched with a partner. Your current partner's scores (out of 10): attractiveness: {current_partner_score_details['attractive']}/10, sincerity: {current_partner_score_details['sincere']}/10, intelligence: {current_partner_score_details['intelligence']}/10, being funny: {current_partner_score_details['funny']}/10, ambition: {current_partner_score_details['ambition']}/10, shared interests: {current_partner_score_details['shared_interests']}/10. A new man is courting you with scores: attractiveness: {proposer_score_details['attractive']}/10, sincerity: {proposer_score_details['sincere']}/10, intelligence: {proposer_score_details['intelligence']}/10, being funny: {proposer_score_details['funny']}/10, ambition: {proposer_score_details['ambition']}/10, shared interests: {proposer_score_details['shared_interests']}/10. Your importance weights are: attractiveness: {proposer_score_details['attractive_importance']}, sincerity: {proposer_score_details['sincere_importance']}, intelligence: {proposer_score_details['intelligence_importance']}, being funny: {proposer_score_details['funny_importance']}, ambition: {proposer_score_details['ambition_importance']}, shared interests: {proposer_score_details['shared_interests_importance']}. Would you leave your current partner for the new man? Your answer must be (YES) or (NO) followed by your reason. You must FOLLOW THIS RULE! You MUST NOT SAY other words."""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a dating advisor. Please answer with YES or NO followed by a reason."},
                {"role": "user", "content": prompt}
            ]
        )
        
        answer = response.choices[0].message.content.strip()
        
        # Extract decision
        if 'YES' in answer.upper():
            decision = True
        elif 'NO' in answer.upper():
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
            default_answer = f"YES, new partner has higher overall score ({proposer_score_details['overall_score']} vs {current_partner_score_details['overall_score']})."
        else:
            default_answer = f"NO, current partner has equal or higher overall score ({current_partner_score_details['overall_score']} vs {proposer_score_details['overall_score']})."
        
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