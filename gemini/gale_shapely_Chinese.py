from read_file_Chinese import * 
from ask_gemini_Chinese import *
import csv
import json
import datetime
import numpy as np
import random

def default_dump(obj):
    """Convert numpy classes to JSON serializable objects."""
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def gale_shapley(men_preferences, women_preferences, men_attr, women_attr, idx):
    men = list(men_preferences.keys())
    women = list(women_preferences.keys())
    single_list = men + women

    # 初始化matching和single_list
    matching = {person: None for person in single_list}
    rejected_list = []
    proposer_indices = {person: 0 for person in men + women}
    logtext = []
    print(single_list)

    while single_list:
      proposer = random.choice(single_list)

      if proposer in men:
        attr = men_attr[proposer]
        preferences = men_preferences[proposer]
        is_man = True
      else:
        attr = women_attr[proposer]
        preferences = women_preferences[proposer]
        is_man = False
      
      index = proposer_indices[proposer]
      print(proposer)
      if index >= len(preferences):
        matching[proposer] = 'rejected'
        rejected_list.append(proposer)
        single_list.remove(proposer)
        continue
      
      best_choice = preferences[index]
      print(best_choice)
      # print(attr)
      proposer_indices[proposer] += 1

      if matching[best_choice] is None or matching[best_choice] == 'rejected':
        handle_single_choice(proposer, best_choice, matching, logtext, attr, single_list,is_man)
      else:
        handle_existing_choice(proposer, best_choice, matching, logtext, attr, men_attr, women_attr,single_list, is_man)
      print(matching)

    log_text("/home/lsy/match/gemini/0728_gemini_Chinese/0728_gemini_Chinese_group"+str(idx)+".csv", logtext)
    js_obj = json.dumps(matching, ensure_ascii=False, default=default_dump)
    with open("/home/lsy/match/gemini/0728_gemini_Chinese/0728_gemini_Chinese_group"+str(idx)+".json", "a") as file:
        file.write(js_obj)
    return matching

def initialize_matching(men, women):
    men_current = {man: None for man in men}
    women_current = {woman: None for woman in women}
    return men_current, women_current

def handle_single_choice(proposer, best_choice, matching, logtext, attr, single_list,is_man):
  # Modified to pass full attribute scores instead of just gpt_score
  proposer_scores = attr[best_choice]
  
  # Extract detailed scores from the attribute list
  # proposer_scores is a list: [age, age_o, career, gpt_score, attractive_partner, sincere_partner, 
  #                             intelligence_partner, funny_partner, ambition_partner, shared_interests_partner,
  #                             attractive_important, sincere_important, intelligence_important,
  #                             funny_important, ambition_important, shared_interests_important]
  
  score_details = {
      'age': proposer_scores[0],            # age
      'age_o': proposer_scores[1],          # age_o (partner's age)
      'career': proposer_scores[2],         # career
      'overall_score': proposer_scores[3],  # gpt_score
      'attractive': proposer_scores[4],     # attractive_partner
      'sincere': proposer_scores[5],        # sincere_partner
      'intelligence': proposer_scores[6],   # intelligence_partner
      'funny': proposer_scores[7],          # funny_partner
      'ambition': proposer_scores[8],       # ambition_partner
      'shared_interests': proposer_scores[9], # shared_interests_partner
      # Importance weights (numeric values)
      'attractive_importance': proposer_scores[10],     # attractive_important
      'sincere_importance': proposer_scores[11],        # sincere_important
      'intelligence_importance': proposer_scores[12],   # intelligence_important
      'funny_importance': proposer_scores[13],          # funny_important
      'ambition_importance': proposer_scores[14],       # ambition_important
      'shared_interests_importance': proposer_scores[15] # shared_interests_important
  }
  
  if is_man:  # 如果提议者是男性
      flag, log = ask_gpt_single_man_propose(score_details)
  else:  # 如果提议者是女性
      flag, log = ask_gpt_single_woman_propose(score_details)
  print(flag)
  
  # Log format: [prompt, response, target, proposer, empty, result]
  log.append(best_choice)
  log.append(proposer)
  log.append("")

  if flag == True:  # 如果对方接受提议
      matching[proposer] = best_choice
      matching[best_choice] = proposer
      log.append(1)
      
      # 更新单身列表，移除提议成功的双方
      single_list.remove(proposer)
      try:
        single_list.remove(best_choice)
      except:
        pass
  else:  # 对方拒绝提议
      log.append(0)
  
  logtext.append(log)

def handle_existing_choice(proposer, best_choice, matching, logtext, attr, men_attr, women_attr, single_list,is_man):
    current_partner = matching[best_choice]  # 当前配对的伴侣
    proposer_scores = attr[best_choice]
    
    # Extract detailed scores for proposer
    proposer_score_details = {
        'age': proposer_scores[0],            # age
        'age_o': proposer_scores[1],          # age_o (partner's age)
        'career': proposer_scores[2],         # career
        'overall_score': proposer_scores[3],  # gpt_score
        'attractive': proposer_scores[4],     # attractive_partner
        'sincere': proposer_scores[5],        # sincere_partner
        'intelligence': proposer_scores[6],   # intelligence_partner
        'funny': proposer_scores[7],          # funny_partner
        'ambition': proposer_scores[8],       # ambition_partner
        'shared_interests': proposer_scores[9], # shared_interests_partner
        # Importance weights (numeric values)
        'attractive_importance': proposer_scores[10],     # attractive_important
        'sincere_importance': proposer_scores[11],        # sincere_important
        'intelligence_importance': proposer_scores[12],   # intelligence_important
        'funny_importance': proposer_scores[13],          # funny_important
        'ambition_importance': proposer_scores[14],       # ambition_important
        'shared_interests_importance': proposer_scores[15] # shared_interests_important
    }
    
    # Get current partner's scores
    if is_man:  # 如果提议者是男性
        current_partner_scores = women_attr[best_choice][current_partner]
        current_partner_score_details = {
            'age': current_partner_scores[0],            # age
            'age_o': current_partner_scores[1],          # age_o
            'career': current_partner_scores[2],         # career
            'overall_score': current_partner_scores[3],  # gpt_score
            'attractive': current_partner_scores[4],     # attractive_partner
            'sincere': current_partner_scores[5],        # sincere_partner
            'intelligence': current_partner_scores[6],   # intelligence_partner
            'funny': current_partner_scores[7],          # funny_partner
            'ambition': current_partner_scores[8],       # ambition_partner
            'shared_interests': current_partner_scores[9], # shared_interests_partner
            # Importance weights (numeric values)
            'attractive_importance': current_partner_scores[10],     # attractive_important
            'sincere_importance': current_partner_scores[11],        # sincere_important
            'intelligence_importance': current_partner_scores[12],   # intelligence_important
            'funny_importance': current_partner_scores[13],          # funny_important
            'ambition_importance': current_partner_scores[14],       # ambition_important
            'shared_interests_importance': current_partner_scores[15] # shared_interests_important
        }
        flag, log = ask_gpt_non_single_man_propose(proposer_score_details, current_partner_score_details)
    else:  # 如果提议者是女性
        current_partner_scores = men_attr[best_choice][current_partner]
        current_partner_score_details = {
            'overall_score': current_partner_scores[3],  # gpt_score
            'attractive': current_partner_scores[4],     # attractive_partner
            'sincere': current_partner_scores[5],        # sincere_partner
            'intelligence': current_partner_scores[6],   # intelligence_partner
            'funny': current_partner_scores[7],          # funny_partner
            'ambition': current_partner_scores[8],       # ambition_partner
            'shared_interests': current_partner_scores[9], # shared_interests_partner
            # Importance weights (numeric values)
            'attractive_importance': current_partner_scores[10],     # attractive_important
            'sincere_importance': current_partner_scores[11],        # sincere_important
            'intelligence_importance': current_partner_scores[12],   # intelligence_important
            'funny_importance': current_partner_scores[13],          # funny_important
            'ambition_importance': current_partner_scores[14],       # ambition_important
            'shared_interests_importance': current_partner_scores[15] # shared_interests_important
        }
        flag, log = ask_gpt_non_single_woman_propose(proposer_score_details, current_partner_score_details)

    log.append(best_choice)
    log.append(proposer)
    log.append(current_partner)
    
    if flag == True:  # 如果对方选择了提议者
        # 解除当前配对关系
        matching[current_partner] = None
        single_list.append(current_partner)  # 把当前伴侣加入单身列表
        
        # 更新提议者和对象的匹配状态
        matching[proposer] = best_choice
        matching[best_choice] = proposer
        log.append(1)
        
        # 更新单身列表，移除成功配对的双方
        single_list.remove(proposer)
        # single_list.remove(best_choice)
        
    else:  # 对方拒绝了提议
        log.append(0)
    
    logtext.append(log)


def log_text(logfile, logtext):
    with open(logfile, mode='a', newline='') as file:
        writer = csv.writer(file)
        for log in logtext:
            writer.writerow(log)


if __name__=='__main__':
  for i in range(1,22):
    # for _ in range(5):
    l1,l2 = read_file(r'/home/lsy/match/dataset/save_merge_select_null_3.xlsx', i)
    dict1, dict2 = read_file_attr(r'/home/lsy/match/dataset/save_merge_select_null_3.xlsx', i)
    gale_shapley(l1, l2, dict1, dict2, i)

# Add this debug version to help identify the issue

# if __name__=='__main__':
#     import traceback
    
#     for i in range(22, 51):
#         print(f"\n{'='*50}")
#         print(f"Processing group {i}")
#         print(f"{'='*50}")
        
#         try:
#             # Read preferences
#             l1, l2 = read_file(r'/home/lsy/match/dataset/save_merge_select_null_3.xlsx', i)
#             print(f"Group {i} preferences loaded:")
#             print(f"  Men preferences: {len(l1)} men")
#             print(f"  Women preferences: {len(l2)} women")
            
#             # Check if data exists
#             if len(l1) == 0 or len(l2) == 0:
#                 print(f"WARNING: Group {i} has no men or women! Skipping...")
#                 continue
            
#             # Read attributes
#             dict1, dict2 = read_file_attr(r'/home/lsy/match/dataset/save_merge_select_null_3.xlsx', i)
#             print(f"  Men attributes: {len(dict1)} entries")
#             print(f"  Women attributes: {len(dict2)} entries")
            
#             # Run Gale-Shapley
#             print(f"Running Gale-Shapley for group {i}...")
#             matching = gale_shapley(l1, l2, dict1, dict2, i)
            
#             print(f"Matching complete. Results: {len(matching)} matches")
#             print(f"Files written for group {i}")
            
#         except Exception as e:
#             print(f"ERROR in group {i}:")
#             print(f"Error type: {type(e).__name__}")
#             print(f"Error message: {str(e)}")
#             traceback.print_exc()
            
#             # Try to write empty files so you know it was attempted
#             try:
#                 with open(f"/home/lsy/match/bahavior_simul/0627_gpt4_eng/0627_gpt4_eng_group{i}_ERROR.txt", "w") as f:
#                     f.write(f"Error processing group {i}:\n")
#                     f.write(f"{type(e).__name__}: {str(e)}\n")
#                     f.write(traceback.format_exc())
#             except:
#                 pass# Add this debug version to help identify the issue

# if __name__=='__main__':
#     import traceback
    
#     for i in range(22, 51):
#         print(f"\n{'='*50}")
#         print(f"Processing group {i}")
#         print(f"{'='*50}")
        
#         try:
#             # Read preferences
#             l1, l2 = read_file(r'/home/lsy/match/dataset/save_merge_select_null_3.xlsx', i)
#             print(f"Group {i} preferences loaded:")
#             print(f"  Men preferences: {len(l1)} men")
#             print(f"  Women preferences: {len(l2)} women")
            
#             # Check if data exists
#             if len(l1) == 0 or len(l2) == 0:
#                 print(f"WARNING: Group {i} has no men or women! Skipping...")
#                 continue
            
#             # Read attributes
#             dict1, dict2 = read_file_attr(r'/home/lsy/match/dataset/save_merge_select_null_3.xlsx', i)
#             print(f"  Men attributes: {len(dict1)} entries")
#             print(f"  Women attributes: {len(dict2)} entries")
            
#             # Run Gale-Shapley
#             print(f"Running Gale-Shapley for group {i}...")
#             matching = gale_shapley(l1, l2, dict1, dict2, i)
            
#             print(f"Matching complete. Results: {len(matching)} matches")
#             print(f"Files written for group {i}")
            
#         except Exception as e:
#             print(f"ERROR in group {i}:")
#             print(f"Error type: {type(e).__name__}")
#             print(f"Error message: {str(e)}")
#             traceback.print_exc()
            
#             # Try to write empty files so you know it was attempted
#             try:
#                 with open(f"/home/lsy/match/bahavior_simul/0627_gpt4_eng/0627_gpt4_eng_group{i}_ERROR.txt", "w") as f:
#                     f.write(f"Error processing group {i}:\n")
#                     f.write(f"{type(e).__name__}: {str(e)}\n")
#                     f.write(traceback.format_exc())
#             except:
#                 pass