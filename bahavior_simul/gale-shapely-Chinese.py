from read_file_Chinese import read_file, read_file_attr
from ask_gpt_Chinese_random import (
    ask_gpt_single_man_propose,
    ask_gpt_single_woman_propose,
    ask_gpt_non_single_man_propose,
    ask_gpt_non_single_woman_propose
)
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

    log_text("/home/lsy/match/bahavior_simul/0619_gpt4_turbo_Chinese/0619_gpt4_turbo_random_group"+str(idx)+".csv", logtext)
    js_obj = json.dumps(matching, ensure_ascii=False, default=default_dump)
    with open("/home/lsy/match/bahavior_simul/0619_gpt4_turbo_Chinese/0619_gpt4_turbo_random_group_Chinese"+str(idx)+".json", "a") as file:
        file.write(js_obj)
    return matching

def initialize_matching(men, women):
    men_current = {man: None for man in men}
    women_current = {woman: None for woman in women}
    return men_current, women_current

def handle_single_choice(proposer, best_choice, matching, logtext, attr, single_list,is_man):
  proposer_score = attr[best_choice]
  if is_man:  # 如果提议者是男性
      flag, log = ask_gpt_single_man_propose(proposer_score)
  else:  # 如果提议者是女性
      flag, log = ask_gpt_single_woman_propose(proposer_score)
  print(flag)
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
    proposer_score = attr[best_choice]
    # current_partner_score = attr[best_choice][current_partner]
    if is_man:  # 如果提议者是男性
        current_partner_score = women_attr[best_choice][current_partner]
        flag, log = ask_gpt_non_single_man_propose(proposer_score, current_partner_score)
    else:  # 如果提议者是女性
        current_partner_score = men_attr[best_choice][current_partner]
        flag, log = ask_gpt_non_single_woman_propose(proposer_score, current_partner_score)

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
  for i in range(21,50):
    # for _ in range(5):
    l1,l2 = read_file(r'/home/lsy/match/dataset/save_merge_select_null_3.xlsx', i)
    dict1, dict2 = read_file_attr(r'/home/lsy/match/dataset/save_merge_select_null_3.xlsx', i)
    gale_shapley(l1, l2, dict1, dict2, i)
