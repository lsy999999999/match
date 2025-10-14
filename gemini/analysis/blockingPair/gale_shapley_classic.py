import pandas as pd

def classic_gale_shapley_matcher(men_prefs, women_prefs):
    """
    执行经典的Gale-Shapley算法 (男性求偶版本)
    
    参数:
    men_prefs (dict): {man_id: [woman_id_1, woman_id_2, ...]}, 按偏好排序
    women_prefs (dict): {woman_id: [man_id_1, man_id_2, ...]}, 按偏好排序
    
    返回:
    dict: 最终的配对结果 {person_id: partner_id}
    """
    
    free_men = list(men_prefs.keys())
    women_partners = {woman: None for woman in women_prefs.keys()}
    men_proposal_index = {man: 0 for man in free_men}
    
    final_matching = {}

    while free_men:
        proposer_man = free_men.pop(0)
        
        # 获取该男性的偏好列表
        man_pref_list = men_prefs[proposer_man]
        
        # 找到他下一个要求偶的女性
        proposal_index = men_proposal_index[proposer_man]
        if proposal_index >= len(man_pref_list):
            # 该男性已被所有人拒绝
            final_matching[proposer_man] = None
            continue
            
        target_woman = man_pref_list[proposal_index]
        men_proposal_index[proposer_man] += 1

        # 检查该女性是否单身
        current_partner = women_partners[target_woman]
        
        if current_partner is None:
            # 女性单身，直接接受
            women_partners[target_woman] = proposer_man
            final_matching[proposer_man] = target_woman
        else:
            # 女性已有伴侣，需要比较
            woman_pref_list = women_prefs[target_woman]
            try:
                rank_new = woman_pref_list.index(proposer_man)
                rank_current = woman_pref_list.index(current_partner)
                
                if rank_new < rank_current: # 新的追求者排名更靠前
                    # 女性接受新的，甩掉旧的
                    women_partners[target_woman] = proposer_man
                    final_matching[proposer_man] = target_woman
                    # 旧的伴侣恢复单身，重新加入求偶列表
                    final_matching[current_partner] = None
                    free_men.append(current_partner)
                else:
                    # 女性拒绝新的，该男性继续单身
                    free_men.append(proposer_man)
            except ValueError:
                # 如果某个男性不在女性的偏好列表中，视为拒绝
                free_men.append(proposer_man)

    # 将女性的配对也加入最终结果
    for woman, man in women_partners.items():
        if man:
            final_matching[woman] = man
        else:
            final_matching[woman] = None
            
    # 将被拒绝的男性标记为 rejected
    for man in men_prefs.keys():
        if final_matching.get(man) is None:
            final_matching[man] = 'rejected'
    for woman in women_prefs.keys():
        if final_matching.get(woman) is None:
            final_matching[woman] = 'rejected'

    return final_matching