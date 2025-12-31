import pandas as pd

def classic_gale_shapley_matcher(men_prefs, women_prefs):
    """
    执行经典的 Gale-Shapley 算法 (男性求偶版本)
    
    修改点 (2024/11):
    1. 偏好列表中包含自己 (self_id)。
    2. Propose 阶段: 如果男性遇到自己，停止求偶，保持单身。
    3. Accept 阶段: 如果女性收到比自己更差的 offer，拒绝。
    
    参数:
    men_prefs (dict): {man_id: [woman_id_1, ..., man_id, ...]}
    women_prefs (dict): {woman_id: [man_id_1, ..., woman_id, ...]}
    """
    
    # 初始状态：所有男性都是自由的
    free_men = list(men_prefs.keys())
    
    # 女性的当前伴侣，初始化为 None (注意：最终结果中单身可能表现为 None 或 self_id，视逻辑而定)
    # 在这里我们用 None 表示尚未匹配，最终结算时填入 self_id
    women_partners = {woman: None for woman in women_prefs.keys()}
    
    # 记录每个男性求偶到了第几个顺位
    men_proposal_index = {man: 0 for man in free_men}
    
    final_matching = {}

    while free_men:
        proposer_man = free_men.pop(0)
        
        # 获取该男性的偏好列表
        man_pref_list = men_prefs.get(proposer_man)
        if not man_pref_list:
            final_matching[proposer_man] = proposer_man # 没有偏好，直接单身
            continue
            
        # 找到他下一个要求偶的对象
        current_idx = men_proposal_index[proposer_man]
        
        # 如果已经遍历完列表 (理论上遇到 self_id 就会 break，防止越界)
        if current_idx >= len(man_pref_list):
            final_matching[proposer_man] = proposer_man
            continue
            
        target = man_pref_list[current_idx]
        men_proposal_index[proposer_man] += 1
        
        # --- 修改点 3: 遇到自己，停止求偶 ---
        if target == proposer_man:
            # 他选择了单身 (self)
            final_matching[proposer_man] = proposer_man
            continue
            
        # 目标是女性，尝试求偶
        target_woman = target
        
        # 检查该女性是否接受
        # 首先检查：该男性是否在女性的“可接受范围”内 (即 rank(man) < rank(woman_self))
        woman_pref_list = women_prefs.get(target_woman)
        if not woman_pref_list:
            free_men.append(proposer_man)
            continue
            
        try:
            rank_man = woman_pref_list.index(proposer_man)
            try:
                rank_self = woman_pref_list.index(target_woman)
            except ValueError:
                # 如果女性偏好里没有自己，假设她永远不愿意单身(rank_self = inf)，或者默认接受
                rank_self = float('inf')
            
            # --- 修改点 3: 截断逻辑 ---
            if rank_man > rank_self:
                # 该男性比单身还差，拒绝
                free_men.append(proposer_man)
                continue
                
            # 比较现任
            current_partner = women_partners[target_woman]
            
            if current_partner is None:
                # 目前单身，且该男性优于/等于单身状态，接受
                women_partners[target_woman] = proposer_man
                final_matching[proposer_man] = target_woman
            else:
                # 已有伴侣，比较排名 (index 越小越好)
                rank_current = woman_pref_list.index(current_partner)
                
                if rank_man < rank_current:
                    # 新欢更好
                    women_partners[target_woman] = proposer_man
                    final_matching[proposer_man] = target_woman
                    
                    # 旧爱恢复自由
                    final_matching[current_partner] = None # 暂时置空
                    free_men.append(current_partner)
                else:
                    # 现任更好，拒绝新欢
                    free_men.append(proposer_man)
                    
        except ValueError:
            # 如果男性不在女性偏好列表里，直接拒绝
            free_men.append(proposer_man)

    # 整理最终结果
    # 将女性字典同步到 final_matching
    for woman, partner in women_partners.items():
        if partner is None:
            final_matching[woman] = woman # 显式标记为单身 (self)
        else:
            final_matching[woman] = partner
            
    # 确保所有男性都有状态
    for man in men_prefs.keys():
        if final_matching.get(man) is None:
            final_matching[man] = man # 默认为单身

    return final_matching