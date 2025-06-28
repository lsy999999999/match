import pandas as pd
import numpy as np

def read_file(path, group=1):
    df = pd.read_excel(path)
    attributes = df[['group', 'iid', 'pid', 'gender', 'gpt_score']]

    group_attr, men_attr, women_attr = extract_attributes(attributes, group)

    men_set_list, women_set_list = get_unique_sets(men_attr, women_attr)

    men_dict_sort = create_sorted_dict(men_set_list, men_attr)
    women_dict_sort = create_sorted_dict(women_set_list, women_attr)

    return men_dict_sort, women_dict_sort


def read_file_attr(path, group=1):
    df = pd.read_excel(path)
    # Modified to include numeric partner scores instead of text satisfaction levels
    attributes = df[['group', 'iid', 'gender', 'pid', 'age', 'age_o', 'career', 'gpt_score', 
                     'attractive_partner', 'sincere_partner', 'intelligence_partner', 
                     'funny_partner', 'ambition_partner', 'shared_interests_partner',
                     'attractive_important', 'sincere_important', 'intelligence_important',
                     'funny_important', 'ambition_important', 'shared_interests_important']]

    women_dict = extract_woman_attributes(attributes, group)
    men_dict = extract_man_attributes(attributes, group)
    return men_dict, women_dict

# Helper functions for read_file
def extract_attributes(attributes, group):
    attributes = attributes.to_numpy()
    group_attr = np.zeros((1, 5))
    men_attr = np.zeros((1, 5))
    women_attr = np.zeros((1, 5))
    flag = 0

    for attr in attributes:
        if attr[0] == group:
            if flag == 0:
                flag = 1
            group_attr = np.append(group_attr, [attr], axis=0)
            if attr[3] == 1:
                men_attr = np.append(men_attr, [attr], axis=0)
            else:
                women_attr = np.append(women_attr, [attr], axis=0)
        else:
            if flag == 1:
                break
    group_attr, men_attr, women_attr = group_attr[1:], men_attr[1:], women_attr[1:]

    return group_attr, men_attr, women_attr


def get_unique_sets(men_attr, women_attr):
    men_set = np.unique(men_attr[:, 1])
    women_set = np.unique(women_attr[:, 1])

    men_set_list = men_set.astype(int).tolist()
    women_set_list = women_set.astype(int).tolist()

    return men_set_list, women_set_list


def create_sorted_dict(men_set_list, men_attr):
    men_dict = {}
    for value in men_set_list:
        dic1 = {}
        for men in men_attr:
            if men[1].astype(int) == value:
                dic1[men[2].astype(int)] = men[4]
        men_dict[value] = dic1

    men_dict_sort = {}
    for key, value in men_dict.items():
        sorted_keys = sorted(value, key=value.get, reverse=True)
        men_dict_sort[key] = sorted_keys

    return men_dict_sort


# Helper functions for read_file_woman
def extract_woman_attributes(attributes, group):
    attributes = attributes.to_numpy()
    women_attr = np.zeros((1, 20))
    flag = 0

    for attr in attributes:
        if attr[0] == group:
            if flag == 0:
                flag = 1
            if attr[2] == 0:
                women_attr = np.append(women_attr, [attr], axis=0)
        else:
            if flag == 1:
                break
    women_attr = women_attr[1:]

    women_set = np.unique(women_attr[:, 1])
    women_set_list = women_set.astype(int).tolist()
    
    women_dict = create_women_dict(women_set_list, women_attr)

    return women_dict

def extract_man_attributes(attributes, group):
    attributes = attributes.to_numpy()
    women_attr = np.zeros((1, 20))
    flag = 0

    for attr in attributes:
        if attr[0] == group:
            if flag == 0:
                flag = 1
            if attr[2] == 1:
                women_attr = np.append(women_attr, [attr], axis=0)
        else:
            if flag == 1:
                break
    women_attr = women_attr[1:]

    women_set = np.unique(women_attr[:, 1])
    women_set_list = women_set.astype(int).tolist()
    
    women_dict = create_women_dict(women_set_list, women_attr)

    return women_dict


def create_women_dict(women_set_list, women_attr):
    women_dict = {}
    for value in women_set_list:
        dic1 = {}
        for women in women_attr:
            if women[1] == value:
                dic1[women[3]] = women[4:].tolist()
        women_dict[value] = dic1

    return women_dict


if __name__=='__main__':
    group = 1
    man_dict = read_file(r'/home/lsy/match/dataset/save_merge_select_null_3.xlsx', group)
    woman_dict = read_file_woman(r'/home/lsy/match/dataset/save_merge_select_null_3.xlsx', group)
    print(man_dict)
    print(woman_dict)