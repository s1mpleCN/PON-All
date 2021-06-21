# -*- coding:utf-8 -*-
'''
__author__ = 'XD'
__mtime__ = 2021/1/18
__project__ = Pon-All
Fix the Problem, Not the Blame.
'''
import os
import pandas as pd

project_path = os.path.dirname(__file__)
aa_index_path = os.path.join(project_path, "./data/aaindexmatrix.txt")
model_path = os.path.join(project_path, "./data/lightgbm_feature_select_20.rfe")
model_path_N = os.path.join(project_path, "./data/lightgbm_feature_select_20_N.rfe")
ancestor_path = os.path.join(project_path, "./data/ANCESTOR.csv")
all_path = os.path.join(project_path, "./data/All_species_train.csv")
all_bootstrap_path = os.path.join(project_path, "./data/Blind_All_Species/All_species/")
human_path = os.path.join(project_path, "./data/Human_train.csv")
human_bootstrap_path = os.path.join(project_path, "./data/Blind_All_Species/Human/")
animal_path = os.path.join(project_path, "./data/Animal_train.csv")
animal_bootstrap_path = os.path.join(project_path, "./data/Blind_All_Species/Animal/")
plant_path = os.path.join(project_path, "./data/Plant_train.csv")
plant_bootstrap_path = os.path.join(project_path, "./data/Blind_All_Species/Plant/")
test_bootstrap_path = os.path.join(project_path, "./data/bootstrap_re_lgbm/")



# data1 = pd.read_csv("./data/train1.csv")
# data2 = pd.read_csv("./data/validation1.csv")
# Blind_train = pd.concat([data1,data2], ignore_index=True)
# Blind_train = Blind_train.iloc[:,1:]
# Blind_train.insert(0,'index',Blind_train.index)
# Blind_train.to_csv('./data/All_train.csv',index=False)
# Blind_train = pd.read_csv("./data/All_train.csv")
# print(Blind_train)