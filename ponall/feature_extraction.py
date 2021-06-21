from collections import defaultdict

import pandas as pd
import os
import sys
import numpy as np
from numpy import sort
import copy
import math
import cmath
import requests
import io
import json
import jsonpath

import logging
from ponall import config,logconfig

logconfig.setup_logging()
log = logging.getLogger("ponall.extract_feature")
# constant
a_list = ('A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y')
aa_list = [i + j for i in a_list for j in a_list]
log.debug("aaindex path: %s", os.path.abspath(config.aa_index_path))
aaindex = pd.read_csv(config.aa_index_path, sep="\t", header=None, names=['name'] + aa_list, index_col='name')
aaindex = aaindex.T


def check_aa(seq, aa):
    seq = "".join(seq.split())
    aaf = aa[0]  # from
    aai = int(aa[1:-1])  # index
    aat = aa[-1]  # to
    return seq, aaf, aat, aai

#record 错误信息记录：
def msg_find(seq, aa):
    seq, aaf, aat, aai = check_aa(seq, aa)
    msg = ""
    if aaf not in a_list:
        msg = "aa error, origin of aa is invalid."
    if aat not in a_list:
        msg = "aa error, nutation of aa is invalid."
    if aai < 1 or aai > len(seq):
        msg = "aa error, index of aa is invalid."
    if seq[aai - 1] != aaf:
        msg = "aa error, seq[{}] = {}, but origin of aa = {}".format(aai, seq[aai - 1], aaf)
    return msg


def get_aaindex(seq, aa):
    seq, aaf, aat, aai = check_aa(seq, aa)
    res = aaindex.loc["{}{}".format(aaf, aat), :]
    return res.to_dict()

def get_residue(seq, aa):
    seq, aaf, aat, aai = check_aa(seq, aa)
    res = {"mut_residue": aai}
    return res

def get_length(seq, aa):
    seq, aaf, aat, aai = check_aa(seq, aa)
    res = {"len": len(seq)}
    return res

def get_nutationAll(seq, aa):
    seq, aaf, aat, aai = check_aa(seq, aa)
    nutationAll = pd.DataFrame(columns=[i[0] + '.' + i[1] for i in aa_list], index=aa_list).fillna(0)
    np.fill_diagonal(nutationAll.values, 1)
    return nutationAll.loc[[aaf+aat]].to_dict(orient='records')[0]

def get_groupAll(seq, aa):
    seq, aaf, aat, aai = check_aa(seq, aa)
    group = {
        'g1': ['V', 'I', 'L', 'F', 'M', 'W', 'Y', 'C'],
        'g2': ['D', 'E'],
        'g3': ['R', 'K', 'H'],
        'g4': ['G', 'P'],
        'g5': ['N', 'Q', 'S'],
        'g6': ['A', 'T']
    }

    # 颠倒键值对用于映射
    group_r = {}
    for k, v in group.items():
        for i in v:
            group_r[i] = k
    # 全对角矩阵
    groupAll = pd.DataFrame(
        index=[i + '.' + j for i in group.keys() for j in group.keys()],
        columns=[i + '.' + j for i in group.keys() for j in group.keys()],
    ).fillna(0)
    np.fill_diagonal(groupAll.values, 1)
    return groupAll.loc[[group_r[aaf]+'.'+group_r[aat]]].to_dict(orient='records')[0]

def get_neighborhood_features(seq, aa):
    seq, aaf, aat, aai = check_aa(seq, aa)

    def find_win(seq, aai):
        # 确定边界
        index = aai - 1
        front = 0 if index - 11 < 0 else index - 11
        after = len(seq) if index + 12 > len(seq) - 1 else index + 12
        return seq[front: after]

    def get_count_a(win):
        """ count number of aa in windows"""
        a_dict = defaultdict(int)
        for i in win:
            a_dict[i] += 1  # 递增
        return {'AA20D.' + i: a_dict[i] for i in a_list}

    win = find_win(seq, aai)
    count_a = get_count_a(win)
    nei_feature = count_a
    # 1.NonPolarAA:Number of nonpolar neighborhood residues
    nei_feature['NonPolarAA'] = nei_feature['AA20D.' + a_list[0]] + nei_feature['AA20D.' + a_list[4]] + nei_feature[
        'AA20D.' + a_list[5]] + nei_feature['AA20D.' + a_list[7]] + nei_feature['AA20D.' + a_list[9]] + nei_feature[
                                    'AA20D.' + a_list[10]] + nei_feature['AA20D.' + a_list[12]] + nei_feature[
                                    'AA20D.' + a_list[17]] + nei_feature['AA20D.' + a_list[18]] + nei_feature[
                                    'AA20D.' + a_list[19]]
    # 2.PolarAA:Number of polar neighborhood residues
    nei_feature['PolarAA'] = nei_feature['AA20D.' + a_list[1]] + nei_feature['AA20D.' + a_list[11]] + nei_feature[
        'AA20D.' + a_list[13]] + nei_feature['AA20D.' + a_list[15]] + nei_feature['AA20D.' + a_list[16]]
    # 3.ChargedAA:Number of charged neighborhood residues
    nei_feature['ChargedAA'] = nei_feature['AA20D.' + a_list[2]] + nei_feature['AA20D.' + a_list[3]] + nei_feature[
        'AA20D.' + a_list[6]] + nei_feature['AA20D.' + a_list[8]] + nei_feature['AA20D.' + a_list[14]]
    # 4.PosAA:Number of Positive charged neighborhood residues
    nei_feature['PosAA'] = nei_feature['AA20D.' + a_list[2]] + nei_feature['AA20D.' + a_list[3]]
    # 5.NegAA:Number of Negative charged neighborhood residues
    nei_feature['NegAA'] = nei_feature['AA20D.' + a_list[6]] + nei_feature['AA20D.' + a_list[8]] + nei_feature[
        'AA20D.' + a_list[14]]
    return nei_feature

#Sift4G
# def get_sift4g(id, seq, aa):
    # seq, aaf, aat, aai = check_aa(seq, aa)
    # with open('seq.fa', 'w') as file_object:
        # file_object.write('>'+id+'\n')
        # file_object.write(seq)
    # if not os.path.exists('subst'):
        # os.makedirs('subst')
    # if not os.path.exists('sift_out'):
        # os.makedirs('sift_out')
    # with open('./subst/{}.subst'.format(id), 'w') as file_object:
        # file_object.write(aa)
    # os.system('sift4g -q ./seq.fa --subst ./subst/ -d ../sift4g/uniprot_sprot.fasta --out ./sift_out/')
    # files = ['./sift_out/{}.SIFTprediction'.format(id)]
    # files
    # sift = []
    # for file in files:
        # with open(file) as f:
            # 读取文件
            # tmp = f.read()
            # tmp = tmp.strip().split('\n')  # 每行分割
            # gi序号
        # (filepath, tempfilename) = os.path.split(file)
        # (filename, extension) = os.path.splitext(tempfilename)
            # 组成json
        # for i in tmp:
            # tmp_1 = []
            # if '\t' in i:
                # tmp_1 = i.split('\t')
                # sift.append({
                    # 'hits': int(tmp_1[5]),
                    # 'score': float(tmp_1[2])
                # }
                # )
    # sift = [{"hits": 29,'score':0.48}]
    # return sift[0]
    
def get_sift4g_file(id, seq):
    with open('seq.fa', 'w') as file_object:
        file_object.write('>'+id+'\n')
        file_object.write(seq+'\n\n')
    if not os.path.exists('sift_out_file'):
        os.makedirs('sift_out_file')
    os.system('sift4g -q ./seq.fa -d ../sift4g/uniprot_sprot.fasta --out ./sift_out_file/')
    filename = './sift_out_file/{}.SIFTprediction'.format(id)
    filename1 = './sift_out_file/{}.SIFTprediction1'.format(id)
    fin = open(filename, 'r')
    a = fin.readlines()
    for i in a:
        i = i.strip()
    fout = open(filename1, 'w')
    l = a[5:]
    l.insert(0, 'A  B  C  D  E  F  G  H  I  K  L  M  N  P  Q  R  S  T  V  W  X  Y  Z  *  -\n')
    b = ''.join(l)
    fout.write(b)

    sift = pd.read_csv(filename1,sep='  ')
    print(sift)
    # sift[aat][aai+1]
    return sift

def get_sift4g_hits(id, seq, aa):
    seq, aaf, aat, aai = check_aa(seq, aa)
    with open('seq.fa', 'w') as file_object:
        file_object.write('>'+id+'\n')
        file_object.write(seq+'\n\n')
    if not os.path.exists('subst'):
        os.makedirs('subst')
    if not os.path.exists('sift_out'):
        os.makedirs('sift_out')
    with open('./subst/{}.subst'.format(id), 'w') as file_object:
        file_object.write(aa)
    os.system('sift4g -q ./seq.fa --subst ./subst/ -d ../sift4g/uniprot_sprot.fasta --out ./sift_out/')
    files = ['./sift_out/{}.SIFTprediction'.format(id)]
    files
    sift = []
    for file in files:
        with open(file) as f:
            # 读取文件
            tmp = f.read()
            tmp = tmp.strip().split('\n')  # 每行分割
            # gi序号
        (filepath, tempfilename) = os.path.split(file)
        (filename, extension) = os.path.splitext(tempfilename)
        #     组成json
        for i in tmp:
            tmp_1 = []
            if '\t' in i:
                tmp_1 = i.split('\t')
                sift.append({
                    'hits': int(tmp_1[5]),
                    'score': float(tmp_1[2])
                }
                )
    return sift[0]['hits']

def get_GO(id):
    requestURL = "https://www.ebi.ac.uk/QuickGO/services/annotation/search?geneProductId={}".format(id)

    r = requests.get(requestURL, headers={"Accept": "application/json"})

    if not r.ok:
        r.raise_for_status()
        sys.exit()

    responseBody = r.text
    responseBody = json.loads(responseBody)
    gos = jsonpath.jsonpath(responseBody, '$..goId')
    if gos!=False:
        #查找祖先
        ancestor = pd.read_csv(config.ancestor_path, sep=',', header=None, low_memory=False).set_index(0)

        l = []
        for i in gos:
            if i in ancestor.index:
                l = l + ancestor.loc[i, :].tolist()
            l.append(i)
        l = list(set(l))
        if np.nan in l:
            l.remove(np.nan)
        if 'all' in l:
            l.remove('all')
        gos = ','.join(l)

    res = {"ancestor": gos}
    return res

def get_pos_1(seq, aa):
    seq, aaf, aat, aai = check_aa(seq, aa)
    pos_1=0
    if aai==1:
        pos_1=1
    res = {"pos_1": pos_1}
    return res

def get_Site_1(id):
    site=''
    if not os.path.exists('Site'):
        os.makedirs('Site')
    url = "https://www.uniprot.org/uniprot/{}.gff".format(id)
    urlData = requests.get(url).content
    rawData = pd.read_csv(io.StringIO('\n'.join(urlData.decode('utf-8').split('\n')[2:])), sep="\t", header=None,
                          names=['ID', 'database', 'site', 'from', 'to', '1', '2', '3', '4', '5'])
    return rawData

def get_Site_2(rawData, seq, aa):
    seq, aaf, aat, aai = check_aa(seq, aa)
    l = []
    for indexs2 in rawData.index:
        if rawData.loc[indexs2]['from'] <= aai and rawData.loc[indexs2]['to'] >= aai:
            l.append(rawData.loc[indexs2, 'site'])
    l = list(set(l))
    site = ','.join(l)
    res = {"site": site}
    return res


def get_all_features(n, seq, aa, kind):
    print('收集特征')
    list = []
    #err_list = []    
    if (kind in ['uniprot id','ensembl id','vcf']):
        print('有GO')
        flag='*'
        GOs=''
        Sites=''
        sift_hits=0
        sift=None
        for i in range(len(n)):
            id_ = n[i]
            seq_ = seq[i]
            aa_ = aa[i]
            features = {}
            features.update({"id" : i})
            features.update({"msg": msg_find(seq_, aa_)})
            if msg_find(seq_, aa_)=="":
                if id_ != flag:
                    sift_hits = get_sift4g_hits(id_, seq_, aa_)
                    sift = get_sift4g_file(id_, seq_)
                    GOs = get_GO(id_)
                    Sites = get_Site_1(id_)
                    flag = id_
                seq1, aaf1, aat1, aai1 = check_aa(seq_, aa_)
                print("hits:{}".format(sift_hits),'score:{}'.format(sift[aat1][aai1-1]))
                features.update(get_residue(seq_, aa_))
                features.update(get_length(seq_, aa_))
                features.update(get_aaindex(seq_, aa_))
                features.update(get_nutationAll(seq_, aa_))
                features.update(get_groupAll(seq_, aa_))
                features.update(get_neighborhood_features(seq_, aa_))
                features.update({"hits": sift_hits,'score':sift[aat1][aai1-1]})
                features.update(GOs)
                features.update(get_pos_1(seq_, aa_))
                features.update(get_Site_2(Sites, seq_, aa_))
                df_features = pd.DataFrame([features])
                list.append(df_features)
            else:
                df_features = pd.DataFrame([features])
                # err_list.append(df_features)
                list.append(df_features)
    else:
        print('无GO')
        flag='*'
        sift_hits=0
        sift=None
        for i in range(len(seq)):
            id_ = n[i]
            seq_ = seq[i]
            aa_ = aa[i]
            features = {}
            features.update({"id" : i})
            features.update({"msg": msg_find(seq_, aa_)})
            if msg_find(seq_, aa_) == "":
                if id_ != flag:
                    sift_hits = get_sift4g_hits(id_, seq_, aa_)
                    sift = get_sift4g_file(id_, seq_)
                    flag = id_
                seq1, aaf1, aat1, aai1 = check_aa(seq_, aa_)
                print("hits:{}".format(sift_hits),'score:{}'.format(sift[aat1][aai1-1]))
                features.update(get_residue(seq_, aa_))
                features.update(get_length(seq_, aa_))
                features.update(get_aaindex(seq_, aa_))
                features.update(get_nutationAll(seq_, aa_))
                features.update(get_groupAll(seq_, aa_))
                features.update(get_neighborhood_features(seq_, aa_))
                features.update({"hits": sift_hits,'score':sift[aat1][aai1-1]})
                features.update(get_pos_1(seq_, aa_))
                df_features = pd.DataFrame([features])
                list.append(df_features)
            else:
                df_features = pd.DataFrame([features])
                list.append(df_features)
                # err_list.append(df_features)
    df2 = pd.concat(list)
    df2 = df2.reset_index()
    del df2['index']
    print('特征')
    print(df2)
    return df2
    # df1 = pd.concat(err_list)
    # df1 = df1.reset_index()
    # del df1['index']
    # print('错误')
    # print(df1)
    # return df1, df2