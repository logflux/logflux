# Copyright 2024 Guineng Zheng and The University of Utah
# All rights reserved.
# This file is part of the logflux project, which is licensed under the
# simplified BSD license.

from logflux import AELParser
from logflux import DLogParser
from logflux import DrainParser
from logflux import FTTreeParser
from logflux import IPLOMParser
from logflux import LenmaParser
from logflux import LFAParser
from logflux import LKEParser
from logflux import LogClusterParser
from logflux import LogmineParser
from logflux import LogSigParser
from logflux import MoLFIParser
from logflux import NuLogParser
from logflux import SHISOParser
from logflux import SLCTParser
from logflux import SpellParser
from logflux import VAParser

def gen_parser(pname, pparas):
    if pname == "ael":
        merge_pct=pparas["merge_pct"]
        min_event_cnt=pparas["min_event_cnt"]
        parser = AELParser(merge_pct, min_event_cnt)
        
    elif pname == "dlog":
        parser = DLogParser()
        
    elif pname == "drain":
        depth=pparas["depth"]
        sim_th = pparas["sim_th"]
        max_children = pparas["max_children"]
        parser = DrainParser(depth, sim_th, max_children)
        
    elif pname == "fttree":
        max_child = pparas["max_child"]
        cut_depth = pparas["cut_depth"]
        parser = FTTreeParser(max_child, cut_depth)
        
    elif pname == "iplom":
        lower_bound = pparas["lower_bound"]
        upper_bound = pparas["upper_bound"]
        parser = IPLOMParser(lower_bound=lower_bound, upper_bound=upper_bound)
        
    elif pname == "lenma":
        threshold = pparas["threshold"]
        parser = LenmaParser(threshold)
        
    elif pname == "lfa":
        parser = LFAParser()
        
    elif pname == "lke":
        split_threshold = pparas["split_threshold"]
        parser = LKEParser(split_threshold)
        
    elif pname == "logcluster":
        freq_bar = pparas["freq_bar"]
        parser = LogClusterParser(freq_bar)
        
    elif pname == "logmine":
        max_dist = pparas["max_dist"]
        level = pparas["level"]
        parser = LogmineParser(max_dist, level)
        
    elif pname == "logsig":
        grp_num = pparas["grp_num"]
        parser = LogSigParser(grp_num)
        
    elif pname == "molfi":
        parser = MoLFIParser()
    
    elif pname == "nulog":
        k = pparas["k"]
        mask_percentage = pparas["mask_percentage"]
        parser = NuLogParser(k=k, mask_percentage=mask_percentage, nr_epochs=8)
        
    elif pname == "shiso":
        threshold = pparas["threshold"]
        max_child = pparas["max_child"]
        parser = SHISOParser(threshold, max_child)
        
    elif pname == "slct":
        wp_supp = pparas["wp_supp"]
        tpl_supp = pparas["tpl_supp"]
        parser = SLCTParser(wp_supp, tpl_supp) 
    
    elif pname == "spell":
        tau = pparas["tau"]
        parser = SpellParser(tau)
    
    elif pname == "varl":
        method = "relative-line"
        threshold = pparas["threshold"]
        parser = VAParser(method, threshold)
        
    elif pname == "varv":
        method = "relative-variable"
        threshold = pparas["threshold"]
        parser = VAParser(method, threshold)
        
    else:
        raise Exception('unknown parser type')
        
    return parser


pconfs = [
    ("ael", {"merge_pct":0.6, "min_event_cnt":3}),
    ("ael", {"merge_pct":0.6, "min_event_cnt":4}),
    ("ael", {"merge_pct":0.7, "min_event_cnt":3}),
    ("ael", {"merge_pct":0.7, "min_event_cnt":4}),

    ("dlog", {}),

    ("drain", {"depth":4, "sim_th":0.4, "max_children":100}),
    ("drain", {"depth":4, "sim_th":0.5, "max_children":100}),
    ("drain", {"depth":5, "sim_th":0.4, "max_children":100}),
    ("drain", {"depth":5, "sim_th":0.5, "max_children":100}),

    ("fttree", {"max_child":3, "cut_depth":3}),
    ("fttree", {"max_child":3, "cut_depth":4}),
    ("fttree", {"max_child":4, "cut_depth":3}),
    ("fttree", {"max_child":4, "cut_depth":4}),

    ("iplom", {"lower_bound":0.2, "upper_bound":0.7}),
    ("iplom", {"lower_bound":0.2, "upper_bound":0.8}),
    ("iplom", {"lower_bound":0.3, "upper_bound":0.7}),
    ("iplom", {"lower_bound":0.3, "upper_bound":0.8}),

    ("lenma", {"threshold":0.7}),
    ("lenma", {"threshold":0.8}),
    ("lenma", {"threshold":0.9}),

    ("lfa", {}),

    ("lke", {"split_threshold":3}),
    ("lke", {"split_threshold":4}),
    ("lke", {"split_threshold":5}),

    ("logcluster", {"freq_bar":40}),
    ("logcluster", {"freq_bar":50}),
    ("logcluster", {"freq_bar":60}),

    ("logmine", {"max_dist":0.2, "level":3}),
    ("logmine", {"max_dist":0.2, "level":4}),
    ("logmine", {"max_dist":0.3, "level":3}),
    ("logmine", {"max_dist":0.3, "level":4}),

    ("logsig", {"grp_num":1}),
    ("logsig", {"grp_num":2}),
    ("logsig", {"grp_num":3}),

    ("molfi", {}),

    ("nulog", {"k":3, "mask_percentage":0.8}),
    ("nulog", {"k":3, "mask_percentage":0.9}),
    ("nulog", {"k":5, "mask_percentage":0.8}),
    ("nulog", {"k":5, "mask_percentage":0.9}),

    ("shiso", {"threshold":0.3, "max_child":60}),
    ("shiso", {"threshold":0.3, "max_child":80}),
    ("shiso", {"threshold":0.4, "max_child":60}),
    ("shiso", {"threshold":0.4, "max_child":80}),

    ("slct", {"wp_supp":40, "tpl_supp":10}),
    ("slct", {"wp_supp":40, "tpl_supp":20}),
    ("slct", {"wp_supp":50, "tpl_supp":10}),
    ("slct", {"wp_supp":50, "tpl_supp":20}),

    ("spell", {"tau":0.4}),
    ("spell", {"tau":0.5}),
    ("spell", {"tau":0.6}),

    ("varl", {"threshold":0.4}),
    ("varl", {"threshold":0.5}),
    ("varl", {"threshold":0.6}),

    ("varv", {"threshold":0.4}),
    ("varv", {"threshold":0.5}),
    ("varv", {"threshold":0.6})
]



def gen_dataset(dname, dnum):
    idx = 0
    unique_logs = []
    for log in open(dname, "r"):
        if log not in unique_logs:
            unique_logs.append(log)

            idx+=1
            if idx>=dnum:
                break

    return unique_logs



dconfs_30 = [
    ('android.log', 30),
    ('bgl.log', 30),
]

dconfs_100 = [
    ('android.log', 100),
    ('bgl.log', 100)
]

dconfs_1 = [
    ('android.log', 1),
    ('bgl.log', 1)
]

dconfs_2 = [
    ('android.log', 2),
    ('bgl.log', 2)
]

dconfs_3 = [
    ('android.log', 3),
    ('bgl.log', 3)
]

import warnings
warnings.filterwarnings("ignore")
algo_test = "logsig"

for dname, dnum in dconfs_3:
    for pname, pparas in pconfs:
        if pname!=algo_test:
            continue
        parser = gen_parser(pname, pparas)
        logs = gen_dataset(dname, dnum)
        try:
            tpls = parser.parse(logs)
            print(dname, dnum, pname, pparas, len(tpls))
        except Exception as e:
            print("running error", pname, pparas, dname, dnum)
            print(e)
            continue

#this parer is shared by different dataset, it requires parser to be reentrable
#right now fttree and shiso from amulog is not reentrable
'''
for pname, pparas in pconfs:
    if pname!=algo_test:
        continue
    parser = gen_parser(pname, pparas)
    for dname, dnum in dconfs_3:
        logs = gen_dataset(dname, dnum)
        try:
            tpls = parser.parse(logs)
            print(dname, dnum, pname, pparas, len(tpls))
        except Exception as e:
            print("running error", pname, pparas, dname, dnum)
            print(e)
            continue
'''

