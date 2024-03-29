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
        
    return parser

pconfs_30 = [
    ("ael", {"merge_pct":0.6, "min_event_cnt":3}),
    ("ael", {"merge_pct":0.6, "min_event_cnt":4}),
    ("ael", {"merge_pct":0.7, "min_event_cnt":3}),
    ("ael", {"merge_pct":0.7, "min_event_cnt":4}),
    
    ("dlog", {}),
    
    ("drain", {"depth":4, "sim_th":0.4, "max_children":20}),
    ("drain", {"depth":4, "sim_th":0.5, "max_children":20}),
    ("drain", {"depth":5, "sim_th":0.4, "max_children":20}),
    ("drain", {"depth":5, "sim_th":0.5, "max_children":20}),
    
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

    ("logcluster", {"freq_bar":10}),
    ("logcluster", {"freq_bar":20}),
    ("logcluster", {"freq_bar":30}),
    
    ("logmine", {"max_dist":0.2, "level":3}),
    ("logmine", {"max_dist":0.2, "level":4}),
    ("logmine", {"max_dist":0.3, "level":3}),
    ("logmine", {"max_dist":0.3, "level":4}),
    
    ("logsig", {"grp_num":80}),
    ("logsig", {"grp_num":100}),
    ("logsig", {"grp_num":120}),
    
    ("molfi", {}),
    
    ("nulog", {"k":3, "mask_percentage":0.8}),
    ("nulog", {"k":3, "mask_percentage":0.9}),
    ("nulog", {"k":5, "mask_percentage":0.8}),
    ("nulog", {"k":5, "mask_percentage":0.9}),
    
    ("shiso", {"threshold":0.3, "max_child":5}),
    ("shiso", {"threshold":0.3, "max_child":10}),
    ("shiso", {"threshold":0.4, "max_child":5}),
    ("shiso", {"threshold":0.4, "max_child":10}),
    
    ("slct", {"wp_supp":4, "tpl_supp":1}),
    ("slct", {"wp_supp":4, "tpl_supp":3}),
    ("slct", {"wp_supp":5, "tpl_supp":1}),
    ("slct", {"wp_supp":5, "tpl_supp":3}),
    
    ("spell", {"tau":0.4}),
    ("spell", {"tau":0.5}),
    ("spell", {"tau":0.6})
]

dconfs_30 = [
    ('android', 30),
    ('bgl', 30)
]


import os
import random

def gen_dataset(datadir, dname, dnum):
    datapath = os.path.join(datadir, "%s.log" % dname)

    idx = 0
    unique_logs = []
    for log in open(datapath, "r"):
        if log not in unique_logs:
            unique_logs.append(log)

            idx+=1
            if idx>=dnum:
                break

    return unique_logs

def gen_exp_groups(exp_grp_num):
    group = []
    for i in range(exp_grp_num):
        for j in range(exp_grp_num):
            if i==j:
                continue

            grp_1 = (i,j)
            grp_2 = (j,i)

            if grp_1 not in group and grp_2 not in group:
                group.append(grp_1)

    return group

def compare_tpls_allsame(tpls1, tpls2):
    if tpls1 is None or tpls2 is None:
        return False

    for tpl1, tpl2 in zip(tpls1, tpls2):
        if tpl1 != tpl2:
            return False
    return True

def gen_tpls(pname, pparas, logs, rseed):
    #run one times of experiment given setups
    parser = gen_parser(pname, pparas)
    parser_identifier = parser.get_parser_identifier()

    random.seed(rseed)
    tpls = parser.parse(logs)

    return tpls

def run_exp(datadir, dconfs, pconfs, exp_grp_num, rseeds):
    #exp_grp_num gen combination
    dconf2res = {}
    for dconf in dconfs:
        dname, dnum = dconf
        logs = gen_dataset(datadir, dname, dnum)

        pconf2res ={}
        for pconf in pconfs:
            pname, pparas = pconf
            rseed2res = {}

            for rseed in rseeds:
                exp_grp2reentrable = {}
                exp_grps = gen_exp_groups(exp_grp_num)

                for exp_grp in exp_grps:
                    tpls1 = gen_tpls(pname, pparas, logs, rseed)
                    tpls2 = gen_tpls(pname, pparas, logs, rseed)
                    reentrable = compare_tpls_allsame(tpls1, tpls2)
                    exp_grp2reentrable[exp_grp] = reentrable

                rseed2res[rseed] = exp_grp2reentrable

            pconf_key = str(pconf)
            pconf2res[pconf_key] = rseed2res

        dconf_key = str(dconf)
        dconf2res[dconf_key] = pconf2res

    return dconf2res

def detect_notentrable(exp_res):
    #exp_res is generated by run_exp,
    #its structure dconf, pconf, rseed, iround, entrable
    notentrable = []
    for dconf, dconfres in exp_res.items():
        for pconf, pconfres in dconfres.items():
            for rseed, rseedres in pconfres.items():
                for exp_grp, exp_grpres in rseedres.items():
                    #reentrable
                    if exp_grpres is False:
                        notentrable.append((dconf, pconf, rseed, exp_grp, exp_grpres))

    return notentrable

import warnings
warnings.filterwarnings("ignore")

datadir = "."
dconf2res = run_exp(datadir, dconfs_30, pconfs_30, 3, [1,2,3])
notentrable = detect_notentrable(dconf2res)

print(notentrable)
