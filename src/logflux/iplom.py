# Copyright 2024 Guineng Zheng and The University of Utah
# All rights reserved.
# This file is part of the logflux project, which is licensed under the
# simplified BSD license.

from collections import namedtuple
#psize is its parent size
Partition = namedtuple("Partition", "tok_logs, stage, size, psize, valid")
Mapping = namedtuple("Mapping", "idx, lval, rval")

def is_valid(partition, PST):
    if PST!=0 and partition.size/partition.psize<PST:
        return False
    else:
        return True
    
def groupby_len(logs):
    len2tokens = {}
    
    for log in logs:
        log_tokens = log.split()
        log_len = len(log_tokens)
        
        if log_len not in len2tokens:
            len2tokens[log_len] = []
            
        len2tokens[log_len].append(log_tokens)
        
    return len2tokens

def partition_by_token_cnt(tok_logs):
    psize = len(tok_logs)
    len2tokens = groupby_len(tok_logs)
    
    partitions = []
    for log_len, log_toks in len2tokens.items():
        size = len(log_toks)
        partition = Partition(log_toks, 1, size, psize, True)
        partitions.append(partition)
        
    return partitions

def gen_pos2uws(tok_logs):
    #tok_logs are logs with same length
    #pos to unqie word set
    pos2uws = {}
    for tok_log in tok_logs:
        for pos, tok in enumerate(tok_log):
            if pos not in pos2uws:
                pos2uws[pos] = []
            if tok not in pos2uws[pos]:
                pos2uws[pos].append(tok)
                
    return pos2uws

def find_split_loc(pos2uws):
    #pos to unique word cnt
    pos2uwc = [len(val) for val in pos2uws.values()]
    min_pos = pos2uwc.index(min(pos2uwc))
    min_cnt = pos2uwc[min_pos]
    
    return min_pos, min_cnt

def split_by_pos(tok_logs, pos):
    tok2logs = {}
    for tok_log in tok_logs:
        pos_tok = tok_log[pos]
        if pos_tok not in tok2logs:
            tok2logs[pos_tok] = []
        tok2logs[pos_tok].append(tok_log)
        
    return tok2logs

def groupby_tokpos(tok_logs):
    pos2uws = gen_pos2uws(tok_logs)
    min_pos, min_cnt = find_split_loc(pos2uws)
    tok2logs = split_by_pos(tok_logs, min_pos)
    
    return tok2logs

def partition_by_token_position(partition):
    psize = len(partition.tok_logs)
    tok2logs = groupby_tokpos(partition.tok_logs)
    
    partitions = []
    for log_toks in tok2logs.values():
        size = len(log_toks)
        partition = Partition(log_toks, 2, size, psize, True)
        partitions.append(partition)
        
    return partitions

def split_stage_2(stage_1_partitions):
    stage_2_partitions = []
    
    for partition in stage_1_partitions:
        if len(partition.tok_logs) == 1:
            #no need to split
            #stage_2_partition = Partition(partition.tok_logs, 2, partition.size, partition.psize, partition.valid)
            stage_2_partition = Partition(partition.tok_logs, 2, 1, 1, partition.valid)
            stage_2_partitions.append(stage_2_partition)
            continue
        
        tokpos_partitions = partition_by_token_position(partition)
        for tokpos_partition in tokpos_partitions:
            stage_2_partitions.append(tokpos_partition)
            
    return stage_2_partitions

def groupby_col(tok_logs):
    #logs with the same length
    loglen = len(tok_logs[0])
    col_stats = [{} for i in range(loglen)]
    
    for i, tok_log in enumerate(tok_logs):
        for j, tok in enumerate(tok_log):
            if tok not in col_stats[j]:
                col_stats[j][tok] = []
            col_stats[j][tok].append(i)
            
    return col_stats

def calc_col_uniq(tok_logs):
    #logs with the same length
    col_stats = groupby_col(tok_logs)
    col_uniqs = []
    
    for col_idx, col_stat in enumerate(col_stats):
        col_uniqs.append(len(col_stat))
        
    return col_uniqs

def calc_uniq_ratio(tok_logs):
    #number of positions with one uniqe value
    #GC in paper
    col_uniq = calc_col_uniq(tok_logs)
    uniq_num = len([x for x in col_uniq if x==1])
    return uniq_num/len(col_uniq)

def gen_cnt2poses(cnt2pos):
    cnt2poses = {}
    
    for pos, cnt in enumerate(cnt2pos):
        if cnt not in cnt2poses:
            cnt2poses[cnt] = []
        cnt2poses[cnt].append(pos)
        
    return cnt2poses

def gen_candidate_poses(cnt2poses):
    #cnt !=1 (=1 no need to split)
    #poses num >=2 so we can find at least 2 positions
    candidate_poses = []
    
    for cnt, poses in cnt2poses.items():
        if cnt!=1 and len(poses)>=2:
            candidate_poses.append(poses)
            
    return candidate_poses


def gen_biposes_strict(cnt2poses):
    #if there are positions with more than 2 candidate positions
    #strictly have 2 candidate positions
    #if it returns none it will got to bijection pose2 
    #the top least 2 cardinaility col will be chosen
    candidate_poses = gen_candidate_poses(cnt2poses)
    
    if len(candidate_poses)>0:
        bi_poses = sorted(candidate_poses, key=lambda x:len(x), reverse=True)[0][:2]
    else:
        bi_poses = None
        
    return bi_poses

def gen_bijection_poses_1(col_uniq):
    cnt2poses = gen_cnt2poses(col_uniq)
    bi_poses = gen_biposes_strict(cnt2poses)
    
    return bi_poses

def gen_bijection_poses_2(col_uniq):
    #there might be positions does not share the same number
    #sort it from small to big and take the two head positions
    #if gen_biject_pose_1 return None use it?
    #col_uniq is a list of distinct num of each col
    #and they are all different
    #take the smallest two columns
    
    #gen bijection 2 
    #return 0 location means all pos only have one value
    #return 1 location means there are only one position is not 1 while others are one
    #return more than 2 locations, normal
    
    pos_cnt = [(pos, cnt) for pos, cnt in enumerate(col_uniq) if cnt!=1]
    sorted_pos_cnt = sorted(pos_cnt, key=lambda x:x[1])
    poses = [x[0] for x in sorted_pos_cnt[:2]]
    return poses

def gen_bijection_poses(tok_logs):
    col_uniq = calc_col_uniq(tok_logs)
    
    bi_poses = gen_bijection_poses_1(col_uniq)
    if bi_poses is None:
        bi_poses = gen_bijection_poses_2(col_uniq)
        
    return bi_poses

def group_vals(vals):
    #key is val value is idxs
    val2idxs = {}
    
    for idx, val in enumerate(vals):
        if val not in val2idxs:
            val2idxs[val] = []
        val2idxs[val].append(idx)
        
    return val2idxs

def gen_oneones_tuple(tok_logs, p1, p2):
    p1_vals = [log[p1] for log in tok_logs]
    p2_vals = [log[p2] for log in tok_logs]
    
    p1_val2idxs = group_vals(p1_vals)
    p2_val2idxs = group_vals(p2_vals)
    
    oneones = []
    for p1_val, p1_idxs in p1_val2idxs.items():
        correspond_p2_vals = []
        for p1_idx in p1_idxs:
            #reverse search
            correspond_p2_val = p2_vals[p1_idx]
            correspond_p2_vals.append(correspond_p2_val)

        if len(set(correspond_p2_vals))>1:
            #not one one
            continue
        else:
            p2_val = correspond_p2_vals[0]
            reverse_p1_idxs = p2_val2idxs[p2_val]

            if sorted(reverse_p1_idxs) == sorted(p1_idxs):
                #one one
                oneone = []
                for p1_idx in p1_idxs:
                    oneone_mapping = Mapping(p1_idx, p1_val, p2_val)
                    oneone.append(oneone_mapping)
                oneones.append(oneone)
                
    return oneones

def gen_onems_tuple(tok_logs, p1, p2):
    p1_vals = [log[p1] for log in tok_logs]
    p2_vals = [log[p2] for log in tok_logs]
    
    p1_val2idxs = group_vals(p1_vals)
    p2_val2idxs = group_vals(p2_vals)
    
    onems = []
    
    for p1_val, p1_idxs in p1_val2idxs.items():
        correspond_p2_idx_vals = []
        for p1_idx in p1_idxs:
            #reverse search
            correspond_p2_val = p2_vals[p1_idx]
            correspond_p2_idx_vals.append((p1_idx, correspond_p2_val))

        #1-m requires right to be more than 1
        #m-side will be uniqued
        unique_p2_idx_vals = set([x[1] for x in correspond_p2_idx_vals])
        if not len(unique_p2_idx_vals)>1:
            continue

        #correspond_p2_val2idxs = construct_val2idxs(correspond_p2_idx_vals)
        correspond_p2_val2idxs = dict([(x[1],p2_val2idxs[x[1]]) for x in correspond_p2_idx_vals])

        #idx, p1_val should be same, p2_val can be diversified
        reverse_vals = []
        for reverse_idxs in correspond_p2_val2idxs.values():
            for idx in reverse_idxs:
                reverse_vals.append((idx, p1_vals[idx], p2_vals[idx]))

        unique_reverse_vals = set([x[1] for x in reverse_vals])

        if len(unique_reverse_vals)==1:
            #1-m
            #reverse_vals is idx, p1_val, p2_val
            onem = []
            for p1_idx in p1_idxs:
                onem_mapping = Mapping(p1_idx, p1_vals[p1_idx], p2_vals[p1_idx])
                onem.append(onem_mapping)
            
            onems.append(onem)
            
    return onems


def gen_mones_tuple(tok_logs, p1, p2):
    mones = []
    
    onems = gen_onems_tuple(tok_logs, p2, p1)
    for onem in onems:
        mone = []
        for onem_i in onem:
            mone_mapping = Mapping(onem_i.idx, onem_i.rval, onem_i.lval)
            mone.append(mone_mapping)
            
        mones.append(mone)
        
    return mones

def gen_mms_tuple(tok_logs, p1, p2, oneones, onems, mones):
    oneone_idxs = [[x.idx for x in oneone] for oneone in oneones]
    oos = sum(oneone_idxs, [])
    
    onem_idxs = [[x.idx for x in onem] for onem in onems]
    oms = sum(onem_idxs, [])
    
    mone_idxs = [[x.idx for x in mone] for mone in mones]
    mos = sum(mone_idxs, [])
    
    oo_om_mo_idxs = sum([oos, oms, mos], [])
    mms = []
    for idx, tok_log in enumerate(tok_logs):
        if idx not in oo_om_mo_idxs:
            mm_mapping = Mapping(idx, tok_logs[idx][p1], tok_logs[idx][p2])
            mms.append(mm_mapping)
            
    return mms

def gen_mappings(tok_logs, lpos, rpos):
    oos = gen_oneones_tuple(tok_logs, lpos, rpos)
    oms = gen_onems_tuple(tok_logs, lpos, rpos)
    mos = gen_mones_tuple(tok_logs, lpos, rpos)
    mms = gen_mms_tuple(tok_logs, lpos, rpos, oos, oms, mos)
    
    return oos, oms, mos, mms


def get_rank_position(mapping, mtype, lower_bound, upper_bound):
    if mtype == "om":
        mset = set([x.rval for x in mapping])
        distance = len(mset)/len(mapping)
    else:
        mset = set([x.lval for x in mapping])
        distance = len(mset)/len(mapping)
        
    if distance <= lower_bound:
        if mtype == "om":
            split_rank = 2
        else:
            split_rank = 1 #mapping is m-1
    elif distance >= upper_bound:
        if mtype == "om":
            split_rank = 1
        else:
            split_rank = 2 #mapping is m-1
    else:
        if mtype == "om":
            split_rank = 1
        else:
            split_rank = 2 #mapping is m-1
    
    return distance, split_rank

def group_by_lr(tok_logs, mappings, lr):
    val2logs = {}
    for mapping in mappings:
        valkey = mapping.lval if lr=="l" else mapping.rval
        
        if valkey not in val2logs:
            val2logs[valkey] = []
            
        val2logs[valkey].append(tok_logs[mapping.idx])
        
    return val2logs

def split_oo(tok_logs, oo):
    #not need to split
    #just retrun one oo
    psize = len(tok_logs)
    size = len(oo)
    
    partition_logs = [tok_logs[ooi.idx] for ooi in oo]
    partition = Partition(partition_logs, 3, size, psize, True)
    
    return partition

def split_om(tok_logs, om, lower_bound, upper_bound):
    #om is a list of mappings
    distance, split_rank = get_rank_position(om, "om", lower_bound, upper_bound)
    
    if split_rank==1:
        #split_pos = p1 no need to split psize = size
        size = len(om)
        psize = len(tok_logs)
        partition_logs = [tok_logs[mapping.idx] for mapping in om]
        partition = Partition(partition_logs, 3, size, psize, True)
        
        return [partition]
    else:
        #group by rval
        rval2logs = group_by_lr(tok_logs, om, "r")
        
        partitions = []
        psize = len(tok_logs)
        for rval, logs in rval2logs.items():
            size = len(logs)
            partition = Partition(logs, 3, size, psize, True)
            partitions.append(partition)
            
        return partitions
    
def split_mo(tok_logs, mo, lower_bound, upper_bound):
    #mo is a list of mappings
    distance, split_rank = get_rank_position(mo, "mo", lower_bound, upper_bound)
    
    if split_rank == 2:
        #split_ps=p2, need to split, group by lval
        lval2logs = group_by_lr(tok_logs, mo, "l")
        
        partitions = []
        psize = len(tok_logs)
        for lval, logs in lval2logs.items():
            size = len(logs)
            partition = Partition(logs, 3, size, psize, True)
            partitions.append(partition)
            
        return partitions
    else:
        #no need to split, psize = size
        size = len(mo)
        psize = len(tok_logs)
        partition_logs = [tok_logs[mapping.idx] for mapping in mo]
        partition = Partition(partition_logs, 3, size, psize, True)
        
        return [partition]
    
def split_mm(tok_logs, mms):
    #split by the small cardinality mms only have one partition
    lset = set(mm.lval for mm in mms)
    rset = set(mm.rval for mm in mms)
    split_pos = "l" if len(lset)<=len(rset) else "r"
    
    val2logs = group_by_lr(tok_logs, mms, split_pos)
    
    partitions = []
    psize = len(tok_logs)
    for val, logs in val2logs.items():
        size = len(logs)
        partition = Partition(logs, 3, size, psize, True)
        partitions.append(partition)

    return partitions

def partition_by_bijection(partition, bi_poses, lower_bound, upper_bound):
    tok_logs = partition.tok_logs
    
    lpos, rpos = bi_poses
    oos, oms, mos, mms = gen_mappings(partition.tok_logs, lpos, rpos)
    
    oo_partitions = [split_oo(tok_logs, oo) for oo in oos]
    
    #since om and mo return a list of lists
    om_partitions = [split_om(tok_logs, om, lower_bound, upper_bound) for om in oms]
    
    mo_partitions = [split_mo(tok_logs, mo, lower_bound, upper_bound) for mo in mos]
    
    mm_partitions = split_mm(tok_logs, mms)
    
    return oo_partitions, om_partitions, mo_partitions, mm_partitions

def split_stage_3(stage_2_partitions, lower_bound, upper_bound):
    stage_3_partitions = []
    
    for partition in stage_2_partitions:
        if len(partition.tok_logs) == 1:
            #no need to split
            stage_3_partition = Partition(partition.tok_logs, 3, partition.size, partition.psize, partition.valid)
            stage_3_partitions.append(stage_3_partition)
            continue
            
        bi_poses = gen_bijection_poses(partition.tok_logs)
        if len(bi_poses)!=2:
            #no need to split
            stage_3_partition = Partition(partition.tok_logs, 3, partition.size, partition.psize, partition.valid)
            stage_3_partitions.append(stage_3_partition)
            continue
        
        oo_partitions, om_partitions, mo_partitions, mm_partitions = partition_by_bijection(partition, bi_poses, lower_bound, upper_bound)
        
        for oo_partition in oo_partitions:
            stage_3_partitions.append(oo_partition)
            
        for om_partition in om_partitions:
            for om_partition_i in om_partition:
                stage_3_partitions.append(om_partition_i)
            
        for mo_partition in mo_partitions:
            for mo_partition_i in mo_partition:
                stage_3_partitions.append(mo_partition_i)
            
        for mm_partition in mm_partitions:
            stage_3_partitions.append(mm_partition)
            
    return stage_3_partitions

def extract_tpl(tok_logs):
    if len(tok_logs)==1:
        return tuple(tok_logs[0])
    else:
        #partition logs are of same size
        loglen = len(tok_logs[0])
        pos2col = [set() for i in range(loglen)]
        
        for tok_log in tok_logs:
            for i, token in enumerate(tok_log):
                pos2col[i].add(token)
                
        tpl=[]
        for col in pos2col:
            if len(col) == 1:
                tpl.append(list(col)[0])
            else:
                tpl.append(None)
                
        return tuple(tpl)

class IPLOMParser():
    def __init__(self, lower_bound=0.3, upper_bound=0.8):
        self.name = "IPLOM"
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        
    def get_parser_identifier(self):
        return {"name":self.name, "lower_bound":self.lower_bound, "upper_bound":self.upper_bound}
    
    def parse(self, logs):
        #here logs is str_logs
        stage_1_partitions = partition_by_token_cnt(logs)
        stage_2_partitions = split_stage_2(stage_1_partitions)
        stage_3_partitions = split_stage_3(stage_2_partitions, self.lower_bound, self.upper_bound)
        
        tpls=[]
        for partition in stage_3_partitions:
            tpl = extract_tpl(partition.tok_logs)
            if tpl not in tpls:
                tpls.append(tpl)
            
        return tpls
