from __future__ import annotations
import math
import numpy as np
import sys
from sklearn.cluster import KMeans
import queue
from collections import Counter
import numpy as np



def entr(x):
    if np.isnan(x):
        return x
    elif x>0:
        return -x*np.log(x)
    elif x==0:
        return 0
    else:
        return -np.inf
    
def rel_entr(x, y):
    if np.isnan(x) or np.isnan(y):
        return np.nan
    elif x>0 and y>0:
        return x*np.log(x/y)
    elif x==0 and y>=0:
        return 0
    else:
        return np.inf
    
    
vfunc_entr = np.vectorize(entr)
vfunc_real_entr = np.vectorize(rel_entr)

def entropy(pk: np.typing.ArrayLike,
            qk: np.typing.ArrayLike | None = None,
            base: float | None = None,
            axis: int = 0
            ) -> np.number | np.ndarray:
        
    if base is not None and base <= 0:
        raise ValueError("`base` must be a positive number or `None`.")

    pk = np.asarray(pk)
    pk = 1.0*pk / np.sum(pk, axis=axis, keepdims=True)
    if qk is None:
        #vec = entr(pk)
        vec = vfunc_entr(pk)
    else:
        qk = np.asarray(qk)
        pk, qk = np.broadcast_arrays(pk, qk)
        qk = 1.0*qk / np.sum(qk, axis=axis, keepdims=True)
        #vec = rel_entr(pk, qk)
        vec = vfunc_real_entr(pk, qk)
    S = np.sum(vec, axis=axis)
    if base is not None:
        S /= np.log(base)
    return S / np.log(base)


def calc_edit_dp(tokens1, tokens2):
    #calc dynamic path
    m = len(tokens1)
    n = len(tokens2)
    
    #dp[i][j] is the distance between tokens1[:i] and tokens2[:j]
    dp = [[0]*(n+1) for _ in range(m+1)]
    
    #if one seqs is empty the edit distance is the lenthg of the other seq
    #the first row and first column
    for i in range(1, m+1):
        dp[i][0] = i
        
    for j in range(1, n+1):
        dp[0][j] = j
        
    for i in range(1, m+1):
        for j in range(1, n+1):
            if tokens1[i-1] == tokens2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j-1], dp[i-1][j], dp[i][j-1]) + 1
                
    return dp

def backpath(tokens1, tokens2, dp):
    i = len(dp)-1
    j = len(dp[0]) - 1
    res = []
    
    while i>0 or j>0:
        a = dp[i-1][j-1] if i>0 and j>0 else float("inf")
        b = dp[i-1][j] if i>0 else float("inf")
        c = dp[i][j-1] if j>0 else float("inf")
        min_val = min([a, b ,c])
        
        if dp[i][j] == a and a == min_val:
            i -= 1
            j -= 1
            #no operation
            
        elif a == min([a,b,c]):
            i -= 1
            j -= 1
            res.append((i, i+1, tokens1[i], tokens2[j], "sub"))
        elif b == min([a,b,c]):
            i -= 1
            res.append((i, i+1, tokens1[i], "", "del"))
        else:
            j -= 1
            res.append((i+1, i+1, "", tokens2[j], "ins"))

    return res

def calc_weighted_edit_distance(operations, wf=1.0):
    #wf is the weight factor v
    weds = [1/(1+math.exp(x[1]-wf)) for x in operations]
    return sum(weds)


def calc_distance(tokens1, tokens2, wf=1.0):
    dp = calc_edit_dp(tokens1, tokens2)
    ops = backpath(tokens1, tokens2, dp)
    wed = calc_weighted_edit_distance(ops, wf)
    
    return wed

#calcute disatances between two logs
def calc_pair_dists(tok_logs, wf=1.0):
    #in this algo the distance is not symmetry
    log_num = len(tok_logs)
    
    dis_matrix = np.zeros((log_num, log_num))
    for i in range(log_num):
        for j in range(i, log_num):
            log_i = tok_logs[i]
            log_j = tok_logs[j]
            dist = calc_distance(log_i, log_j, wf)
            dis_matrix[i][j] = dis_matrix[j][i] = dist
            
    return dis_matrix


#get the split threshold cluster them to two get maxInner
def calc_cc_threshold(pair_dists):
    flatten_pair_dists = np.array(pair_dists).flatten()
    uniq_pair_dists = np.unique(flatten_pair_dists)
    uniq_pair_dists = uniq_pair_dists.reshape(-1,1)

    cluster = KMeans(n_clusters=2, random_state=0).fit(uniq_pair_dists)
    
    c0 = uniq_pair_dists[cluster.labels_==0].flatten()
    c1 = uniq_pair_dists[cluster.labels_==1].flatten()
    c0_max = np.max(c0)
    c0_min = np.min(c0)
    c1_max = np.max(c1)
    c1_min = np.min(c1)
    
    threhold = sorted([c0_max, c0_min, c1_max, c1_min])[1]
    
    return threhold

class Graph:
    def __init__(self, V):
        self.V = V
        self.adj = [[] for i in range(V)]
 
    def DFSUtil(self, temp, v, visited):
        # Mark the current vertex as visited
        visited[v] = True
 
        # Store the vertex to list
        temp.append(v)
 
        # Repeat for all vertices adjacent
        # to this vertex v
        for i in self.adj[v]:
            if visited[i] == False:
                # Update the list
                temp = self.DFSUtil(temp, i, visited)
        return temp
 
    # method to add an undirected edge
    def addEdge(self, v, w):
        self.adj[v].append(w)
        self.adj[w].append(v)
 
    # Method to retrieve connected components
    # in an undirected graph
    def connectedComponents(self):
        visited = []
        ccs = []
        
        for i in range(self.V):
            visited.append(False)
            
        for v in range(self.V):
            if visited[v] == False:
                temp = []
                ccs.append(self.DFSUtil(temp, v, visited))
        
        return ccs
    
def gen_connected_components(pair_dists, tok_logs, cc_threshold):
    connection_matrix = pair_dists<cc_threshold
    lognum = len(tok_logs)
    
    sys.setrecursionlimit(lognum*lognum+1)
    
    graph = Graph(lognum)
    for i in range(lognum):
        for j in range(i+1, lognum):
            if connection_matrix[i][j] == True:
                graph.addEdge(i,j)
                
    ccs_groups = graph.connectedComponents()
    ccs = []
    for ccs_group in ccs_groups:
        ccs_tok_logs = []
        for idx in ccs_group:
            ccs_tok_logs.append(tok_logs[idx])
        ccs.append(ccs_tok_logs)
    
    return ccs

def calc_lcs(seq1, seq2):
    lengths = [[0 for j in range(len(seq2)+1)] for i in range(len(seq1)+1)]
    
    # row 0 and column 0 are initialized to 0 already
    for i in range(len(seq1)):
        for j in range(len(seq2)):
            if seq1[i] == seq2[j]:
                lengths[i+1][j+1] = lengths[i][j] + 1
            else:
                lengths[i+1][j+1] = max(lengths[i+1][j], lengths[i][j+1])
                
    # read the substring out from the matrix
    result = []
    len_seq1, len_seq2 = len(seq1), len(seq2)
    
    while len_seq1!=0 and len_seq2 != 0:
        if lengths[len_seq1][len_seq2] == lengths[len_seq1-1][len_seq2]:
            len_seq1 -= 1
        elif lengths[len_seq1][len_seq2] == lengths[len_seq1][len_seq2-1]:
            len_seq2 -= 1
        else:
            assert seq1[len_seq1-1] == seq2[len_seq2-1]
            result.insert(0,seq1[len_seq1-1])
            len_seq1 -= 1
            len_seq2 -= 1
            
    return result

def gen_common(tok_logs):
    common = tok_logs[0]
    for i in range(1, len(tok_logs)):
        common = calc_lcs(common, tok_logs[i])
        
    return common


def extract(tok_log, common_words):
    res = []
    start_idx=0
    for common in common_words:
        end_idx = tok_log[start_idx:].index(common)

        #print(log[start_idx:start_idx+end_idx])
        res.append(tok_log[start_idx:start_idx+end_idx])

        start_idx = start_idx+end_idx+1

    res.append(tok_log[start_idx:])
    
    return res


def gen_privates(logs, common):
    #logs is a list of list
    common_pos_num = len(common) + 1
    privates = [[] for x in range(common_pos_num)]
    
    for log in logs:
        extaction = extract(log, common)
        for i, ext_i in enumerate(extaction):
            privates[i].append(tuple(ext_i))
            
    return privates


def gen_split_pos(group, split_threshold):
    #group is a list of logs
    #if split_threshold is large more change to split
    common = gen_common(group)
    privates = gen_privates(group, common)
    
    #uniq_privates = [set(private) for private in privates]
    freqs = [Counter(private).values() for private in privates]
    #entropies = [entropy(list(x), base=2) for x in freqs]
    
    stats = []
    for i, freq in enumerate(freqs):
        pos_distnum = len(freq)
        pos_entropy = entropy(list(freq), base=2)
        pos = i

        stats.append([pos_distnum, pos_entropy, pos])
        
    sorted_stats = sorted(stats, key=lambda x:(x[0],x[1],x[2]))
    
    split_idx = None
    for stat in sorted_stats:
        if stat[0]>1 and stat[1]<=split_threshold:
            split_idx = stat[2]
            break

    return split_idx

def gen_posvar_splits(privates, split_pos):
    grp_idxs = [[]] * len(set(privates[split_pos]))
    private2idxs = dict([((split_pos, x), []) for x in set(privates[split_pos])])
    
    for idx, private in enumerate(privates[split_pos]):
        private2idxs[(split_pos, private)].append(idx)

    return private2idxs

def process_group(logs, common_part, privates, split_threshold):
    #a group of logs share the same common part
    pos = gen_split_pos(logs, split_threshold)
    if pos is None:
        #no need to split
        return {(None, None):logs}
    
    #posvar_idxs = gen_posvar_splits(privates, pos)
    #print(posvar_idxs)
    posvar_idxs = gen_posvar_splits(privates, pos)
    posvar_logs = {}
    for posvar, idxs in posvar_idxs.items():
        if posvar not in posvar_logs:
            posvar_logs[posvar] = []
        
        for idx in idxs:
            posvar_logs[posvar].append(logs[idx])
    
    return posvar_logs

def split(logs, split_threshold):
    common_part = gen_common(logs)
    privates = gen_privates(logs, common_part)
    
    posvar_logs = process_group(logs, common_part, privates, split_threshold)
    return posvar_logs

def iter_split(pair_dists, tok_logs, cc_threshold, split_threshold):
    #cc_threshold is calculated from 4 values but it might be not good
    #split threshold is specified for a position 
    #the bigger split_threshold, the more tpls
    ccs = gen_connected_components(pair_dists, tok_logs, cc_threshold)
    
    undecided = queue.Queue()
    decided = []
    
    for ccs_i in ccs:
        undecided.put(ccs_i)
        
    while not undecided.empty():
        cur_grp=undecided.get()
        split_pos = gen_split_pos(cur_grp, split_threshold)
        
        if split_pos is None:
            decided.append(cur_grp)
        else:
            splited = split(cur_grp, split_threshold)
            for splited_i in splited.values():
                undecided.put(splited_i)
       
    return decided

def extract(tok_log, common_words):
    res = []
    start_idx=0
    for common in common_words:
        end_idx = tok_log[start_idx:].index(common)

        #print(log[start_idx:start_idx+end_idx])
        res.append(tok_log[start_idx:start_idx+end_idx])

        start_idx = start_idx+end_idx+1

    res.append(tok_log[start_idx:])
    
    return res

def gen_tpl(private, common):
    assert(len(private) == (len(common)+1))
    
    words = []
    for i in range(len(private)-1):
        for p in private[i]:
            words.append(None)
        words.append(common[i])
     
    for p in private[-1]:
        words.append(None)
        
    return words

def gen_tpls(decided):
    tpls = []

    for decided_i in decided:
        common = gen_common(decided_i)

        for decided_ii in decided_i:
            extracted = extract(decided_ii, common)
            tpl = tuple(gen_tpl(extracted, common))

            if tpl not in tpls:
                tpls.append(tpl)
                
    return tpls



class LKEParser:
    def __init__(self, split_threshold, cc_threshold_ratio=1.0):
        self.name = "LKE"
        
        #split_threshold increase, more tpls
        self.split_threshold = split_threshold
        self.cc_threshold_ratio = cc_threshold_ratio
        
    def get_parser_identifier(self):
        return {"name":self.name, "split_threshold":self.split_threshold}
    
    def gen_unique_toklogs(self, logs):
        #order senstive do not use set
        idx = 0
        rtn_logs = []

        for log in logs:
            tok_log = log.split()
            if tok_log not in rtn_logs:
                rtn_logs.append(tok_log)
                idx += 1

        return rtn_logs
    
    def parse(self, logs):
        tok_logs = self.gen_unique_toklogs(logs)
        pair_dists = calc_pair_dists(tok_logs)
        
        #this cc_threshold algorithm is very bad
        cc_threshold = calc_cc_threshold(pair_dists)
        self.cc_threshold = cc_threshold*self.cc_threshold_ratio
        
        decided = iter_split(pair_dists, tok_logs, self.cc_threshold, self.split_threshold)
        tpls = gen_tpls(decided)
        
        return tpls