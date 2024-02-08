class lcs_obj:
    """ Class object to store a log group with the same template
    """
    def __init__(self, log_tpl='', log_idxs=[]):
        self.log_tpl = log_tpl
        self.log_idxs = log_idxs
        
class node:
    """ A node in prefix tree data structure
    """
    def __init__(self, token='', tpl_no=0):
        self.log_cluster = None
        self.token = token
        self.tpl_no = tpl_no
        self.child_dict = dict()

def compare_template(tpl1, tpl2):
    if len(tpl1) != len(tpl2):
        return False
    
    for w1, w2 in zip(tpl1, tpl2):
        if w1!=w2:
            return False
    
    return True

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
            

def loop_match(clusters, seq):
    for cluster in clusters:
        if float(len(cluster.log_tpl)) < 0.5*len(seq):
            continue
            
        #check the template is a subsequence of seq 
        #(we use set checking as a proxy here for speedup since
        #incorrect-ordering bad cases rarely occur in logs)
        
        token_set = set(seq)
        if all(token in token_set or token is None for token in cluster.log_tpl):
            return cluster
        
    return None

def prefix_tree_match(parent, seq, idx, tau):
    seq_len = len(seq)
    
    for i in range(idx, seq_len):
        #seq_i is the token id?
        seq_i = seq[i]
        
        if seq_i in parent.child_dict:
            child_node = parent.child_dict[seq_i]
            
            if child_node.log_cluster is not None:
                const_lm = [w for w in child_node.log_cluster.log_tpl if w!=None]
                
                if len(const_lm) >= seq_len * tau:
                    return child_node.log_cluster
                
            else:
                return prefix_tree_match(child_node, seq, i+1, tau)
            
    return None


def lcs_match(clusters, seq, tau):
    set_seq = set(seq)
    seq_len = len(seq)
    
    max_len = -1
    max_cluster = None
    max_lcs = []
    
    for cluster in clusters:
        set_tpl = set(cluster.log_tpl)
        
        if len(set_seq & set_tpl) < 0.5*seq_len:
            continue
            
        lcs_value = calc_lcs(seq, cluster.log_tpl)
        
        if len(lcs_value) > max_len or (len(lcs_value) == max_len and len(cluster.log_tpl)<len(max_cluster.log_tpl)):
            max_len = len(lcs_value)
            max_lcs = lcs_value
            max_cluster = cluster
            
    if max_len >= len(seq) * tau:
        return max_cluster
    else:
        return None
            
def add_seq_to_prefix_tree(root, cluster):
    parent = root
    seq = [w for w in cluster.log_tpl if w != None]
    
    for token in seq:
        if token in parent.child_dict:
            parent.child_dict[token].tpl_no += 1
        else:
            parent.child_dict[token] = node(token, 1)
            
        parent = parent.child_dict[token]
        
    if parent.log_cluster is None:
        parent.log_cluster = cluster
        
def remove_seq_from_prefix_tree(root, cluster):
    parent = root
    seq = [w for w in cluster.log_tpl if w != None]
    
    for token in seq:
        if token in parent.child_dict:
            matched_node = parent.child_dict[token]
            
            if matched_node.tpl_no == 1:
                del parent.child_dict[token]
                break
            else:
                matched_node.tpl_no -= 1
                parent = matched_node
                
def get_tpl(lcs, seq):
    ret_val = []
    
    if not lcs:
        return ret_val
    
    lcs = lcs[::-1]
    i = 0
    for token in seq:
        i += 1
        if token == lcs[-1]:
            ret_val.append(token)
            lcs.pop()
        else:
            ret_val.append(None)
            
        if not lcs:
            break
    
    if i<len(seq):
        ret_val.append(None)
        
    return ret_val


class SpellParser:
    def __init__(self, tau=0.5):
        self.name = "Spell"
        self.tau = tau
        
    def get_parser_identifier(self):
        return {"name":self.name, "tau":self.tau}
    
    def parse(self, logs):
        root_node = node()
        clusters = []

        count = 0
        for idx, log in enumerate(logs):
            log_idx = idx
            log_msg = log.split()
            const_log_msg = [w for w in log_msg if w is not None]

            matched_cluster = prefix_tree_match(root_node, const_log_msg, 0, self.tau)

            if matched_cluster is None:
                matched_cluster = loop_match(clusters, log_msg)

                if matched_cluster is None:
                    matched_cluster = lcs_match(clusters, log_msg, self.tau)

                    if matched_cluster is None:
                        #create new cluster
                        new_cluster = lcs_obj(log_tpl=log_msg, log_idxs=[log_idx])
                        clusters.append(new_cluster)
                        add_seq_to_prefix_tree(root_node, new_cluster)
                    else:
                        #add to existing cluster
                        lcs = calc_lcs(log_msg, matched_cluster.log_tpl)
                        new_tpl = get_tpl(lcs, matched_cluster.log_tpl)

                        if not compare_template(new_tpl, matched_cluster.log_tpl):
                            remove_seq_from_prefix_tree(root_node, matched_cluster)
                            matched_cluster.log_tpl = new_tpl
                            add_seq_to_prefix_tree(root_node, matched_cluster)

            else:
                matched_cluster.log_idxs.append(log_idx)

        tpls = [tuple(cluster.log_tpl) for cluster in clusters]

        return tpls