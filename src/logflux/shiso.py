import numpy as np
from collections import defaultdict

class TemplateTable:
    """Temporal template table for log template generator."""

    def __init__(self):
        self._d_tpl = {}  # key = tid, val = template
        self._d_rtpl = {}  # key = key_template, val = tid
        self._d_ltid = {}  # key = tid, val = ltid
        self._d_cand = defaultdict(list)  # key = tid, val = List[ltid]
        self._last_modified = None  # used for LTGenJoint

    def __str__(self):
        ret = []
        for tid, tpl in self._d_tpl.items():
            ret.append(" ".join([str(tid)] + tpl))
        return "\n".join(ret)

    def __iter__(self):
        return self._generator()

    def _generator(self):
        for tid in self._d_tpl:
            yield self._d_tpl[tid]

    def __getitem__(self, key):
        assert isinstance(key, int)
        if key not in self._d_tpl:
            raise IndexError("index out of range")
        return self._d_tpl[key]

    def __len__(self):
        return len(self._d_tpl)

    def next_tid(self):
        cnt = 0
        while cnt in self._d_tpl:
            cnt += 1
        else:
            return cnt

    def tids(self):
        return self._d_tpl.keys()

    @staticmethod
    def _key_template(template):
        # l_word = [strutil.add_esc(w) for w in template]
        # return "@".join(l_word)
        return tuple(template)

    def exists(self, template):
        key = self._key_template(template)
        return key in self._d_rtpl

    def get_tid(self, template):
        key = self._key_template(template)
        return self._d_rtpl[key]

    def get_template(self, tid):
        return self._d_tpl[tid]

    def add(self, template):
        tid = self.next_tid()
        self._d_tpl[tid] = template
        self._d_rtpl[self._key_template(template)] = tid
        return tid

    def replace(self, tid, template):
        self._last_modified = self._d_tpl[tid]
        self._d_tpl[tid] = template
        self._d_rtpl[self._key_template(template)] = tid

    def get_updated(self):
        return self._last_modified

    def add_ltid(self, tid, ltid):
        self._d_ltid[tid] = ltid

    def get_ltid(self, tid):
        return self._d_ltid[tid]

    def load(self, obj):
        self._d_tpl, self._d_cand = obj
        for tid, tpl in self._d_tpl.items():
            self._d_rtpl[self._key_template(tpl)] = tid

    def dumpobj(self):
        return self._d_tpl, self._d_cand

def merged_template(m1, m2):
    """Return common area of log message (to be log template)"""
    ret = []
    for w1, w2 in zip(m1, m2):
        if w1 == w2:
            ret.append(w1)
        else:
            ret.append(None)
    return ret

class LTGen():
    state_added = 0
    state_changed = 1
    state_unchanged = 2

    def __init__(self, table):
        if table is None:
            self._table = TemplateTable()
        else:
            self._table = table

    def is_stateful(self):
        return True

    def get_tpl(self, tid):
        return self._table[tid]

    def add_tpl(self, ltw, tid=None):
        if tid is None:
            tid = self._table.add(ltw)
        return tid

    def update_tpl(self, ltw, tid):
        self._table.replace(tid, ltw)

    def merge_tpl(self, l_w, tid):
        old_tpl = self._table[tid]
        new_tpl = merged_template(old_tpl, l_w)
        if old_tpl == new_tpl:
            return self.state_unchanged
        else:
            self.update_tpl(new_tpl, tid)
            return self.state_changed

    def update_table(self, tpl):
        if tpl is None:
            return None, None
        elif self._table.exists(tpl):
            tid = self._table.get_tid(tpl)
            return tid, self.state_unchanged
        else:
            tid = self._table.add(tpl)
            return tid, self.state_added

def edit_distance(m1, m2):
    """Calculate Levenshtein edit distance of 2 tokenized messages.
    This function considers wildcards."""

    table = [[0] * (len(m2) + 1) for _ in range(len(m1) + 1)]

    for i in range(len(m1) + 1):
        table[i][0] = i

    for j in range(len(m2) + 1):
        table[0][j] = j

    for i in range(1, len(m1) + 1):
        for j in range(1, len(m2) + 1):
            if m1[i - 1] == m2[j - 1] or \
                    m1[i - 1] == None or \
                    m2[j - 1] == None:
                cost = 0
            else:
                cost = 1
            table[i][j] = min(table[i - 1][j] + 1, table[i][j - 1] + 1,
                              table[i - 1][j - 1] + cost)
    return table[-1][-1]
    
class LTGenNode:

    def __init__(self, tid=None):
        self.l_child = []
        self.tid = tid

    def __len__(self):
        return len(self.l_child)

    def __iter__(self):
        return self.l_child.__iter__()

    def join(self, node):
        self.l_child.append(node)
        
class SHISOParser():
    def __init__(self, threshold, max_child, cfunc=None):
        self.name = "SHISO"
        
        self._n_root = LTGenNode()
        self._threshold = threshold
        self._max_child = max_child
        
        if cfunc is None:
            self._cfunc = self.c_alphabet
        else:
            self._cfunc = cfunc
            
        self.table = TemplateTable()
        self.ltgen = LTGen(self.table)
        
    def get_parser_identifier(self):
        return {"name":self.name, "threshold":self._threshold, "max_child":self._max_child}
        
    def process_line(self, pline):
        l_w = pline.split()
        n_parent = self._n_root
        
        while True:
            for n_child in n_parent:
                nc_tpl = self.ltgen._table[n_child.tid]
                sr = self._seq_ratio(nc_tpl, l_w, self._cfunc)
                
                if sr>=self._threshold:
                    state = self.ltgen.merge_tpl(l_w, n_child.tid)
                    return n_child.tid, state
            else:
                if len(n_parent)<self._max_child:
                    new_tid = self.ltgen.add_tpl(l_w)
                    n_child = LTGenNode(new_tid)
                    n_parent.join(n_child)
                    return new_tid, self.ltgen.state_added
                else:
                    a_sim = np.array([self._sim(self.ltgen.get_tpl(n_child.tid), l_w)
                                  for n_child in n_parent])
                    n_parent = n_parent.l_child[a_sim.argmin()]
                        
    @staticmethod
    def c_original(w):
        a_cnt = np.zeros(4)  # upper_alphabet, lower_alphabet, digit, symbol
        for c in w:
            if c.isupper():
                a_cnt[0] += 1
            elif c.islower():
                a_cnt[1] += 1
            elif c.isdigit():
                a_cnt[2] += 1
            else:
                a_cnt[3] += 1
        return a_cnt / np.linalg.norm(a_cnt)
    
    @staticmethod
    def c_alphabet(w):
        a_cnt = np.zeros(26 + 26 + 2)  # A-Z, a-z, digit, symbol
        for c in w:
            if c.isupper():
                abc_index = ord(c) - 65
                a_cnt[abc_index] += 1.0
            elif c.islower():
                abc_index = ord(c) - 97
                a_cnt[26 + abc_index] += 1
            elif c.isdigit():
                a_cnt[-2] += 1.0
            else:
                a_cnt[-1] += 1.0
        return a_cnt / np.linalg.norm(a_cnt)

    @staticmethod
    def _seq_ratio(m1, m2, cfunc):
        if len(m1) == len(m2):
            length = len(m1)
            if length == 0:
                return 1.0

            sum_dist = 0.0
            for w1, w2 in zip(m1, m2):
                if None in (w1, w2):
                    pass
                else:
                    sum_dist += sum(np.power(cfunc(w1) - cfunc(w2), 2))
            return 1.0 - (sum_dist / (2.0 * length))
        else:
            return 0.0

    @staticmethod
    def _sim(m1, m2):
        """Sim is different from SeqRatio, because Sim allows
        word sequence of different length. Original SHISO uses
        Levenshtein edit distance, and amulog follows that."""
        return edit_distance(m1, m2)

    '''
    @staticmethod
    def _equal(m1, m2):
        if not len(m1) == len(m2):
            return False
        for w1, w2 in zip(m1, m2):
            if w1 == w2 or w1 == lt_common.REPLACER or w2 == lt_common.REPLACER:
                pass
            else:
                return False
        else:
            return True
    '''
        
    def parse(self, logs):
        for str_log in logs:
            self.process_line(str_log)
            
        return [tuple(x) for x in self.table._d_tpl.values()]
    
