# This file adapted from the amulog project, which is available from
# https://github.com/amulog/amulog

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
        
def merged_template(m1, m2):
    """Return common area of log message (to be log template)"""
    ret = []
    for w1, w2 in zip(m1, m2):
        if w1 == w2:
            ret.append(w1)
        else:
            ret.append(None)
    return ret
    
class Node:

    def __init__(self, word, depth):
        self.word = word
        self.depth = depth
        self.childs = {}
        

    def __len__(self):
        return len(self.childs)
    

class FTTreeParser():
    def __init__(self, max_child=6, cut_depth=3, message_type_func=None):
        self.name = "FTTree"
        self._max_child = max_child
        self._cut_depth = cut_depth
        self._d_words = defaultdict(int)
        self._tree = {}

        if message_type_func is None:
            self._type_func = self.message_type_none
        else:
            self._type_func = message_type_func
            
        self.table = TemplateTable()
        self.ltgen = LTGen(self.table)
        
    def get_parser_identifier(self):
        return {"name":self.name, "max_child":self._max_child, "cut_depth":self._cut_depth}
            
    @staticmethod
    def message_type_none(words):
        return None, words

    @staticmethod
    def message_type_top(words):
        return words[0], words[1:]

    @staticmethod
    def message_type_length(words):
        return str(len(words)), words
    
    def _add_line(self, pline):
        tokens = pline.split()
        
        list_words = [(w, self._d_words[w]) for w in tokens]
        message_type, list_words = self._type_func(list_words)
        list_words.sort(key=lambda x: (x[1], x[0]), reverse=True)

        if message_type not in self._tree:
            self._tree[message_type] = Node(message_type, 0)
        current = self._tree[message_type]
        
        
        s_tpl = set()
        for w, _ in list_words:
            if current.childs is None:
                break
            elif w not in current.childs:
                current.childs[w] = Node(w, current.depth + 1)
                # pruning
                if len(current.childs) >= self._max_child and \
                        current.depth >= self._cut_depth:
                    current.childs = None
                    break
            s_tpl.add(w)
            current = current.childs[w]
            
        tpl = [w if w in s_tpl else None for w in tokens]
        
        return tpl
    
    def process_line(self, pline):
        #update word dict
        for word in pline:
            self._d_words[word] += 1
            
        #update tree
        tpl = self._add_line(pline)
        self.ltgen.update_table(tpl)
        
    def parse(self, logs):
        for str_log in logs:
            self.process_line(str_log)
            
        return [tuple(x) for x in self.table._d_tpl.values()]
