#this is a recursvie algorithms
#so it does not suit for large size for the recursive depth limitation

from collections import Counter



class Node:
    def __init__(self, word, depth):
        self.word = word
        self.depth = depth
        self.count = 0
        self.childs = {}
        self.part_of_tpl = False

    def __len__(self):
        return len(self.childs)

    def __getitem__(self, key):
        if key not in self.childs:
            raise IndexError("index out of range")
        return self.childs[key]

    def add(self, num=1):
        self.count += num
        
class DLogParser:
    def __init__(self):
        self.name = "DLog"
        self.root = Node(None, 0)
        
    def get_parser_identifier(self):
        return {"name":self.name}
        
    def add_line(self, primary_tpl):
        words = [w for w in primary_tpl if w is not None]
        current = self.root
        
        for word in words:
            if word not in current.childs:
                current.childs[word] = Node(word, current.depth+1)
                
            current.childs[word].add()
            current = current.childs[word]
    
    @classmethod
    def merge_subtree(cls, current_node, new_node):
        # replace a child node of the current_node into the new one
        if current_node.depth + 1 == new_node.depth:
            child = current_node.childs[new_node.word]
            for gchild_word in child.childs:
                # merge grandchild nodes
                if gchild_word in new_node.childs:
                    new_gchild = Node(gchild_word, new_node.depth + 1)
                    cls.merge_subtree(child, new_gchild)
                    cls.merge_subtree(new_node, new_gchild)
                else:
                    new_node.childs[gchild_word] = child.childs[gchild_word]
            current_node.childs[new_node.word] = new_node
        elif current_node.depth + 1 < new_node.depth:
            for child in current_node.childs.values():
                cls.merge_subtree(child, new_node)
        else:
            raise ValueError("subtree merge failure")
            
    @classmethod
    def aggregate_tree(cls, node):
        d_cnt = Counter()
        # merge counts of all child nodes
        for child in node.childs.values():
            d_cnt += cls.aggregate_tree(child)

        # merge nodes of same count value
        for (word, depth), cnt in d_cnt.items():
            if cnt == node.count and cnt > 1:
                new_node = Node(word, depth)
                new_node.add(cnt)
                node.part_of_tpl = True
                new_node.part_of_tpl = True
                cls.merge_subtree(node, new_node)

        # add counts of current node
        d_cnt += Counter({(node.word, node.depth): node.count})
        return d_cnt
        

    #def get_primary_tpl(self, words):
    #    return words
            
    
    def restore_tpl(self, primary_tpl):
        node = self.root
        tpl = []
        
        for w in primary_tpl:
            if w is None:
                tpl.append(w)
            else:
                node = node.childs[w]
                tpl.append(w if node.part_of_tpl else None)
                
        return tuple(tpl)
        
    def parse(self, logs):
        #make tree
        for mid, log in enumerate(logs):
            words = log.split()
            self.add_line(words)
        
        #aggregate tree
        self.aggregate_tree(self.root)
        
        #restore tpl
        ret = []
        for mid, log in enumerate(logs):
            words = log.split()
            tpl = self.restore_tpl(words)
            
            if tpl not in ret:
                ret.append(tpl)
            
        return ret