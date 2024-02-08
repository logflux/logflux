import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity

class LenmaTemplate():
    #this algorithm use word length to represent word
    #it will tend to generate very coarse templates, so the template number is small
    def __init__(self, index=None, words=None):
        assert(index is not None)
        assert(words is not None)
        
        self.index = index
        self.words = words
        self.nwords = len(words)
        self.wordlens = [len(w) for w in words]
        self.counts = 1
        
    def __str__(self):
        str_tpl = ' '.join([self.words[idx] if self.words[idx] != '' else '*' for idx in range(self.nwords)])
        return '{index}({nwords})({counts}):{template}'.format(
            index=self.index,
            nwords=self.nwords,
            counts=self.counts,
            template=str_tpl)
    
    def _get_accuracy_score(self, new_words):
        # accuracy score
        # wildcard word matches any words
        assert(self.nwords == len(new_words))
        #fill_wildcard = [self.words[idx] if self.words[idx] != ''
        #                 else new_words[idx] for idx in range(self.nwords)]
        
        fill_wildcard = [self.words[idx] if self.words[idx] != '' 
                         else new_words[idx] for idx in range(len(self.words))]
        
        ac_score = accuracy_score(fill_wildcard, new_words)
        return ac_score
      
    def _get_wcr(self):
        #wildcard ratio
        #variable number / log length
        return self.words.count('') / len(self.words)
    
    
    def _get_similarity_score_cosine(self, new_words):
        # cosine similarity
        wordlens = np.asarray(self.wordlens).reshape(1, -1)
        new_wordlens = np.asarray([len(w) for w in new_words]).reshape(1, -1)
        cos_score = cosine_similarity(wordlens, new_wordlens)
        return cos_score
    
    def _count_same_word_positions(self, new_words):
        #this is for heauristic
        c = 0
        for idx in range(self.nwords):
            if self.words[idx] == new_words[idx]:
                c = c + 1
        return c
    
    
    def get_similarity_score(self, new_words):
        # heuristic judge: the first word (process name) must be equal
        #if self.words[0] != new_words[0]:
        #    return 0
        
        # check exact match
        ac_score = self._get_accuracy_score(new_words)
        if  ac_score == 1:
            return 1
        
        cos_score = self._get_similarity_score_cosine(new_words)
        
        # heuristic judge: same word position<3 return 0
        #if self._count_same_word_positions(new_words) < 3:
        #    return 0
        
        return cos_score
    
    def update(self, new_words):
        self.counts += 1
        self.wordlens = [len(w) for w in new_words] 
        self.words = [self.words[idx] if self.words[idx] == new_words[idx]
                       else '' for idx in range(self.nwords)]
        

class LenmaTemplateManager():
    def __init__(self, threshold=0.9991):
        #threshold should be very big(from 0.9991 -0.9999)?
        
        self.templates = []
        self.threshold = threshold
        
    def append_template(self, template):
        assert(template.index == len(self.templates))
        self.templates.append(template)
        
        return template
    
    def infer_template(self, words):
        #there is a representative template
        #for each log message, find its representative template (closest) or generate it as itself
        #if found, update the represenative template with merged template
        
        nwords = len(words)
        
        candidates = []
        for (index, template) in enumerate(self.templates):
            if nwords != template.nwords:
                continue
            
            score = template.get_similarity_score(words)
            if score < self.threshold:
                continue
                
            candidates.append((index, score))
        #print("candidates", candidates)
        
        candidates.sort(key=lambda c: c[1], reverse=True)
        
        if len(candidates)>0:
            #found, update reprentative and return
            index = candidates[0][0]
            self.templates[index].update(words)
            return self.templates[index]
        
        
        #not find, generate new template with itself
        new_index = len(self.templates)
        new_template = self.append_template(LenmaTemplate(new_index, words))
        
        return new_template
    
class LenmaParser():
    def __init__(self, threshold=0.9995):
        #threshold 0.9995 -> 0.9999
        #it is a very bad algorithm
        self.name = "Lenma"
        self.threshold = threshold
        
    def get_parser_identifier(self):
        return {"name":self.name, "threshold":self.threshold}
        
    def parse(self, logs):
        template_mgr = LenmaTemplateManager(threshold=self.threshold)
        
        for idx, str_log in enumerate(logs):
            tok_log = str_log.split()
            new_tpl = template_mgr.infer_template(tok_log)
            
            #print(idx, new_tpl)
            
        
        tpls = []
        for template in template_mgr.templates:
            tpl_words = template.words
            template = tuple([x if x!='' else None for x in tpl_words])
            tpls.append(template)
            
        return tpls