from collections import defaultdict

class LFAParser:
    reserve_words = []
    
    def __init__(self):
        self.wordseqs = []
        self.wordpos_count = defaultdict(int)
        
    def get_parser_identifier(self):
        return {"name":"LFA"}
    
    def load_data(self, logs):
        self.logs = logs
    
    def firstpass(self):    
        for idx, line in enumerate(self.logs):
            wordseq = line.split()
            self.wordseqs.append(wordseq)

            for pos, word in enumerate(wordseq):
                self.wordpos_count[(pos,word)] += 1
                
        #print('First pass done.')
        
    def secondpass(self):
        self.tpls = []
        
        for wordseq in self.wordseqs:
            
            countsl = [self.wordpos_count[(pos,word)] for pos,word in enumerate(wordseq) if word != None]
            
            if len(countsl) > 1:
                # find max gap
                countsl_sorted = sorted(countsl)
                gaps = [(countsl_sorted[idx + 1] - countsl_sorted[idx], idx) for idx in range(len(countsl_sorted) - 1)]
                split_value = countsl_sorted[max(gaps, key=lambda x: x[0])[1]]
                if max(countsl) != min(countsl):
                    countsl = [self.wordpos_count[(pos, word)] for pos, word in enumerate(wordseq)]
                    wordseq = [wordseq[pos] if count > split_value else None for pos, count in enumerate(countsl)]
            tpl = tuple(wordseq)
            
            if tpl not in self.tpls:
                self.tpls.append(tpl)
            
        #print('Second pass done.')
            
    def parse(self, logs):
        self.load_data(logs)
        self.firstpass()
        self.secondpass()
        
        return self.tpls