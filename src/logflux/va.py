# Copyright 2024 Guineng Zheng and The University of Utah
# All rights reserved.
# This file is part of the logflux project, which is licensed under the
# simplified BSD license.

def get_word_freq(seqs):
    word2freq = {}
    for seq in seqs:
        for word in seq:
            if word not in word2freq:
                word2freq[word] = 1
            else:
                word2freq[word] += 1
    return word2freq

def va_gen_tpls_static(seqs, bar):
    #large word number will be constant
    word2freq = get_word_freq(seqs)
    
    tpls = []
    for seq in seqs:
        tpl = []
        for word in seq:
            if word2freq[word]>bar:
                tpl.append(word)
            else:
                tpl.append(None)
                
        tpl=tuple(tpl)
        if tpl not in tpls:
            tpls.append(tpl)
            
    tuple_tpls = [tuple(tpl) for tpl in tpls]
    return tpls

def va_gen_tpls_relative_line(seqs, bar):
    #word num > bar*line num
    word2freq = get_word_freq(seqs)
    lognum = len(seqs)
    
    tpls = []
    for seq in seqs:
        tpl = []
        for word in seq:
            freq = word2freq[word]
            ratio = freq/lognum
            if ratio > bar:
                tpl.append(word)
            else:
                tpl.append(None)
        
        tpl = tuple(tpl)
        if tpl not in tpls:
            tpls.append(tpl)
            
    return tpls


def va_gen_tpls_relative_variable(seqs, bar):
    #sorted with word number for one seq
    #get percentile over bar as constant
    word2freq = get_word_freq(seqs)
    
    tpls = []
    for seq in seqs:
        wordfreqs_in_seq = []
        for word in seq:
            word_freq = word2freq[word]
            wordfreqs_in_seq.append((word,word_freq))
            
        seqlen = len(seq)
        sorted_wordfreqs = sorted(wordfreqs_in_seq, key=lambda x:x[1])
        position = int(seqlen*bar)
        
        constants = [x[0] for x in sorted_wordfreqs[position:]]
        
        #print(sorted_wordfreqs, constants)
        #break
        
        tpl = []
        for word in seq:
            if word in constants:
                tpl.append(word)
            else:
                tpl.append(None)
        
        tpl = tuple(tpl)
        if tpl not in tpls:
            tpls.append(tpl)
        
    return tpls

class VAParser():
    def __init__(self, method, threshold):
        self.name = "VA"
        self.method = method
        self.th = threshold
    
    def get_parser_identifier(self):
        return {"name":self.name, "method":self.method, "threshold":self.th}
        
    def parse(self,logs):
        tok_logs = [log.split() for log in logs]
        
        if self.method == "static":
            tpls = va_gen_tpls_static(tok_logs, self.th)
        elif self.method == "relative-line":
            tpls = va_gen_tpls_relative_line(tok_logs, self.th)
        elif self.method == "relative-variable":
            tpls = va_gen_tpls_relative_variable(tok_logs, self.th)
        else:
            raise NotImplementedError
            
        return tpls
