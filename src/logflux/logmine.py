# Copyright 2024 Guineng Zheng and The University of Utah
# All rights reserved.
# This file is part of the logflux project, which is licensed under the
# simplified BSD license.

class MessageScorer():
    def __init__(self, k=1.0):
        self.k = k
        
    def distance(self, tokens1, tokens2):
        if not (isinstance(tokens1, list) and isinstance(tokens1, list)):
            raise TypeError('log must be a list')
            
        max_len = max(len(tokens1), len(tokens2))
        min_len = min(len(tokens1), len(tokens2))
        
        dis = 1.0
        for i in range(min_len):
            dis -= (self.k if tokens1[i]==tokens2[i] else 0 * 1.0) / max_len
        
        return dis
    

def process_message_level(tok_logs, message_scorer, max_dist):
    clusters = []
    
    for idx, tokens in enumerate(tok_logs):
        found = False
        
        for i in range(len(clusters)):
            [representative, log_idxs] = clusters[i]
            score = message_scorer.distance(representative, tokens)
            
            if score<=max_dist:
                clusters[i][1].append(idx)
                found = True
                
        if not found:
            representative = tokens
            log_idxs = [idx]
            clusters.append([representative, log_idxs])
    
    msg_clusters = [x[0] for x in clusters]
    
    return msg_clusters

def zeros(shape):
    retval = []
    for x in range(shape[0]):
        retval.append([])
        for y in range(shape[1]):
            retval[-1].append(0)
    return retval


def match_score(alpha, beta, match_award=10, mismatch_penalty=1, gap_penalty=0):
    if alpha == beta:
        return match_award
    elif alpha is None or beta is None:
        return gap_penalty
    else:
        return mismatch_penalty

def water(seq1, seq2, match_award=10, mismatch_penalty=1, gap_penalty=0):
    seq1 = [str(x) for x in seq1]
    seq2 = [str(x) for x in seq2]
    
    m, n = len(seq1), len(seq2)  # length of two sequences
    
    # Generate DP table and traceback path pointer matrix
    score = zeros((m+1, n+1))      # the DP table
    pointer = zeros((m+1, n+1))    # to store the traceback path
    
    max_score = 0        # initial maximum score in DP table
    # Calculate DP table and mark pointers
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            score_diagonal = score[i-1][j-1] + match_score(seq1[i-1], seq2[j-1], match_award, mismatch_penalty, gap_penalty)
            score_up = score[i][j-1] + gap_penalty
            score_left = score[i-1][j] + gap_penalty
            score[i][j] = max(0,score_left, score_up, score_diagonal)
            if score[i][j] == 0:
                pointer[i][j] = 0 # 0 means end of the path
            if score[i][j] == score_left:
                pointer[i][j] = 1 # 1 means trace up
            if score[i][j] == score_up:
                pointer[i][j] = 2 # 2 means trace left
            if score[i][j] == score_diagonal:
                pointer[i][j] = 3 # 3 means trace diagonal
            if score[i][j] >= max_score:
                max_i = i
                max_j = j
                max_score = score[i][j];
    
    align1 = []
    align2 = []    # initial sequences
    
    i,j = max_i,max_j    # indices of path starting point
    
    #traceback, follow pointers
    while pointer[i][j] != 0:
        if pointer[i][j] == 3:
            align1.append(seq1[i-1])
            align2.append(seq2[j-1])
            i -= 1
            j -= 1
        elif pointer[i][j] == 2:
            align1.append(None)
            align2.append(seq2[j-1])
            j -= 1
        elif pointer[i][j] == 1:
            align1.append(seq1[i-1])
            align2.append(None)
            i -= 1

    #return finalize(align1, align2)
    align1.reverse()
    align2.reverse()
    return align1, align2

class PatternScorer():
    def __init__(self, k1=1.0, k2=1.0):
        self.k1 = k1
        self.k2 = k2
        
    def distance(self, tokens1, tokens2):
        if not (isinstance(tokens1, list) and isinstance(tokens1, list)):
            raise TypeError('log must be a list')
            
        max_len = max(len(tokens1), len(tokens2))
        min_len = min(len(tokens1), len(tokens2))
        
        dis = 1.0
        for i in range(min_len):
            if tokens1[i] == tokens2[i]:
                if tokens1[i] == None:
                    dis -= self.k2 * 1.0 / max_len
                else:
                    dis -= self.k1 * 1.0 / max_len
                
        return dis
    
class PatternGenerator():
    def create_pattern(self, a, b):
        if len(a) == 0 and len(b) == 0:
            return []
        (a, b) = water(a, b)
        new = []
        for i in range(len(a)):
            if a[i] == b[i]:
                new.append(a[i])
            else:
                new.append(None)
        return new



def process_pattern_level(tok_patterns, pattern_scorer, pattern_generator, max_dist):
    clusters = []
    
    for pattern_idx, tokens in enumerate(tok_patterns):
        found = False
        
        for i in range(len(clusters)):
        
            [representative, count, pattern, matched] = clusters[i]
            #print(representative, count, pattern)
            
            score = pattern_scorer.distance(representative, tokens)
            #print(score)
            
            if score <= max_dist:
                found = True
                clusters[i][1] += 1
                merged_pattern = pattern_generator.create_pattern(pattern, tokens)
                clusters[i][2] = merged_pattern
                clusters[i][3].append(pattern_idx)
                
                #here do not break, every cluster will be scaned against all representative
                #break
                
        if not found:
            #representative, cnt, pattern, self idx
            #if one is matched by previous pattern, it will not be qualified to be represntative
            #pattern_idx is the first match, if it has been matched by previous cluster
            #it will not show up
            clusters.append([tokens, 1, tokens, [pattern_idx]])
            
    return clusters

class LogmineParser():
    def __init__(self, max_dist=0.2, level=3):
        #max_dist large, coarse and less templates
        #max_dist small, fine and more templates
        #larger level num, coarse and less templates, it will return the top level
        
        self.name = "Logmine"
        self.max_dist = max_dist
        self.level = level

        self.message_scorer = MessageScorer(k=1.0)
        self.pattern_scorer = PatternScorer(k1=1.0, k2=1.0)
        self.pattern_generator = PatternGenerator()
        self.alpha = 1.05    #attenuation coeffcient
        
        
    def get_parser_identifier(self):
        return {"name":self.name, "max_dist":self.max_dist, "level":self.level}
        
    def parse(self, logs):
        tok_logs = [x.split() for x in logs]
        message_clusters = process_message_level(tok_logs, self.message_scorer, self.max_dist)
        
        #all_patterns are layer by layer
        #all_patterns = []
        templates = message_clusters
        #all_patterns.append(templates)
        
        max_dist = self.max_dist
        for level_i in range(self.level):
            #print(level_i)
            max_dist *= self.alpha
            #print(level_i, max_dist, self.max_dist)
            level_patterns = process_pattern_level(templates, 
                                                   self.pattern_scorer, 
                                                   self.pattern_generator,
                                                   max_dist)
            templates = [x[2] for x in level_patterns]
            
            #all_patterns.append(templates)
            
        return [tuple(template) for template in templates]
