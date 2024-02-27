# Copyright 2024 Guineng Zheng and The University of Utah
# All rights reserved.
# This file is part of the logflux project, which is licensed under the
# simplified BSD license.

from copy import deepcopy
import random

def gen_unique(logs):
    unique_logs = []
    for log in logs:
        if log not in unique_logs:
            unique_logs.append(log)
    return unique_logs

def is_variaible(tok):
    return tok==None

def generate_random_template(tok_log):
    tok_log = deepcopy(tok_log)
    modifiable_indexes = [idx for idx, tok in enumerate(tok_log) if not is_variaible(tok)]
    if len(modifiable_indexes)>0:
        index = random.choice(modifiable_indexes)
        token = tok_log[index]
        tok_log[index] = None
        
    return tok_log

def gen_chromosome(tok_logs):
    #a chromsome is a group of tpls
    tpls = []
    for tok_log in tok_logs:
        tpl = generate_random_template(tok_log)
        #if tpl not in tpls:
        #    tpls.append(tpl)
        tpls.append(tpl)
    return tpls

def gen_pc_pairs(tpls):
    #generate parent-child pare
    pairs = []
    
    if len(tpls) <= 1:
        return pairs
    
    for i, ptpl in enumerate(tpls):
        for j, ctpl in enumerate(tpls):
            if i==j:
                continue
            elif len(ptpl)!=len(ctpl):
                continue
            else:
                is_pc = True
                for pword, cword in zip(ptpl, ctpl):
                    if pword != cword and pword != None:
                        is_pc = False
                        break
                if is_pc:
                    pairs.append((ptpl, ctpl))
                    
    return pairs

def remove_sub_tpls(tpls):
    pc_pairs = gen_pc_pairs(tpls)
    tobe_removed = []
    
    for pair in pc_pairs:
        tobe_removed.append(pair[1])
        
    new_tpls = []
    for tpl in tpls:
        if tpl not in tobe_removed:
            new_tpls.append(tpl)
            
    return new_tpls

def deduplicate(tpls):
    unique_tpls = []
    
    for tpl in tpls:
        if tpl not in unique_tpls:
            unique_tpls.append(tpl)
            
    return unique_tpls

def preprocess(chromosome):
    #deduplicate, remove sub
    deduped_chromosome = deduplicate(chromosome)
    subremoved_chromsome = remove_sub_tpls(deduped_chromosome)
    
    return subremoved_chromsome


def initialize(tok_logs, N=20):
    #generate N chromosomes 
    #a chromosome is a solution with tpls
    chromosomes = []
    
    for i in range(N):
        chromosome = gen_chromosome(tok_logs)
        chromosome = preprocess(chromosome)
        
        chromosomes.append(chromosome)
        
    return chromosomes

def match(log_toks, tpl_toks):
    if len(log_toks) != len(tpl_toks):
        return False
    else:
        for log_tok, tpl_tok in zip(log_toks, tpl_toks):
            if log_tok != tpl_tok and tpl_tok != None:
                return False
        return True

def gen_matched_logs(tpl, tok_logs):
    matched_logs = []
    for tok_log in tok_logs:
        if match(tok_log, tpl):
            matched_logs.append(tok_log)
    return matched_logs


def calc_specificity(tpl):
    constant_num = len([x for x in tpl if x!=None])
    return constant_num/len(tpl)

def calc_frequency(tpl, tok_logs):
    matched_num = 0
    for tok_log in tok_logs:
        if match(tok_log, tpl):
            matched_num += 1
    return matched_num

def score(tok_logs, chromosome):
    #chromosome is tpl2num
    specificities = [calc_specificity(tpl) for tpl in chromosome]
    avg_specificity = sum(specificities)/len(specificities)
    
    frequencies = [calc_frequency(tpl, tok_logs) for tpl in chromosome]
    avg_frequency = sum(frequencies)/len(frequencies)
    
    return (avg_specificity+avg_frequency)/2

def binary_tournament(ch1, ch2, tok_logs):
    if score(tok_logs, ch1) > score(tok_logs, ch2):
        #print("winner 1")
        return ch1
    else:
        #print("winner 2")
        return ch2

def select_chromosomes(chromosomes, tok_logs, N):
    #sample two solutions N times and keep the better one
    #there might be duplicated and that is what we want
    new_chromosomes = []
    for i in range(N):
        idx1, idx2 = random.sample([i for i in range(len(chromosomes))], 2)
        #print("indexes", idx1, idx2)
        
        ch1 = chromosomes[idx1]
        ch2 = chromosomes[idx2]
        #ch1, ch2 = random.sample(chromosomes,2)
        
        winner = binary_tournament(ch1, ch2, tok_logs)
        new_chromosomes.append(winner)
        
    return new_chromosomes

def construct_pairs(chromosome_num):
    #make pairs to be crossovered (1, n-1)
    if chromosome_num%2==0:
        pairs = []
        for i in range(chromosome_num//2):
            pairs.append([i, chromosome_num-1-i])
    else:
        pairs = []
        for i in range(chromosome_num//2):
            pairs.append([i, chromosome_num-1-i])
        pairs.append([chromosome_num//2])
            
    return pairs

def crossover(ch1, ch2):
    ch1_len = len(ch1)
    ch2_len = len(ch2)

    ch1_mid = ch1_len//2
    ch2_mid = ch2_len//2
    
    ch1_head, ch1_tail = ch1[:ch1_mid], ch1[ch1_mid:]
    ch2_head, ch2_tail = ch2[:ch2_mid], ch2[ch2_mid:]
    
    new_ch1 = ch1_head+ch2_tail
    new_ch2 = ch2_head+ch1_tail
    
    return new_ch1, new_ch2

def crossover_chromosomes(chromosomes):
    chromosome_num = len(chromosomes)
    chromosome_pairs = construct_pairs(chromosome_num)
    
    new_chromosomes = []
    for ch_pair in chromosome_pairs:
        if len(ch_pair) == 1:
            #only one chrosome no need to cross over
            ch_idx = ch_pair[0]
            ch = chromosomes[ch_idx]
            new_chromosomes.append(ch)
        elif len(ch_pair) == 2:
            #two chrosomes crossover it
            ch1_idx, ch2_idx = ch_pair
            ch1, ch2 = chromosomes[ch1_idx], chromosomes[ch2_idx]
            
            new_ch1, new_ch2 = crossover(ch1, ch2)
            
            new_chromosomes.append(new_ch1)
            new_chromosomes.append(new_ch2)
        else:
            raise Exception("pair should have 1 or 2 elements")
            
    return new_chromosomes

def change_constant(tok_tpl):
    constant_idxs = [idx for idx, word in enumerate(tok_tpl) if word!=None]
    
    if len(constant_idxs)>0:
        random_idx = random.choice(constant_idxs)
        
        tok_tpl_c = deepcopy(tok_tpl)
        tok_tpl_c[random_idx] = None
        
        return tok_tpl_c
    else:
        return tok_tpl

def change_variable(tok_tpl, matched_logs):
    variable_idxs = [idx for idx, tok in enumerate(tok_tpl) if tok==None]
    
    if len(variable_idxs)>0:
        random_msg = random.choice(matched_logs)
        random_idx = random.choice(variable_idxs)

        #print(random_msg, variable_idxs, random_idx, random_msg[random_idx])

        tok_tpl_c = deepcopy(tok_tpl)
        tok_tpl_c[random_idx] = random_msg[random_idx]

        return tok_tpl_c
    else:
        return tok_tpl
    

def mutation(chromosome, tok_logs):
    #chromosome is a set of templates
    new_chromosome = []
    for tpl in chromosome:
        if random.random() <= 0.5:
            matched_logs = gen_matched_logs(tpl, tok_logs)
            new_tpl = change_variable(tpl, matched_logs)
            new_chromosome.append(new_tpl)
        else:
            new_tpl = change_constant(tpl)
            new_chromosome.append(new_tpl)
            
    return new_chromosome

def mutation_chromosomes(chromosomes, tok_logs):
    return [mutation(chromosome, tok_logs) for chromosome in chromosomes]

def calc_missed_logs(tok_tpls, tok_logs):
    missed_logs = []
    
    for tok_log in tok_logs:
        matched = False
        for tok_tpl in tok_tpls:
            if match(tok_log, tok_tpl):
                matched = True
                break
        if not matched:
            missed_logs.append(tok_log)
            
    return missed_logs

def add_to_100_cov(tok_tpls, tok_logs):
    tok_missed_logs = calc_missed_logs(tok_tpls, tok_logs)
    #print(tok_missed_logs)
    
    tok_tpls_c = deepcopy(tok_tpls) 
    for tok_log in tok_missed_logs:
        tpl = generate_random_template(tok_log)
        if tpl not in tok_tpls_c:
            tok_tpls_c.append(tpl)
            
    return tok_tpls_c

def postprocess(chromosome, tok_logs):
    #add to 100 cov, deduplicate, remove sub
    tpls_100_cov = add_to_100_cov(chromosome, tok_logs)
    deduped_chromosome = deduplicate(tpls_100_cov)
    subremoved_chromsome = remove_sub_tpls(deduped_chromosome)
    
    return subremoved_chromsome

def one_round(tok_logs, chromosomes, N=20):
    #select N times the winner will be corssed and mutationed
    winner_chromosomes = select_chromosomes(chromosomes, tok_logs, N)
    crossed_chromosomes = crossover_chromosomes(winner_chromosomes)
    mutationed_chromosomes = mutation_chromosomes(crossed_chromosomes, tok_logs)
    
    return mutationed_chromosomes

def get_best(chromosomes, tok_logs):
    scores = [score(chromosome, tok_logs) for chromosome in chromosomes]
    best_idx = scores.index(max(scores))
    return chromosomes[best_idx]

class MoLFIParser:
    def __init__(self, chromsome_num=5, iter_round=10, pprogress=True):
        self.name = "molfi"
        self.chromsome_num = chromsome_num
        self.iter_round = iter_round
        self.pprogress = pprogress
        
    def get_parser_identifier(self):
        return {"name":self.name, "chromsome_num":self.chromsome_num, "iter_round":self.iter_round}
    
    def parse(self, logs):
        unique_logs = gen_unique(logs)
        tok_logs = [x.split() for x in unique_logs]
        
        chromosomes = initialize(tok_logs, self.chromsome_num)
        for i in range(self.iter_round):
            
            if self.pprogress:
                print("round", i)
            
            chromosomes = one_round(tok_logs, chromosomes, self.chromsome_num)
            chromosomes = [postprocess(chromosome, tok_logs) for chromosome in chromosomes]
            
        best_chromosome = get_best(chromosomes, tok_logs)
        final_tpls = [tuple(x) for x in best_chromosome]
        
        return final_tpls
