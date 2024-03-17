# Copyright 2024 Guineng Zheng and The University of Utah
# All rights reserved.
# This file is part of the logflux project, which is licensed under the
# simplified BSD license.

def gen_freq_words(log_words, freq_bar):
    #gen word equal or greater than freq_bar
    word_freq = dict()
    
    for words in log_words:
        for word in words:
            if word not in word_freq:
                word_freq[word] = 1
            else:
                word_freq[word] += 1
    
    freq_words = [word for word, freq in word_freq.items() if freq>=freq_bar]
    
    return freq_words

def gen_templates(tokenized_logs, freq_words):
    templates = []
    for tokenized_log in tokenized_logs:
        template = []
        for token in tokenized_log:
            if token in freq_words:
                template.append(token)
            else:
                template.append(None)
        tuple_tpl = tuple(template)
        
        if tuple_tpl not in templates:
            templates.append(tuple_tpl)
            
    return templates
    
def extract_pattern(tokenized_log, freq_words):
    skip = 0
    freq_word_pattern = []
    pattern = []

    for word in tokenized_log:
        if word in freq_words:
            if skip!=0:
                pattern.append(skip)
                skip=0

            freq_word_pattern.append(word)
            pattern.append(word)
        else:
            #if freq_word_pattern[-1] is not None:
            #    freq_word_pattern.append(None)
            skip += 1
            
    if skip!=0:
        pattern.append(skip)
            
    return tuple(freq_word_pattern), pattern

def gen_meta2patterns(logs, freq_words):
    #logs are a sequence of tokenized log tokens
    meta2pattern={}
    
    for log in logs:
        freq_word_pattern, pattern = extract_pattern(log, freq_words)
        
        if freq_word_pattern not in meta2pattern:
            meta2pattern[freq_word_pattern] = []
            
        meta2pattern[freq_word_pattern].append(pattern)
        
    return meta2pattern

def collapse_patterns(input_pattern_tuple):
    freq_word_pattern, patterns = input_pattern_tuple
    
    patterns = set([tuple(pattern) for pattern in patterns])
    
    # Turn original patters from [w1, w2, w3] => [set(), w1, set(), w2, set(), w3, set()]
    # Sets will be used to keep track of an skip words
    aggregate_pattern = [set()]
    for word in freq_word_pattern:
        aggregate_pattern.append(word)
        aggregate_pattern.append(set())

    # Iterate over patters keeping track of number of skips that occur
    for pattern in patterns:
        output_loc = 0

        prev_val = 0
        for word in pattern:
            if isinstance(word, int):
                aggregate_pattern[output_loc].add(word)
                output_loc += 1
                prev_val = 1
            else:
                if prev_val ==0:
                    aggregate_pattern[output_loc].add(0)
                    output_loc += 2
                else:
                    output_loc += 1

                prev_val=0
    

    #create final pattern where sets collapased into skip num in regex
    final_pattern = []
    for word in aggregate_pattern:
        if isinstance(word, set):
            if len(word)>=2:
                #print(min(word), max(word))
                final_pattern.append([min(word), max(word)])
            elif len(word) == 1 and 0 not in word:
                num_var = list(word)[0]
                final_pattern.append([num_var])
        else:
            final_pattern.append(word)
            
    return final_pattern
    
class LogClusterParser:
    def __init__(self, freq_bar, return_regex=False):
        self.freq_bar = freq_bar
        self.return_regex = return_regex
        
    def get_parser_identifier(self):
        return {"name":"logcluster", "freq_bar":self.freq_bar}
    
    def parse(self, logs):
        tokenized_logs = [log.split() for log in logs]
        freq_words = gen_freq_words(tokenized_logs, self.freq_bar)
        
        if self.return_regex:
            meta2patterns = gen_meta2patterns(tokenized_logs, freq_words)
            res = [collapse_patterns(input_pattern_tuple) for input_pattern_tuple in meta2patterns.items()]
        else:
            res = gen_templates(tokenized_logs, freq_words)
            
        return res
