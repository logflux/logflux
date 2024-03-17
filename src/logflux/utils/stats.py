import torch
import numpy as np

def gen_tpl_log_tokens(tpls, logs):
    #tpl are tok_tpl, logs are tok_logs
    #tpl and log are all words
    tokens = set()
    for tpl in tpls:
        for token in tpl:
            tokens.add(token)
            
    for log in logs:
        for token in log:
            tokens.add(token)
            
    return tokens

def gen_vocab(tokens):
    w2i = {}
    i2w = []
    
    w2i[None] = 0
    i2w.append(None)
    
    for word in tokens:
        if word in w2i:
            continue
            
        w2i[word] = len(i2w)
        i2w.append(word)
        
    return w2i, i2w

def get_minibatch(data, batch_size = 256):
    batch = []
    
    for example in data:
        batch.append(example)
        
        if len(batch) == batch_size:
            yield batch
            batch = []
            
    if len(batch) > 0:
        yield batch
        

def prepare_minibatch(mb, w2i, device=None):
    x = [[w2i.get(t,-1) for t in ex] for ex in mb]
    x = np.array(x)
    
    x = torch.from_numpy(x).to(device)
    
    return x


def calc_tpl_logs_match(tpl, logs, w2i, batch_size, use_gpu=True):
    #tpl are tok_tpl, logs are tok_logs
    #tpl and logs are all words, need w2i
    #tpl and logs have same length, no need to pad
    #tpl = [w2i.get(t,0) for t in tpl]
    #None shoudld be 0 in w2i
    device = torch.device('cuda' if torch.cuda.is_available() & use_gpu else 'cpu')
    
    tpl = [w2i[t] for t in tpl]
    tpl = np.array(tpl)
    tpl = torch.from_numpy(tpl).to(device)
    
    matches = []
    for mb in get_minibatch(logs):
        log_mb = prepare_minibatch(mb, w2i, device)
        matched = torch.all((((tpl != 0) * log_mb) == tpl), axis=1)
        
        log_matched = log_mb[matched]
        log_matched_cpu = log_matched.cpu()
        matches.append(log_matched_cpu.detach().numpy())
    
    concated_matches = np.concatenate(matches, axis=0)
    
    return concated_matches


def calc_log_tpls_match(log, tpls, w2i, batch_size, use_gpu=True):
    #log are tok_log, tpls are tok_tpls
    #tpl and logs are all words, need w2i
    #tpl and logs have same length, no need to pad
    #log = [w2i.get(t,0) for t in log]
    #None shoudld be 0 in w2i
    device = torch.device('cuda' if torch.cuda.is_available() & use_gpu else 'cpu')
    
    log = [w2i[t] for t in log]
    log = np.array(log)
    log = torch.from_numpy(log).to(device)
    
    matches = []
    for mb in get_minibatch(tpls):
        tpl_mb = prepare_minibatch(mb, w2i, device)
        matched = torch.all((((tpl_mb!=0)*log) == tpl_mb), axis=1)
        
        tpl_matched = tpl_mb[matched]
        tpl_matched_cpu = tpl_matched.cpu()
        matches.append(tpl_matched_cpu.detach().numpy())
        
    concated_matches = np.concatenate(matches, axis=0)
    
    return concated_matches

def groupby_loglen(logs):
    #logs is tok_logs
    len2logs = {}
    
    for log in logs:
        loglen = len(log)
        if loglen not in len2logs:
            len2logs[loglen] = []
            
        len2logs[loglen].append(log)
        
    return len2logs


def gen_tpl2logs(tpls, logs, batch_size, use_gpu=True):
    #tpls has to be a list of tuple
    tokens = gen_tpl_log_tokens(tpls, logs)
    w2i, i2w = gen_vocab(tokens)
    
    #tpls is tok tpls
    #logs is tok logs
    len2tpls = groupby_loglen(tpls)
    len2logs = groupby_loglen(logs)
    
    tpl2logs = {}
    for tpl in tpls:
        tpllen = len(tpl)
        
        #no match put empty []
        if tpllen not in len2logs:
            tpl2logs[tpl] = []
            continue
        
        logs = len2logs[tpllen]
        matched_logs_arr = calc_tpl_logs_match(tpl, logs, w2i, batch_size, use_gpu)
        
        tok_logs = [tuple([i2w[idx] for idx in log_arr]) for log_arr in matched_logs_arr]
        
        tpl2logs[tpl] = tok_logs
        
    return tpl2logs

def gen_log2tpls(logs, tpls, batch_size, use_gpu=True):
    #logs has to be a list of tuple
    tokens = gen_tpl_log_tokens(tpls, logs)
    w2i, i2w = gen_vocab(tokens)
    
    #tpls is tok tpls
    #logs is tok logs
    len2tpls = groupby_loglen(tpls)
    len2logs = groupby_loglen(logs)
    
    log2tpls = {}
    for log in logs:
        loglen = len(log)
        
        #miss match on length put empty
        if loglen not in len2tpls:
            log2tpls[log] = []
            continue
            
        tpls = len2tpls[loglen]
        matched_tpls_arr = calc_log_tpls_match(log, tpls, w2i, batch_size, use_gpu)
        
        tok_tpls = [tuple([i2w[idx] for idx in tpl_arr]) for tpl_arr in matched_tpls_arr]
        
        log2tpls[log] = tok_tpls
        
    return log2tpls

def gen_missed_logs(tpls, logs, batch_size, use_gpu=True):
    #tpls are tok tpls, logs are tok logs
    #they should be tuple, for dict key
    log2tpls = gen_log2tpls(logs, tpls, batch_size, use_gpu)
    
    missed_logs = []
    for log in logs:
        matched_tpls = log2tpls[log]
        if len(matched_tpls) == 0:
            missed_logs.append(log)
            
    return missed_logs

def gen_missed_tpls(tpls, logs, batch_size, use_gpu=True):
    #tpls are tok tpls, logs are tok logs
    #they should be tuple, for dict key
    tpl2logs = gen_tpl2logs(tpls, logs, batch_size, use_gpu)
    
    missed_tpls = []
    for tpl in tpls:
        matched_logs = tpl2logs[tpl]
        if len(matched_logs) == 0:
            missed_tpls.append(tpl)
            
    return missed_tpls