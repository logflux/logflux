class AELParser():
    #this algorithm is order senstive
    
    def __init__(self, merge_pct=0.5, min_event_cnt=3):
        self.name = "AEL"
        self.min_event_cnt = min_event_cnt
        self.merge_pct = merge_pct
        
    def get_parser_identifier(self):
        return {"name":self.name, "min_event_cnt":self.min_event_cnt, "merge_pct":self.merge_pct}
    
    
    def groupby_len_var(self, tok_logs):
        #here all logs has no var(originall it has zero for domain knowledge)
        lenvar2tokens = {}

        for tok_log in tok_logs:
            lenvar = (len(tok_log), 0)
            if lenvar not in lenvar2tokens:
                lenvar2tokens[lenvar] = []

            lenvar2tokens[lenvar].append(tok_log)

        return lenvar2tokens
    
    def has_diff(self, tokens1, tokens2, merge_pct):
        assert(len(tokens1) == len(tokens2))
        seqlen = len(tokens1)

        diff = 0
        for idx in range(seqlen):
            if tokens1[idx] != tokens2[idx]:
                diff += 1

        diff_ratio = float(diff/seqlen)

        if 0 < diff_ratio <= merge_pct:
            return True
        else:
            return False
        
    def merge_group(self, logs):
        #logs is a group of log tokens with same length
        #if the column is unique return it else return none

        grp_tpl = None
        for log in logs:
            if grp_tpl is None:
                grp_tpl = log
            else:
                new_tokens = []
                for log_token, tpl_token in zip(log, grp_tpl):
                    if log_token == tpl_token:
                        new_tokens.append(log_token)
                    else:
                        new_tokens.append(None)

                grp_tpl = new_tokens

        return grp_tpl
    
    def reconcile(self, logs, merge_pct):
        #logs is a group of tokens with same length
        visisted = [False for i in range(len(logs))]
        tobe_merged = []

        for idx_out, tok_out in enumerate(logs):
            if visisted[idx_out]:
                continue

            tobe_merged.append([tok_out])
            visisted[idx_out] = True

            for idx_inner, tok_inner in enumerate(logs):
                if visisted[idx_inner]:
                    continue

                if self.has_diff(tok_out, tok_inner, merge_pct):
                    tobe_merged[-1].append(tok_inner)
                    visisted[idx_inner] = True

        return tobe_merged
    
    def parse(self, logs):
        tok_logs = [log.split() for log in logs]
        lenvar2toklogs = self.groupby_len_var(tok_logs)
        
        tpls = []
        for toklogs in lenvar2toklogs.values():
            if len(toklogs) < self.min_event_cnt:
                for toklog in toklogs:
                    if toklog not in tpls:
                        tpls.append(toklog)
            else:
                tobe_merged = self.reconcile(toklogs, self.merge_pct)
                for group in tobe_merged:
                    tpl = self.merge_group(group)
                    
                    if tpl not in tpls: 
                        tpls.append(tpl)
                        
        return [tuple(tpl) for tpl in tpls]