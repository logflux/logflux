class SLCTParser:
    reserve_words = []
    
    def __init__(self, wp_supp=50, tpl_supp=20):
        #wp_supp: word position support
        #tpl_supp: template support
        self.wp_supp = wp_supp
        self.tpl_supp = tpl_supp
        
    def get_parser_identifier(self):
        return {"name":"SLCT",
                "wp_supp":self.wp_supp, 
                "tpl_supp":self.tpl_supp}
        
        
    def parse(self, logs):
        #word position frequency
        wp_freq = {}
        for log in logs:
            for word_pos, word in enumerate(log.split()):
                wp = (word_pos, word)
                
                if wp not in wp_freq:
                    wp_freq[wp] = 1
                else:
                    wp_freq[wp] += 1
                    
        valid_wp = [wp for wp, freq in wp_freq.items() if freq>self.wp_supp]
        
        #gen tpl
        tpl_cnt = {}
        for log in logs:
            words = []
            
            for word_pos, word in enumerate(log.split()):
                if (word_pos, word) in valid_wp:
                    words.append(word)
                else:
                    words.append(None)
                    
            candidate_tpl = tuple(words)
            
            if candidate_tpl not in tpl_cnt:
                tpl_cnt[candidate_tpl] = 1
            else:
                tpl_cnt[candidate_tpl] += 1
                
        valid_tpls = [tpl for tpl, cnt in tpl_cnt.items() if cnt>self.tpl_supp]
        
        return valid_tpls