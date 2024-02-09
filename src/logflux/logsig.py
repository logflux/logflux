#this is not very quick and fittable for larage datasize

import random
import math
import operator

class LogSigParser:
    def __init__(self, grp_num=30, seed=0):
        self.name = "LogSig"
        self.grp_num = grp_num

        self.seed = seed
        
    def get_parser_identifier(self):
        return {"name":self.name, "grp_num":self.grp_num}
        
    def load_logs(self, logs):
        for log in logs:
            tok_log = log.strip().split()
            self.tok_logs.append(tuple(tok_log))
    
    def gen_termpair_logs(self):
        print('Generating term pairs...')
        
        for tok_log in self.tok_logs:
            termpairs = []
            for j in range(len(tok_log)):
                for k in range(j+1, len(tok_log), 1):
                    termpair = (tok_log[j], tok_log[k])
                    termpairs.append(termpair)
            
            self.termpair_logs.append(termpairs)
            
    def randomize_termpair(self):
        # grp_termpair_cnt, used to account the occurrence of each termpair of each group
        # initialize the item value to zero
        for i in range(self.grp_num):
            self.grp_termpair_cnt.append(dict())
            self.grp_sizes.append(0)

        # divide logs into initial groupNum groups randomly, the group number of each log is stored in the groupIndex
        random.seed(self.seed)
        
        log_num = len(self.tok_logs)
        for i in range(log_num):
            ran = random.randint(0, self.grp_num-1)  # group number from 0 to k-1
            self.log2grp[i] = ran
            self.grp_sizes[ran] += 1  # count the number of loglines per group

        # count the frequency of each termpairs per group
        for i, termpair_log in enumerate(self.termpair_logs):
            for key in termpair_log:
                cur_grp_idx = self.log2grp[i]
                if key not in self.grp_termpair_cnt[cur_grp_idx]:
                    self.grp_termpair_cnt[cur_grp_idx][key] = 1
                else:
                    self.grp_termpair_cnt[cur_grp_idx][key] += 1

    def logmsg_partition(self):
        """ Use local search, for each log, find the group that it should be moved to.
            in this process, termpairs occurange should also make some changes and logNumber 
            of corresponding should be changed
        """
        print('Log message partitioning...')
        changed = True
        while changed:
            changed = False
            i = 0
            for termpair_log in self.termpair_logs:
                cur_grp = self.log2grp[i]
                alter_grp = potenFunc(cur_grp, self.grp_termpair_cnt, self.grp_sizes, i, termpair_log, self.grp_num)
                if cur_grp != alter_grp:
                    changed = True
                    self.log2grp[i] = alter_grp
                    
                    # update the dictionary of each group
                    for key in termpair_log:
                        
                        # minus 1 from the current group count on this key
                        self.grp_termpair_cnt[cur_grp][key] -= 1
                        if self.grp_termpair_cnt[cur_grp][key] == 0:
                            del self.grp_termpair_cnt[cur_grp][key]
                        
                        # add 1 to the alter group
                        if key not in self.grp_termpair_cnt[alter_grp]:
                            self.grp_termpair_cnt[alter_grp][key] = 1
                        else:
                            self.grp_termpair_cnt[alter_grp][key] += 1
                    
                    self.grp_sizes[cur_grp] -= 1
                    self.grp_sizes[alter_grp] += 1
                i += 1

    def construct_signature(self):
        """ 
        Calculate the occurancy of each word of each group, and for each group, save the words that
        happen more than half all log number to be candidateTerms(list of dict, words:frequency),
        """
        print('Log message signature construction...')
        # create the folder to save the resulted templates
        #if not os.path.exists(self.para.savePath):
        #    os.makedirs(self.para.savePath)

        wordFreqPerGroup = []
        candidateTerm = []
        candidateSeq = []
        
        self.signature = []

        # save the all the log indexs of each group: logidxs_per_grp
        for t in range(self.grp_num):
            wordFreqPerGroup.append(dict())
            candidateSeq.append(dict())
            
            self.logidxs_per_grp.append(list())

        # count the occurence of each word of each log per group
        # and save into the wordFreqPerGroup, which is a list of dictionary,
        # where each dictionary represents a group, key is the word, value is the occurence
        
        for logidx, tok_log in enumerate(self.tok_logs):
            grp_idx = self.log2grp[logidx]
            self.logidxs_per_grp[grp_idx].append(logidx)
            
            for token in tok_log:
                if token not in wordFreqPerGroup[grp_idx]:
                    wordFreqPerGroup[grp_idx][token] = 1
                else:
                    wordFreqPerGroup[grp_idx][token] += 1

        # calculate the halfLogNum and select those words whose occurence is larger than halfLogNum
        # as constant part and save into candidateTerm
        for i in range(self.grp_num):
            halfLogNum = math.ceil(self.grp_sizes[i] / 2.0)
            candidate = dict((k, v) for k, v in wordFreqPerGroup[i].items() if v >= halfLogNum)
            candidateTerm.append(candidate)

        # scan each logline's each word that also is a part of candidateTerm, put these words together
        # as a new candidate sequence, thus, each raw log will have a corresponding candidate sequence
        # and count the occurence of these candidate sequence of each group and select the most frequent
        # candidate sequence as the signature, i.e. the templates

        for log_idx, tok_log in enumerate(self.tok_logs):
            cur_grp = self.log2grp[log_idx]
            newCandiSeq = []

            for token in tok_log:
                if token in candidateTerm[cur_grp]:
                    newCandiSeq.append(token)
                else:
                    newCandiSeq.append(None)

            keySeq = tuple(newCandiSeq)
            if keySeq not in candidateSeq[cur_grp]:
                candidateSeq[cur_grp][keySeq] = 1
            else:
                candidateSeq[cur_grp][keySeq] += 1


        for i in range(self.grp_num):
            if len(candidateSeq[i]) == 0:
                continue
                
            sig = max(candidateSeq[i].items(), key=operator.itemgetter(1))[0]
            self.signature.append(sig)

            
    def parse(self, logs):
        self.tok_logs = []
        self.load_logs(logs)
        
        self.termpair_logs = []
        self.grp_sizes = []
        self.log2grp = dict()  # each line corresponding to which group
        self.grp_termpair_cnt = [] # each group is a dict of termpair to its cnt
        self.logidxs_per_grp = []
        
        
        self.gen_termpair_logs()
        self.randomize_termpair()
        
        self.logmsg_partition()
        self.construct_signature()
        
        return self.signature

        
def potenFunc(curGroupIndex, grp_termpair_cnt, grp_sizes, lineNum, termpairLT, k):
    maxDeltaD = 0
    maxJ = curGroupIndex
    for i in range(k):
        returnedDeltaD = getDeltaD(grp_sizes, grp_termpair_cnt, curGroupIndex, i, lineNum, termpairLT)
        if returnedDeltaD > maxDeltaD:
            maxDeltaD = returnedDeltaD
            maxJ = i
    return maxJ


# part of the potential function
def getDeltaD(grp_sizes, grp_termpair_cnt, groupI, groupJ, lineNum, termpairLT):
    deltaD = 0
    Ci = grp_sizes[groupI]
    Cj = grp_sizes[groupJ]
    for r in termpairLT:
        if r in grp_termpair_cnt[groupJ]:
            deltaD += (pow(((grp_termpair_cnt[groupJ][r] + 1) / (Cj + 1.0)), 2) \
                       - pow((grp_termpair_cnt[groupI][r] / (Ci + 0.0)), 2))
        else:
            deltaD += (pow((1 / (Cj + 1.0)), 2) - pow((grp_termpair_cnt[groupI][r] / (Ci + 0.0)), 2))
    deltaD = deltaD * 3
    return deltaD

#generate word paris
def gen_wps(words):
    wps = []
    for i, wi in enumerate(words):
        for j, wj in enumerate(words):
            if i!=j:
                wps.append((wi, wj))
                
    return wps
