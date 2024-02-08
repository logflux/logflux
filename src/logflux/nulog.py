import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import SequentialSampler, RandomSampler
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.autograd import Variable
import copy

def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):
    """Pads sequences to the same length.

    This function transforms a list of
    `num_samples` sequences (lists of integers)
    into a 2D Numpy array of shape `(num_samples, num_timesteps)`.
    `num_timesteps` is either the `maxlen` argument if provided,
    or the length of the longest sequence otherwise.

    Sequences that are shorter than `num_timesteps`
    are padded with `value` at the beginning or the end
    if padding='post.

    Sequences longer than `num_timesteps` are truncated
    so that they fit the desired length.
    The position where padding or truncation happens is determined by
    the arguments `padding` and `truncating`, respectively.

    Pre-padding is the default.

    # Arguments
        sequences: List of lists, where each element is a sequence.
        maxlen: Int, maximum length of all sequences.
        dtype: Type of the output sequences.
            To pad sequences with variable length strings, you can use `object`.
        padding: String, 'pre' or 'post':
            pad either before or after each sequence.
        truncating: String, 'pre' or 'post':
            remove values from sequences larger than
            `maxlen`, either at the beginning or at the end of the sequences.
        value: Float or String, padding value.

    # Returns
        x: Numpy array with shape `(len(sequences), maxlen)`

    # Raises
        ValueError: In case of invalid values for `truncating` or `padding`,
            or in case of invalid shape for a `sequences` entry.
    """
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    num_samples = len(sequences)

    lengths = []
    sample_shape = ()
    flag = True

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.

    for x in sequences:
        try:
            lengths.append(len(x))
            if flag and len(x):
                sample_shape = np.asarray(x).shape[1:]
                flag = False
        except TypeError:
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))

    if maxlen is None:
        maxlen = np.max(lengths)

    is_dtype_str = np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.unicode_)
    if isinstance(value, str) and dtype != object and not is_dtype_str:
        raise ValueError("`dtype` {} is not compatible with `value`'s type: {}\n"
                         "You should set `dtype=object` for variable length strings."
                         .format(dtype, type(value)))

    x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" '
                             'not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s '
                             'is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x

class LogTokenizer:
    def __init__(self):
        self.word2index = {}
        self.index2word = {}
        self.n_words = 3  # Count SOS and EOS
        
        self.pad_idx = 0
        self.cls_idx = 1
        self.msk_idx = 2
        
    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1
            
    def tokenize(self, sent):
        filtered = sent.split()
        
        new_filtered = []
        for f in filtered:
            new_filtered.append(f)
        
        for w in range(len(new_filtered)):
            self.add_word(new_filtered[w])
            new_filtered[w] = self.word2index[new_filtered[w]]
        
        return new_filtered
    
    def get_pad_idx(self):
        return 0
    
    def get_cls_idx(self):
        return 1
    
    def get_msk_idx(self):
        return 2

class MaskedDataset(Dataset):
    def __init__(self, data, tokenizer, pad_len=32):
        self.data = data
        self.tokenizer = tokenizer
        self.pad_val = self.tokenizer.get_pad_idx()
        self.pad_len = pad_len
        
        #pad will do deep copy
        self.padded_data = pad_sequences(self.data, maxlen=self.pad_len,
                                         value=self.pad_val,
                                         dtype="long",
                                         truncating="post", 
                                         padding="post")
        
    def __getitem__(self, idx):
        src = self.padded_data[idx]
        offset = 1
        data_len = len(self.data[idx])-1 if len(self.data[idx])<self.pad_len else self.pad_len-1
        
        return src, offset, data_len, idx
    
    def __len__(self):
        #for dataloader sampler
        return self.padded_data.shape[0]
    
def do_mask_yield(batch, mask_id, mask_pct, seed=None):
    srcs, offsets, data_lens, indices = batch
    
    for i, tokens in enumerate(srcs):
        data_len = data_lens[i].item()
        offset = offsets[i].item()
        
        num_masks = round(mask_pct*data_len)
        if mask_pct<1.0:
            np.random.seed(seed)
            masked_idxs = np.random.choice(np.arange(offset, offset+data_len),
                                          size=num_masks if num_masks>0 else 1,
                                          replace=False)
        else:
            masked_idxs = np.arange(offset, offset+data_len)
            
        masked_idxs.sort()
        
        for j in masked_idxs:
            label = tokens[j].item()
            
            #here to list is a deep copy
            tokens_d = tokens.tolist()
            tokens_d[j] = mask_id
            
            #idx in batch
            idx_in_batch = i
            
            #yield (tokens_d.tolist(), label.item())
            yield (tokens_d, label, idx_in_batch)
            
def subsequent_mask_torch(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)
    #subsequent_mask = torch.triu(torch.ones(attn_shape))
    return subsequent_mask == 0

def make_std_mask_torch(tgt, pad):
    "Create a mask to hide padding and future words."
    tgt_mask = (tgt != pad).unsqueeze(-2)
    tgt_mask = tgt_mask & Variable(
            subsequent_mask_torch(tgt.size(-1)).type_as(tgt_mask.data))
    return tgt_mask

class TorchBatch:
    #hold a batch of data, all are torch tensor
    def __init__(self, src, tgt, pad):
        self.src = src
        self.src_mask = (src!=pad).unsqueeze(-2)
        
        self.tgt = tgt
        self.tgt_mask = make_std_mask_torch(tgt, pad)
        
        self.ntokens = (self.tgt != pad).data.sum().item()
        


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        #         return self.decode(self.encode(src, src_mask), src_mask,
        #                             tgt, tgt_mask)
        out = self.encode(src, src_mask)
        return out

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.d_model = d_model
        self.proj = nn.Linear(self.d_model, vocab)

    def forward(self, x):
        # print(torch.mean(x, axis=1).shape)
        out = self.proj(x[:, 0, :])
        # out = self.proj(torch.mean(x, axis=1))
        # print(out.shape)
        return out

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)



class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2



class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return self.norm(x + self.dropout(sublayer(x)))



class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)




class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)



class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)

class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator, criterion, opt=None, is_test=False):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
        self.is_test = is_test

    def __call__(self, x, y, norm):
        x = self.generator(x)
        y = y.reshape(-1)
        loss = self.criterion(x, y)
        if not self.is_test:
            loss.backward()
            if self.opt is not None:
                self.opt.step()
                #                 self.opt.optimizer.zero_grad()
                self.opt.zero_grad()

        return loss.item() * norm
    
    
def make_model(src_vocab_size, tgt_vocab_size, N=3, d_model=128, d_ff=128, h=8, dropout=0.1, max_len=20):
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout, max_len)
    
    c = copy.deepcopy
    model = EncoderDecoder(
            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
            Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
            nn.Sequential(Embeddings(d_model, src_vocab_size), c(position)),
            nn.Sequential(Embeddings(d_model, tgt_vocab_size), c(position)),
            Generator(d_model, tgt_vocab_size))
    
    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
            #nn.init.xavier_uniform_(p)
            
    return model

def derive_batch_result(x, y, ind):
    #for a batch, we get id_in_batch which represent a log
    #pos_argprob for each position we have the arg probablity of all words the last the larger
    #pos_val for each position we have its orig_val it has the same length with argprob
    uniq_ids = np.unique(ind).tolist()
    pos_argprob = x
    pos_words = y
    
    batch_result = []
    for id_in_batch, uniq_id in enumerate(uniq_ids):
        poses_in_batch = np.argwhere(ind==uniq_id).flatten().tolist()
        argprob_in_batch = [pos_argprob[idx] for idx in poses_in_batch]
        words = [pos_words[idx] for idx in poses_in_batch]
        batch_result.append((id_in_batch, poses_in_batch, words, argprob_in_batch))
        
    return batch_result

def construct_tpl(log_seq, log_pos_argprob, idx2word, topk):
    assert(len(log_seq) == len(log_pos_argprob))
    
    tpl = []
    for log_wordidx, pos_argprob in zip(log_seq, log_pos_argprob):
        if log_wordidx in pos_argprob[-topk:].tolist():
            tpl.append(idx2word[log_wordidx])
        else:
            tpl.append(None)

    return tuple(tpl)




class NuLogParser:
    def __init__(self, 
                 k,
                 mask_percentage,
                 batch_size=32,
                 pad_len=32, 
                 N=1, 
                 d_model=128,
                 d_ff=128,
                 h=8,
                 dropout=0.1,  
                 lr=0.001, 
                 betas=(0.9, 0.999), 
                 weight_decay=0.005, 
                 nr_epochs=50,
                 step_size=10,
                 print_progress=False
                ):
        
        self.name = "NuLog"
        self.k = k
        self.mask_percentage=mask_percentage
        
        
        self.batch_size = batch_size
        self.pad_len = pad_len
        self.N = N
        self.d_model = d_model
        self.d_ff = d_ff
        self.h = h
        
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.nr_epochs = nr_epochs
        
    def get_parser_identifier(self):
        return {"name":self.name, "k":self.k, "mask_percentage":self.mask_percentage}
    
    
    def parse(self, str_logs):
        
        self.tokenizer = tokenizer = LogTokenizer()
        self.data_tokenized = []
        for log in str_logs:
            tokenized = tokenizer.tokenize(log)
            self.data_tokenized.append([tokenizer.get_cls_idx()] + tokenized)
            
        data_tokenized = self.data_tokenized
            
        train_data = MaskedDataset(data_tokenized, tokenizer)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.batch_size)
        
        test_data = MaskedDataset(data_tokenized, tokenizer)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=self.batch_size)
        
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        
        self.src_vocab_size = src_vocab_size = tokenizer.n_words
        self.tgt_vocab_size = tgt_vocab_size = tokenizer.n_words
        
        model = make_model(src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size,
                          N=self.N, d_model=self.d_model, d_ff=self.d_ff, h=self.h, max_len=self.pad_len)
        self.model = model
        
        criterion = nn.CrossEntropyLoss()
        model_opt = torch.optim.Adam(model.parameters(), 
                                     lr=self.lr, 
                                     betas=self.betas, 
                                     weight_decay=self.weight_decay)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.train()
        
        mask_idx = tokenizer.get_msk_idx()
        pad_idx = tokenizer.get_pad_idx()
        mask_pct = self.mask_percentage
        
        for epoch in range(self.nr_epochs):
            epoch_loss = 0
            
            for i, batch in enumerate(train_dataloader):
                masked_batch = []
                for tokens, label, idx_in_batch in do_mask_yield(batch, mask_idx, mask_pct):
                    masked_batch.append([tokens, label, idx_in_batch])
                    
                b_input = np.array([x[0] for x in masked_batch])
                b_label = np.array([x[1] for x in masked_batch])
                
                b_input_torch = torch.from_numpy(b_input)
                b_label_torch = torch.from_numpy(b_label)
                
                b_batch_torch = TorchBatch(b_input_torch, b_label_torch, pad_idx)
                
                src_d = b_batch_torch.src.to(device)
                tgt_d = b_batch_torch.tgt.to(device)
                src_mask_d = b_batch_torch.src_mask.to(device)
                tgt_mask_d = b_batch_torch.tgt_mask.to(device)
                
                out = model(src_d, tgt_d, src_mask_d, tgt_mask_d)
                out_p = model.generator(out)
                
                loss = criterion(out_p, tgt_d.reshape(-1))
                loss.backward()
                
                model_opt.step()
                model_opt.zero_grad()
                
                epoch_loss+=loss.item()
                #print(epoch, i, loss.item(), loss.item()/b_batch_torch.ntokens)
                
            #print("epochs loss", epoch, epoch_loss)
            
        #test_results = self.run_test(test_dataloader)
        #tpls = generate_tpls(test_results, self.tokenizer.index2word, self.k)
        #return tpls
        #return None
        
        tpls = self.run_test_yield(self.test_dataloader,
                                  self.tokenizer.index2word,
                                  self.k)
        
        return tpls
            
            

    
    def run_test_yield(self, test_dataloader, index2word, topk):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = self.model.to(device)
        model.eval()

        mask_idx = self.tokenizer.get_msk_idx()
        mask_pct = 1.0
        pad_idx = self.tokenizer.get_pad_idx()

        tpls = []

        for i, batch in enumerate(test_dataloader):
            #print(i)

            masked_batch = []
            for tokens, label, idx_in_batch in do_mask_yield(batch, mask_idx, mask_pct):
                masked_batch.append([tokens, label, idx_in_batch])

            b_input = np.array([x[0] for x in masked_batch])
            b_label = np.array([x[1] for x in masked_batch])
            b_idxs = np.array([x[2] for x in masked_batch])

            b_input_torch = torch.from_numpy(b_input)
            b_label_torch = torch.from_numpy(b_label)

            b_batch_torch = TorchBatch(b_input_torch, b_label_torch, pad_idx)

            src_d = b_batch_torch.src.to(device)
            tgt_d = b_batch_torch.tgt.to(device)
            src_mask_d = b_batch_torch.src_mask.to(device)
            tgt_mask_d = b_batch_torch.tgt_mask.to(device)

            out = model(src_d, tgt_d, src_mask_d, tgt_mask_d)
            out_p = model.generator(out)

            x = out_p.cpu().detach().numpy().argsort(axis=1)
            y = b_label
            ind = b_idxs
            batch_result = derive_batch_result(x,y,ind)

            for batch_item in batch_result:
                log_seq = batch_item[2]
                log_pos_argprob = batch_item[3]
                tpl = construct_tpl(log_seq, log_pos_argprob, index2word, topk)

                if tpl not in tpls:
                    tpls.append(tpl)
                    #print(len(tpls))

        return tpls