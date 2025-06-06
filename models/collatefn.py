import numpy as np
import torch
class CollateFn:
    def __init__(self, tokenizer, seq_len=16, train=True):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.train = train
    def random_word(self, tokens):
        ''' a parallel implementation of random_word function'''
        
        # convert tokens to numpy array for random operations
        tokens_arr = np.array(tokens)
        
        # Randomly mask some tokens
        mask_prob = np.random.rand(len(tokens_arr)) < 0.15

        # generate a random strategy for each token
        # < 8: replace with <MASK>
        # == 8: replace with random token
        # > 8: keep original token
        strategy = np.random.randint(0, 10, size=len(tokens_arr))

        output = np.copy(tokens_arr)
        labels = np.zeros_like(tokens_arr)

        # get the mask token id
        mask_token_id = self.tokenizer.vocab['<MASK>']
        # get the random token ids
        random_tokens = np.random.randint(0, len(self.tokenizer.vocab), size=len(tokens_arr))
        # apply the masking strategy
        if np.any(mask_prob):
            # create boolean masks for each strategy
            mask_strategy = mask_prob & (strategy < 8)    # 80% MASK
            rand_strategy = mask_prob & (strategy == 8)   # 10% 随机词
            
            # apply the strategies
            output[mask_strategy] = mask_token_id
            output[rand_strategy] = random_tokens[rand_strategy]
            
            # set the labels
            labels[mask_prob] = tokens_arr[mask_prob]
        assert(len(output) == len(labels))
        return output.tolist(), labels.tolist()

    def __call__(self, batch):
        bert_inputs = []
        bert_labels = []
        for instr_pairs in batch:
            t1=instr_pairs['instr1']
            t2=instr_pairs['instr2']
            t1_tokens = self.tokenizer.encode(t1)
            t2_tokens = self.tokenizer.encode(t2)
            if self.train:
                t1_random, t1_label = self.random_word(t1_tokens)
                t2_random, t2_label = self.random_word(t2_tokens)
            else:
                t1_random = t1_tokens
                t2_random = t2_tokens
            t1_random = t1_random[:self.seq_len] + [self.tokenizer.vocab['<PAD>']] * (self.seq_len - len(t1_random))
            t2_random = t2_random[:self.seq_len] + [self.tokenizer.vocab['<PAD>']] * (self.seq_len - len(t2_random))
            t1_label = t1_label[:self.seq_len] + [0] * (self.seq_len - len(t1_label))
            t2_label = t2_label[:self.seq_len] + [0] * (self.seq_len - len(t2_label))
            # Adding CLS and SEP tokens
            t1 = [self.tokenizer.vocab['<CLS>']] + t1_random + [self.tokenizer.vocab['<SEP>']]
            t2 = t2_random + [self.tokenizer.vocab['<SEP>']]
            t1_label = [0] + t1_label + [0]
            t2_label = t2_label + [0]
            # Pad to fixed length
            bert_input = t1 + t2
            bert_label = (t1_label + t2_label)
            bert_inputs.append(bert_input)
            bert_labels.append(bert_label)
        return {
            "bert_input": torch.tensor(bert_inputs, dtype=torch.long),
            "bert_label": torch.tensor(bert_labels, dtype=torch.long)
        }
	