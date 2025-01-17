import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

# -------------

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

#  create a mapping of characters to integers

stoi = {s:i for i, s in enumerate(chars)}
itos = {i:s for s,i in stoi.items()}
encode = lambda s: [stoi[ch] for ch in s]
decode = lambda l: ''.join(itos[i] for i in l)



# Splitting the data into training and validation data
data = torch.tensor(encode(text), dtype = torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate small batch of data for inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval() # setting the model to evaluation phase
    for split in ['train', 'val']:
        losses  = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # resetting the model to training phase
    return out

class Head(nn.Module):
    """one head of self attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.query = nn.Linear(n_embd, head_size, bias = False)
        self.value = nn.Linear(n_embd, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        # compute attention scores (affinities)
        wei = q@k.transpose(-2,-1)*C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T]==0, float('-inf')) # B, T, T
        wei = F.softmax(wei, dim=-1) # B, T, T
        wei = self.dropout(wei)
        # perform weighted aggregation of the values
        out = wei@v
        return out
    
class MultiHeadedAttention(nn.Module):
    """multiple heads of self attention running in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class Feedforward(nn.Module):
    """a simple linear layer followed by a non linearity"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout)
            )
    
    def forward(self, x):
        out = self.net(x)
        return out


class Block(nn.Module):

    def __init__(self, n_embd, n_head):
        # n_embd is the embedding dimension, n_head is the number of heads we would like
        super().__init__()
        head_size = n_embd//n_head
        self.sa = MultiHeadedAttention(n_head, head_size)
        self.ffwd = Feedforward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))   # residual connection
        x = x + self.ffwd(self.ln2(x)) # residual connection
        return x


# super simple bigram language model
class BigramLanguageModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size) # the un-embedding matrix
    
    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B, T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # B, T, C
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # T, C (this operation returns the embedding table inside the object self.position_embedding_table)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) # B, T, vocab_size

        
        if targets == None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        #idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop the idx before till block size for feeding into the bigram model for generation
            idx_cond = idx[:,-block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:,-1,:] # logits is treated as 3 dimensional array because while calling self(idx), no targets were provided so the function doesn't venture into the else block where the dimension of logits is modified
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples = 1) # (B,1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) #(B, T+1)
        
        return idx

model = BigramLanguageModel()
m = model.to(device)

# create a PyTorch Optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr = learning_rate)

for iter in range(max_iters):
    
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")


    # sample the batch of data
    xb, yb = get_batch('train')
    
    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context,max_new_tokens=500)[0].tolist()))