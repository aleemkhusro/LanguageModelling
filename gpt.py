import torch
import torch.nn as nn
from torch.nn import functional as F

#hyperparameters
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32

#-------

torch.manual_seed(1337)
# read it in to inspect it
with open('Data\\input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

#create train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% is train, rest val
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    xappender = []
    yappender = []
    for i in ix:
        xappender.append(data[i:i+block_size])
    x = torch.stack(xappender)
    for i in ix:
        yappender.append(data[i+1:i+block_size+1])
    y = torch.stack(yappender)
    x,y = x.to(device), y.to(device)


    return x, y

@torch.no_grad()
def estimate_loss():
    out= {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class BigramLanguageModel(nn.Module):

    def __init__(self):
        #no need to pass the vocab size arouond as it is already defined uptop
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        #positional emebdinng Notice how the second dimension is the same as the token embedding dimension
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        #this is now new linear layer that will apply soon after the token emmbeddings.
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B,T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        #this won't be called logits anymore since we are increasing the complexity.
        #instead these will just be the token embeddings.
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        #this will right now just create the same block_size by n_embd tensor. It just indexes the position embedding table with 0-7 indices, and it will just
        #pluck out the entire position embedding table
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        #the below linear layer is basically a matrix multiplaction
        #the (4,8,32) @ (32,65) produces a tensor of (4,8,65)
        logits = self.lm_head(x) # (B,T,Vocab_Size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel()
#model and m will refer to the same location in memory
m = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-03)
for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}")
    xb,yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens = 500)[0].tolist()))





