import torch
import torch.nn as nn
from torch.nn import functional as F

#hyperparameters
batch_size = 16
block_size = 8
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3 #decrease the learning because self attention cant tolerate very high learning rate
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

class Head(nn.Module):
    """
    one head of self attention
    """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias = False)
        self.value = nn.Linear(n_embd, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size,block_size)))

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x) #shape of B,T,head_size # head_size is also referred to as the channel dimension in the comments below
        q = self.query(x) #shape of B,T,head_size
        v = self.value(x) #shape of B,T,head_size

        #compute attention scores
        wei = q @ k.transpose(-1,-2) * C**-0.5 # (B,T,C) @ (B,C,T) -> (B,T,T) these will be the attention affinities
        wei = wei.masked_fill(self.tril[:T, :T]==0, float('-inf'))
        wei = F.softmax(wei, dim=-1) #B,T,T #the softmax will give the attention score
        # perform weighted aggregation
        out = wei @ v #wei: B,T,T and v: B,T,head_size -> B,T,head_size
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        #this linear projecttion is added to make sure that the concatenation of outputs of multi attention are compatible with the dmodel. 
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        #the shape of the output of each attention head is B,T,head_seize. So concating along the last dim -1 means that the new shape will be like
        # B,T,head_size*num_heads
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd,4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd,n_embd)
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head ):
        super().__init__()
        #the head_size is 32//4 = 8 dimensional.
        head_size = n_embd // n_head
        self.sa_heads = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)

    def forward(self, x):
        #x: input is B,T,C (B,8,32)
        # apply multiple head of self attention with residual connection, the output is of size B,T,C here C is 32 dimensional. Because
        # 4* 8, or "num_heads*head_size" is equal to 32. Remember that the each attention heads output was concatenated
        x = x + self.sa_heads(x)
        # output is B,T,C this feedforward is token by token, like each token is just gonna go to the linear layer and get multiplied
        #the linear layer has output neurons of size n_embd, so naturally C is also n_embd.
        x = x + self.ffwd(x)
        return x





class BigramLanguageModel(nn.Module):
    #we are going to modify this to incldue the attention head
    def __init__(self):
        #no need to pass the vocab size arouond as it is already defined uptop
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        #positional emebdinng Notice how the second dimension is the same as the token embedding dimension
        #for the positional encoding we just index it with arange tensor, and uptill the context we have and less than block size always during forward and inference.
        #see below for more details on positional embedding.
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            Block(n_embd, 4),
            Block(n_embd, 4),
            Block(n_embd, 4),
        )

        #this is now new linear layer that will apply soon after the token emmbeddings.
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B,T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        #this won't be called logits anymore since we are increasing the complexity.
        #instead these will just be the token embeddings.
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        #this will right now just create the same block_size by n_embd tensor. It just indexes the position embedding table with 0-7 indices, and it will just
        #pluck out the entire position embedding table, or uptill T if we are doing inference with incremental context size.
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb # (B,T,C) + (T,C) -> B,T,C
        x = self.blocks(x)

        #the below linear layer is basically a matrix multiplaction
        #the (B,8,32) @ (32,65) produces a tensor of (B,8,65)
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
            #one more thing we need to add because we have added positional encoding is to crop the context other wise
            # the positional encoding table will run out of scope, as the positional encoding is defined as
            #self.position_embedding_table = nn.Embedding(block_size, n_embd) so we need to crop the input
            # doing -block_size as the indexer for the column dimension is okay even if the idx tensor is say of size (8,3) and our block size is of 9 for eg, and if index it like this
            #idx[:,-9:] this means that i want all the rows and i want all the columns starting from 9 places to the end of the tensor. But it will still work if we only have 3 columns so far, and
            # we dont have to worry about going out of bounds with negative indexing, it will just pick out the elements that are available.
            idx_cond = idx[:, -block_size:] #reassign to new variable so that the current generation doesn't get overwritten.
            # get the predictions
            logits, loss = self(idx_cond)
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
    if iter % eval_interval == 0 and iter > 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}")
    xb,yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

context = torch.zeros((1,1), dtype=torch.long, device=device)
generated = m.generate(context, max_new_tokens = 500)[0].tolist()
print(len(generated))
print(decode(generated))





