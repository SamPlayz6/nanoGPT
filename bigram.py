import torch
import torch.nn as nn
from torch.nn import functional as F


#Hyper parameters
batch_size = 32 # Amount of independant sequences to process in parrallel
block_size = 8 # Maximum content length for a predictor. Never look at size bigger than block size when predicting
max_iters = 3000
eval_interval =  300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters =200
n_embed = 32 # Number of embedding dimensions

#----------------------
torch.manual_seed(1337) # Same randomly generated sequence of numbers

with open('tiny-shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()


# print("Dataset of length " ,len(text))
# print(text[:1000])


#Find all unique characters in the text
chars = sorted(list(set(text)))
vocab_size  = len(chars)

# print(''.join(chars))
# print(len(chars))


# Encoding and Decoding functions
stoi = {ch:i for i,ch in enumerate(chars)} # Dictionaries are being created here. Here, {char:int, char:int etc}
itos = {i:ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s] # Encoding character by character. Now this object acts a Function with an input
decode = lambda l: "".join([itos[c] for c in l])
# print(stoi);print(itos)


#----------
# Hmm, didnt know you could do this. This piece of code is redundant
p = [i for i,ch in enumerate(chars)]
# print(p)
#-----------

# print("\nEncoding and decoding")
# print(encode("hii there"))
# print(decode(encode("hii there")))



#Now to encode our data into a torch Tensor
data = torch.tensor(encode(text), dtype = torch.long)
# print(data.shape, data.dtype)
# print(data[:100])






#Seperate the dataset into Training data and Test Data
n = int(0.9*len(data))

train_data = data[:n]
val_data = data[n:]


#train_data[:block_size+1]

# block_size = 4
def get_batch(split):
    #Generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data  #Had no idea you could do this!!!
    ix = torch.randint(len(data) - block_size, (batch_size,)) # Getting batch_size amount of random ints between 0 and len(data) - block_size
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+1+block_size] for i in ix]) # Ofset by one character of x
    x, y = x.to(device), y.to(device)  # Loading data to the GPU - Make sure to load it to device
    
    return x,y


xb, yb = get_batch('train')

# print(xb)
# print(yb)


#This fucntion is used to get a more accurate maeasurment of loss by averaging it over a bunch of batches.
@torch.no_grad() # Tell Pytroch there is no backprogation - So dont need to carry all the extra storage of variables needed to backpropgate  
def estimate_loss():
    out = {}
    model.eval()  #Setting model to evaluation phase. Whatever that means
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # Setting model to training phase - Some models behave differently in different modes. THats why it is important  
    return out

class BigramLangageModel(nn.Module):

    def __init__(self):
        super().__init__()
        #Each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed) 
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size )

    def forward(self,idx,targets = None):
        B, T = idx.shape

        #idx and tagets are both (B, T) tensors of integers
        tok_emb= self.token_embedding_table(idx) # (B, T, C) B - Batch, T - Time, C - Length of Batch?:  batch, time, channels
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb
        logits = self.lm_head(x) # (B,T,vocab_size) 

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss =  F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        #idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            #Getting predictions from forward function
            logits, loss = self(idx)

            #focusing only on the last time step
            logits = logits[:,-1,:] # Become (B,C)

            #Softmax applied
            probs = F.softmax(logits, dim = 1) # (B, 1)

            idx_next = torch.multinomial(probs, num_samples = 1) # (B,1)

            #Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim =1) # (B, T+1)

        return idx

         

model = BigramLangageModel()
m = model.to(device)

# logits, loss = m(xb,yb)
# print(logits.shape)
# print(loss)


# context = torch.zeros((1,1), dtype=torch.long)


# print(m.generate(context, max_new_tokens=100)[0])

# print(decode(m.generate(context, max_new_tokens=100)[0].tolist()))




#Creating a Pytorch Optimizer Object

optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)


for iter in range(max_iters):

    #Every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    #sample a batch of data
    xb, yb = get_batch('train')

    #Evaluate Loss
    logits, loss = model(xb,yb)
    optimizer.zero_grad(set_to_none = True)

    loss.backward()
    optimizer.step()



context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))








# batch_size = 32

# for steps in range(10000):

#     #sample a batch of data
#     xb, yb = get_batch('train')

#     #Evaluate the loss
#     logits, loss = m(xb,yb)
#     optimizer.zero_grad(set_to_none=True)
#     loss.backward()
#     optimizer.step() 

# print(loss.item())


# print(decode(m.generate(context, max_new_tokens=300)[0].tolist()))
