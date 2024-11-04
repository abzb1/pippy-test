import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.pipelining import pipeline, SplitPoint

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.tok_embeddings = nn.Embedding(1234, 768)

        # ensuring checkpoints will correctly save and load.
        self.layers = torch.nn.ModuleList(
            TransformerBlock(768) for _ in range(2)
        )

        self.output = nn.Linear(768, 1234)

    def forward(self, tokens: torch.Tensor):
        # Handling layers being 'None' at runtime enables easy pipeline splitting
        h = self.tok_embeddings(tokens) if self.tok_embeddings else tokens

        for layer in self.layers:
            h = layer(h)

        output = self.output(h).float() if self.output else h
        return output
    
class TransformerBlock(nn.Module):
    def __init__(self, h_dim: int):
        super().__init__()

        self.attention = nn.MultiheadAttention(h_dim, 4)
        self.norm1 = nn.LayerNorm(h_dim)
        self.mlp1 = nn.Linear(h_dim, 4*h_dim)
        self.gelu = nn.GELU()
        self.mlp2 = nn.Linear(4*h_dim, h_dim)
        self.norm2 = nn.LayerNorm(h_dim)

    def forward(self, x: torch.Tensor):
        h = self.norm1(x + self.attention(x, x, x)[0])
        h2 = self.gelu(self.mlp1(h))
        return self.norm2(h + self.mlp2(h2))
        
model = Transformer()
print(model)

# forward test
input_ = (torch.rand(1, 10)*100).long()
print(input_)
output = model(input_)
print(output.device)

# An example micro-batch input
x = torch.LongTensor([[1, 2, 4, 5], [4, 2, 3, 2]])
print(x.shape)
pipe = pipeline(
    module=model,
    mb_args=(x,),
    split_spec={
        "layers.1": SplitPoint.BEGINNING,
    }
)

print(pipe)

info = pipe.info()
print(info)
# stage = build_stage(dp_mod, stage_idx, info, device, group)