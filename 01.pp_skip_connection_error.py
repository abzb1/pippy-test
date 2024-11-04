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
        self.mlp = nn.Sequential(
            nn.Linear(h_dim, 4*h_dim),
            nn.GELU(),
            nn.Linear(4*h_dim, h_dim)
        )
        self.norm2 = nn.LayerNorm(h_dim)

    def forward(self, x: torch.Tensor):
        h = self.norm1(x + self.attention(x, x, x)[0])
        return self.norm2(h + self.mlp(h))
        
        
model = Transformer()
print(model)

# forward test
input_ = (torch.rand(1, 10)*100).long()
print(input_)
output = model(input_)
print(output.device)

# An example micro-batch input
x = torch.LongTensor([1, 2, 4, 5])

pipe = pipeline(
    module=model,
    mb_args=(x,),
    split_spec={
        "layers.1": SplitPoint.BEGINNING,
    }
)

print(pipe)

# ## error
# def _compute_accessor(parent_fqn: str, child_fqn: str) -> str:
#     if parent_fqn == "":
#         # Handle the root module correctly.
#         return child_fqn

#     parent_split = parent_fqn.split(".")
#     child_split = child_fqn.split(".")

#     # TODO: support skip connection by inlining the child module.
#     if child_split[: len(parent_split)] != parent_split:
#         raise RuntimeError(
#             f"Child module '{child_fqn}' is not a descendant of parent mldule '{parent_fqn}'."
#             "This is currently unsupported."
#             "Please try to make child module attach to parent module direclty."
#         )
#     return ".".join(child_split[len(parent_split) :])

# RuntimeError: Child module 'layers.1._modules.mlp.0' is not a descendant of parent mldule 'layers.1.mlp'.This is currently unsupported.Please try to make child module attach to parent module direclty.