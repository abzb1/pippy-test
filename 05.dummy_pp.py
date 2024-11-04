import os

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.pipelining import pipeline, SplitPoint, ScheduleGPipe, Schedule1F1B

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.tok_embeddings = nn.Embedding(1234, 768)

        # ensuring checkpoints will correctly save and load.
        self.layers = torch.nn.ModuleList(
            TransformerBlock(768) for _ in range(16)
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

rank = int(os.environ['RANK'])
world_size = int(os.environ['WORLD_SIZE'])
dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
    )

model = Transformer()

# An example micro-batch input
num_b = 16
num_mb = 4

dummy_b = torch.randint(0, 1234, (num_b, 10), device=f"cuda:{rank}", dtype=torch.long)
dummy_mb = torch.randint(0, 1234, (num_mb, 10), device=f"cuda:{rank}", dtype=torch.long)

# dummy_target 1-hot encoding
one_idx = torch.randint(0, 1234, (num_b, 1), device=f"cuda:{rank}")
dummy_target = torch.zeros((num_b, 1234), device=f"cuda:{rank}").to(torch.long)
dummy_target.scatter_(1, one_idx, 1)


layers_per_rank = 16 // world_size
pipe = pipeline(
    module=model,
    mb_args=(dummy_mb,),
    split_spec={
        f"layers.{i * layers_per_rank}": SplitPoint.BEGINNING for i in range(1, world_size)
    },

)

stage = pipe.build_stage(
        rank,
        device=f"cuda:{rank}",
    )

loss_fn=torch.nn.CrossEntropyLoss()

schedule = ScheduleGPipe(stage=stage, n_microbatches=num_mb, loss_fn=loss_fn)
# schedule = Schedule1F1B(stage=stage, n_microbatches=num_mb, loss_fn=loss_fn)

optimizer = torch.optim.Adam(pipe.parameters(), lr=1e-4)

# Run
optimizer.zero_grad()

if rank == 0:
    schedule.step(dummy_b)
elif rank == world_size - 1:
    losses = []
    out = schedule.step(target=dummy_target, losses=losses)
    print(f"Losses: {losses}")
else:
    out = schedule.step()

# print gradients
print(f"Rank {rank} gradients:")
for name, param in pipe.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.abs().mean()}")

optimizer.step()

dist.barrier()
dist.destroy_process_group()

print(f"Rank {rank} completes")