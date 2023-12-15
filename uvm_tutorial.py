import os
import torch
import torchrec
import torch.distributed as dist

os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"

# Note - you will need a V100 or A100 to run tutorial!
# If using an older GPU (such as colab free K80), 
# you will need to compile fbgemm with the appripriate CUDA architecture
# or run with "gloo" on CPUs 
dist.init_process_group(backend="nccl")
gpu_device=torch.device("cuda")
hbm_cap_2x = 2 * torch.cuda.get_device_properties(gpu_device).total_memory

embedding_dim = 8
# By default, each element is FP32, hence, we divide by sizeof(FP32) == 4.
num_embeddings = hbm_cap_2x // 4 // embedding_dim
ebc = torchrec.EmbeddingBagCollection(
    device="meta",
    tables=[
        torchrec.EmbeddingBagConfig(
            name="large_table",
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
            feature_names=["my_feature"],
            pooling=torchrec.PoolingType.SUM,
        ),
    ],
)
from typing import cast

from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.distributed.planner.types import ParameterConstraints
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
from torchrec.distributed.types import ModuleSharder


topology = Topology(world_size=1, compute_device="cuda")
constraints = {
    "large_table": ParameterConstraints(
        sharding_types=["table_wise"],
        compute_kernels=[EmbeddingComputeKernel.BATCHED_FUSED_UVM.value],
    )
}
plan = EmbeddingShardingPlanner(
    topology=topology, constraints=constraints
).plan(
    ebc, [cast(ModuleSharder[torch.nn.Module], EmbeddingBagCollectionSharder())]
)

uvm_model = torchrec.distributed.DistributedModelParallel(
    ebc,
    device=torch.device("cuda"),
    plan=plan
)

# Notice "batched_fused_uvm" in compute_kernel.
print(uvm_model.plan)

# Notice ShardedEmbeddingBagCollection.
print(uvm_model)
mb = torchrec.KeyedJaggedTensor(
    keys = ["my_feature"],
    values = torch.tensor([101, 202, 303, 404, 505, 606]).cuda(),
    lengths = torch.tensor([2, 0, 1, 1, 1, 1], dtype=torch.int64).cuda(),
)
uvm_result = uvm_model(mb).wait()
print(uvm_result)
