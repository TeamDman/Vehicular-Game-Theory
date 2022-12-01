# RL

Some jot notes for what would be good in an RL framework / tutorial

## Custom tensor dataclasses

Would be good to have a dataclass-inspired class for people to group together tensors and be able to do bulk operations.
Example: the state tensor batch object used in this project represents two tensors, one for the vehicles and one for the vulnerabilities. Sometimes these groups of tensors need to be repeated or sent to the GPU, which means re-implementing the pytorch methods to work on dataclasses containing tuples. Something the dataclasses could inherit to act as a torch.Tensor would be good