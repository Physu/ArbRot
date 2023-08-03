import torch.nn as nn
import torch

loss = nn.CrossEntropyLoss()

input1 = torch.randn(3, 5, requires_grad=True)
target1 = torch.empty(3, dtype=torch.long).random_(5)

input2 = torch.randn(2, 3, requires_grad=True)
target2 = torch.empty(2, dtype=torch.long).random_(3)

input3 = torch.randn(2, 2, requires_grad=True)
target3 = torch.empty(2, dtype=torch.long).random_(2)

output1 = loss(input1, target1)
output2 = loss(input2, target2)
output3 = loss(input3, target3)

print(f"output1:{output1}\noutput2:{output2}\noutput3:{output3}")