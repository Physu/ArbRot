import torch

input_ = torch.tensor([[1., 2.], [3., 4.]], requires_grad=False)
w1 = torch.tensor(2.0, requires_grad=True)
w2 = torch.tensor(3.0, requires_grad=True)

l1 = input_ * w1
l2 = l1 + w2
loss1 = l2.mean()
loss1.backward(retain_graph=True)

print(w1.grad)  # 输出：tensor(2.5)
print(w2.grad)  # 输出：tensor(1.)

loss2 = l2.sum()
loss2.backward()

print(w1.grad)  # 输出：tensor(12.5)
print(w2.grad)  # 输出：tensor(5.)


