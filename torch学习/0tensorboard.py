from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("log")

for i in range(100):
    writer.add_scalar("y=3x", 3*i, i)

writer.add_image("others")

writer.close()

# 使用代码查看结果
# tensorboard --logdir=log --port=6007