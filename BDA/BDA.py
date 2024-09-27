import time
import torch
from BDA_Aux import *
from torch.optim import Adam, SGD
from torch.utils.tensorboard import SummaryWriter

from torch import nn

def main():
    # 处理外部使用的参数问题
    args = parser()

    # 使用GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 设置随机种子
    set_random_set(args)

    # 导入数据
    train_loader, validation_loader, test_loader = data_load(args.batch_size, args.datasets, args.pollute, args.proxy)

    # 设定两个训练参数
    model_x = ModelX(device)
    model_x.x.retain_grad()  # x不是叶子节点，其梯度的计算需要本命令

    model_y = ModelY(device, args.flag)

    par = Paras(device, args.niu, args.niu1,args.niu2, args.s_u, args.s_l, args.lr_x, args.lr_y, args.batch_size,
                args.t1, args.t2)

    optim_x = Adam(model_x.parameters(), lr=par.lr_x)
    optim_y = SGD(model_y.parameters(), lr=par.lr_y)

    if args.logdir is not None:
        writer = SummaryWriter(args.logdir)

    F = Loss_F(device, args.flag)
    f = Loss_f(device)

    elements = args.output.split("+")
    print(elements)

    start_time = time.time()
    for epo in range(args.epochs):
        model_y.y.requires_grad_(True)
        if "epochs" in elements:
            print(epo, end='|', flush=True)
        for k in range(args.K):

            # 开始累积y的梯度
            model_y.y.requires_grad_(True)

            # 计算gF_y(上层损失函数)
            total_loss_F = F.exec(validation_loader, model_x, model_y, par)
            total_loss_F.backward()
            gF_y = model_y.y.grad.clone()

            # 计算gf_y(下层损失函数)
            total_loss_f = f.exec(train_loader, model_x, model_y, par)
            total_loss_f.backward()
            gf_y = model_y.y.grad.clone()

            # 根据BDA的算法迭代
            direction = get_direction(gF_y, gf_y, par, k)

            # model_y.update_opt(optim_y, direction)
            model_y.update_man(args.lr_y, direction)
            if args.logdir is not None:
                if "norm_of_grad" in elements:
                    writer.add_scalars("norm_of_grad/F",{'{}'.format(args.flag):torch.torch.norm(gF_y, p=2)},k+args.K*epo)
                    writer.add_scalars("norm_of_grad/f",{'{}'.format(args.flag):torch.torch.norm(gf_y, p=2)},k+args.K*epo)

        optim_x.step()

    end_time = time.time()
    total_train_time = end_time-start_time
    print("train duration:{} sec".format(total_train_time))

    validation_acc = acc(model_y, validation_loader,device)
    test_acc = acc(model_y, test_loader,device)
    print("validation_acc: {}, test_acc: {}".format(validation_acc,test_acc))
    if args.logdir is not None:
        # writer.add_scalars("K{}to epochs/Accuracy, ".format(args.K), {'test':test_acc,'validation':validation_acc}, args.epochs)
        writer.close()

if __name__ == "__main__":
    main()