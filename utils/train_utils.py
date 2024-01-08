import os
import torch
import numpy as np
import matplotlib.pyplot as plt


def draw_loss_curve(loss_list, epoch, save_path):
    """ 绘制损失曲线 """
    loss_list = np.array(loss_list)
    epochs = loss_list[:, 0]
    losses = loss_list[:, 1]
    min_loss = min(losses)
    min_loss_epoch = epochs[np.argmin(losses)]
    plt.plot(epochs, losses, '--o', label='train loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.text(min_loss_epoch, min_loss, "min loss: {:.4f}".format(min_loss))
    plt.scatter(min_loss_epoch, min_loss, c='g', marker='D')
    plt.savefig(os.path.join(save_path, "{}.png".format(str(epoch).zfill(3))),
                dpi=200, bbox_inches='tight')
    plt.close()


def train_model(loader, model, optimizer, epoch, args, loss_list):
    """ 训练模型 """
    # 训练模式
    model.train()
    # GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    # 迭代
    s = args.start_deep
    z = args.deep_dim
    losses = []
    for idx, (data, label) in enumerate(loader):
        data = data[:, s:s+z].to(device=args.device, dtype=torch.float32)
        label = label.to(device=args.device, dtype=torch.float32)
        # 清空梯度
        optimizer.zero_grad()
        # 前向传播
        pred = model(data)
        # 计算损失
        pred = pred.squeeze(1)
        pred = pred.to(device=args.device, dtype=torch.float32)
        loss = model.criterion(pred, label)
        losses.append(loss.item())
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        # 打印信息
        max_value = pred.max().item()
        print("Epoch: {}/{} | Iter: {}/{} | Loss: {:.4f} | Max: {:.4f}"
              .format(str(epoch + 1).rjust(3), args.epochs,
                      str(idx + 1).rjust(4), len(loader),
                      loss.item(), max_value))
    # 平均损失
    mean_loss = sum(losses) / len(losses)
    print("[INFO] Epoch: {}/{} | Mean Train Loss: {:.4f}"
          .format(str(epoch + 1).rjust(3), args.epochs,
                  mean_loss))
    loss_list.append([epoch + 1, mean_loss])
    # 绘制损失曲线
    print("[INFO] Draw train loss curve...")
    loss_path = os.path.join(args.save_path, "train loss") + "/"
    if not os.path.exists(loss_path):
        os.makedirs(loss_path)
    draw_loss_curve(loss_list, epoch + 1, loss_path)
    # 间隔保存
    model_path = os.path.join(args.save_path, "checkpoints") + "/"
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if epoch ==0 or (epoch + 1) % args.val_epoch == 0 or epoch == args.epochs - 1:
        print("[INFO] Save model...")
        torch.save(model.state_dict(),
                   os.path.join(model_path, '{}.pth'.format(str(epoch + 1).zfill(3))))
    # 保存最好的模型
    best_model_path = os.path.join(model_path, "best_model.pth") + "/"
    if mean_loss < args.loss_th:
        print("[INFO] Save best model...")
        torch.save(model.state_dict(), best_model_path)
        args.loss_th = mean_loss
    # 返回平均损失
    return loss_list




