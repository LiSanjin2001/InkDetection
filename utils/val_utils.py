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
    plt.plot(epochs, losses, '--o', label='val loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.text(min_loss_epoch, min_loss, "min loss: {:.4f}".format(min_loss))
    plt.scatter(min_loss_epoch, min_loss, c='g', marker='D')
    plt.savefig(os.path.join(save_path, "{}.png".format(str(epoch).zfill(3))),
                dpi=200, bbox_inches='tight')
    plt.close()


def get_eval(args, pred, label):
    """评估指标"""
    # 调整维度
    # pred = torch.sigmoid(pred)
    # 二值化
    pred = (pred > args.threshold).int()
    pred = pred[0].int().cpu()
    label = label[0].int().cpu()
    # 计算指标
    TP = (pred & label).sum()
    TN = ((1 - pred) & (1 - label)).sum()
    FP = (pred & (1 - label)).sum()
    FN = ((1 - pred) & label).sum()
    eps = 1e-8
    # 计算指标
    iou = TP / (TP + FP + FN + eps)
    acc = (TP + TN) / (TP + TN + FP + FN + eps)
    recall = TP / (TP + FN + eps)
    precision = TP / (TP + FP + eps)
    F1 = 2 * precision * recall / (precision + recall + eps)
    tnr = TN / (TN + FP + eps)
    beta = 0.5
    dice = (1+beta**2) * precision * recall / (beta**2 * precision + recall + eps)
    # 保存信息
    eval_list = [iou, acc, recall, precision, F1, tnr, dice]
    # 返回
    return eval_list


def plot_img(output, label, epoch, arg, idx, dice):
    """可视化"""
    output = (output > arg.threshold).int()
    output = output[0].int().cpu().numpy()
    label = label[0].int().cpu().numpy()
    plt.ioff()  # 关闭交互模式
    plt.subplot(1, 2, 1)
    plt.axis('off')
    plt.title('pred | dice: {:.4f}'.format(dice))
    plt.imshow(output, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.axis('off')
    plt.title('ground truth')
    plt.imshow(label, cmap='gray')
    save_path = os.path.join(arg.save_path, 'images', str(epoch + 1).zfill(3))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, '{}.png'.format(str(idx + 1).zfill(3))),
                bbox_inches='tight',
                dpi=200,
                transparent=True)
    plt.close()  # 关闭图像，释放内存


def draw_eval_curve(val_list, epoch, eval_path):
    """绘制评估曲线"""
    eval_list = np.array(val_list)
    plt.figure()
    plt.plot(eval_list[:, 0], eval_list[:, 1], '--v', label='IoU')
    plt.plot(eval_list[:, 0], eval_list[:, 2], '--^', label='Acc')
    plt.plot(eval_list[:, 0], eval_list[:, 3], '-->', label='Recall')
    plt.plot(eval_list[:, 0], eval_list[:, 4], '--<', label='Precision')
    plt.plot(eval_list[:, 0], eval_list[:, 5], '--s', label='F1')
    plt.plot(eval_list[:, 0], eval_list[:, 6], '--p', label='TNR')
    plt.plot(eval_list[:, 0], eval_list[:, 7], '--d', label='Dice')
    plt.legend()
    plt.grid()
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Evaluation Curve')
    plt.savefig(os.path.join(eval_path, '{}.png'.format(str(epoch + 1).zfill(3))),
                bbox_inches='tight',
                dpi=200)
    plt.close()


def val_model(loader, model, epoch, args, val_list, loss_list):
    """验证模型"""
    print('[INFO] Validating {} epoch...'.format(epoch + 1))
    # 验证模式
    model.eval()
    # GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    # 迭代
    s = args.start_deep
    z = args.deep_dim
    eval_list = []
    losses = []
    with torch.no_grad():
        for idx, (data, label) in enumerate(loader):
            data = data[:, s:s+z].to(device=args.device, dtype=torch.float32)
            label = label.to(device=args.device, dtype=torch.float32)
            # 前向传播
            pred = model(data)
            # 计算损失
            pred = pred.squeeze(1)
            pred = pred.to(device=args.device, dtype=torch.float32)
            # 损失函数
            loss = model.criterion(pred, label)
            losses.append(loss.item())
            # 评估指标
            evals = get_eval(args, pred, label)
            iou, acc, recall, precision, F1, tnr, dice = evals
            eval_list.append(evals)
            # 打印信息
            print(
                "[INFO] Iter: {}/{} | IoU: {:.4f} | Acc: {:.4f} | R: {:.4f} | P: {:.4f} | F1: {:.4f} | TNR: {:.4f} | Dice: {:.4f}"
                .format(str(idx + 1).rjust(3), args.val_num,
                        iou, acc, recall, precision, F1, tnr, dice))
            # 可视化
            plot_img(pred, label, epoch, args, idx, dice)
    # 平均指标
    mean_loss = sum(losses) / len(losses)
    print("[INFO] Epoch: {}/{} | Mean Val Loss: {:.4f}"
          .format(str(epoch + 1).rjust(3), args.epochs,
                  mean_loss))
    loss_list.append([epoch + 1, mean_loss])
    eval_list = np.array(eval_list)
    mean_eval = np.mean(eval_list, axis=0)
    print(
        "[INFO] Epoch: {}/{} | Mean Dice: {:.4f}"
        .format(str(epoch + 1).rjust(3), args.epochs,
                mean_eval[-1]))
    mean_eval = mean_eval.tolist()
    mean_eval.insert(0, epoch + 1)
    val_list.append(mean_eval)
    # 绘制损失曲线
    print("[INFO] Draw val loss curve...")
    loss_path = os.path.join(args.save_path, "val loss") + "/"
    if not os.path.exists(loss_path):
        os.makedirs(loss_path)
    draw_loss_curve(loss_list, epoch + 1, loss_path)
    # 绘制指标曲线
    print("[INFO] Draw eval curve...")
    eval_path = os.path.join(args.save_path, "val") + "/"
    if not os.path.exists(eval_path):
        os.makedirs(eval_path)
    draw_eval_curve(val_list, epoch, eval_path)
    # 返回
    return val_list, loss_list

