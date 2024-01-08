import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt


def merge_images(results, outputs, save_path, masks):
    """ 合并图片 """
    for folder, x, y, pred in outputs:
        results[folder][0][x:x + 512, y:y + 512] = pred
    out = {}
    save_test_path = os.path.join(save_path, "test") + "/"
    if not os.path.exists(save_test_path):
        os.makedirs(save_test_path)
    for folder in results.keys():
        result, shape_original = results[folder]
        result = result[:shape_original[0], :shape_original[1]]
        result = result * masks[folder]
        save_path_file = os.path.join(save_test_path, "{}.png".format(folder))
        cv2.imwrite(save_path_file, result * 255)
        out[folder] = result
    return out


def save_csv(results, save_path):
    """ 保存csv文件 """
    # 生成空白rle
    rles = {folder: None for folder in results.keys()}
    # 生成rle
    for folder in results.keys():
        pixels = np.where(results[folder].flatten(), 1, 0).astype(np.uint8)
        pixels[0] = 0
        pixels[-1] = 0
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
        runs[1::2] = runs[1::2] - runs[:-1:2]
        rle = ' '.join(str(x) for x in runs)
        rles[folder] = rle
    # 保存
    out_str = 'Id,Predicted\n'
    for folder in rles.keys():
        out_str += folder + ',' + rles[folder] + '\n'
    save_path = os.path.join(save_path, 'submission.csv')
    with open(save_path, 'w', newline='') as f:
        f.write(out_str)
        f.close()


def draw_curve(train_losses, val_losses, val_lists, save_path):
    """ 绘制曲线 """
    # 保存 txt
    save_path_eval = os.path.join(save_path, 'eval.txt')
    with open(save_path_eval, 'w') as f:
        for val in val_lists:
            f.write(str(val) + '\n')
        f.close()
    save_path_loss = os.path.join(save_path, 'train_loss.txt')
    with open(save_path_loss, 'w') as f:
        for train in train_losses:
            f.write(str(train) + '\n')
        f.close()
    save_path_loss = os.path.join(save_path, 'val_loss.txt')
    with open(save_path_loss, 'w') as f:
        for val in val_losses:
            f.write(str(val) + '\n')
        f.close()
    # 绘制损失曲线
    train_losses = np.array(train_losses)
    val_losses = np.array(val_losses)
    plt.plot(train_losses[:, 0], train_losses[:, 1], '--bo', label='train loss')
    plt.plot(val_losses[:, 0], val_losses[:, 1], '--rd', label='val loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_path, "loss.png"),
                dpi=300, bbox_inches='tight')
    plt.close()
    # 绘制评估曲线
    val_list = np.array(val_lists)
    plt.figure()
    plt.plot(val_list[:, 0], val_list[:, 1], '--v', label='IoU')
    plt.plot(val_list[:, 0], val_list[:, 2], '--^', label='Acc')
    plt.plot(val_list[:, 0], val_list[:, 3], '-->', label='Recall')
    plt.plot(val_list[:, 0], val_list[:, 4], '--<', label='Precision')
    plt.plot(val_list[:, 0], val_list[:, 5], '--s', label='F1')
    plt.plot(val_list[:, 0], val_list[:, 6], '--p', label='TNR')
    plt.plot(val_list[:, 0], val_list[:, 7], '--d', label='Dice')
    plt.legend()
    plt.grid()
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Evaluation Curve')
    plt.savefig(os.path.join(save_path, 'eval.png'),
                bbox_inches='tight')
    plt.close()


def test_model(loader, model, args, train_loss, val_loss, val_list):
    """ 测试模型 """
    print("[INFO] Testing...")
    # 参数设置
    root_path = args.data_path
    size = args.size
    test_path = os.path.join(root_path, "test") + "/"
    folder_list = sorted(os.listdir(test_path))
    # 初始化
    results = {folder_list[i]: None for i in range(len(folder_list))}
    # 生成空白图片
    masks = {}
    for folder in folder_list:
        # 导入mask
        mask_path = os.path.join(test_path, folder, "mask.png")
        mask = cv2.imread(mask_path, 0)
        masks[folder] = mask
        shape_original = mask.shape
        # 填充
        pad0 = (size - shape_original[0] % size)
        pad1 = (size - shape_original[1] % size)
        mask = np.pad(mask, ((0, pad0), (0, pad1)), 'constant', constant_values=0)
        # 生成空白图片
        shape_pad = mask.shape
        result = np.zeros(shape_pad)
        # 保存
        print("[INFO] Generating blank image, file {} | shape: {} -> {}"
              .format(folder, shape_original, shape_pad))
        results[folder] = [result, shape_original]
    # 测试
    model.eval()
    outputs = []
    s = args.start_deep
    z = args.deep_dim
    with torch.no_grad():
        for idx, (data, label) in enumerate(loader):
            data = data[:, s:s+z].to(device=args.device, dtype=torch.float32)
            # 前向传播
            pred = model(data)
            # 处理预测值
            pred = pred[0, 0]
            pred = (pred > args.threshold).int()
            pred = pred.cpu().numpy()
            # 处理标签
            label = label[0]
            label = label.split('/')[-1]
            folder, x, y = label.split('_')
            y = y.split('.')[0]
            x, y = int(x), int(y)
            # 保存
            print("[INFO] Generating image, file {} | x: {} | y: {}"
                  .format(folder, str(x).ljust(4), str(y).ljust(4)))
            outputs.append([folder, x, y, pred])
    # 合并
    print("[INFO] Merging images...")
    results = merge_images(results, outputs, args.save_path, masks)
    # 输出 submission
    print("[INFO] Generating submission...")
    save_csv(results, args.save_path)
    # 绘制曲线
    print("[INFO] Draw evaluation and loss curve...")
    draw_curve(train_loss, val_loss, val_list, args.save_path)
