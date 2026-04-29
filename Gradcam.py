import cv2
import numpy as np
import torch
import os
from ultralytics import YOLO
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# ================= 配置区域 =================
MODEL_PATH = r'D:\桌面\best11.pt'
IMG_DIR = r'D:\桌面\our'
SAVE_DIR = r'D:\桌面\res_gradcam_pro1'
TARGET_LAYER_INDEX = 21
# 指定你想要观察的缺陷类别索引（如果你只想看某一类缺陷，请修改这里）
# 如果不确定，设为 None，代码将自动选取图中置信度最高的类别
TARGET_CATEGORY = None


# ===========================================

class AdvancedTarget:
    def __init__(self, category=None):
        self.category = category

    def __call__(self, model_output):
        # model_output 现在是 Wrapper 传来的 [num_classes, num_boxes]
        if self.category is not None:
            # 只取指定类别的总分
            return model_output[self.category, :].sum()
        else:
            # 自动取当前最强类别的分数
            return model_output.max(dim=0)[0].sum()


class YOLO11TargetWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, x):
        result = self.model(x)
        if isinstance(result, (list, tuple)):
            result = result[0]

            # result 形状: [1, 4 + num_classes, num_boxes]
        # 1. 过滤掉坐标信息，只保留类别得分
        # 2. 应用 Sigmoid 确保得分在 0-1 之间，增强梯度稳定性
        output = result[0, 4:, :].sigmoid()

        # 3. 阈值过滤：将得分极低的框设为0，防止背景杂色干扰
        # 只有得分 > 0.25 的框才参与热力图计算
        output = torch.where(output > 0.4, output, torch.zeros_like(output))

        return output


def main():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    yolo = YOLO(MODEL_PATH)
    base_model = yolo.model.to(device).eval()

    wrapped_model = YOLO11TargetWrapper(base_model)

    # 获取目标层
    target_layers = [base_model.model[TARGET_LAYER_INDEX]]

    # 4. 使用 Grad-CAM
    cam = GradCAM(model=wrapped_model, target_layers=target_layers)

    img_list = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for img_name in img_list:
        img_path = os.path.join(IMG_DIR, img_name)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None: continue

        # 预处理
        img_input = cv2.resize(img_bgr, (640, 640))
        img_rgb = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
        img_float = np.float32(img_rgb) / 255

        input_tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0).to(device)
        input_tensor.requires_grad_(True)

        try:
            # 开启 aug_smooth 会进行多轮推理合并，极大提升热力图的集中度
            # 虽然速度慢，但对于“缺陷定位”效果最好
            grayscale_cam = cam(input_tensor=input_tensor,
                                targets=[AdvancedTarget(TARGET_CATEGORY)],
                                eigen_smooth=True,
                                aug_smooth=True)

            grayscale_cam = grayscale_cam[0, :]

            # 生成热力图
            cam_image = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)
            res_img = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

            cv2.imwrite(os.path.join(SAVE_DIR, f"pro_cam_{img_name}"), res_img)
            print(f"热力图已生成: {img_name}")

        except Exception as e:
            print(f"错误 {img_name}: {str(e)}")


if __name__ == "__main__":
    main()