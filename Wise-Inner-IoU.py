import torch
import torch.nn as nn


# -----------------------------
# IoU计算（标准）
# -----------------------------
def bbox_iou(box1, box2, eps=1e-7):
    """
    box: [x1, y1, x2, y2]
    """
    # 交集
    inter_x1 = torch.max(box1[..., 0], box2[..., 0])
    inter_y1 = torch.max(box1[..., 1], box2[..., 1])
    inter_x2 = torch.min(box1[..., 2], box2[..., 2])
    inter_y2 = torch.min(box1[..., 3], box2[..., 3])

    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

    # 面积
    area1 = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
    area2 = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])

    union = area1 + area2 - inter + eps

    return inter / union


# -----------------------------
# Inner Box（核心创新）
# -----------------------------
def inner_box(box, ratio=0.5):
    """
    收缩box，强调内部区域
    """
    x1, y1, x2, y2 = box.unbind(-1)

    w = (x2 - x1) * ratio
    h = (y2 - y1) * ratio

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    inner_x1 = cx - w / 2
    inner_y1 = cy - h / 2
    inner_x2 = cx + w / 2
    inner_y2 = cy + h / 2

    return torch.stack([inner_x1, inner_y1, inner_x2, inner_y2], dim=-1)


# -----------------------------
# Wise权重（质量感知）
# -----------------------------
def wise_weight(iou, gamma=2.0):
    """
    高质量样本权重更高 / 或难样本增强
    """
    return torch.pow(1 - iou, gamma)


# -----------------------------
# Wise-Inner-IoU Loss
# -----------------------------
class WiseInnerIoULoss(nn.Module):
    def __init__(self, inner_ratio=0.5, gamma=2.0):
        super().__init__()
        self.inner_ratio = inner_ratio
        self.gamma = gamma

    def forward(self, pred, target):
        """
        pred, target: [B, 4]  -> xyxy
        """

        # -----------------------
        # 1. 标准IoU
        # -----------------------
        iou = bbox_iou(pred, target)

        # -----------------------
        # 2. Inner IoU（关键创新）
        # -----------------------
        pred_inner = inner_box(pred, self.inner_ratio)
        target_inner = inner_box(target, self.inner_ratio)

        inner_iou = bbox_iou(pred_inner, target_inner)

        # -----------------------
        # 3. Wise权重（动态质量）
        # -----------------------
        weight = wise_weight(iou, self.gamma)

        # -----------------------
        # 4. 最终Loss
        # -----------------------
        loss = weight * (1 - 0.5 * (iou + inner_iou))

        return loss.mean()