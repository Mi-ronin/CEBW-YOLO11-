# check_image_file.py
import os
import cv2
from pathlib import Path


def diagnose_image_file(file_path):
    """诊断单个图像文件的问题"""
    print(f"\n=== 诊断文件: {file_path} ===")

    # 1. 检查文件是否存在
    if not os.path.exists(file_path):
        print("❌ 文件不存在")
        return False

    # 2. 检查文件大小
    file_size = os.path.getsize(file_path)
    print(f"文件大小: {file_size} 字节")
    if file_size == 0:
        print("❌ 文件大小为0，文件可能已损坏")
        return False

    # 3. 尝试用不同模式读取
    print("\n尝试不同读取模式:")

    # 模式1：常规读取（默认BGR）
    img1 = cv2.imread(file_path, cv2.IMREAD_COLOR)
    print(f"  IMREAD_COLOR: {'✅ 成功' if img1 is not None else '❌ 失败'}")

    # 模式2：灰度读取
    img2 = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    print(f"  IMREAD_GRAYSCALE: {'✅ 成功' if img2 is not None else '❌ 失败'}")

    # 模式3：任意深度
    img3 = cv2.imread(file_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    print(f"  IMREAD_ANYCOLOR: {'✅ 成功' if img3 is not None else '❌ 失败'}")

    # 4. 如果任何一种方式成功，显示基本信息
    successful_img = img1 if img1 is not None else (img2 if img2 is not None else img3)
    if successful_img is not None:
        print(f"\n✅ 文件可读取，形状: {successful_img.shape}")
        print(f"   数据类型: {successful_img.dtype}")
        print(f"   数值范围: [{successful_img.min()}, {successful_img.max()}]")
        return True
    else:
        print("\n❌ 所有读取方式都失败")

        # 5. 尝试用PIL读取（备用方法）
        try:
            from PIL import Image
            pil_img = Image.open(file_path)
            print(f"✅ PIL可以读取: 格式={pil_img.format}, 尺寸={pil_img.size}, 模式={pil_img.mode}")
            return True
        except Exception as e:
            print(f"❌ PIL也无法读取: {e}")

        return False


# 测试一个具体的文件（请修改为您的实际路径）
test_file = r'D:\桌面\battery_defect\images\train\241.jpg'
diagnose_image_file(test_file)