import os
import glob
from datetime import datetime


def batch_convert_labels_to_zero(input_path, backup=True):
    """
    将指定路径下所有YOLO格式标签文件的类别索引改为0并保存到原位置

    Args:
        input_path: 包含txt文件的目录路径
        backup: 是否备份原文件
    """
    print("=" * 70)
    print("批量修改YOLO标签文件索引为0")
    print(f"处理目录: {input_path}")
    print("=" * 70)

    # 检查目录是否存在
    if not os.path.exists(input_path):
        print(f"❌ 错误：目录不存在 - {input_path}")
        return False

    # 获取所有txt文件
    txt_files = glob.glob(os.path.join(input_path, "*.txt"))

    if not txt_files:
        print("❌ 错误：目录中没有找到txt文件")
        return False

    print(f"📁 找到 {len(txt_files)} 个txt文件")

    # 备份目录（可选）
    backup_path = None
    if backup:
        backup_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(os.path.dirname(input_path), f"backup_{backup_time}")
        os.makedirs(backup_path, exist_ok=True)
        print(f"📂 创建备份目录: {backup_path}")

    # 处理统计
    processed_files = 0
    processed_lines = 0
    skipped_files = 0

    # 处理每个txt文件
    for i, txt_file in enumerate(txt_files):
        filename = os.path.basename(txt_file)

        try:
            # 读取文件内容
            with open(txt_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            if not lines:
                skipped_files += 1
                continue

            # 备份文件（如果需要）
            if backup and backup_path:
                backup_file = os.path.join(backup_path, filename)
                with open(backup_file, 'w', encoding='utf-8') as f:
                    f.writelines(lines)

            # 处理每一行
            new_lines = []
            modified_lines = 0

            for line in lines:
                line = line.strip()
                if not line:  # 跳过空行
                    new_lines.append("")
                    continue

                parts = line.split()

                # 检查是否为YOLO格式（至少5个值：class x_center y_center width height）
                if len(parts) >= 5:
                    # 检查当前类别索引是否为0
                    current_class = parts[0]

                    # 如果已经是0，则不需要修改
                    if current_class == "0":
                        new_lines.append(line)
                    else:
                        # 修改类别索引为0
                        parts[0] = "0"
                        new_line = " ".join(parts)
                        new_lines.append(new_line)
                        modified_lines += 1
                else:
                    # 如果不是标准YOLO格式，保留原样
                    new_lines.append(line)

            # 如果有修改，写回原文件
            if modified_lines > 0:
                with open(txt_file, 'w', encoding='utf-8') as f:
                    f.write("\n".join(new_lines))

                processed_files += 1
                processed_lines += modified_lines

            # 显示进度
            if (i + 1) % 10 == 0 or (i + 1) == len(txt_files):
                print(f"🔄 进度: {i + 1}/{len(txt_files)} - {filename} ({modified_lines}行已修改)")

        except Exception as e:
            print(f"❌ 处理文件失败 {filename}: {e}")
            skipped_files += 1

    # 显示统计结果
    print("\n" + "=" * 70)
    print("✅ 处理完成！")
    print("=" * 70)
    print("📊 统计信息:")
    print(f"   总文件数: {len(txt_files)}")
    print(f"   已修改文件: {processed_files}")
    print(f"   已修改标注行: {processed_lines}")
    print(f"   跳过文件: {skipped_files}")

    if backup_path:
        print(f"\n📂 备份位置: {backup_path}")
        print("   原文件已备份，如需恢复可复制回原目录")

    return True


def verify_sample_files(input_path, sample_count=5):
    """随机抽样验证修改结果"""
    print("\n🔍 抽样验证修改结果:")
    print("-" * 50)

    txt_files = glob.glob(os.path.join(input_path, "*.txt"))

    if not txt_files:
        print("没有找到txt文件")
        return

    # 取前几个文件作为样本
    sample_files = txt_files[:min(sample_count, len(txt_files))]

    for i, txt_file in enumerate(sample_files):
        filename = os.path.basename(txt_file)
        print(f"\n[{i + 1}] {filename}:")

        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            if not lines:
                print("   (空文件)")
                continue

            # 显示前3行（如果有）
            for j, line in enumerate(lines[:3]):
                if line.strip():  # 非空行
                    print(f"   第{j + 1}行: {line.strip()}")
        except Exception as e:
            print(f"   读取失败: {e}")

    print("-" * 50)


def check_class_distribution(input_path):
    """检查修改后的类别分布"""
    print("\n📈 类别分布检查:")
    print("-" * 50)

    txt_files = glob.glob(os.path.join(input_path, "*.txt"))

    if not txt_files:
        print("没有找到txt文件")
        return

    class_counts = {}
    total_objects = 0

    for txt_file in txt_files[:50]:  # 只检查前50个文件加快速度
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if parts:
                    class_id = parts[0]
                    class_counts[class_id] = class_counts.get(class_id, 0) + 1
                    total_objects += 1
        except:
            continue

    # 打印结果
    if total_objects == 0:
        print("未找到标注对象")
        return

    print(f"检查文件数: {min(50, len(txt_files))}")
    print(f"总标注对象: {total_objects}")
    print("\n类别分布:")
    for class_id in sorted(class_counts.keys()):
        count = class_counts[class_id]
        percentage = (count / total_objects) * 100
        print(f"  类别 {class_id}: {count} 个 ({percentage:.1f}%)")

    # 检查是否有非0类别
    non_zero_classes = [c for c in class_counts.keys() if c != "0"]
    if non_zero_classes:
        print(f"\n⚠️  警告：发现非0类别: {', '.join(non_zero_classes)}")
        print("  可能需要重新运行转换脚本")
    else:
        print(f"\n✅ 所有标注对象类别均为0")


def main():
    """主函数"""
    # ================== 配置部分 ==================
    # 请修改这里的路径为你的实际路径
    LABELS_DIR = r"D:\桌面\battery_defect\labels\train"

    # 是否创建备份（建议设为True，以防出错）
    CREATE_BACKUP = True

    # ================== 执行部分 ==================
    print("🚀 YOLO标签批量转换工具")
    print("功能：将所有txt标签文件的类别索引改为0")
    print(f"目标目录: {LABELS_DIR}")
    print()

    # 1. 确认操作
    confirm = input("⚠️  确认要修改此目录下所有txt文件吗？(y/n): ")
    if confirm.lower() != 'y':
        print("操作已取消")
        return

    # 2. 执行批量转换
    success = batch_convert_labels_to_zero(LABELS_DIR, backup=CREATE_BACKUP)

    if not success:
        return

    # 3. 验证修改结果
    verify_sample_files(LABELS_DIR, sample_count=5)

    # 4. 检查类别分布
    check_class_distribution(LABELS_DIR)

    # 5. 完成提示
    print("\n" + "=" * 70)
    print("🎉 操作完成！")
    print("=" * 70)
    print("\n📋 使用说明:")
    print("1. 现在所有标签文件的类别索引已改为0")
    print("2. 可以开始训练scratch缺陷检测模型")
    print("3. 如需恢复原文件，请从备份目录复制回原位置")
    print("\n🔧 下一步:")
    print("运行训练命令: python train_battery_defect.py")

    # 等待用户按Enter键退出
    input("\n按Enter键退出...")


if __name__ == "__main__":
    main()