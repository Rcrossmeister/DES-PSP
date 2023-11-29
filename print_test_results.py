import os
import json
from collections import defaultdict

# 主目录路径，请将此路径替换为你的实际目录路径
main_directory_1 = "results/baseline88"
# main_directorys = ["results/baseline88_new_data", "results/des_psp88_new_data"]
# main_directorys = ["results/baseline88", "results/des_psp88"]
main_directorys = ["results/des_psp_a_hy", "results/des_psp_l_hy"]


# 创建一个字典来根据target存储metrics
target_metrics = defaultdict(lambda: defaultdict(dict))

# 存储所有模型名称
model_names = set()

# 定义每个target对应的指标列表
metrics_by_target = {
    "price": ["MSE", "RMSE", "MAE", "ADE", "FDE"],
    "movement": ["Accuracy", "F1", "MCC"],
    "target": ["Accuracy", "F1", "MCC"]
}

# 遍历所有子目录，读取metrics.json文件
for main_directory in main_directorys:
    for root, dirs, files in os.walk(main_directory):
        for name in files:
            if name == 'metrics.json':
                file_path = os.path.join(root, name)
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    target = data.get("target")
                    model_name = data.get("Model")
                    metrics = data.get("Metrics")
                    # 添加模型名称
                    model_names.add(model_name)
                    # 仅添加目标指标集合中存在的指标
                    target_metrics[target][model_name] = {
                        k: metrics[k] for k in metrics_by_target[target] if k in metrics
                    }

# 将模型名称排序
sorted_model_names = sorted(model_names)

# 格式化指标值，保留两位小数
def format_metric(value, decimal_places=2):
    try:
        return f"{float(value):.{decimal_places}f}"
    except (ValueError, TypeError):
        return value

# 打印表格的函数
def print_table(target, metrics_dict, metrics_keys, model_order):
    print(f"Target: {target}")
    headers = ["Model"] + metrics_keys
    header_row = "| " + f"{headers[0]:<16}" + " | " + " | ".join(f"{h:<10}" for h in headers[1:]) + " |"
    print(header_row)
    separator_row = "|-" + "-" * 17 + "|" + "-|".join("-" * 11 for _ in headers[1:]) + "|"
    print(separator_row)

    for model_name in model_order:
        metrics = metrics_dict.get(model_name, {})
        row = f"| {model_name:<16} "
        for key in metrics_keys:
            metric_value = format_metric(metrics.get(key, 'N/A'), 4)
            row += f"| {metric_value:<10} "
        row += "|"
        print(row)

# 对每个target的指标打印表格
for target, metrics in target_metrics.items():
    print_table(target, metrics, metrics_by_target[target], sorted_model_names)
    print("\n")  # 用于分隔不同target的空行
