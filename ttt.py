def check_empty_lines(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    empty_lines = [i + 1 for i, line in enumerate(lines) if not line.strip()]  # 获取空行的行号

    if empty_lines:
        print(f"空行出现在以下行号: {empty_lines}")
    else:
        print("没有空行。")

# 使用示例
file_path = '/home/hlh/datasets/nlpdata/result903.txt'
check_empty_lines(file_path)