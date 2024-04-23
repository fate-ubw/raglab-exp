import json
import re

# 读取JSON行文件
with open('full_output_1005.jsonl', 'r') as file:
    data = file.readlines()

# 提取output字段并去除特殊标记
processed_data = []

for line in data:
    json_data = json.loads(line.strip())
    output = json_data['output']
    
    # 去除特殊标记
    output_cleaned = re.sub(r'<[^>]*>', '', output)
    output_cleaned = re.sub(r'\[.*?\]', '', output_cleaned)
    
    # 添加到处理后的数据列表
    json_data['output_cleaned'] = output_cleaned.strip()
    processed_data.append(json_data)

# 保存处理后的数据
with open('processed_output_1005.jsonl', 'w') as file:
    for item in processed_data:
        json.dump(item, file)
        file.write('\n')

print('Data preprocessing completed.')
