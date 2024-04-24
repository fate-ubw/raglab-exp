import json

# 打开输入和输出文件
with open('/workspace/train_data/full_output_1005.jsonl', 'r') as f_in, open('/workspace/train_data/full_output_1005-10samples.jsonl', 'w') as f_out:
    # 迭代输入文件的每一行
    for i, line in enumerate(f_in):
        # 将JSON行解析为Python字典
        data = json.loads(line)
        
        # 将字典写入输出文件,转换为JSON字符串
        f_out.write(json.dumps(data) + '\n')
        
        # 如果已经写入了20条数据,则退出循环
        if i == 49:
            break