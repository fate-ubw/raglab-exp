import matplotlib.pyplot as plt

# 数据集列表
datasets = ['PopQA', 'TQA', 'HopPopQA', 'WikiMultiHopQA', 'ARC', 'MMLU', 'Pub', 'StreategyQA']

# 算法和模型的准确率列表

selfrag_llama3_8B_always_retrieval = [0.38, 0.618, 0.23, 0.186, 0.586, 0.392, 0.678, 0.48]
selfrag_llama3_8B_adaptive_retrieval = [0.356, 0.564, 0.21, 0.204, 0.58, 0.392, 0.678, 0.464]
selfrag_llama3_8b_no_retrieval = [0.148, 0.314, 0.112, 0.21, 0.58, 0.396, 0.684, 0.472]

selfrag_llama3_70B_always_retrieval = [0.452, 0.776, 0.406, 0.38, 0.894, 0.728, 0.794, 0.68]
selfrag_llama3_70B_adaptive_retrieval = [0.488, 0.774, 0.406, 0.382, 0.9, 0.724, 0.794, 0.68]
selfrag_llama3_70b_no_retrieval = [0.3, 0.766, 0.308, 0.31, 0.9, 0.726, 0.804, 0.694]




# 绘制折线图
plt.figure(figsize=(14, 10))

# Llama3_8b_baseline 的算法颜色为灰色

plt.plot(datasets, selfrag_llama3_8B_always_retrieval, marker='>', color='gray', label='selfrag-llama3_8B-always_retrieval')
plt.plot(datasets, selfrag_llama3_8B_adaptive_retrieval, marker='1', color='gray', label='selfrag-llama3_8B-adaptive_retrieval')
plt.plot(datasets, selfrag_llama3_8b_no_retrieval, marker='2', color='gray', label='selfrag-llama3_8b-no_retrieval')

# Llama3_70b_baseline 算法的颜色为蓝色

plt.plot(datasets, selfrag_llama3_70B_always_retrieval, marker='>', color='blue', label='selfrag-llama3_70B-always_retrieval')
plt.plot(datasets, selfrag_llama3_70B_adaptive_retrieval, marker='1', color='blue', label='selfrag-llama3_70B-adaptive_retrieval')
plt.plot(datasets, selfrag_llama3_70b_no_retrieval, marker='2', color='blue', label='selfrag-llama3_70b-no_retrieval')

# Chat_Direct 开头的算法调整为红色


# 设置图表标题和轴标签
plt.title('Performance Comparison of Different Algorithms and Models')
plt.xlabel('Datasets')
plt.ylabel('Accuracy')

# 设置x轴刻度
plt.xticks(rotation=45)

# 显示图例
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# 显示图表
plt.tight_layout()
plt.show()