import matplotlib.pyplot as plt

# 数据集列表
datasets = ['PopQA', 'TQA', 'HopPopQA', 'WikiMultiHopQA', 'ARC', 'MMLU', 'Pub', 'StreategyQA']

# 算法和模型的准确率列表
llama3_8b_baseline_Direct = [0.256, 0.68, 0.206, 0.248, 0.774, 0.58, 0.76, 0.564]
llama3_8b_baseline_NaiveRag = [0.388, 0.648, 0.282, 0.18, 0.692, 0.508, 0.662, 0.582]
llama3_8b_baseline_query_rewrite = [0.384, 0.584, 0.216, 0.214, 0.696, 0.5, 0.654, 0.576]
llama3_baseline_Iter_Gen = [0.348, 0.654, 0.288, 0.198, 0.688, 0.526, 0.442, 0.57]
Llama3_8b_baseline_Active_Rag = [0.346, 0.642, 0.288, 0.16, 0.662, 0.52, 0.418, 0.568]
Llama3_8b_baseline_Self_Ask = [0.118, 0.356, 0.162, 0.152, 0.556, 0.454, 0.376, 0.504]
selfrag_llama3_8B_always_retrieval = [0.38, 0.618, 0.23, 0.186, 0.586, 0.392, 0.678, 0.48]
selfrag_llama3_8B_adaptive_retrieval = [0.356, 0.564, 0.21, 0.204, 0.58, 0.392, 0.678, 0.464]
selfrag_llama3_8b_no_retrieval = [0.148, 0.314, 0.112, 0.21, 0.58, 0.396, 0.684, 0.472]


Chat_Direct = [0.266, 0.77, 0.338, 0.38, 0.796, 0.636, 0.78, 0.682]
Chat_NaiveRag = [0.45, 0.728, 0.416, 0.332, 0.674, 0.544, 0.538, 0.618]
Chat_Query_Rewrite_Rag = [0.462, 0.736, 0.372, 0.33, 0.684, 0.544, 0.548, 0.636]
Chatgpt_Iter_Gen = [0.442, 0.73, 0.448, 0.346, 0.698, 0.55, 0.392, 0.562]
chatpgt_Active_Rag = [0.442, 0.728, 0.438, 0.342, 0.7, 0.558, 0.5, 0.612]
chatpgt_Self_Ask = [0.382, 0.686, 0.364, 0.416, 0.634, 0.486, 0.452, 0.396]


llama3_70b_baseline_Direct = [0.256, 0.764, 0.278, 0.282, 0.904, 0.734, 0.772, 0.706]
Llama3_70b_baseline_NaiveRag = [0.396, 0.736, 0.338, 0.282, 0.894, 0.706, 0.752, 0.636]
Llama3_70b_baseline_query_rewrite = [0.39, 0.728, 0.314, 0.276, 0.884, 0.726, 0.744, 0.626]
Llama3_70b_baseline_Iter_Gen = [0.362, 0.744, 0.336, 0.264, 0.894, 0.724, 0.626, 0.592]

Llama3_70b_baseline_Active_Rag = [0.37, 0.736, 0.332, 0.266, 0.892, 0.718, 0.58, 0.61]

Llama3_70b_baseline_Self_Ask = [0.208, 0.658, 0.334, 0.35, 0.804, 0.674, 0.604, 0.496]

selfrag_llama3_70B_always_retrieval = [0.452, 0.776, 0.406, 0.38, 0.894, 0.728, 0.794, 0.68]
selfrag_llama3_70B_adaptive_retrieval = [0.488, 0.774, 0.406, 0.382, 0.9, 0.724, 0.794, 0.68]
selfrag_llama3_70b_no_retrieval = [0.3, 0.766, 0.308, 0.31, 0.9, 0.726, 0.804, 0.694]




# 绘制折线图
plt.figure(figsize=(14, 10))

# Llama3_8b_baseline 的算法颜色为灰色
plt.plot(datasets, llama3_8b_baseline_Direct, marker='o', color='gray', label='llama3-8b-baseline Direct')
plt.plot(datasets, llama3_8b_baseline_NaiveRag, marker='s', color='gray', label='llama3-8b-baseline NaiveRag')
plt.plot(datasets, llama3_8b_baseline_query_rewrite, marker='^', color='gray', label='llama3-8b-baseline query rewrite')
plt.plot(datasets, llama3_baseline_Iter_Gen, marker='D', color='gray', label='llama3-baseline Iter-Gen')
plt.plot(datasets, Llama3_8b_baseline_Active_Rag, marker='v', color='gray', label='Llama3-8b-baseline Active Rag')
plt.plot(datasets, Llama3_8b_baseline_Self_Ask, marker='<', color='gray', label='Llama3-8b-baseline Self Ask')
plt.plot(datasets, selfrag_llama3_8B_always_retrieval, marker='>', color='gray', label='selfrag-llama3_8B-always_retrieval')
plt.plot(datasets, selfrag_llama3_8B_adaptive_retrieval, marker='1', color='gray', label='selfrag-llama3_8B-adaptive_retrieval')
plt.plot(datasets, selfrag_llama3_8b_no_retrieval, marker='2', color='gray', label='selfrag-llama3_8b-no_retrieval')

# Llama3_70b_baseline 算法的颜色为蓝色
plt.plot(datasets, llama3_70b_baseline_Direct, marker='o', color='blue', label='llama3-70b-baseline Direct')
plt.plot(datasets, Llama3_70b_baseline_NaiveRag, marker='s', color='blue', label='Llama3-70b-baseline NaiveRag')
plt.plot(datasets, Llama3_70b_baseline_query_rewrite, marker='^', color='blue', label='Llama3-70b-baseline query rewrite')
plt.plot(datasets, Llama3_70b_baseline_Iter_Gen, marker='D', color='blue', label='Llama3-70b-baseline Iter-Gen')
plt.plot(datasets, Llama3_70b_baseline_Active_Rag, marker='v', color='blue', label='Llama3-70b-baseline Active Rag')
plt.plot(datasets, Llama3_70b_baseline_Self_Ask, marker='<', color='blue', label='Llama3-70b-baseline Self Ask')
plt.plot(datasets, selfrag_llama3_70B_always_retrieval, marker='>', color='blue', label='selfrag-llama3_70B-always_retrieval')
plt.plot(datasets, selfrag_llama3_70B_adaptive_retrieval, marker='1', color='blue', label='selfrag-llama3_70B-adaptive_retrieval')
plt.plot(datasets, selfrag_llama3_70b_no_retrieval, marker='2', color='blue', label='selfrag-llama3_70b-no_retrieval')

# Chat_Direct 开头的算法调整为红色
plt.plot(datasets, Chat_Direct, marker='o', color='red', label='Chat_Direct')
plt.plot(datasets, Chat_NaiveRag, marker='s', color='red', label='Chat_Direct-NaiveRag')
plt.plot(datasets, Chat_Query_Rewrite_Rag, marker='^', color='red', label='Chat_Direct-Query Rewrite Rag')
plt.plot(datasets, Chatgpt_Iter_Gen, marker='D', color='red', label='Chatgpt-Iter-Gen')
plt.plot(datasets, chatpgt_Active_Rag, marker='v', color='red', label='chatpgt-Active Rag')
plt.plot(datasets, chatpgt_Self_Ask, marker='<', color='red', label='chatpgt-Self Ask')

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