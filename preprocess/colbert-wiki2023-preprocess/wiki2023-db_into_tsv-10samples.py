import sqlite3
import csv
import pdb
# 连接到SQLite数据库
conn = sqlite3.connect('/home/wyd/data/2-factscore/enwiki-20230401.db')
cursor = conn.cursor()
# 获取所有表名
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = [table[0] for table in cursor.fetchall()]
tsv_data = []
for table_name in tables: # 这里面应该只有 1 个表格
    cursor.execute(f"SELECT * FROM {table_name}")
    rows = cursor.fetchall() # 到这里才开始读取数据，
    columns = [desc[0] for desc in cursor.description] # ['title', 'text']
    samples_10 = rows[0:10]
    pdb.set_trace()
    for id,row in enumerate(samples_10): # 这里面开始处理即可 id \t title  \t text 
        title = row[0]
        text = row[1]
        id = str(id)
        data = id + '\t' + text + '\t' + title +'\n'#为什么程序会间隔这么大呢？？？属实是没看懂，存储的时候是不是有什么秘诀
        tsv_data.append(data)
# 创建TSV文件
tsv_file = "/home/wyd/data/2-factscore/enwiki-20230401.tsv"
with open(tsv_file, "w") as file:
    file.writelines(tsv_data)
    print(f"{table_name} exported to {tsv_file}")

cursor.close()
conn.close()
