import pickle
import multiprocessing
from python_structured import *
from sqlang_structured import *

def multipro_python_query(data_list):
    return [python_query_parse(line) for line in data_list]

def multipro_python_code(data_list):
    return [python_code_parse(line) for line in data_list]

def multipro_python_context(data_list):
    result = []
    for line in data_list:
        if line == '-10000':
            result.append(['-10000'])
        else:
            result.append(python_context_parse(line))
    return result

def multipro_sqlang_query(data_list):
    return [sqlang_query_parse(line) for line in data_list]

def multipro_sqlang_code(data_list):
    return [sqlang_code_parse(line) for line in data_list]

def multipro_sqlang_context(data_list):
    result = []
    for line in data_list:
        if line == '-10000':
            result.append(['-10000'])
        else:
            result.append(sqlang_context_parse(line))
    return result

def parse(data_list, split_num, context_func, query_func, code_func):
    pool = multiprocessing.Pool()
    split_list = [data_list[i:i + split_num] for i in range(0, len(data_list), split_num)]
    results = pool.map(context_func, split_list)
    context_data = [item for sublist in results for item in sublist]
    print(f'context条数：{len(context_data)}')

    results = pool.map(query_func, split_list)
    query_data = [item for sublist in results for item in sublist]
    print(f'query条数：{len(query_data)}')

    results = pool.map(code_func, split_list)
    code_data = [item for sublist in results for item in sublist]
    print(f'code条数：{len(code_data)}')

    pool.close()
    pool.join()

    return context_data, query_data, code_data

def main(lang_type, split_num, source_path, save_path, context_func, query_func, code_func):
    with open(source_path, 'rb') as f:
        corpus_lis = pickle.load(f)

    context_data, query_data, code_data = parse(corpus_lis, split_num, context_func, query_func, code_func)
    qids = [item[0] for item in corpus_lis]

    total_data = [[qids[i], context_data[i], code_data[i], query_data[i]] for i in range(len(qids))]

    with open(save_path, 'wb') as f:
        pickle.dump(total_data, f)

if __name__ == '__main__':
    staqc_python_path = '.ulabel_data/python_staqc_qid2index_blocks_unlabeled.txt'
    staqc_python_save = '../hnn_process/ulabel_data/staqc/python_staqc_unlabled_data.pkl'

    staqc_sql_path = './ulabel_data/sql_staqc_qid2index_blocks_unlabeled.txt'
    staqc_sql_save = './ulabel_data/staqc/sql_staqc_unlabled_data.pkl'

    main(python_type, split_num, staqc_python_path, staqc_python_save, multipro_python_context, multipro_python_query, multipro_python_code)
    main(sqlang_type, split_num, staqc_sql_path, staqc_sql_save, multipro_sqlang_context, multipro_sqlang_query, multipro_sqlang_code)

    large_python_path = './ulabel_data/large_corpus/multiple/python_large_multiple.pickle'
    large_python_save = '../hnn_process/ulabel_data/large_corpus/multiple/python_large_multiple_unlable.pkl'

    large_sql_path = './ulabel_data/large_corpus/multiple/sql_large_multiple.pickle'
    large_sql_save = './ulabel_data/large_corpus/multiple/sql_large_multiple_unlable.pkl'

    main(python_type, split_num, large_python_path, large_python_save, multipro_python_context, multipro_python_query, multipro_python_code)
    main(sqlang_type, split_num, large_sql_path, large_sql_save, multipro_sqlang_context, multipro_sqlang_query, multipro_sqlang_code)


import pickle
import multiprocessing
from python_structured import *
from sqlang_structured import *

# 定义多进程函数
def multipro_python_query(data_list):
    # 对 Python 语言进行查询处理
    return [python_query_parse(line) for line in data_list]

def multipro_python_code(data_list):
    # 对 Python 语言进行代码处理
    return [python_code_parse(line) for line in data_list]

def multipro_python_context(data_list):
    # 对 Python 语言进行上下文处理
    result = []
    for line in data_list:
        if line == '-10000':
            result.append(['-10000'])
        else:
            result.append(python_context_parse(line))
    return result

def multipro_sqlang_query(data_list):
    # 对 SQLang 语言进行查询处理
    return [sqlang_query_parse(line) for line in data_list]

def multipro_sqlang_code(data_list):
    # 对 SQLang 语言进行代码处理
    return [sqlang_code_parse(line) for line in data_list]

def multipro_sqlang_context(data_list):
    # 对 SQLang 语言进行上下文处理
    result = []
    for line in data_list:
        if line == '-10000':
            result.append(['-10000'])
        else:
            result.append(sqlang_context_parse(line))
    return result

# 定义并行处理函数
def parse(data_list, split_num, context_func, query_func, code_func):
    # 创建多进程池
    pool = multiprocessing.Pool()
    # 将数据分割成小块
    split_list = [data_list[i:i + split_num] for i in range(0, len(data_list), split_num)]
    # 并行处理数据
    results = pool.map(context_func, split_list)
    context_data = [item for sublist in results for item in sublist]
    print(f'context条数：{len(context_data)}')

    results = pool.map(query_func, split_list)
    query_data = [item for sublist in results for item in sublist]
    print(f'query条数：{len(query_data)}')

    results = pool.map(code_func, split_list)
    code_data = [item for sublist in results for item in sublist]
    print(f'code条数：{len(code_data)}')

    # 关闭多进程池
    pool.close()
    pool.join()

    return context_data, query_data, code_data

# 定义主函数
def main(lang_type, split_num, source_path, save_path, context_func, query_func, code_func):
    # 读取源数据
    with open(source_path, 'rb') as f:
        corpus_lis = pickle.load(f)

    # 并行处理数据
    context_data, query_data, code_data = parse(corpus_lis, split_num, context_func, query_func, code_func)
    qids = [item[0] for item in corpus_lis]

    # 组合处理结果
    total_data = [[qids[i], context_data[i], code_data[i], query_data[i]] for i in range(len(qids))]

    # 保存处理结果
    with open(save_path, 'wb') as f:
        pickle.dump(total_data, f)

if __name__ == '__main__':

    # 设置数据路径
    staqc_python_path = '.ulabel_data/python_staqc_qid2index_blocks_unlabeled.txt'
    staqc_python_save = '../hnn_process/ulabel_data/staqc/python_staqc_unlabled_data.pkl'

    staqc_sql_path = './ulabel_data/sql_staqc_qid2index_blocks_unlabeled.txt'
    staqc_sql_save = './ulabel_data/staqc/sql_staqc_unlabled_data.pkl'

    # 运行主函数
    main(python_type, split_num, staqc_python_path, staqc_python_save, multipro_python_context, multipro_python_query, multipro_python_code)
    main(sqlang_type, split_num, staqc_sql_path, staqc_sql_save, multipro_sqlang_context, multipro_sqlang_query, multipro_sqlang_code)

    # 设置大规模数据路径
    large_python_path = './ulabel_data/large_corpus/multiple/python_large_multiple.pickle'
    large_python_save = '../hnn_process/ulabel_data/large_corpus/multiple/python_large_multiple_unlable.pkl'

    large_sql_path = './ulabel_data/large_corpus/multiple/sql_large_multiple.pickle'
    large_sql_save = './ulabel_data/large_corpus/multiple/sql_large_multiple_unlable.pkl'

    main(python_type, split_num, large_python_path, large_python_save, multipro_python_context, multipro_python_query,
         multipro_python_code)
    main(sqlang_type, split_num, large_sql_path, large_sql_save, multipro_sqlang_context, multipro_sqlang_query,
         multipro_sqlang_code)
