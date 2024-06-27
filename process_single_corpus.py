import pickle
from collections import Counter


def load_pickle(filename):
    """
        加载 pickle 格式的文件并返回数据对象。

        Args:
        - filename (str): pickle 文件路径

        Returns:
        - data (object): 从 pickle 文件中加载的数据对象
        """
    with open(filename, 'rb') as f:
        data = pickle.load(f, encoding='iso-8859-1')
    return data


def split_data(total_data, qids):
    """
       根据 qids 中每个元素的出现次数，将 total_data 分成单一出现和多次出现的两部分数据。

       Args:
       - total_data (list): 包含数据的列表，每个数据包含一个或多个元素
       - qids (list): 包含与每个数据关联的qid的列表

       Returns:
       - total_data_single (list): 单一出现的数据列表
       - total_data_multiple (list): 多次出现的数据列表
       """
    result = Counter(qids)
    total_data_single = []
    total_data_multiple = []
    for data in total_data:
        if result[data[0][0]] == 1:
            total_data_single.append(data)
        else:
            total_data_multiple.append(data)
    return total_data_single, total_data_multiple


def data_staqc_processing(filepath, save_single_path, save_multiple_path):
    """
        从文件中加载数据，根据 qids 将数据分为单一出现和多次出现的两部分，
        然后分别保存到指定路径的文件中。

        Args:
        - filepath (str): 包含数据的文件路径
        - save_single_path (str): 保存单一出现数据的文件路径
        - save_multiple_path (str): 保存多次出现数据的文件路径
        """
    with open(filepath, 'r') as f:
        total_data = eval(f.read())
    qids = [data[0][0] for data in total_data]
    total_data_single, total_data_multiple = split_data(total_data, qids)

    with open(save_single_path, "w") as f:
        f.write(str(total_data_single))
    with open(save_multiple_path, "w") as f:
        f.write(str(total_data_multiple))


def data_large_processing(filepath, save_single_path, save_multiple_path):
    """
        加载大型 pickle 文件中的数据，根据 qids 将数据分为单一出现和多次出现的两部分，
        然后分别保存到指定路径的 pickle 文件中。

        Args:
        - filepath (str): 包含数据的大型 pickle 文件路径
        - save_single_path (str): 保存单一出现数据的 pickle 文件路径
        - save_multiple_path (str): 保存多次出现数据的 pickle 文件路径
        """
    total_data = load_pickle(filepath)  # 加载大型 pickle 文件中的数据
    qids = [data[0][0] for data in total_data] # 提取每个数据的 qid
    total_data_single, total_data_multiple = split_data(total_data, qids) # 分割数据

    # 将单一出现和多次出现的数据分别保存到 pickle 文件
    with open(save_single_path, 'wb') as f:
        pickle.dump(total_data_single, f)
    with open(save_multiple_path, 'wb') as f:
        pickle.dump(total_data_multiple, f)


def single_unlabeled_to_labeled(input_path, output_path):
    """
       加载 pickle 文件中的数据，并将每个数据标记为 1，然后按第一个元素和标签值排序，
       最后将排序后的结果写入到指定路径的文件中。

       Args:
       - input_path (str): 包含数据的 pickle 文件路径
       - output_path (str): 保存排序后数据的文件路径
       """
    total_data = load_pickle(input_path)## 加载 pickle 文件中的数据
    labels = [[data[0], 1] for data in total_data] # 为每个数据创建标签为 1
    total_data_sort = sorted(labels, key=lambda x: (x[0], x[1])) # 根据第一个元素和标签值排序

    # 将排序后的结果写入文件
    with open(output_path, "w") as f:
        f.write(str(total_data_sort))


if __name__ == "__main__":
    staqc_python_path = './ulabel_data/python_staqc_qid2index_blocks_unlabeled.txt'
    staqc_python_single_save = './ulabel_data/staqc/single/python_staqc_single.txt'
    staqc_python_multiple_save = './ulabel_data/staqc/multiple/python_staqc_multiple.txt'
    data_staqc_processing(staqc_python_path, staqc_python_single_save, staqc_python_multiple_save)

    staqc_sql_path = './ulabel_data/sql_staqc_qid2index_blocks_unlabeled.txt'
    staqc_sql_single_save = './ulabel_data/staqc/single/sql_staqc_single.txt'
    staqc_sql_multiple_save = './ulabel_data/staqc/multiple/sql_staqc_multiple.txt'
    data_staqc_processing(staqc_sql_path, staqc_sql_single_save, staqc_sql_multiple_save)

    large_python_path = './ulabel_data/python_codedb_qid2index_blocks_unlabeled.pickle'
    large_python_single_save = './ulabel_data/large_corpus/single/python_large_single.pickle'
    large_python_multiple_save = './ulabel_data/large_corpus/multiple/python_large_multiple.pickle'
    data_large_processing(large_python_path, large_python_single_save, large_python_multiple_save)

    large_sql_path = './ulabel_data/sql_codedb_qid2index_blocks_unlabeled.pickle'
    large_sql_single_save = './ulabel_data/large_corpus/single/sql_large_single.pickle'
    large_sql_multiple_save = './ulabel_data/large_corpus/multiple/sql_large_multiple.pickle'
    data_large_processing(large_sql_path, large_sql_single_save, large_sql_multiple_save)

    large_sql_single_label_save = './ulabel_data/large_corpus/single/sql_large_single_label.txt'
    large_python_single_label_save = './ulabel_data/large_corpus/single/python_large_single_label.txt'
    single_unlabeled_to_labeled(large_sql_single_save, large_sql_single_label_save)
    single_unlabeled_to_labeled(large_python_single_save, large_python_single_label_save)
