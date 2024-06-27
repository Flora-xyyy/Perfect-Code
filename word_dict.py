import pickle


def get_vocab(corpus1, corpus2):
    """
        从两个语料库构建词汇表。

        Args:
        - corpus1 (list): 第一个语料库，包含特定格式的数据。
        - corpus2 (list): 第二个语料库，包含特定格式的数据。

        Returns:
        - word_vocab (set): 从两个语料库中提取的唯一词汇集合。
        """
    word_vocab = set()
    for corpus in [corpus1, corpus2]:
        for i in range(len(corpus)):
            # 假设每个语料库项的结构为元组或列表，其中：
            # corpus[i][1][0]、corpus[i][1][1]、corpus[i][2][0]、corpus[i][3] 包含单词的可迭代对象。
            word_vocab.update(corpus[i][1][0])  # 更新词汇表，添加 corpus[i][1][0] 中的单词
            word_vocab.update(corpus[i][1][1])  # 更新词汇表，添加 corpus[i][1][1] 中的单词
            word_vocab.update(corpus[i][2][0])  # 更新词汇表，添加 corpus[i][2][0] 中的单词
            word_vocab.update(corpus[i][3])  # 更新词汇表，添加 corpus[i][3] 中的单词
    print(len(word_vocab))
    return word_vocab


def load_pickle(filename):
    """
        加载 pickle 文件中的数据。

        Args:
        - filename (str): pickle 文件的路径。

        Returns:
        - data (object): 从 pickle 文件中加载的数据。
        """
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def vocab_processing(filepath1, filepath2, save_path):
    """
        处理两个文本文件中的词汇，并将结果保存到另一个文件中。

        Args:
        - filepath1 (str): 第一个文本文件的路径。
        - filepath2 (str): 第二个文本文件的路径。
        - save_path (str): 保存最终词汇集合的文件路径。

        这个函数从文本文件中读取数据，计算词汇表，根据另一个集合排除某些词汇，并将最终的词汇集合保存到指定文件中。
        """
    with open(filepath1, 'r') as f:
        total_data1 = set(eval(f.read()))  ## 读取并评估 filepath1 中的数据
    with open(filepath2, 'r') as f:
        total_data2 = eval(f.read())     ## 读取并评估 filepath2 中的数据

    #从total_data1 和total_data2获取词汇表
    word_set = get_vocab(total_data2, total_data2)

    # 排除在 total_data1 中存在的词汇
    excluded_words = total_data1.intersection(word_set)
    word_set = word_set - excluded_words

    print(len(total_data1))
    print(len(word_set))

    # 将 word_set 保存到 save_path 中
    with open(save_path, 'w') as f:
        f.write(str(word_set))


if __name__ == "__main__":
    python_hnn = './data/python_hnn_data_teacher.txt'
    python_staqc = './data/staqc/python_staqc_data.txt'
    python_word_dict = './data/word_dict/python_word_vocab_dict.txt'

    sql_hnn = './data/sql_hnn_data_teacher.txt'
    sql_staqc = './data/staqc/sql_staqc_data.txt'
    sql_word_dict = './data/word_dict/sql_word_vocab_dict.txt'

    new_sql_staqc = './ulabel_data/staqc/sql_staqc_unlabled_data.txt'
    new_sql_large = './ulabel_data/large_corpus/multiple/sql_large_multiple_unlable.txt'
    large_word_dict_sql = './ulabel_data/sql_word_dict.txt'

    final_vocab_processing(sql_word_dict, new_sql_large, large_word_dict_sql)
