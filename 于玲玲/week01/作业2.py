from openai import OpenAI
import jieba
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier

# 机器学习分类
# 读取数据集，获得DataFrame对象
dataset = pd.read_csv("dataset.csv", sep="\t", header=None)
# print(dataset.head(5))
# print(len(dataset))
# 对数据集中中文文本内容分词,获得分词结果
jieba_result = dataset[0].apply(lambda x: " ".join(jieba.lcut(x)))
# print(jieba_result)
# 对文本提取特征 默认是使用标点符号分词
vector = CountVectorizer()
# 学习词表
vector.fit(jieba_result.values)
# 将文本转换为词频向量
input_feature = vector.transform(jieba_result.values)
# 查看转换结果维度
# print(input_feature)
# print(input_feature.shape)
# KNN模型训练
knn_model = KNeighborsClassifier()
knn_model.fit(input_feature, dataset[1].values)


# print(dataset[1].values)
def text_classify_using_ml(text: str) -> str:
    """
    文本分类（机器学习），输入文本完成类别划分
    :param text: 待分类文本
    :return: 文本分类名称
    """
    # print(dataset[1].values)
    return knn_model.predict(vector.transform([" ".join(jieba.lcut(text))]))[0]


def text_classify_using_llm(text: str) -> str:
    """
    文本分类（大语言模型），输入文本完成类别划分
    :param text: 待分类文本
    :return: 文本分类名称
    """
    # 大语言模型分类
    # 创建客户端连接云端模型
    client = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        # https://bailian.console.aliyun.com/?tab=model#/api-key
        api_key="sk-685d5da74c2047dbb7d80c0e80fcb05d",

        # 大模型厂商的地址
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    # 调用
    completion = client.chat.completions.create(
        # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        model="qwen-plus",

        messages=[
            {"role": "user", "content": f"帮我进行文本分类：{text}，分类范围是{dataset[1].values}，只输出分类结果"},
        ]
    )
    return completion.choices[0].message.content


if __name__ == '__main__':
    text1 = "导航去万寿宫"
    text2 = "想听一首周杰伦的歌"
    print("机器学习预测结果：")
    print("待预测文本：", text1, "预测结果：", text_classify_using_ml(text1))
    print("待预测文本：", text2, "预测结果：", text_classify_using_ml(text2))
    print("大语言模型预测结果：")
    print("待预测文本：", text1, "预测结果：", text_classify_using_llm(text1))
    print("待预测文本：", text2, "预测结果：", text_classify_using_llm(text2))
