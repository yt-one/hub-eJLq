import os

# 设置环境变量
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import torch
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification

# 获取当前脚本所在目录
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# 构建训练好的模型路径（在homework目录下）
trained_model_path = os.path.join(current_script_dir, "anime_results", "checkpoint-500")

print(f"当前脚本目录: {current_script_dir}")
print(f"检查模型路径: {trained_model_path}")
print(f"模型路径是否存在: {os.path.exists(trained_model_path)}")

# 只使用训练好的模型
if not os.path.exists(trained_model_path):
    print("错误：训练好的模型不存在！")
    print("请先运行 bert_anime_classification.py 进行训练")
    exit(1)

print("加载训练好的模型...")
# 使用原始BERT模型的tokenizer
tokenizer_path = os.path.join(current_script_dir, "..", "..", "models", "google-bert", "bert-base-chinese")
print(f"Tokenizer路径: {tokenizer_path}")

try:
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    model = BertForSequenceClassification.from_pretrained(trained_model_path)
    print("成功加载训练好的模型！")
except Exception as e:
    print(f"加载模型失败: {e}")
    exit(1)

# 标签编码器
lbl = LabelEncoder()
lbl.fit(['国漫', '日漫', '美漫'])

def predict_anime_type(text):
    """预测文本属于哪个动漫类别"""
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_class = torch.argmax(outputs.logits, dim=-1).item()
    return lbl.inverse_transform([predicted_class])[0]

# 测试样本
print("\n=== 动漫分类预测测试 ===")
test_samples = [
    "虹猫蓝兔七侠传",
    "我想看火影忍者", 
    "播放蜘蛛侠电影",
    "查找海贼王漫画",
    "续播死神到最新话",
    "我想看变形金刚",
    "播放铠甲勇士",
    "查找龙珠超",
    "我想看喜羊羊与灰太狼",
    "播放钢铁侠3"
]

print("预测结果:")
correct_predictions = 0
for sample in test_samples:
    try:
        prediction = predict_anime_type(sample)
        # 简单的预期结果判断
        expected = ""
        if "虹猫" in sample or "铠甲" in sample or "喜羊羊" in sample:
            expected = "国漫"
        elif "火影" in sample or "海贼王" in sample or "死神" in sample or "龙珠" in sample:
            expected = "日漫"
        elif "蜘蛛侠" in sample or "变形金刚" in sample or "钢铁侠" in sample:
            expected = "美漫"
        
        status = "✓" if prediction == expected else "✗"
        print(f"{status} '{sample}' -> {prediction}")
        if prediction == expected:
            correct_predictions += 1
            
    except Exception as e:
        print(f"✗ '{sample}' -> 预测出错: {e}")

print(f"\n准确率: {correct_predictions}/{len(test_samples)} ({correct_predictions/len(test_samples)*100:.1f}%)")
