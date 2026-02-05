# 推理
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

if __name__ == "__main__":
    # # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # # 加载微调后的模型
    model = AutoModelForQuestionAnswering.from_pretrained("./bert-squad-final")

    # 创建 QA pipeline（最简单方式，避免自己实现后处理）
    qa_pipeline = pipeline(
        "question-answering",  # 这个名字不能随便命名，走问题回答的预测流程
        model=model,
        tokenizer=tokenizer,
        device=0  # 用 GPU
    )

    # 示例1
    context = """
  The University of Oxford is a collegiate research university in Oxford, England.
  There is evidence of teaching as early as 1096, making it the oldest university
  in the English-speaking world and the world's second-oldest university in continuous operation.
  """

    question = "What is the oldest university in the English-speaking world?"

    result = qa_pipeline(question=question, context=context)

    print(f"Answer: {result['answer']}")
    print(f"Score: {result['score']:.4f}")
    print(f"Start: {result['start']}, End: {result['end']}")

    # 示例2
    context = """
Photosynthesis is a process used by plants and other organisms to convert light energy
into chemical energy that can later be released to fuel the organism's activities.
This chemical energy is stored in carbohydrate molecules, such as sugars, which are
synthesized from carbon dioxide and water. In most cases, oxygen is also released as
a waste product. The process of photosynthesis primarily occurs in the chloroplasts,
specialized organelles found in plant cells. Chlorophyll, the green pigment in chloroplasts,
absorbs light most efficiently in the blue and red wavelengths.
"""

    question = "What pigment in chloroplasts absorbs light during photosynthesis?"

    result = qa_pipeline(question=question, context=context)

    print(f"Answer: {result['answer']}")
    print(f"Score: {result['score']:.4f}")
    print(f"Start: {result['start']}, End: {result['end']}")
