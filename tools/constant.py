from enum import Enum
from transformers import AutoModel, AutoModelForCausalLM, AutoModelForMaskedLM, AutoModelForSeq2SeqLM, AutoModelForQuestionAnswering, AutoModelForTokenClassification, AutoModelForSequenceClassification

class AutoModelType(Enum):
    r"""[Hugging Face的AutoModel類型](https://blog.csdn.net/weixin_42426841/article/details/142236561)"""
    Default = AutoModel                                         # 特徵提取或自訂任務
    CausalLM = AutoModelForCausalLM                             # 文字生成
    MaskedLM = AutoModelForMaskedLM                             # 填空任務
    Seq2SeqLM = AutoModelForSeq2SeqLM                           # 機器翻譯、文字摘要
    QuestionAnswering = AutoModelForQuestionAnswering           # 抽取式問答
    TokenClassification = AutoModelForTokenClassification       # 命名實體識別
    SequenceClassification = AutoModelForSequenceClassification # 文字分類
