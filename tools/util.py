import torch, argparse, coremltools as ct, numpy as np
import tools.model as model
from transformers import AutoTokenizer
from tools.constant import AutoModelType

def load_hugging_face_model(model_id: str, model_type: str):
    r"""從 Hugging Face Hub 載入模型。

    :param model_id: Hugging Face 模型的 ID。

    :return: 載入的模型和分詞器。
    """

    model_type = autoModelType(model_type)

    print(model_type)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        pytorch_model = model_type.value.from_pretrained(model_id, torch_dtype=torch.float32)
        pytorch_model.eval()  # 將模型設定為評估模式
        return model.Result(value=(tokenizer, pytorch_model))
    except Exception as error:
        return model.Result(error=error)

def convert_traced_model(wrapped_model, dummy_input):
    r"""將 PyTorch 模型追蹤為 TorchScript 模型。

    :param wrapped_model: 包裝過的 PyTorch 模型。
    :param dummy_input: 用於追蹤的虛擬輸入。

    :return: 追蹤後的 TorchScript 模型。
    """

    try:
        traced_model = torch.jit.trace(wrapped_model, dummy_input)
        return model.Result(value=traced_model)
    except Exception as error:
        return model.Result(error=error)

def wrapped_model_eval(pytorch_model):
    r"""將 PyTorch 模型包裝為評估模式的簡單輸出包裝器。
    :param pytorch_model: 原始的 PyTorch 模型。
    :return: 包裝後的模型。
    """
    wrapped_model = model.SimpleOutputWrapper(pytorch_model)
    wrapped_model.eval()

    return wrapped_model

def torchscript_to_coreml(traced_model, dummy_input, coreml_filename: str):
    r"""將追蹤後的 TorchScript 模型轉換為 Core ML 格式。

    :param traced_model: 追蹤後的 TorchScript 模型。
    :param dummy_input: 用於定義輸入形狀的虛擬輸入。
    :param coreml_filename: 儲存 Core ML 模型的檔名。

    :return: 轉換後的 Core ML 模型。
    """
    try:
        coreml_model = ct.convert(
            traced_model,
            # 定義輸入的名稱、形狀和資料類型
            inputs=[ct.TensorType(name="input_ids", shape=dummy_input.shape, dtype=np.int64)],
            convert_to="mlprogram",  # 使用現代的 'mlprogram' 格式
            compute_units=ct.ComputeUnit.CPU_ONLY # 強制使用 CPU 進行轉換，以提高相容性
        )

        coreml_model.save(coreml_filename)
        return model.Result(value=coreml_model)

    except Exception as error:
        return model.Result(error=error)

def parse_args():
    r"""解析命令行參數。
    :return: 解析後的命令行參數。
    """
    parser = argparse.ArgumentParser(description="將 Hugging Face 模型轉換為 Core ML 格式")
    parser.add_argument("--model_id", type=str, required=True, help="Hugging Face 模型的 ID，例如 'distilbert-base-uncased'")
    parser.add_argument("--model_type", type=str, default=None, help="Hugging Face 模型的類型（預設為 masked）")
    parser.add_argument("--output", type=str, default=None, help="輸出的 Core ML 檔名（預設為 <model_id>.mlpackage）")

    return parser.parse_args()

def autoModelType(type: str) -> AutoModelType:
    r"""輸入參數轉成transformers的AutoModelType"""
    type = f'{type.lower()}'.strip()

    match type:
        case "masked": return AutoModelType.MaskedLM
        case "causal": return AutoModelType.CausalLM
        case "seq2seq": return AutoModelType.Seq2SeqLM
        case "question": return AutoModelType.QuestionAnswering
        case "token": return AutoModelType.TokenClassification
        case "sequence": return AutoModelType.SequenceClassification
        case _: return AutoModelType.Default