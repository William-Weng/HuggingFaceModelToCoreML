import torch
import tools.util as util

def convert_hugging_face_to_coreml():
    r"""將 Hugging Face 模型轉換為 Core ML 格式。"""

    args = util.parse_args()
    model_id = args.model_id
    model_type = args.model_type if args.model_type else 'masked'
    coreml_filename = args.output if args.output else f"{model_id}.mlpackage"

    # --- 步驟 1: 載入 Hugging Face 模型 ---
    print(f"正在從 Hugging Face Hub 載入模型 '{model_id}'...")

    result = util.load_hugging_face_model(model_id, model_type)
    if result.is_err: print(f"載入模型時發生錯誤: {result.error}"); return
    tokenizer, pytorch_model = result.value

    # --- 步驟 2: 將 PyTorch 模型轉換為 Core ML ---
    print("正在將 PyTorch 模型轉換為 Core ML...")

    wrapped_model = util.wrapped_model_eval(pytorch_model)
    dummy_input = torch.randint(0, tokenizer.vocab_size, (1, 128), dtype=torch.int64)

    result = util.convert_traced_model(wrapped_model, dummy_input)

    if result.is_err: print(f"追蹤 PyTorch 模型時發生錯誤: {result.error}"); return
    traced_model = result.value

    result = util.torchscript_to_coreml(traced_model, dummy_input, coreml_filename)
    if result.is_err: print(f"轉換為 Core ML 時發生錯誤: {result.error}"); return

    print(f"成功轉換為 Core ML 並儲存為 '{coreml_filename}'")

if __name__ == "__main__":
    convert_hugging_face_to_coreml()