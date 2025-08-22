import torch

class SimpleOutputWrapper(torch.nn.Module):
    r"""一個簡單的 PyTorch 模型包裝類別，用於將 Hugging Face 模型的輸出簡化為 logits。
    這個類別將 Hugging Face 模型的輸出轉換為 logits，這是模型的原始預測分數。
    這樣做是為了確保在轉換過程中，Core ML 工具能夠正確地處理模型的輸出。
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    # 對於轉換，我們只關心 logits（模型的原始預測分數）輸出
    def forward(self, input_ids):
        outputs = self.model(input_ids)
        return outputs.logits

class Result:
    def __init__(self, value=None, error=None):
        self.value = value
        self.error = error

    @property
    def is_ok(self):
        return self.error is None

    @property
    def is_err(self):
        return self.error is not None