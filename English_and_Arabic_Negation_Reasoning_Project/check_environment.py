import os

os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["USE_FLAX"] = "0"

from importlib import import_module, metadata
from pathlib import Path
import platform
import sys


def main():
    root = Path(__file__).resolve().parent
    print(f'Python: {platform.python_version()}', flush=True)
    for line in (root / 'requirements.txt').read_text().splitlines():
        package, expected = line.split('==', 1)
        actual = metadata.version(package)
        print(f'{package}: {actual}', flush=True)
        if actual != expected:
            raise RuntimeError(f'{package}: expected {expected}, found {actual}')
    for module in ('numpy', 'pandas', 'sentencepiece', 'torch', 'transformers', 'accelerate', 'bitsandbytes', 'datasets', 'peft', 'wandb', 'yaml', 'huggingface_hub'):
        import_module(module)
    import torch
    import bitsandbytes as bnb
    from datasets import Dataset
    from transformers import HfArgumentParser

    sys.path.insert(0, str(root / 'English_Code'))
    for module in ('run', 'load_model', 'dataset', 'evaluate', 'optimizer'):
        import_module(module)
    from config import DataTrainingArguments, ModelArguments

    HfArgumentParser((ModelArguments, DataTrainingArguments))
    if len(Dataset.from_dict({'statement': ['synthetic check']})) != 1:
        raise RuntimeError('Dataset construction failed')
    print(f'PyTorch: {torch.__version__}; CUDA: {torch.version.cuda}', flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError('Select a CUDA GPU runtime in Colab before running the experiments.')
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
    print(f'GPU: {torch.cuda.get_device_name(0)}; compute dtype: {dtype}', flush=True)
    with torch.inference_mode():
        layer = bnb.nn.Linear4bit(64, 64, bias=False, compute_dtype=dtype, compress_statistics=True, quant_type='nf4').to('cuda')
        inputs = torch.ones((2, 64), device='cuda', dtype=dtype)
        output = layer(inputs)
        torch.cuda.synchronize()
        if output.shape != (2, 64) or not torch.isfinite(output).all().item():
            raise RuntimeError('NF4 GPU smoke test failed')
    print('PASS: dependency versions, experiment imports, dataset construction and synthetic 4-bit CUDA operation. No research inference performed.')


if __name__ == '__main__':
    main()
