"""Offline synthetic CPU gate. Prohibits optimizer updates, network and CUDA init."""
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch

if __name__=='__main__':
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'))
    with patch('torch.optim.AdamW.step',side_effect=AssertionError('Optimizer updates are prohibited in CPU gate')),patch('torch.cuda._lazy_init',side_effect=AssertionError('CUDA initialization is prohibited in CPU gate')),patch('urllib.request.urlopen',side_effect=AssertionError('Network requests are prohibited in CPU gate')):
        result=unittest.TextTestRunner(verbosity=2).run(suite)
    print('CPU_GATE: optimizer/network/CUDA guards enabled; only synthetic tiny weights, local tokenizer and fixtures used.')
    sys.exit(not result.wasSuccessful())
