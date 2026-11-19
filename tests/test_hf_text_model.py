"""Random tiny CPU HF structure fixture. No real model weights or optimizer."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from transformers import Qwen3_5ForCausalLM,Qwen3_5TextConfig,Qwen3_5Config,Qwen3_5ForConditionalGeneration
from ops_repro.models import SelectedPolicy,load_policy


class HFTextModelTests(unittest.TestCase):
    def test_selected_projection_and_text_checkpoint_roundtrip(self):
        cfg=Qwen3_5TextConfig(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=1,
            num_attention_heads=2,num_key_value_heads=1,head_dim=8,layer_types=['full_attention'],
            linear_key_head_dim=8,linear_value_head_dim=8,linear_num_key_heads=1,linear_num_value_heads=2,
            max_position_embeddings=128,eos_token_id=2,pad_token_id=0)
        model=Qwen3_5ForCausalLM(cfg).eval()
        ids=torch.tensor([[1,3,4,5,2]])
        labels=torch.tensor([[-100,-100,4,5,2]])
        with torch.no_grad():
            selected=SelectedPolicy(model,chunk_size=1)(ids,torch.ones_like(ids),labels)['log_probs']
            logits=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).logits
            full=logits[0,1:4].float().log_softmax(-1).gather(-1,ids[0,2:].unsqueeze(-1)).squeeze(-1)
        self.assertTrue(torch.allclose(selected,full,atol=1e-6))
        with tempfile.TemporaryDirectory() as folder:
            model.save_pretrained(folder,safe_serialization=True)
            loaded=load_policy(folder)
            self.assertEqual(loaded._loading_evidence['missing_keys'],set() if isinstance(loaded._loading_evidence['missing_keys'],set) else [])
            for name,value in model.state_dict().items():self.assertTrue(torch.equal(value,loaded.state_dict()[name]))
        self.assertFalse(torch.cuda.is_initialized())

    def test_full_multimodal_checkpoint_extracts_text_weights(self):
        cfg=Qwen3_5Config(text_config=dict(vocab_size=32,hidden_size=16,intermediate_size=32,
            num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,head_dim=8,layer_types=['full_attention'],
            linear_key_head_dim=8,linear_value_head_dim=8,linear_num_key_heads=1,linear_num_value_heads=2,
            max_position_embeddings=128),vision_config=dict(depth=1,hidden_size=16,intermediate_size=32,
            num_heads=2,patch_size=2,temporal_patch_size=1,spatial_merge_size=1,num_position_embeddings=16,out_hidden_size=16))
        original=Qwen3_5ForConditionalGeneration(cfg).eval()
        with tempfile.TemporaryDirectory() as folder:
            original.save_pretrained(folder,safe_serialization=True)
            loaded=load_policy(folder)
            expected=original.model.language_model.state_dict()
            for name,value in expected.items():self.assertTrue(torch.equal(value,loaded.model.state_dict()[name]))
            self.assertTrue(torch.equal(original.lm_head.weight,loaded.lm_head.weight))
        self.assertFalse(torch.cuda.is_initialized())


if __name__=='__main__':unittest.main()
