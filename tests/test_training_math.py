"""Synthetic tensor/autograd checks only; no optimizer steps or LLM loading."""
import copy
import ast
from typing import Optional
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from ops_repro.models import SelectedPolicy
from ops_repro.tensor_ops import grpo_advantages,mixture,ppo_token_losses,rlsd_weights


class LossTests(unittest.TestCase):
    def test_grpo_zero_variance_and_unbiased_std(self):
        self.assertEqual(grpo_advantages([0,0,0,0]),[0.,0.,0.,0.])
        values=grpo_advantages([0,1])
        self.assertAlmostEqual(values[0],-2**-.5,places=5)
        self.assertAlmostEqual(values[1],2**-.5,places=5)

    def test_rlsd_keeps_direction_and_detaches_privileged_evidence(self):
        teacher=torch.tensor([100.,-100.,0.],requires_grad=True)
        old=torch.tensor([0.,0.,0.],requires_grad=True)
        new=torch.tensor([0.,0.,0.],requires_grad=True)
        positive=rlsd_weights(1.,teacher,old)
        negative=rlsd_weights(-1.,teacher,old)
        self.assertTrue(torch.allclose(positive,torch.tensor([1.1,.9,1.])))
        self.assertTrue(torch.allclose(negative,torch.tensor([.9,1.1,1.])))
        ppo_token_losses(new,old,-1.,negative).sum().backward()
        self.assertIsNone(teacher.grad)
        self.assertIsNone(old.grad)
        self.assertTrue(torch.all(new.grad>0))

    def test_lambda_zero_is_exact_grpo(self):
        t=torch.tensor([-.2,-3.]);s=torch.tensor([-.7,-1.])
        self.assertTrue(torch.equal(rlsd_weights(1,t,s,lam=0),torch.ones_like(t)))
        self.assertEqual(mixture(50),0)
        self.assertAlmostEqual(mixture(1),.49)

    def test_weights_match_locked_official_implementation(self):
        source=Path(__file__).resolve().parents[1]/'tests/fixtures/rlsd_official_formula.py'
        tree=ast.parse(source.read_text())
        method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_build_stgca_advantages')
        method.decorator_list=[]
        module=ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[]))
        class VF:
            @staticmethod
            def masked_mean(x,m):return (x*m).sum()/m.sum()
        namespace={'torch':torch,'Optional':Optional,'VF':VF}
        exec(compile(module,str(source),'exec'),namespace)
        teacher=torch.tensor([[-1.,-2.,-3.],[-3.,-1.,-2.]])
        student=torch.tensor([[-2.,-2.,-2.],[-2.,-2.,-2.]])
        advantage=torch.tensor([[.7,.7,.7],[-.7,-.7,-.7]])
        official,_=namespace['_build_stgca_advantages'](teacher_log_probs=teacher,student_log_probs=student,
            advantages=advantage,response_mask=torch.ones_like(advantage),lam=.5,clip_range=.2)
        actual=torch.stack([advantage[i]*rlsd_weights(advantage[i,0].item(),teacher[i],student[i]) for i in range(2)])
        self.assertTrue(torch.allclose(official,actual,atol=1e-7))

    def test_ppo_clip_and_dual_clip(self):
        old=torch.zeros(3);new=torch.log(torch.tensor([.1,1.,10.]))
        self.assertTrue(torch.allclose(ppo_token_losses(new,old,1.,torch.ones(3)),torch.tensor([-.1,-1.,-1.28])))
        self.assertTrue(torch.allclose(ppo_token_losses(new,old,-1.,torch.ones(3)),torch.tensor([.8,1.,3.])))

    def test_global_token_normalization_across_unequal_microbatches(self):
        x=torch.tensor([1.,2.,3.,4.],requires_grad=True)
        # Two ranks, two microbatches each; no implicit /accumulation in our loop.
        rank_grad=[]
        groups=[[[0],[1,2]],[[3],[]]]
        for rank in groups:
            objective=sum((x[indexes].sum()*2/len(x) for indexes in rank),torch.tensor(0.))
            rank_grad.append(torch.autograd.grad(objective,x,retain_graph=True)[0])
        observed=sum(rank_grad)/2
        expected=torch.autograd.grad(x.mean(),x)[0]
        self.assertTrue(torch.equal(observed,expected))

    def test_selected_projection_equals_full_ce_and_gradients(self):
        class Backbone(torch.nn.Module):
            def __init__(self):super().__init__();self.embedding=torch.nn.Embedding(13,5)
            def forward(self,input_ids,**kwargs):
                return type('Output',(),{'last_hidden_state':self.embedding(input_ids)})()
        class Tiny(torch.nn.Module):
            def __init__(self):super().__init__();self.model=Backbone();self.lm_head=torch.nn.Linear(5,13,bias=False)
        torch.manual_seed(7)
        first=Tiny();second=copy.deepcopy(first)
        ids=torch.tensor([[1,2,3,4,5,6]])
        labels=torch.tensor([[-100,-100,3,-100,5,6]])
        value=SelectedPolicy(first,chunk_size=1)(ids,torch.ones_like(ids),labels)
        loss=-value['log_probs'].sum();loss.backward()
        logits=second.lm_head(second.model(input_ids=ids).last_hidden_state)
        reference=torch.nn.functional.cross_entropy(logits[:,:-1,:].reshape(-1,13),labels[:,1:].reshape(-1),ignore_index=-100,reduction='sum')
        reference.backward()
        self.assertTrue(torch.allclose(loss,reference,atol=1e-6))
        for a,b in zip(first.parameters(),second.parameters()):self.assertTrue(torch.allclose(a.grad,b.grad,atol=1e-6))

    def test_padding_is_never_supervised(self):
        class Empty(torch.nn.Module):pass
        policy=SelectedPolicy(Empty())
        with self.assertRaisesRegex(ValueError,'Padding'):
            policy(torch.tensor([[1,2]]),torch.tensor([[1,0]]),torch.tensor([[-100,2]]))


if __name__=='__main__':unittest.main()
