"""RLSD formula and PPO objective. Inputs contain generated action tokens only."""
import math


def grpo_advantages(rewards,epsilon=1e-6):
    import torch
    if len(rewards)<2:raise ValueError('GRPO needs at least two samples per prompt')
    values=torch.tensor(rewards,dtype=torch.float64)
    if not torch.isfinite(values).all():raise ValueError('Nonfinite rewards')
    # torch.std uses correction=1, matching the locked official GRPO source.
    result=(values-values.mean())/(values.std(correction=1)+epsilon)
    return result.tolist()


def rlsd_weights(advantage,teacher_logp,student_logp,lam=.5,evidence_clip=.2):
    import torch
    if not 0<=lam<=1 or not 0<=evidence_clip<1:raise ValueError('Invalid RLSD mixture/clip')
    if teacher_logp.shape!=student_logp.shape:raise ValueError('Teacher/student token alignment differs')
    sign=1 if advantage>0 else -1 if advantage<0 else 0
    delta=(teacher_logp.detach()-student_logp.detach())*sign
    if not torch.isfinite(delta).all():raise ValueError('Nonfinite RLSD log probabilities')
    # Stable algebraic equivalent of exp(delta).clamp(1-eps,1+eps).
    evidence=delta.clamp(math.log1p(-evidence_clip),math.log1p(evidence_clip)).exp()
    return ((1-lam)+lam*evidence).detach()


def ppo_token_losses(new_logp,old_logp,advantage,weights,clip_low=.2,clip_high=.28,clip_dual=3.):
    import torch
    if new_logp.shape!=old_logp.shape or weights.shape!=new_logp.shape:raise ValueError('PPO token arrays differ')
    if not torch.isfinite(new_logp).all() or not torch.isfinite(old_logp).all():raise ValueError('Nonfinite PPO log probabilities')
    ratio=(new_logp-old_logp.detach()).clamp(-20,20).exp()
    token_adv=advantage*weights.detach()
    clipped=torch.maximum(-token_adv*ratio,-token_adv*ratio.clamp(1-clip_low,1+clip_high))
    return torch.minimum(clipped,-token_adv*clip_dual) if advantage<0 else clipped


def mixture(step,start=.5,decay_steps=50):
    return start*max(0.,1.-step/decay_steps) if decay_steps else start
