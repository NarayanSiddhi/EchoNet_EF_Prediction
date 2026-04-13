import torch
import torch.nn.functional as F


def supcon_loss(z, labels, temperature=0.2):  # ← slightly higher temp (more stable)
    z = F.normalize(z, dim=1)

    sim = torch.matmul(z, z.T) / temperature

    labels = labels.unsqueeze(1)
    mask = torch.eq(labels, labels.T).float()

    # remove self-comparisons
    logits_mask = torch.ones_like(mask) - torch.eye(mask.shape[0]).to(z.device)
    mask = mask * logits_mask

    exp_sim = torch.exp(sim) * logits_mask
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

    mean_log_prob = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)

    loss = -mean_log_prob.mean()
    return loss


def hierarchical_loss(z, sex, age, bmi):

    loss_sex = supcon_loss(z, sex)
    loss_age = supcon_loss(z, age)
    loss_bmi = supcon_loss(z, bmi)

    # 🔥 FIX: normalize instead of summing
    loss = (loss_sex + loss_age + loss_bmi) / 3

    return loss