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


def _masked_mean(values, mask, eps=1e-8):
    denom = mask.sum().clamp_min(eps)
    return (values * mask).sum() / denom


def _groupwise_anchor_average(anchor_losses, anchor_valid_mask, group_ids):
    valid_idx = torch.where(anchor_valid_mask > 0)[0]
    if valid_idx.numel() == 0:
        return anchor_losses.new_tensor(0.0)

    uniq_groups = torch.unique(group_ids[valid_idx])
    group_means = []
    for g in uniq_groups:
        g_mask = (group_ids == g).float() * anchor_valid_mask
        if g_mask.sum() > 0:
            group_means.append(_masked_mean(anchor_losses, g_mask))

    if not group_means:
        return anchor_losses.new_tensor(0.0)
    return torch.stack(group_means).mean()


def hierarchical_multidemographic_loss(
    z,
    task_labels,
    sex,
    age,
    bmi,
    lambda_h2=0.0,
    temp_h1=0.2,
    temp_h2=0.2,
    alpha_intra=0.5,
    alpha_inter=1.0,
    eps=1e-8,
):
    """
    FairHICON-style 2-level hierarchical contrastive loss generalized
    to multi-demographic conditions (sex + age + bmi).
    """
    z = F.normalize(z, dim=1)
    n = z.size(0)
    if n < 2:
        zero = z.new_tensor(0.0)
        return zero, zero, zero

    device = z.device
    eye_mask = torch.eye(n, device=device, dtype=torch.bool)
    sim = torch.matmul(z, z.T)

    task_labels = task_labels.view(-1).long()
    sex = sex.view(-1).long()
    age = age.view(-1).long()
    bmi = bmi.view(-1).long()

    # Composite subgroup id over all demographic attributes.
    subgroup = sex * 100 + age * 10 + bmi

    same_task = task_labels.unsqueeze(1).eq(task_labels.unsqueeze(0))
    same_group = subgroup.unsqueeze(1).eq(subgroup.unsqueeze(0))
    not_self = ~eye_mask

    # ---------------- Hierarchy 1 ----------------
    # Positives: same task label regardless of demographics.
    pos_h1 = same_task & not_self
    exp_h1 = torch.exp(sim / temp_h1) * not_self.float()
    denom_h1 = exp_h1.sum(dim=1) + eps
    log_prob_h1 = sim / temp_h1 - torch.log(denom_h1.unsqueeze(1))

    pos_count_h1 = pos_h1.float().sum(dim=1)
    anchor_valid_h1 = (pos_count_h1 > 0).float()
    mean_pos_log_prob_h1 = (pos_h1.float() * log_prob_h1).sum(dim=1) / (pos_count_h1 + eps)
    anchor_loss_h1 = -mean_pos_log_prob_h1
    loss_h1 = _groupwise_anchor_average(anchor_loss_h1, anchor_valid_h1, subgroup)

    # ---------------- Hierarchy 2 ----------------
    # Positives: strict same subgroup.
    pos_h2 = same_group & not_self

    # Compute subgroup prototypes in normalized space.
    prototypes = {}
    for g in torch.unique(subgroup):
        g_mask = subgroup == g
        c = z[g_mask].mean(dim=0)
        prototypes[int(g.item())] = F.normalize(c, dim=0)

    weights = torch.ones((n, n), device=device, dtype=z.dtype)

    for i in range(n):
        g_i = int(subgroup[i].item())
        c_i = prototypes[g_i]
        sim_proto = torch.dot(z[i], c_i)

        for j in range(n):
            if i == j or pos_h2[i, j]:
                continue

            # Intra-class negative: same task, different subgroup.
            if same_task[i, j]:
                alpha = alpha_intra
            # Inter-class negative: different task.
            else:
                alpha = alpha_inter

            sim_ij = sim[i, j]
            if sim_ij > sim_proto:
                weights[i, j] = torch.exp(alpha * (sim_ij - sim_proto))

    exp_h2 = torch.exp(sim / temp_h2) * not_self.float()
    weighted_exp_h2 = exp_h2 * weights
    denom_h2 = weighted_exp_h2.sum(dim=1) + eps
    log_prob_h2 = sim / temp_h2 - torch.log(denom_h2.unsqueeze(1))

    pos_count_h2 = pos_h2.float().sum(dim=1)
    anchor_valid_h2 = (pos_count_h2 > 0).float()
    mean_pos_log_prob_h2 = (pos_h2.float() * log_prob_h2).sum(dim=1) / (pos_count_h2 + eps)
    anchor_loss_h2 = -mean_pos_log_prob_h2
    loss_h2 = _groupwise_anchor_average(anchor_loss_h2, anchor_valid_h2, subgroup)

    total = loss_h1 + lambda_h2 * loss_h2
    return total, loss_h1, loss_h2