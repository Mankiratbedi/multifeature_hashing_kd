
import torch
import torch.nn.functional as F


# ==========================================
# 1. SIMILARITY PRESERVATION LOSS
# ==========================================

def similarity_loss(hash_codes, similarity_labels):

    num_bits = hash_codes.shape[1]

    similarity_scores = (
        hash_codes @ hash_codes.T
    ) / num_bits

    mask = ~torch.eye(
        hash_codes.shape[0],
        dtype=torch.bool,
        device=hash_codes.device,
    )

    return F.binary_cross_entropy_with_logits(
        similarity_scores[mask],
        similarity_labels[mask].float(),
    )


# ==========================================
# 2. QUANTIZATION LOSS
# ==========================================

def quantization_loss(hash_codes):

    # Distance to the nearest binary value (-1 or +1)
    distance_to_binary = torch.minimum(
        (hash_codes - 1.0).pow(2),
        (hash_codes + 1.0).pow(2),
    )

    return distance_to_binary.mean()


# ==========================================
# 3. BIT BALANCE LOSS
# ==========================================

def bit_balance_loss(hash_codes):

    # Encourage the average of every bit to approach zero
    bit_means = hash_codes.mean(dim=0)

    return bit_means.pow(2).mean()


# ==========================================
# 4. BIT DECORRELATION LOSS
# ==========================================

def bit_decorrelation_loss(hash_codes):

    batch_size, num_bits = hash_codes.shape

    if batch_size < 2:
        raise ValueError("Batch size must be at least 2.")

    centered = (
        hash_codes - hash_codes.mean(dim=0, keepdim=True)
    )

    covariance = (
        centered.T @ centered
    ) / (batch_size - 1)

    variance = torch.diagonal(covariance).clamp_min(1e-6)

    std = torch.sqrt(variance)

    correlation = covariance / (
        std[:, None] * std[None, :]
    )

    # Penalize only correlations between different bits
    mask = ~torch.eye(
        num_bits,
        dtype=torch.bool,
        device=hash_codes.device,
    )

    return correlation[mask].pow(2).mean()


# ==========================================
# 5. CLASSIFICATION LOSS
# ==========================================

def classification_loss(class_logits, labels):

    return F.cross_entropy(
        class_logits,
        labels,
    )


# ==========================================
# 6. KNOWLEDGE DISTILLATION LOSS
# ==========================================

def distillation_loss(student_features, projected_teacher):

    student_normalized = F.normalize(
        student_features,
        p=2,
        dim=1,
    )

    teacher_normalized = F.normalize(
        projected_teacher,
        p=2,
        dim=1,
    )

    return F.mse_loss(
        student_normalized,
        teacher_normalized,
    )


# ==========================================
# COMBINED HASHING OBJECTIVE
# ==========================================

def total_hashing_loss(
    hash_codes,
    similarity_labels,
    class_logits,
    labels,
    student_features,
    projected_teacher,
    alpha=1.0,
    beta=1.0,
    lambda_d=0.1,
    gamma=1.0,
    eta=1.0,
):

    lp = similarity_loss(
        hash_codes,
        similarity_labels,
    )

    lq = quantization_loss(hash_codes)

    lb = bit_balance_loss(hash_codes)

    ld = bit_decorrelation_loss(hash_codes)

    lc = classification_loss(
        class_logits,
        labels,
    )

    ldistill = distillation_loss(
        student_features,
        projected_teacher,
    )

    total = (
        lp
        + alpha * lq
        + beta * lb
        + lambda_d * ld
        + gamma * lc
        + eta * ldistill
    )

    return total, {
        "similarity": lp,
        "quantization": lq,
        "balance": lb,
        "decorrelation": ld,
        "classification": lc,
        "distillation": ldistill,
    }
