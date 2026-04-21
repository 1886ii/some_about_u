
"""
Config — SEG-Doc Reproduction + Innovation
============================================
Reproduction: Paper's concat fusion + LAM-GraphSAGE on THGC graph
Innovation:   三图独立GNN → 跨模态对齐 → 任务感知融合
"""
import os, torch
from dataclasses import dataclass, field

SER_LABEL_MAP = {"question": 0, "answer": 1, "header": 2, "other": 3}

@dataclass
class Config:
    # ══════════ 模式切换 ══════════
    use_innovations: bool = False    # False=复现, True=创新

    # ══════════ 编码器 ══════════
    encoder_name: str = "./layoutlmv3_base"
    max_seq_len: int = 512
    freeze_encoder_epochs: int = 2
    hidden_dim: int = 512            # 主GNN hidden (论文900, 我们512)

    # ══════════ THGC ══════════
    thgc_threshold: float = 0.5
    thgc_angle_tol: float = 45.0

    # ══════════ 主GNN (复现) ══════════
    num_gnn_layers: int = 3
    gnn_dropout: float = 0.3

    # ══════════ 创新: 三图 + 对齐 + 融合 ══════════
    modal_dim: int = 256             # 每个模态的投影维度
    vis_knn_k: int = 5               # 视觉KNN图的K
    spa_knn_k: int = 3               # 空间KNN图的K
    modal_gnn_layers: int = 2        # 每个模态GNN的层数
    align_dim: int = 256             # 对齐空间维度
    lambda_align: float = 0.1        # 对齐损失权重
    fusion_heads: int = 4            # 任务感知融合的注意力头数

    # ══════════ SER ══════════
    ser_num_classes: int = 4
    ser_class_weights: tuple = (1.0, 1.0, 6.0, 2.5)

    # ══════════ 训练 ══════════
    lr_encoder: float = 2e-5
    lr_head: float = 1e-3
    weight_decay: float = 1e-4
    num_epochs: int = 200
    warmup_ratio: float = 0.06
    grad_clip: float = 1.0
    eval_every: int = 1

    # ══════════ Loss ══════════
    lambda_re: float = 1.0
    lambda_ser: float = 1.0

    # ══════════ 模态消融实验 ══════════
    use_text: bool   = True
    use_layout: bool = True
    use_visual: bool = True

    # ══════════ 推理 ══════════
    re_threshold: float = 0.5

    # ══════════ 杂项 ══════════
    use_amp: bool = True
    debug_every: int = 10
    data_dir: str = "./data/funsd"
    output_dir: str = "./output"
    checkpoint_dir: str = "./checkpoints"
    best_ckpt: str = "./checkpoints/best_model.pt"
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    def make_dirs(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
