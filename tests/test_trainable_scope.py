"""Tests for strict task-head-only checkpoint refinement scope."""
import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from experiments.run_config import configure_trainable_scope


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(3, 4)
        self.heads = torch.nn.ModuleDict({
            "HC": torch.nn.Linear(4, 2),
            "A4C": torch.nn.Linear(4, 2),
        })


class ToyFusionEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Linear(3, 4)
        self.fusion = torch.nn.Linear(4, 4, bias=False)


class ToyFusionModel(ToyModel):
    def __init__(self):
        super().__init__()
        self.encoder = ToyFusionEncoder()


class ToyBlockEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Module()
        self.backbone.blocks = torch.nn.ModuleList([
            torch.nn.Linear(4, 4), torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)
        ])
        self.backbone.norm = torch.nn.LayerNorm(4)


class ToyBlockModel(ToyModel):
    def __init__(self):
        super().__init__()
        self.encoder = ToyBlockEncoder()


def test_task_scope_trains_only_named_head():
    model = ToyModel()
    trainable = configure_trainable_scope(model, train_task="HC")
    assert trainable
    assert all(not parameter.requires_grad for parameter in model.encoder.parameters())
    assert all(parameter.requires_grad for parameter in model.heads["HC"].parameters())
    assert all(not parameter.requires_grad for parameter in model.heads["A4C"].parameters())
    assert {id(parameter) for parameter in trainable} == {
        id(parameter) for parameter in model.heads["HC"].parameters()
    }


def test_unknown_task_is_rejected():
    with pytest.raises(ValueError, match="unknown --train-task"):
        configure_trainable_scope(ToyModel(), train_task="missing")


def test_historical_freeze_encoder_scope_is_preserved():
    model = ToyModel()
    configure_trainable_scope(model, freeze_encoder=True)
    assert all(not parameter.requires_grad for parameter in model.encoder.parameters())
    assert all(parameter.requires_grad for head in model.heads.values()
               for parameter in head.parameters())


def test_fusion_scope_trains_only_fusion_adapter():
    model = ToyFusionModel()
    trainable = configure_trainable_scope(model, train_fusion_only=True)
    assert {id(parameter) for parameter in trainable} == {
        id(parameter) for parameter in model.encoder.fusion.parameters()
    }
    assert all(not parameter.requires_grad for parameter in model.encoder.backbone.parameters())
    assert all(not parameter.requires_grad for head in model.heads.values()
               for parameter in head.parameters())


def test_fusion_scope_requires_fusion_encoder():
    with pytest.raises(ValueError, match="requires a fusion encoder"):
        configure_trainable_scope(ToyModel(), train_fusion_only=True)


def test_task_and_fusion_scopes_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        configure_trainable_scope(ToyFusionModel(), train_task="HC", train_fusion_only=True)


def test_task_scope_can_add_only_last_encoder_block_and_norm():
    model = ToyBlockModel()
    trainable = configure_trainable_scope(model, train_task="HC", unfreeze_last_blocks=1)
    trainable_ids = {id(parameter) for parameter in trainable}
    expected = list(model.heads["HC"].parameters())
    expected += list(model.encoder.backbone.blocks[-1].parameters())
    expected += list(model.encoder.backbone.norm.parameters())
    assert trainable_ids == {id(parameter) for parameter in expected}
    assert all(not parameter.requires_grad
               for block in model.encoder.backbone.blocks[:-1]
               for parameter in block.parameters())
    assert all(not parameter.requires_grad for parameter in model.heads["A4C"].parameters())


def test_unfreeze_last_blocks_requires_task_scope():
    with pytest.raises(ValueError, match="requires --train-task"):
        configure_trainable_scope(ToyBlockModel(), unfreeze_last_blocks=1)


def test_too_many_last_blocks_are_rejected():
    with pytest.raises(ValueError, match="does not expose"):
        configure_trainable_scope(ToyBlockModel(), train_task="HC", unfreeze_last_blocks=4)
