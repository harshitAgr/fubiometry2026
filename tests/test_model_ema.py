"""Numerical and non-interference tests for training-time EMA checkpoints."""

import pytest

torch = pytest.importorskip("torch")

from experiments.model_ema import StateDictEMA


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1, bias=False)
        self.register_buffer("counter", torch.tensor(0, dtype=torch.int64))


def test_ema_initially_equals_raw_model():
    model = TinyModel()
    ema = StateDictEMA(model, 0.9)
    for key, value in model.state_dict().items():
        assert torch.equal(ema.state_dict()[key], value)


def test_update_matches_closed_form_and_copies_integer_buffers():
    model = TinyModel()
    with torch.no_grad():
        model.linear.weight.fill_(2.0)
    ema = StateDictEMA(model, 0.9)
    with torch.no_grad():
        model.linear.weight.fill_(4.0)
        model.counter.fill_(7)
    ema.update(model)
    torch.testing.assert_close(ema.state_dict()["linear.weight"],
                               torch.full_like(model.linear.weight, 2.2))
    assert ema.state_dict()["counter"].item() == 7


def test_ema_update_does_not_mutate_raw_model():
    model = TinyModel()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    ema = StateDictEMA(model, 0.9999)
    ema.update(model)
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key])


def test_override_decay_matches_powered_periodic_update():
    model = TinyModel()
    with torch.no_grad():
        model.linear.weight.fill_(2.0)
    ema = StateDictEMA(model, 0.9)
    with torch.no_grad():
        model.linear.weight.fill_(4.0)
    ema.update(model, decay=0.9 ** 3)
    expected = 0.9 ** 3 * 2.0 + (1.0 - 0.9 ** 3) * 4.0
    torch.testing.assert_close(ema.state_dict()["linear.weight"],
                               torch.full_like(model.linear.weight, expected))


@pytest.mark.parametrize("decay", [0.0, 1.0, -0.1, 1.1])
def test_invalid_decay_rejected(decay):
    with pytest.raises(ValueError):
        StateDictEMA(TinyModel(), decay)
