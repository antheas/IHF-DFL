import random
import pytest
import torch
import ml.attacks as attacks
import ml.visualization as visualization
import ml.training as training


def test_manipulate_adds_noise():
    model = torch.nn.Linear(2, 2)
    original = {k: v.clone() for k, v in model.state_dict().items()}

    updated = attacks.manipulate(model, scale=0.0)

    # With scale 0.0, floating tensors should be unchanged
    for key, tensor in updated.items():
        assert torch.equal(tensor, original[key])


def test_add_noise_modifies_target_tensor():
    random.seed(0)
    model = torch.nn.Linear(3, 1)
    state_dict = attacks.add_noise(model, offset_from_end=1)

    items = list(model.state_dict().items())
    target_idx = max(0, len(items) - 1)
    target_key, target_tensor = items[target_idx]

    new_tensor = state_dict[target_key]
    diff = (new_tensor - target_tensor).abs().max().item()
    assert diff == pytest.approx(1e-05, rel=0, abs=2e-06)


def test_get_color_handles_known_and_unknown_indices():
    assert visualization.get_color(0, "bad") == visualization.bad_c
    assert visualization.get_color(0, "freerider") == visualization.free_c
    assert visualization.get_color(100, "good") is None


def test_device_label_formats_output():
    assert training.device_label(torch.device("cpu")) == "CPU"
    assert training.device_label(torch.device("cuda"), 2) == "GPU 2"


def test_training_model_compiles_only_once_per_process(monkeypatch):
    created = []
    compiled = []

    class FakeModel:
        def to(self, device):
            self.device = device
            return self

    def make_model():
        model = FakeModel()
        created.append(model)
        return model

    def compile_model(model):
        wrapper = object()
        compiled.append((model, wrapper))
        return wrapper

    monkeypatch.setattr(training, "Net_CIFAR", make_model)
    monkeypatch.setattr(training.torch, "compile", compile_model)
    training._TRAINING_MODEL_CACHE.clear()

    first = training._training_model("cifar", torch.device("cuda:0"))
    second = training._training_model("cifar", torch.device("cuda:0"))

    assert first is second
    assert first == (created[0], compiled[0][1])
    assert len(created) == 1
    assert len(compiled) == 1

    training._TRAINING_MODEL_CACHE.clear()
