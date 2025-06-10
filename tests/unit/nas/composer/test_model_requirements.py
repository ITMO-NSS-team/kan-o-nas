from __future__ import annotations

from nas.composer.requirements import ModelRequirements


def test_input_shape():
    requirements = ModelRequirements(input_shape=[32, 32], output_shape=10, color_mode='grayscale')
    assert requirements.input_shape == [32, 32]
    assert requirements.channels_num == 1
    requirements = ModelRequirements(input_shape=[128, 128], output_shape=10, color_mode='color')
    assert requirements.input_shape == [128, 128,]
    assert requirements.channels_num == 3


def test_color_mode_case_sensitivity():
    requirements = ModelRequirements(input_shape=[128, 128], output_shape=10, color_mode='COLOR')
    assert requirements.input_shape == [128, 128,]
    assert requirements.channels_num == 3
    requirements = ModelRequirements(input_shape=[128, 128], output_shape=10, color_mode='GraYsCalE')
    assert requirements.input_shape == [128, 128,]
    assert requirements.channels_num == 1


def test_color_modes():
    try:
        ModelRequirements(input_shape=[128, 128], output_shape=3, color_mode='Lab')
    except ValueError:
        return True
    return False
