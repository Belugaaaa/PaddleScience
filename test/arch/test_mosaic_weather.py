# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Independent checks for weather metric weighting and invalid inputs."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples/mosaic"))
from evaluate_weather import weighted_mse  # noqa: E402
from rollout import forecast_rollout  # noqa: E402
from rollout import validate_era5_sequence  # noqa: E402


class TestMosaicWeather(unittest.TestCase):
    def test_rollout_sequence_validation(self):
        data = dict(
            state=np.array([[0, 1], [1, 2], [2, 3]]), target=np.array([2, 3, 4])
        )
        validate_era5_sequence(data, 0, 3)
        with self.assertRaises(ValueError):
            validate_era5_sequence(data, 0, 4)
        data["state"][1, 1] = 100
        with self.assertRaises(AssertionError):
            validate_era5_sequence(data, 0, 3)

    def test_rollout_feedback_and_noise(self):
        calls = []

        def predict(history, phase, noise):
            calls.append((history.copy(), phase.copy(), noise.copy()))
            return history[:, -1:] + 1

        history = np.zeros((1, 2, 2, 3, 82), dtype="float32")
        phase = np.zeros((1, 2), dtype="float32")
        result = forecast_rollout(
            predict, history, phase, np.zeros(82), np.ones(82), 4, 2026, 3
        )
        np.testing.assert_allclose(result[:, :, 0, 0, 0], [[1, 2, 3]])
        np.testing.assert_array_equal(calls[1][0][:, -1], 1)
        np.testing.assert_allclose(calls[2][1], [[2, 8 / 365.25]])
        self.assertFalse(np.array_equal(calls[0][2], calls[1][2]))
        expected_noise = (
            np.random.default_rng(2026).standard_normal((1, 4)).astype("float32")
        )
        np.testing.assert_array_equal(calls[0][2], expected_noise)
        np.testing.assert_array_equal(history, 0)

    def test_cosine_weighted_mse(self):
        prediction = np.ones((2, 1, 4, 3, 82))
        target = np.zeros_like(prediction)
        np.testing.assert_allclose(weighted_mse(prediction, target, [-60, 0, 60]), 1)
        prediction.fill(0)
        prediction[..., 1, :] = 2
        # Equator carries half the total weight; squared error there is four.
        np.testing.assert_allclose(weighted_mse(prediction, target, [-60, 0, 60]), 2)

    def test_invalid_inputs(self):
        prediction = np.ones((1, 1, 4, 3, 82))
        with self.assertRaises(ValueError):
            weighted_mse(prediction, prediction[..., :1], [-60, 0, 60])
        with self.assertRaises(ValueError):
            weighted_mse(prediction, prediction, [0, 60])
        prediction[..., 0] = float("nan")
        with self.assertRaises(ValueError):
            weighted_mse(prediction, prediction, [-60, 0, 60])


if __name__ == "__main__":
    unittest.main()
