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

"""Mosaic routing gradients and native checkpoint regression tests."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import paddle

from ppsci.arch.mosaic import Mosaic
from ppsci.arch.mosaic_attention import sparse_attention


class TestMosaic(unittest.TestCase):
    def test_preserved_forward_remains_dynamic_after_to_static(self):
        # to_static mutates Layer.forward; benchmark baselines must retain the
        # original bound method rather than calling that mutated layer again.
        class Tiny(paddle.nn.Layer):
            def forward(self, x):
                return x + 1

        model = Tiny()
        original = model.forward
        compiled = paddle.jit.to_static(model, full_graph=True)
        self.assertIs(compiled, model)
        self.assertIsNot(model.forward, original)
        np.testing.assert_array_equal(original(paddle.to_tensor([2.0])).numpy(), [3.0])

    def setUp(self):
        paddle.set_device("cpu")
        paddle.seed(2026)

    def test_selected_attention_matches_dense_mask_and_gradients(self):
        q = paddle.randn([2, 12, 4, 8])
        k = paddle.randn([2, 12, 2, 8])
        v = paddle.randn([2, 12, 2, 8])
        indices = paddle.to_tensor(
            np.tile(
                np.array([[[[2, 0], [1, 2]], [[0, 1], [2, 0]], [[1, 2], [0, 1]]]]),
                (2, 1, 1, 1),
            )
        )
        for tensor in (q, k, v):
            tensor.stop_gradient = False
        selected = sparse_attention(q, k, v, indices, 4, query_chunk_size=2)
        mask = np.zeros((2, 4, 12, 12), dtype=bool)
        for b in range(2):
            for h in range(4):
                for block in range(3):
                    for choice in indices.numpy()[b, block, h // 2]:
                        mask[
                            b,
                            h,
                            block * 4 : (block + 1) * 4,
                            choice * 4 : (choice + 1) * 4,
                        ] = True
        keys = paddle.repeat_interleave(k, 2, axis=2).transpose([0, 2, 1, 3])
        values = paddle.repeat_interleave(v, 2, axis=2).transpose([0, 2, 1, 3])
        scores = paddle.matmul(
            q.transpose([0, 2, 1, 3]), keys, transpose_y=True
        ) / np.sqrt(8)
        scores = paddle.where(
            paddle.to_tensor(mask), scores, paddle.full_like(scores, float("-inf"))
        )
        dense = paddle.matmul(
            paddle.nn.functional.softmax(scores, axis=-1), values
        ).transpose([0, 2, 1, 3])
        np.testing.assert_allclose(
            selected.numpy(), dense.numpy(), atol=1e-6, rtol=1e-6
        )
        grads_sparse = paddle.grad(
            selected.square().mean(), [q, k, v], retain_graph=True
        )
        grads_dense = paddle.grad(dense.square().mean(), [q, k, v])
        for actual, expected in zip(grads_sparse, grads_dense):
            np.testing.assert_allclose(
                actual.numpy(), expected.numpy(), atol=1e-6, rtol=1e-5
            )

    def test_native_checkpoint_roundtrip_with_geometry(self):
        config = dict(
            dim=16,
            num_heads=2,
            k_neighbors=2,
            qk_norm=False,
            rope=True,
            rope_theta=10000,
            sparse_every=1,
            variables=["u"],
            static_variables=["s"],
            qkv_compress_ratio=1,
            noise_dim=4,
            rmsnorm_elementwise_affine=False,
            cg_stage_cfgs=[
                dict(
                    nside=2,
                    dim=16,
                    num_heads=2,
                    block_attn_size=4,
                    sparse_block_size=4,
                    sparse_block_count=2,
                    encoder_depth=1,
                    decoder_depth=1,
                    mlp_ratio=2,
                    gqa_ratio=2,
                )
            ],
            bottleneck_cfg=dict(
                nside=1,
                dim=32,
                num_heads=2,
                block_attn_size=4,
                sparse_block_size=4,
                sparse_block_count=2,
                depth=1,
                mlp_ratio=2,
                gqa_ratio=2,
            ),
        )
        lon, lat = paddle.arange(4, dtype="float32") * 90, paddle.to_tensor(
            [-60.0, 0.0, 60.0]
        )
        static = paddle.randn([4, 3, 1])

        def create():
            model = Mosaic(config)
            model.initialize_static_vars(static, lon, lat)
            model.initialize_interpolation(lon, lat)
            return model.eval()

        model = create()
        inputs = dict(
            state=paddle.randn([1, 1, 1, 4, 3, 1]),
            time=paddle.to_tensor([[[0.25, 0.5]]]),
            noise=paddle.randn([1, 4]),
        )
        with paddle.no_grad():
            expected = model(inputs)["forecast"].numpy()
            with tempfile.TemporaryDirectory() as directory:
                path = str(Path(directory) / "model.pdparams")
                paddle.save(model.state_dict(), path)
                loaded = create()
                loaded.set_state_dict(paddle.load(path))
                actual = loaded(inputs)["forecast"].numpy()
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.shape, (1, 1, 4, 3, 1))

    def test_full_grid_static_normalization(self):
        # Exercise the production-size reduction without constructing the 214M model.
        from types import SimpleNamespace

        from ppsci.arch.mosaic import rad_to_xyz

        lon = paddle.arange(240, dtype="float32") * 1.5
        lat = paddle.arange(121, dtype="float32") * 1.5 - 90
        data = (
            np.random.default_rng(42).normal(1e4, 3e3, (240, 121, 3)).astype("float32")
        )
        model = SimpleNamespace()
        Mosaic.initialize_static_vars(model, paddle.to_tensor(data), lon, lat)
        grid = paddle.deg2rad(paddle.stack(paddle.meshgrid(lon, lat), axis=-1))
        fields = np.concatenate([data, rad_to_xyz(grid).numpy()], axis=-1).astype(
            "float64"
        )
        expected = (fields - fields.mean((0, 1), keepdims=True)) / (
            fields.std((0, 1), ddof=1, keepdims=True) + 1e-6
        )
        np.testing.assert_allclose(
            model.static_vars.numpy().reshape(240, 121, 6),
            expected,
            atol=5e-7,
            rtol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
