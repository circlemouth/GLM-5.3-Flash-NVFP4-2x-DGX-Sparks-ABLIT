import copy
import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from glm53_setup import axl_ablit, launch_assets, server, server_config

try:
    import torch
except ImportError:
    torch = None


def read_raw(path, key):
    header_len, header = axl_ablit.read_header(path)
    start, end = header[key]["data_offsets"]
    with path.open("rb") as stream:
        stream.seek(8 + header_len + start)
        return stream.read(end - start)


def write_shard(path, tensors):
    header = {}
    body = bytearray()
    for key, (dtype, shape, raw) in tensors.items():
        start = len(body)
        body.extend(raw)
        header[key] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [start, len(body)],
        }
    encoded = json.dumps(header, separators=(",", ":")).encode()
    padding = (-len(encoded)) % 8
    encoded += b" " * padding
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + body)


class SafetensorsRewriteTests(unittest.TestCase):
    def test_rewrite_replaces_only_named_raw_tensor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.safetensors"
            output = root / "output.safetensors"
            write_shard(
                source,
                {
                    "keep": ("U8", [3], b"abc"),
                    "change": ("U8", [2], b"de"),
                },
            )
            replacement = root / "replacement.bin"
            replacement.write_bytes(b"WXYZ")
            axl_ablit._rewrite_shard(
                source,
                output,
                {
                    "change": {
                        "dtype": "U8",
                        "shape": [4],
                        "nbytes": 4,
                        "path": replacement,
                    }
                },
                [],
            )
            self.assertEqual(read_raw(output, "keep"), b"abc")
            self.assertEqual(read_raw(output, "change"), b"WXYZ")
            _, header = axl_ablit.read_header(output)
            self.assertEqual(header["change"]["shape"], [4])
            self.assertEqual(header["change"]["dtype"], "U8")

    def test_rewrite_rejects_a_missing_replacement_key(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.safetensors"
            write_shard(source, {"present": ("U8", [1], b"a")})
            with self.assertRaisesRegex(axl_ablit.AxlAblitError, "not found"):
                axl_ablit._rewrite_shard(
                    source,
                    root / "output.safetensors",
                    {"absent": ("U8", [1], b"b")},
                    [],
                )

    def test_reader_rejects_duplicate_tensor_names(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.safetensors"
            item = '{"dtype":"U8","shape":[1],"data_offsets":[0,1]}'
            encoded = f'{{"same":{item},"same":{item}}}'.encode()
            path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"x")
            with self.assertRaisesRegex(axl_ablit.AxlAblitError, "duplicate JSON key"):
                axl_ablit.read_header(path)


class BuildSafetyTests(unittest.TestCase):
    def test_source_tree_authentication_rejects_extra_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a").write_bytes(b"one")
            row = ("a", 3, hashlib.sha256(b"one").hexdigest())
            aggregate = hashlib.sha256(
                json.dumps(row, separators=(",", ":")).encode() + b"\n"
            ).hexdigest()
            expected = {
                "repo": "example/repo",
                "revision": "a" * 40,
                "tree": {
                    "file_count": 1,
                    "total_bytes": 3,
                    "aggregate_sha256": aggregate,
                },
            }
            self.assertEqual(
                axl_ablit.validate_source_tree(root, expected, "fixture")["tree"],
                expected["tree"],
            )
            (root / "extra").write_bytes(b"two")
            with self.assertRaisesRegex(axl_ablit.AxlAblitError, "pinned revision"):
                axl_ablit.validate_source_tree(root, expected, "fixture")

    def test_build_rejects_output_inside_an_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            axl = root / "axl"
            nvidia = root / "nvidia"
            donor = root / "donor"
            axl.mkdir()
            nvidia.mkdir()
            donor.mkdir()
            with self.assertRaisesRegex(axl_ablit.AxlAblitError, "input directory"):
                axl_ablit.build(
                    axl,
                    nvidia,
                    donor,
                    root / "report.json",
                    axl / "derived",
                )


class QuantizerTests(unittest.TestCase):
    @unittest.skipIf(torch is None, "torch is not installed")
    def test_e2m1_midpoints_use_round_to_even(self):
        values = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
        self.assertEqual(
            axl_ablit.e2m1_codes(values, torch).tolist(), [0, 2, 2, 4, 4, 6, 6]
        )
        self.assertEqual(
            axl_ablit.e2m1_codes(-values, torch).tolist(),
            [8, 10, 10, 12, 12, 14, 14],
        )

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_quantizer_packs_even_column_into_low_nibble(self):
        weight = torch.zeros((1, 16), dtype=torch.bfloat16)
        packed, scale, scale2 = axl_ablit.quant_nvfp4(weight, torch)
        self.assertEqual(list(packed.shape), [1, 8])
        self.assertEqual(list(scale.shape), [1, 1])
        self.assertEqual(list(scale2.shape), [])
        self.assertEqual(packed.tolist(), [[0] * 8])
        self.assertEqual(str(scale.dtype), "torch.float8_e4m3fn")
        self.assertEqual(str(scale2.dtype), "torch.float32")


class ManifestAndProfileTests(unittest.TestCase):
    def reproduction_report(self):
        lock = axl_ablit.load_lock()
        rows = []
        for key in axl_ablit.main_keys():
            suffix = key.rsplit(".", 1)[1]
            dtype = {
                "weight": "U8",
                "weight_scale": "F8_E4M3",
                "weight_scale_2": "F32",
            }[suffix]
            value = {
                "dtype": dtype,
                "shape": [1],
                "nbytes": 4 if dtype == "F32" else 1,
                "sha256": "0" * 64,
            }
            rows.append(
                {
                    "key": key,
                    "expected": value,
                    "actual": copy.deepcopy(value),
                    "match": True,
                }
            )
        return {
            "schema": 1,
            "kind": "axl-reproduction",
            "passed": True,
            "tensor_count": 87,
            "main_layers": list(axl_ablit.MAIN_LAYERS),
            "axl": {
                "repo": lock["axl"]["repo"],
                "revision": lock["axl"]["revision"],
                "tree": lock["axl"]["tree"],
                "files": lock["axl"]["identity_sha256"],
            },
            "nvidia": lock["nvidia"],
            "requantizer": lock["requantizer"],
            "converter": {
                "module_sha256": axl_ablit.sha256_file(Path(axl_ablit.__file__))
            },
            "environment": {"torch": lock["requantizer"]["torch"]},
            "tensors": rows,
        }

    def profile(self):
        profile = server_config.load(
            Path(__file__).resolve().parents[1] / "examples/server.axl.example.toml"
        )
        profile["runtime"]["derived_checkpoint"]["manifest_sha256"] = "1" * 64
        return profile

    def test_reproduction_report_rejects_unpinned_toolchain(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            report = self.reproduction_report()
            path.write_text(json.dumps(report), encoding="utf-8")
            self.assertEqual(axl_ablit.validate_reproduction(path), report)
            report["environment"]["torch"] = "different"
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(axl_ablit.AxlAblitError, "invalid"):
                axl_ablit.validate_reproduction(path)

    def test_manifest_identity_is_optional_but_strict_when_present(self):
        server_config.validate(self.profile())
        profile = self.profile()
        profile["runtime"]["derived_checkpoint"]["manifest_sha256"] = "BAD"
        with self.assertRaisesRegex(ValueError, "manifest_sha256"):
            server_config.validate(profile)

    def test_server_preflight_gate_checks_configured_manifest(self):
        profile = self.profile()
        metadata = {
            "quantization_config": {
                "quant_algo": "MIXED_PRECISION",
                "producer": {"requant_target": "l"},
                "quantized_layers": {},
            }
        }
        report = {"manifest": {"transformed_key_count": 88}}
        with mock.patch.object(
            axl_ablit, "verify_manifest", return_value=report
        ) as verify:
            checks = server.derived_checks(profile, metadata)
        self.assertIs(checks["derived_manifest"], True)
        verify.assert_called_once_with(
            Path(profile["runtime"]["derived_checkpoint"]["path"]), "1" * 64
        )

    def test_launch_identity_keeps_legacy_and_records_manifest_provenance(self):
        self.assertEqual(
            launch_assets.derived_identity(Path("/legacy"), {"path": "/legacy"}),
            {"enabled": True, "manifest_sha256": None},
        )
        manifest = {
            "output": {"shards": {"model.safetensors": "0" * 64}},
            "base": {"revision": "base-revision"},
            "donor": {"revision": "donor-revision"},
            "transformed_key_count": 88,
        }
        with mock.patch.object(
            axl_ablit,
            "verify_manifest",
            return_value={"manifest": manifest, "manifest_sha256": "1" * 64},
        ):
            identity = launch_assets.derived_identity(
                Path("/derived"), {"path": "/derived", "manifest_sha256": "1" * 64}
            )
        self.assertEqual(identity["manifest_sha256"], "1" * 64)
        self.assertEqual(identity["base_revision"], "base-revision")
        self.assertEqual(identity["donor_revision"], "donor-revision")
        self.assertEqual(identity["transformed_key_count"], 88)
        self.assertRegex(identity["artifact_sha256"], r"^[0-9a-f]{64}$")

    def test_build_publishes_only_after_full_synthetic_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            axl = root / "axl"
            nvidia = root / "nvidia"
            donor = root / "donor"
            axl.mkdir()
            nvidia.mkdir()
            donor.mkdir()
            shard = "model-00001-of-00001.safetensors"
            tensors = {"model.keep": ("U8", [3], b"old")}
            for key in axl_ablit.transformed_keys():
                if key == axl_ablit.mtp_key():
                    tensors[key] = ("BF16", [1, 16], b"\0" * 32)
                else:
                    suffix = key.rsplit(".", 1)[1]
                    dtype, shape, raw = {
                        "weight": ("U8", [1, 8], b"\0" * 8),
                        "weight_scale": ("F8_E4M3", [1, 1], b"\0"),
                        "weight_scale_2": ("F32", [], b"\0" * 4),
                    }[suffix]
                    tensors[key] = (dtype, shape, raw)
            write_shard(axl / shard, tensors)
            index = {"weight_map": {key: shard for key in tensors}}
            (axl / axl_ablit.INDEX_NAME).write_text(json.dumps(index))

            report = self.reproduction_report()

            def fake_reproduce(_nvidia, _axl, path):
                path.write_text(json.dumps(report), encoding="utf-8")
                return copy.deepcopy(report)

            def fake_replacements(_donor, temp):
                replacements = {}
                transformed = {}
                for position, key in enumerate(axl_ablit.transformed_keys(), start=1):
                    dtype, shape, raw = tensors[key]
                    data = bytes([position % 251 or 1]) * len(raw)
                    info = axl_ablit._replacement_file(
                        temp / f"replacement-{position}.bin", data, dtype, shape
                    )
                    replacements[key] = info
                    transformed[key] = {
                        "output_sha256": info["sha256"],
                        "dtype": dtype,
                        "shape": shape,
                        "nbytes": len(raw),
                        (
                            "donor_sha256"
                            if key == axl_ablit.mtp_key()
                            else "donor_weight_sha256"
                        ): "1" * 64,
                    }
                return (
                    replacements,
                    transformed,
                    {
                        "tensor_count": 30,
                        "tensor_bytes": 1,
                        "manifest_sha256": axl_ablit.load_lock()["donor"][
                            "overlay_manifest_sha256"
                        ],
                    },
                )

            lock = axl_ablit.load_lock()["axl"]
            identity = {
                "repo": lock["repo"],
                "revision": lock["revision"],
                "tree": lock["tree"],
                "files": lock["identity_sha256"],
            }
            output = root / "derived"
            reproduction = root / "reproduction.json"
            with (
                mock.patch.object(
                    axl_ablit, "reproduce_axl", side_effect=fake_reproduce
                ),
                mock.patch.object(
                    axl_ablit, "_prepare_replacements", side_effect=fake_replacements
                ),
                mock.patch.object(
                    axl_ablit, "validate_axl_identity", return_value=identity
                ),
            ):
                manifest = axl_ablit.build(axl, nvidia, donor, reproduction, output)
            self.assertTrue(output.is_dir())
            self.assertFalse((output / ".incomplete").exists())
            self.assertEqual(manifest["transformed_key_count"], 88)
            self.assertEqual(
                axl_ablit.verify(
                    output,
                    expected_manifest_sha256=axl_ablit.sha256_file(
                        output / axl_ablit.MANIFEST_NAME
                    ),
                )["transformed_key_count"],
                88,
            )

            failed = root / "failed"
            with (
                mock.patch.object(
                    axl_ablit, "reproduce_axl", side_effect=fake_reproduce
                ),
                mock.patch.object(
                    axl_ablit,
                    "_prepare_replacements",
                    side_effect=axl_ablit.AxlAblitError("synthetic failure"),
                ),
            ):
                with self.assertRaisesRegex(
                    axl_ablit.AxlAblitError, "synthetic failure"
                ):
                    axl_ablit.build(
                        axl, nvidia, donor, root / "failed-report.json", failed
                    )
            self.assertFalse(failed.exists())
            self.assertEqual(list(root.glob(".failed.incomplete-*")), [])

    def test_lightweight_manifest_check_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard = "model-00001-of-00001.safetensors"
            (root / shard).write_bytes(b"test-shard")
            index = {
                "weight_map": {
                    "tensor": shard,
                    **{key: shard for key in axl_ablit.transformed_keys()},
                }
            }
            (root / axl_ablit.INDEX_NAME).write_text(json.dumps(index))
            lock = axl_ablit.load_lock()
            transformed = {}
            for key in sorted(axl_ablit.transformed_keys()):
                suffix = key.rsplit(".", 1)[1]
                dtype = (
                    "BF16"
                    if key == axl_ablit.mtp_key()
                    else {
                        "weight": "U8",
                        "weight_scale": "F8_E4M3",
                        "weight_scale_2": "F32",
                    }[suffix]
                )
                row = {
                    "dtype": dtype,
                    "shape": [0],
                    "nbytes": 0,
                    "output_sha256": "0" * 64,
                    "base_axl_sha256": "0" * 64,
                }
                row[
                    "donor_sha256"
                    if key == axl_ablit.mtp_key()
                    else "donor_weight_sha256"
                ] = "0" * 64
                transformed[key] = row
            manifest = {
                "schema": 1,
                "kind": "axl-ablit-derived-checkpoint",
                "complete": True,
                "base": {
                    "repo": lock["axl"]["repo"],
                    "revision": lock["axl"]["revision"],
                    "tree": copy.deepcopy(lock["axl"]["tree"]),
                    "files": copy.deepcopy(lock["axl"]["identity_sha256"]),
                },
                "donor": {
                    **copy.deepcopy(lock["donor"]),
                    "asset_report": {
                        "manifest_sha256": lock["donor"]["overlay_manifest_sha256"],
                        "tensor_count": 30,
                        "tensor_bytes": 1,
                    },
                },
                "requantizer": copy.deepcopy(lock["requantizer"]),
                "converter": {
                    "module_sha256": axl_ablit.sha256_file(Path(axl_ablit.__file__)),
                    "command_schema": "build-v1",
                },
                "nvidia_reproduction": {
                    "source": copy.deepcopy(lock["nvidia"]),
                    "converter_module_sha256": axl_ablit.sha256_file(
                        Path(axl_ablit.__file__)
                    ),
                    "passed": True,
                    "tensor_count": 87,
                    "report_sha256": "0" * 64,
                },
                "transformed_key_count": 88,
                "transformed_keys": transformed,
                "invariance": {
                    "affected_shard_unchanged_tensor_count": 0,
                    "affected_shard_unchanged_tensor_aggregate_sha256": hashlib.sha256().hexdigest(),
                    "unchanged_shards": [],
                    "auxiliary_files": {
                        axl_ablit.INDEX_NAME: axl_ablit.sha256_file(
                            root / axl_ablit.INDEX_NAME
                        )
                    },
                },
                "output": {
                    "index_sha256": axl_ablit.sha256_file(root / axl_ablit.INDEX_NAME),
                    "shards": {shard: hashlib.sha256(b"test-shard").hexdigest()},
                    "shard_sizes": {shard: len(b"test-shard")},
                },
            }
            manifest_path = root / axl_ablit.MANIFEST_NAME
            manifest_path.write_text(json.dumps(manifest, sort_keys=True))
            digest = axl_ablit.sha256_file(manifest_path)
            report = axl_ablit.verify_manifest(root, digest)
            self.assertEqual(report["shard_count"], 1)
            (root / "unexpected").write_bytes(b"extra")
            with self.assertRaisesRegex(axl_ablit.AxlAblitError, "file set drifted"):
                axl_ablit.verify_manifest(root, digest)
            (root / "unexpected").unlink()

            with self.assertRaisesRegex(axl_ablit.AxlAblitError, "SHA-256 mismatch"):
                axl_ablit.verify_manifest(root, "f" * 64)
            (root / ".incomplete").write_text("incomplete\n")
            with self.assertRaisesRegex(axl_ablit.AxlAblitError, "incomplete"):
                axl_ablit.verify_manifest(root, digest)
            (root / ".incomplete").unlink()
            (root / shard).write_bytes(b"other-data")
            with self.assertRaisesRegex(axl_ablit.AxlAblitError, "content drifted"):
                axl_ablit.verify_manifest(root, digest)


if __name__ == "__main__":
    unittest.main()
