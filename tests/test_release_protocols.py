import csv
import io
import tempfile
import unittest
from contextlib import redirect_stderr
from types import SimpleNamespace
from pathlib import Path

from data_utils.prompt_utils import get_prompt_template
from eemf.evaluation.cli import parse_args
from eemf.evaluation.protocols import (
    CORRUPTIONS,
    build_dataset_spec,
    get_backbone_spec,
    run_dir_name,
)
from eemf.evaluation.result_writer import RESULT_COLUMNS, make_result_row, upsert_result_rows


class ReleaseProtocolTests(unittest.TestCase):
    def test_corrupted_modelnet_expands_by_type_then_level(self):
        spec = build_dataset_spec("modelnet_c")

        self.assertEqual(len(spec.corruption_specs), 35)
        self.assertEqual(
            [item.cor_type for item in spec.corruption_specs[:5]],
            [
                "add_global_0",
                "add_global_1",
                "add_global_2",
                "add_global_3",
                "add_global_4",
            ],
        )
        self.assertEqual({item.corruption for item in spec.corruption_specs}, set(CORRUPTIONS))
        self.assertTrue(all(item.cor_type != "all" for item in spec.corruption_specs))

    def test_severity_filter_keeps_all_corruption_types_for_one_level(self):
        spec = build_dataset_spec("scanobjnn_c", severity=2)

        self.assertEqual(len(spec.corruption_specs), 7)
        self.assertEqual(
            [item.cor_type for item in spec.corruption_specs],
            [
                "add_global_2",
                "add_local_2",
                "dropout_global_2",
                "dropout_local_2",
                "rotate_2",
                "scale_2",
                "jitter_2",
            ],
        )

    def test_clean_dataset_has_one_clean_stream_and_no_severity(self):
        spec = build_dataset_spec("modelnet")

        self.assertEqual(len(spec.corruption_specs), 1)
        self.assertEqual(spec.corruption_specs[0].corruption, "clean")
        self.assertEqual(spec.corruption_specs[0].cor_type, "clean")
        with self.assertRaisesRegex(ValueError, "severity"):
            build_dataset_spec("modelnet", severity=1)

    def test_result_csv_schema_and_upsert(self):
        spec = build_dataset_spec("modelnet_c", severity=0)
        backbone = get_backbone_spec("ulip2")

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "result.csv"
            row = make_result_row(spec, backbone, spec.corruption_specs[0], 42.345, "done")
            upsert_result_rows(path, [row])
            upsert_result_rows(path, [make_result_row(spec, backbone, spec.corruption_specs[0], 43.0, "done")])

            with path.open(newline="", encoding="utf-8") as f:
                rows = list(csv.reader(f))

        self.assertEqual(rows[0], RESULT_COLUMNS)
        self.assertEqual(len(rows), 2)
        self.assertNotIn("log" + "_path", rows[0])
        self.assertNotIn("severity", rows[0])
        self.assertEqual(rows[1], ["modelnet_c", "add_global", "add_global_0", "ULIP-2", "43.00", "done"])

    def test_run_dir_name_has_no_legacy_id(self):
        spec = build_dataset_spec("scanobjnn_c")
        backbone = get_backbone_spec("uni3d")

        name = run_dir_name(backbone, spec)
        self.assertEqual(name, "uni3d_scanobjnnc_hardest")
        self.assertNotIn("E" + "12", name)
        self.assertNotIn("E" + "11", name)

    def test_weighted_prompt_template_loads_from_release_cache(self):
        args = SimpleNamespace(
            prompt_source="manualfull_llm_dynamic_init",
            prompt_cache_dir="llm",
            prompt_cache_file="",
            llm_provider="deepseek",
            llm_model="deepseek-v4-pro",
            llm_prompt_mode="multiview_2d3d",
            dynamic_prompt_count=10,
            prompt_static_weight=0.75,
            prompt_dynamic_weight=0.25,
            force_regenerate_prompts=False,
        )

        template = get_prompt_template(args, ["airplane", "bathtub"], dataset_name="modelnet_c")

        self.assertEqual(template["__eemf_prompt_type__"], "weighted_fusion")
        self.assertEqual(len(template["dynamic_template"]["airplane"]), 10)
        self.assertEqual(len(template["dynamic_template"]["bathtub"]), 10)

    def test_individual_corruption_argument_is_not_supported(self):
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args([
                    "--backbone",
                    "ulip",
                    "--dataset",
                    "modelnet_c",
                    "--corruption",
                    "add_global_0",
                ])


if __name__ == "__main__":
    unittest.main()
