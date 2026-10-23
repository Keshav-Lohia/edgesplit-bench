"""Unit tests (no Docker needed):  python -m unittest discover -s tests -v"""
import json
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from edgesplit import bench, model, partition  # noqa: E402


class ParseTests(unittest.TestCase):
    def test_parse_llama_bench_json_with_noise(self):
        rows = [
            {"n_prompt": 64, "n_gen": 0, "avg_ts": 55.5, "stddev_ts": 1.5},
            {"n_prompt": 0, "n_gen": 32, "avg_ts": 9.25, "stddev_ts": 0.25},
        ]
        text = "some log line\n" + json.dumps(rows)
        r = bench.parse_bench_json(text)
        self.assertEqual(r["pp_ts"], 55.5)
        self.assertEqual(r["tg_ts"], 9.25)

    def test_parse_fails_without_json(self):
        with self.assertRaises(bench.BenchError):
            bench.parse_bench_json("error: model not found")

    def test_build_cmd_rpc_and_local(self):
        cfg = bench.BenchConfig(model_path="/models/m.gguf", rpc=["w1:50052", "w2:50052"], tensor_split="3/1")
        cmd = bench.build_cmd(cfg)
        self.assertIn("--rpc", cmd)
        self.assertEqual(cmd[cmd.index("--rpc") + 1], "w1:50052,w2:50052")
        self.assertEqual(cmd[cmd.index("-ts") + 1], "3/1")
        local = bench.build_cmd(bench.BenchConfig(model_path="/models/m.gguf"))
        self.assertNotIn("--rpc", local)
        self.assertEqual(local[local.index("-ngl") + 1], "0")


class PartitionTests(unittest.TestCase):
    def nodes(self):
        return [partition.Node("fast", 0.01, 16), partition.Node("mid", 0.02, 11), partition.Node("slow", 0.04, 11)]

    def test_compositions_respect_caps_and_total(self):
        for comp in partition.compositions(28, [16, 11, 11]):
            self.assertEqual(sum(comp), 28)
            self.assertTrue(all(0 <= l <= c for l, c in zip(comp, [16, 11, 11])))

    def test_latency_plan_fills_fastest_first(self):
        plan = partition.plan(self.nodes(), 28, hop_cost=0.001, objective="latency")
        self.assertEqual(plan.layers[0], 16)           # fastest node full
        self.assertGreaterEqual(plan.layers[1], plan.layers[2])

    def test_latency_plan_uses_single_node_if_it_fits(self):
        nodes = [partition.Node("a", 0.01, 40), partition.Node("b", 0.02, 40)]
        self.assertEqual(partition.plan(nodes, 28, 0.005, "latency").layers, (28, 0))

    def test_throughput_plan_balances_stage_times(self):
        nodes = [partition.Node("a", 0.01, 40), partition.Node("b", 0.02, 40), partition.Node("c", 0.04, 40)]
        lay = partition.plan(nodes, 28, 0.0, "throughput").layers
        stage = [l * n.sec_per_layer for l, n in zip(lay, nodes)]
        self.assertLess(max(stage) - min(stage), 0.04)  # within one slow layer
        self.assertGreater(lay[0], lay[1])
        self.assertGreater(lay[1], lay[2])

    def test_throughput_plan_never_worse_than_even_split(self):
        nodes = self.nodes()
        plan = partition.plan(nodes, 28, 0.001, "throughput")
        even = partition.even_split(28, 3)
        self.assertLessEqual(plan.cost, partition.stage_cost(even, nodes, 0.001) + 1e-12)

    def test_infeasible(self):
        with self.assertRaises(ValueError):
            partition.plan([partition.Node("a", 0.01, 5)], 28, 0.0)

    def test_even_and_proportional_sum(self):
        self.assertEqual(sum(partition.even_split(28, 3)), 28)
        self.assertEqual(partition.even_split(28, 3), (10, 9, 9))
        prop = partition.proportional_split(self.nodes(), 28)
        self.assertEqual(sum(prop), 28)
        self.assertEqual(prop, (16, 8, 4))

    def test_tensor_split_string(self):
        self.assertEqual(partition.Plan((16, 11, 1), "latency", 0.0).tensor_split(), "16/11/1")


class ModelTests(unittest.TestCase):
    def test_fit_recovers_parameters(self):
        true = model.TgParams(t0=0.080, o=0.004, k=2.0, beta=1.0)
        n_embd = 1536
        ns, rtts, rates, ts = [], [], [], []
        for n in (1, 2, 3):
            for rtt in (0, 1, 5, 10, 20, 50):
                ns.append(n); rtts.append(rtt); rates.append(None)
                ts.append(model.predict_tg_ts(true, n, rtt, None, n_embd))
            for rate in (200, 50, 10, 1):
                ns.append(n); rtts.append(1); rates.append(rate)
                ts.append(model.predict_tg_ts(true, n, 1, rate, n_embd))
        fit = model.fit_tg(ns, rtts, rates, ts, n_embd)
        self.assertAlmostEqual(fit.k, 2.0, places=3)
        self.assertAlmostEqual(fit.beta, 1.0, places=3)
        self.assertAlmostEqual(fit.t0, 0.080, places=4)
        self.assertGreater(fit.r2, 0.9999)

    def test_bw_term(self):
        # 1536 * 4 B = 6144 B = 49152 bit; at 100 Mbit/s -> 0.49 ms
        self.assertAlmostEqual(model.bw_term(1536, 100), 49152 / 100e6)
        self.assertEqual(model.bw_term(1536, None), 0.0)

    def test_prefill_is_bandwidth_bound_decode_is_not(self):
        n_embd, rate = 1536, 10
        dec = model.bw_term(n_embd, rate, tokens=1)
        pre = model.bw_term(n_embd, rate, tokens=128)
        self.assertAlmostEqual(pre / dec, 128)


class MockPipelineTests(unittest.TestCase):
    def test_all_experiments_in_mock_mode(self):
        import tempfile
        os.environ["EDGESPLIT_MOCK"] = "1"
        from edgesplit import config as C, experiments, analysis, plots
        # redirect results into a temp dir
        tmp = Path(tempfile.mkdtemp())
        C.ROOT = tmp
        (tmp / "models").mkdir()
        for cmd in ("e1", "e2", "e3", "e4", "e5"):
            experiments.main([cmd, "--rtts", "0", "10", "50", "--node-counts", "1", "2"])
        df = analysis.load()
        self.assertGreater(len(df), 20)
        self.assertTrue((df.source == "mock").all())
        prm = analysis.fit_from_results("main")
        self.assertIsNotNone(prm)
        self.assertGreater(prm.r2, 0.95)
        plots.main([])
        self.assertTrue((C.fig_dir() / "e2_latency_decode.png").exists())
        # decode must slow down with latency; prefill must slow down with low bandwidth
        e2 = df[(df.experiment == "e2") & (df.n_nodes == 2)].sort_values("rtt_ms")
        self.assertGreater(e2.tg_ts.iloc[0], e2.tg_ts.iloc[-1])
        # latency-aware plan must beat the even split in the simulator
        e5 = df[df.experiment == "e5"].set_index(df[df.experiment == "e5"].note.str.split().str[0])
        self.assertGreater(e5.loc["plan-latency", "tg_ts"], e5.loc["even", "tg_ts"])
        os.environ.pop("EDGESPLIT_MOCK")


if __name__ == "__main__":
    unittest.main()
