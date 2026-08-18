import tempfile
import unittest
import copy
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from scipy.sparse import coo_matrix
from torch.utils.data import Dataset

from bert4_task2 import collate as task2_collate, run_epoch as run_task2_epoch
from models.collatefn import MLMCollateFn
from models.model import CFGFusionModel
from models.retrieval import build_retrieval_sets
from models.tokenizer import AsmTokenizer
from models.batch_sampler import WorkloadBatchSampler
from models.finetune_trainer import BERTFinetuneTrainer
from models.graph_dataset import _encode_blocks
from normalize_instr import discover_binary_paths


class _FakeBert(nn.Module):
    def __init__(self, hidden_size=4):
        super().__init__()
        self.config = SimpleNamespace(pad_token_id=0)
        self.embedding = nn.Embedding(64, hidden_size)

    def encode_block_embeddings(self, input_ids, attention_mask=None,
                                token_type_ids=None):
        values = self.embedding(input_ids)
        mask = attention_mask.unsqueeze(-1)
        return (values * mask).sum(1) / mask.sum(1).clamp_min(1)


def _graph(n, seq_len=8):
    ids = torch.randint(1, 32, (n, seq_len))
    adj = torch.zeros(n, n)
    adj[torch.arange(n - 1), torch.arange(1, n)] = 1
    return ids, adj


class NativeGraphForwardTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = CFGFusionModel(
            _FakeBert(), d_model=4, mpnn_readout_dim=4,
            cnn_out=3, hidden_dim=5,
        ).eval()

    def test_embedding_does_not_depend_on_batch_companion(self):
        small_ids, small_adj = _graph(6)
        large_ids, large_adj = _graph(15)
        with torch.no_grad():
            alone = self.model([small_ids], [small_adj])[0]
            together = self.model(
                [small_ids, large_ids], [small_adj, large_adj])[0]
        torch.testing.assert_close(alone, together)

    def test_dense_tensor_input_returns_embeddings(self):
        ids, adj = _graph(6)
        with torch.no_grad():
            result = self.model(ids.unsqueeze(0), adj.unsqueeze(0))
        self.assertEqual(result.shape, (1, 5))

    def test_sparse_adjacency_matches_dense_adjacency(self):
        ids, adj = _graph(9)
        with torch.no_grad():
            dense_result = self.model([ids], [adj])
            sparse_result = self.model([ids], [adj.to_sparse().coalesce()])
        torch.testing.assert_close(dense_result, sparse_result)

    def test_same_size_fusion_preserves_independent_training_output(self):
        first_ids, first_adj = _graph(8)
        second_ids, second_adj = _graph(8)
        batched = copy.deepcopy(self.model).train()
        separate = copy.deepcopy(self.model).train()

        together = batched(
            [first_ids, second_ids], [first_adj, second_adj])
        alone = torch.cat([
            separate([first_ids], [first_adj]),
            separate([second_ids], [second_adj]),
        ])
        torch.testing.assert_close(together, alone, rtol=2e-5, atol=2e-5)

    def test_multiple_native_size_groups_support_backward(self):
        graphs = [_graph(n) for n in (8, 8, 9, 9)]
        model = copy.deepcopy(self.model).train()
        output = model(
            [ids for ids, _ in graphs], [adj for _, adj in graphs])
        output.square().mean().backward()
        self.assertIsNotNone(model.order_cnn.conv_in.weight.grad)


class BinaryDiscoveryTests(unittest.TestCase):
    def test_angr_sidecar_directory_is_not_a_binary(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as root:
            project = Path(root) / "openssl"
            project.mkdir()
            binary = project / "x64-gcc-9-O2_libcrypto.so.3"
            binary.write_bytes(b"\x7fELF" + b"\0" * 12)
            (project / "x64-gcc-9-O2_libcrypto.so.3_angr_rtdb").mkdir()
            (project / "arm64-gcc-9-O2_not-a-binary.txt").write_text("no")

            selected = discover_binary_paths(
                root, ["x64", "arm64"], ["gcc"])

        self.assertEqual(selected, [str(binary)])


class RetrievalProtocolTests(unittest.TestCase):
    @staticmethod
    def _key(version, arch, target, name="func", opt="O2"):
        return (name, "gcc", version, opt, arch,
                f"{arch}-gcc-{version}-{opt}_{target}")

    def test_all_same_source_versions_are_relevant(self):
        anchors = [self._key("4.8", "x64", "binary"),
                   self._key("9.0", "x64", "binary")]
        candidates = [self._key(v, "arm64", "binary")
                      for v in ("4.8", "5.0", "7.0", "9.0")]
        candidates.append(self._key("9.0", "arm64", "other", name="other"))
        retrieval, pool = build_retrieval_sets(anchors, candidates, "O2")
        self.assertEqual(len(retrieval), 2)
        self.assertEqual(len(pool), 5)
        self.assertEqual([positions for _, positions in retrieval],
                         [[0, 1, 2, 3], [0, 1, 2, 3]])


class MLMValidationTests(unittest.TestCase):
    def test_validation_input_is_actually_masked(self):
        tokenizer = AsmTokenizer()
        text = " ".join(["mov rax rbx add rcx rdx"] * 20)
        tokenizer.build_vocab([text])
        collate = MLMCollateFn(
            tokenizer, seq_len=64, max_samples=1, train=False)
        torch.manual_seed(11)
        batch = collate([{"instruction_blocks": [text]}])
        supervised = batch["labels"] != -100
        self.assertTrue(supervised.any())
        self.assertTrue((batch["input_ids"][supervised]
                         == tokenizer.mask_token_id).any())


class GraphTokenizationTests(unittest.TestCase):
    def test_fast_block_encoding_matches_hf_api(self):
        tokenizer = AsmTokenizer()
        blocks = ["mov rax <IMM>", "add [rbp+rax] verylongtoken " * 20]
        tokenizer.build_vocab(blocks)
        expected = torch.stack([
            torch.tensor(tokenizer(
                block, max_length=16, padding="max_length",
                truncation=True, verbose=False)["input_ids"])
            for block in blocks
        ])
        actual = _encode_blocks(tokenizer, blocks, 16)
        torch.testing.assert_close(actual, expected)


class _PairDataset(Dataset):
    def __len__(self):
        return 3

    def graph_sizes(self):
        return [3, 4, 5]

    def __getitem__(self, index):
        n = index + 3
        a = torch.full((n, 4), index + 1, dtype=torch.long)
        b = torch.full((n, 4), index + 2, dtype=torch.long)
        adj = coo_matrix((n, n), dtype=float)
        label = torch.tensor(1.0 if index != 1 else -1.0)
        return a, adj, b, adj, label


class _TinyPairModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, ids, adj):
        outputs = []
        for graph in ids:
            mean = graph.float().mean()
            outputs.append(torch.stack(
                (self.weight * mean, self.weight.square() + mean)))
        return torch.stack(outputs)


class FinetuneTrainerTests(unittest.TestCase):
    def test_epoch_handles_float_loss_and_accumulation_tail(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as checkpoint_dir:
            config = SimpleNamespace(
                device="cpu", lr=1e-4, betas=(0.9, 0.999),
                weight_decay=0.0, margin=0.0, use_amp=False,
                grad_accum=2, checkpoint_save_path=checkpoint_dir,
                num_workers=0, batch_size=2, seed=3, use_bucketing=True,
                node_budget=100,
            )
            trainer = BERTFinetuneTrainer(_TinyPairModel(), config)
            loss, accuracy = trainer._run_epoch(
                _PairDataset(), epoch=0, train=True)
        self.assertIsInstance(loss, float)
        self.assertGreaterEqual(loss, 0.0)
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)


class WorkloadBatchSamplerTests(unittest.TestCase):
    def test_skewed_work_is_balanced_without_losing_samples(self):
        costs = list(range(1, 101))
        sampler = WorkloadBatchSampler(
            costs, batch_size=10, shuffle=False, seed=3)
        batches = list(sampler)
        flattened = [index for batch in batches for index in batch]
        work = [sum(costs[index] for index in batch) for batch in batches]

        self.assertEqual(sorted(flattened), list(range(len(costs))))
        self.assertTrue(all(len(batch) == 10 for batch in batches))
        self.assertLessEqual(max(work) - min(work), 10)

    def test_epoch_changes_shuffle_but_not_coverage(self):
        sampler = WorkloadBatchSampler(
            [index // 3 for index in range(40)], batch_size=4,
            shuffle=True, seed=7)
        first = list(sampler)
        sampler.set_epoch(1)
        second = list(sampler)

        self.assertNotEqual(first, second)
        self.assertEqual(sorted(sum(first, [])), list(range(40)))
        self.assertEqual(sorted(sum(second, [])), list(range(40)))

    def test_exact_shapes_form_full_native_batches_first(self):
        shapes = ["small"] * 23 + ["large"] * 12 + ["odd"] * 4
        sampler = WorkloadBatchSampler(
            [1] * 23 + [100] * 12 + [7] * 4,
            batch_size=10, shuffle=False, shapes=shapes)
        batches = list(sampler)
        full_shape_batches = [batch for batch in batches
                              if len(batch) == 10
                              and len({shapes[index] for index in batch}) == 1]

        self.assertEqual(len(full_shape_batches), 3)
        self.assertEqual(sorted(sum(batches, [])), list(range(len(shapes))))


class _Task2Dataset(Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, index):
        n = index + 2
        ids = torch.full((n, 4), index + 1, dtype=torch.long)
        return ids, coo_matrix((n, n), dtype=float), torch.tensor(index)


class _CountingSGD(torch.optim.SGD):
    def __init__(self, parameters, **kwargs):
        super().__init__(parameters, **kwargs)
        self.step_count = 0

    def step(self, closure=None):
        self.step_count += 1
        return super().step(closure)


class Task2BatchTests(unittest.TestCase):
    def test_memory_groups_share_one_logical_batch_step(self):
        model = _TinyPairModel()
        classifier = nn.Linear(2, 2)
        optimizer = _CountingSGD(
            list(model.parameters()) + list(classifier.parameters()), lr=1e-3)
        loader = torch.utils.data.DataLoader(
            _Task2Dataset(), batch_size=2, collate_fn=task2_collate)
        config = SimpleNamespace(use_amp=False, node_budget=1)
        loss, accuracy = run_task2_epoch(
            model, classifier, loader, torch.device("cpu"), True, config,
            optimizer, scaler=None)
        self.assertEqual(optimizer.step_count, 1)
        self.assertGreaterEqual(loss, 0.0)
        self.assertGreaterEqual(accuracy, 0.0)


if __name__ == "__main__":
    unittest.main()
