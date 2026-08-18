"""Work-balanced logical batch scheduling for variable-size CFGs."""
import math
import random


class WorkloadBatchSampler:
    """Build fixed-size batches with approximately equal estimated work.

    ``workloads`` may be a node count or any monotonic work estimate.
    Serpentine rank assignment pairs expensive samples with cheap ones,
    keeping the paper's logical batch size while smoothing GPU work.
    """

    def __init__(self, workloads, batch_size, shuffle=True, seed=42,
                 shapes=None):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.workloads = list(workloads)
        self.shapes = list(shapes) if shapes is not None else None
        if self.shapes is not None and len(self.shapes) != len(self.workloads):
            raise ValueError("shapes and workloads must have equal lengths")
        self._len = math.ceil(len(self.workloads) / batch_size)

    def set_epoch(self, epoch):
        """Change only the stochastic ordering, not the work balance."""
        self.epoch = epoch

    def __iter__(self):
        if not self.workloads:
            return iter(())
        rng = random.Random(self.seed + self.epoch)
        if self.shapes is None:
            batches = self._balance(list(range(len(self.workloads))), rng)
        else:
            # Full exact-shape batches unlock native (unpadded) MPNN/OrderCNN
            # fusion.  Only each shape bucket's remainder needs balancing.
            shape_groups = {}
            for index, shape in enumerate(self.shapes):
                shape_groups.setdefault(shape, []).append(index)
            batches = []
            remainder = []
            for group in shape_groups.values():
                if self.shuffle:
                    rng.shuffle(group)
                full = len(group) // self.batch_size * self.batch_size
                batches.extend(group[start:start + self.batch_size]
                               for start in range(0, full, self.batch_size))
                remainder.extend(group[full:])
            batches.extend(self._balance(remainder, rng))

        if self.shuffle:
            rng.shuffle(batches)
        return iter(batches)

    def _balance(self, indices, rng):
        if not indices:
            return []
        ranked = list(indices)
        if self.shuffle:
            # Stable sort preserves this random tie order.
            rng.shuffle(ranked)
        ranked.sort(key=self.workloads.__getitem__, reverse=True)

        count = math.ceil(len(ranked) / self.batch_size)
        batches = [[] for _ in range(count)]
        for row, start in enumerate(range(0, len(ranked), count)):
            stratum = ranked[start:start + count]
            if row % 2:
                stratum.reverse()
            for batch, sample in zip(batches, stratum):
                batch.append(sample)

        if self.shuffle:
            for batch in batches:
                rng.shuffle(batch)
        return batches

    def __len__(self):
        return self._len
