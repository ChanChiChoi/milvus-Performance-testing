from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from typing import Any

import h5py
import numpy as np
from tqdm import tqdm

from milvus_bench_common import (
    PLATFORMS,
    H5SliceSpec,
    MilvusConfig,
    decode_bytes,
    get_thread_client,
    init_common_datasets,
    make_text_array,
    make_uuid_array,
    maybe_flush,
    milvus_string_literal,
    percentile_stats,
    shard_row_count,
)


@dataclass
class UpdateOperationResult:
    ok_count: int
    fail_count: int
    elapsed: float | None = None
    error: str | None = None
    read_elapsed: float | None = None
    prep_elapsed: float | None = None
    query_elapsed: float | None = None
    delete_elapsed: float | None = None
    insert_elapsed: float | None = None
    rpc_elapsed: float | None = None
    query_count: int = 0
    delete_count: int = 0
    insert_count: int = 0
    deleted_rows: int = 0


@dataclass
class UpdateBenchmarkResult:
    ok_count: int
    fail_count: int
    latencies: list[float]
    read_latencies: list[float]
    prep_latencies: list[float]
    query_latencies: list[float]
    delete_latencies: list[float]
    insert_latencies: list[float]
    rpc_latencies: list[float]
    wall_elapsed: float
    operation_count: int
    successful_operations: int
    failed_operations: int
    query_count: int
    delete_count: int
    insert_count: int
    deleted_rows: int

    @property
    def total_count(self) -> int:
        return self.ok_count + self.fail_count

    @property
    def success_rate(self) -> float:
        return self.ok_count / self.total_count if self.total_count else 0.0

    @property
    def success_rows_per_second(self) -> float:
        return self.ok_count / self.wall_elapsed if self.wall_elapsed > 0 else 0.0

    @property
    def total_rows_per_second(self) -> float:
        return self.total_count / self.wall_elapsed if self.wall_elapsed > 0 else 0.0

    @property
    def operations_per_second(self) -> float:
        return self.operation_count / self.wall_elapsed if self.wall_elapsed > 0 else 0.0


_PROCESS_STATE: dict[str, object] = {}


def init_update_process_worker(config: MilvusConfig, collection_name: str, timeout: float, mode: str) -> None:
    _PROCESS_STATE.clear()
    _PROCESS_STATE.update(
        {
            'config': config,
            'collection_name': collection_name,
            'timeout': timeout,
            'mode': mode,
        }
    )


def get_process_state() -> dict[str, object]:
    return _PROCESS_STATE


def expected_shard_paths(data_dir: str | Path, total_rows: int, shard_rows: int) -> list[Path]:
    data_path = Path(data_dir)
    shard_count = math.ceil(total_rows / shard_rows)
    return [data_path / f'part-{idx:04d}.h5' for idx in range(shard_count)]


def h5_rows(path: Path) -> int:
    with h5py.File(path, 'r') as h5:
        return int(h5.attrs['rows'])


def update_shards_are_complete(
    data_dir: str | Path,
    total_rows: int,
    shard_rows: int,
    *,
    required_attrs: dict[str, Any],
) -> bool:
    paths = expected_shard_paths(data_dir, total_rows, shard_rows)
    if not paths or not all(path.exists() for path in paths):
        return False
    row_sum = 0
    for idx, path in enumerate(paths):
        expected_rows = shard_row_count(total_rows, shard_rows, idx)
        try:
            with h5py.File(path, 'r') as h5:
                rows = int(h5.attrs['rows'])
                if rows != expected_rows or 'source' not in h5:
                    return False
                for key, expected_value in required_attrs.items():
                    actual = h5.attrs.get(key)
                    if isinstance(actual, bytes):
                        actual = actual.decode('utf-8')
                    if actual != expected_value:
                        return False
                row_sum += rows
        except Exception:
            return False
    return row_sum == total_rows


def required_ready_rows(update_rows: int, random_ratio: int) -> int:
    ready_rows = 0
    for block_start in range(0, update_rows, 100):
        block_rows = min(100, update_rows - block_start)
        random_rows = block_rows * random_ratio // 100
        ready_rows += block_rows - random_rows
    return ready_rows


def validate_ready_shards(
    ready_data_dir: str | Path,
    total_rows: int,
    shard_rows: int,
    *,
    random_ratio: int,
    kind: str,
    attrs: dict[str, Any],
    required_datasets: tuple[str, ...],
) -> list[Path]:
    paths = expected_shard_paths(ready_data_dir, total_rows, shard_rows)
    if not paths or not all(path.exists() for path in paths):
        raise SystemExit(
            f'--ready-data-dir is incomplete: {ready_data_dir}. '
            'Run the matching insert script first with enough --total-rows/--shard-rows.'
        )
    for idx, path in enumerate(paths):
        update_rows = shard_row_count(total_rows, shard_rows, idx)
        need_rows = required_ready_rows(update_rows, random_ratio)
        with h5py.File(path, 'r') as h5:
            ready_rows = int(h5.attrs['rows'])
            if ready_rows < need_rows:
                raise SystemExit(
                    f'{path} has only {ready_rows} ready rows, but update generation needs {need_rows} '
                    f'for shard {idx} (--total-rows={total_rows}, --shard-rows={shard_rows}, '
                    f'--random-ratio={random_ratio}). Run the matching insert script first to regenerate enough ready data.'
                )
            if h5.attrs.get('kind') != kind:
                raise SystemExit(f'{path} kind mismatch: expected {kind!r}, got {h5.attrs.get("kind")!r}')
            for key, expected_value in attrs.items():
                actual = h5.attrs.get(key)
                if isinstance(actual, bytes):
                    actual = actual.decode('utf-8')
                if actual != expected_value:
                    raise SystemExit(f'{path} attr {key} mismatch: expected {expected_value!r}, got {actual!r}')
            missing = [name for name in required_datasets if name not in h5]
            if missing:
                raise SystemExit(f'{path} missing datasets: {missing}')
    return paths


def iter_update_h5_slice_specs(paths: Iterable[Path], batch_size: int) -> Iterator[H5SliceSpec]:
    for path in paths:
        with h5py.File(path, 'r') as h5:
            rows = int(h5.attrs['rows'])
            for start in range(0, rows, batch_size):
                end = min(start + batch_size, rows)
                yield H5SliceSpec(path=path, start=start, end=end, rows=end - start)


def iter_update_h5_slices(paths: Iterable[Path], batch_size: int) -> Iterator[dict[str, Any]]:
    names = ('id', 'uuid1', 'uuid2', 'platform', 'text', 'vector', 'source')
    for path in paths:
        with h5py.File(path, 'r') as h5:
            rows = int(h5.attrs['rows'])
            for start in range(0, rows, batch_size):
                end = min(start + batch_size, rows)
                read_start = time.perf_counter()
                batch = {name: h5[name][start:end] for name in names}
                batch['_read_elapsed'] = time.perf_counter() - read_start
                yield batch


def copy_ready_rows(ready_h5: h5py.File, out: dict[str, list[np.ndarray]], start: int, end: int) -> None:
    for name in ('id', 'uuid1', 'uuid2', 'platform', 'text', 'vector'):
        out[name].append(ready_h5[name][start:end])
    out['source'].append(np.asarray([b'ready'] * (end - start), dtype='S8'))


def make_random_common_rows(
    *,
    row_count: int,
    random_start_id: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    ids = np.arange(random_start_id, random_start_id + row_count, dtype=np.int64)
    return {
        'id': ids,
        'uuid1': make_uuid_array(ids + 10_000_000_000, salt='update-random-uuid1'),
        'uuid2': make_uuid_array(ids + 10_000_000_000, salt='update-random-uuid2'),
        'platform': rng.choice(PLATFORMS, size=row_count),
        'text': make_text_array(rng, row_count, 16),
        'source': np.asarray([b'random'] * row_count, dtype='S8'),
    }


def write_mixed_update_shard(
    *,
    tmp_path: Path,
    ready_path: Path,
    rows: int,
    global_start_id: int,
    random_start_id: int,
    random_ratio: int,
    rng: np.random.Generator,
    attrs: dict[str, Any],
    vector_shape_tail: tuple[int, ...],
    vector_chunks: tuple[int, ...],
) -> None:
    columns: dict[str, list[np.ndarray]] = {name: [] for name in ('id', 'uuid1', 'uuid2', 'platform', 'text', 'vector', 'source')}
    ready_offset = 0
    next_random_id = random_start_id
    with h5py.File(ready_path, 'r') as ready_h5:
        for block_start in range(0, rows, 100):
            block_rows = min(100, rows - block_start)
            random_rows = block_rows * random_ratio // 100
            ready_rows = block_rows - random_rows
            block: dict[str, list[np.ndarray]] = {name: [] for name in columns}
            if ready_rows:
                copy_ready_rows(ready_h5, block, ready_offset, ready_offset + ready_rows)
                ready_offset += ready_rows
            if random_rows:
                common = make_random_common_rows(row_count=random_rows, random_start_id=next_random_id, rng=rng)
                next_random_id += random_rows
                for name in ('id', 'uuid1', 'uuid2', 'platform', 'text', 'source'):
                    block[name].append(common[name])
                vector_shape = (random_rows, *vector_shape_tail)
                block['vector'].append(rng.random(vector_shape, dtype=np.float32).astype(np.float16))
            block_arrays = {name: np.concatenate(parts, axis=0) for name, parts in block.items()}
            order = rng.permutation(block_rows)
            for name, values in block_arrays.items():
                columns[name].append(values[order])

    arrays = {name: np.concatenate(parts, axis=0) for name, parts in columns.items()}
    with h5py.File(tmp_path, 'w') as h5:
        for key, value in attrs.items():
            h5.attrs[key] = value
        init_common_datasets(h5, rows, global_start_id)
        h5.create_dataset('vector', shape=(rows, *vector_shape_tail), dtype='float16', chunks=vector_chunks)
        h5.create_dataset('source', shape=(rows,), dtype='S8')
        for name, values in arrays.items():
            h5[name][:] = values


def decode_record_common(batch: dict[str, np.ndarray], idx: int) -> dict[str, object]:
    return {
        'pk': int(batch['id'][idx]),
        'id': int(batch['id'][idx]),
        'uuid1': decode_bytes(batch['uuid1'][idx]),
        'uuid2': decode_bytes(batch['uuid2'][idx]),
        'platform': decode_bytes(batch['platform'][idx]),
        'text': decode_bytes(batch['text'][idx]),
        'source': decode_bytes(batch['source'][idx]),
    }


def record_delete_filter(record: dict[str, object]) -> str:
    uuid1 = milvus_string_literal(record['uuid1'])
    uuid2 = milvus_string_literal(record['uuid2'])
    return f'uuid1 == {uuid1} and uuid2 == {uuid2}'


def batch_query_filter(records: list[dict[str, object]]) -> str:
    uuid_values = ', '.join(milvus_string_literal(record['uuid1']) for record in records)
    return f'uuid1 in [{uuid_values}]'


def batch_pk_filter(pk_values: list[int]) -> str:
    return 'pk in [' + ', '.join(str(pk) for pk in pk_values) + ']'


def query_delete_insert_records(
    client: Any,
    *,
    collection_name: str,
    records: list[dict[str, object]],
    timeout: float,
) -> tuple[float, float, float, int, int, int, int]:
    if not records:
        return 0.0, 0.0, 0.0, 0, 0, 0, 0

    expected_keys = {(str(record['uuid1']), str(record['uuid2'])) for record in records}
    query_start = time.perf_counter()
    rows = client.query(
        collection_name=collection_name,
        filter=batch_query_filter(records),
        output_fields=['pk', 'uuid1', 'uuid2'],
        timeout=timeout,
    )
    query_elapsed = time.perf_counter() - query_start
    query_count = 1

    delete_pks: list[int] = []
    seen_pks: set[int] = set()
    for row in rows or []:
        key = (str(row.get('uuid1')), str(row.get('uuid2')))
        pk = row.get('pk')
        if key in expected_keys and pk is not None:
            pk_value = int(pk)
            if pk_value not in seen_pks:
                seen_pks.add(pk_value)
                delete_pks.append(pk_value)

    delete_elapsed = 0.0
    delete_count = 0
    if delete_pks:
        delete_start = time.perf_counter()
        client.delete(collection_name=collection_name, filter=batch_pk_filter(delete_pks), timeout=timeout)
        delete_elapsed = time.perf_counter() - delete_start
        delete_count = 1

    insert_records = [{key: value for key, value in record.items() if key != 'source'} for record in records]
    insert_start = time.perf_counter()
    client.insert(collection_name=collection_name, data=insert_records, timeout=timeout)
    insert_elapsed = time.perf_counter() - insert_start
    insert_count = 1
    return query_elapsed, delete_elapsed, insert_elapsed, len(delete_pks), query_count, delete_count, insert_count


def prefetch_iterable(batches: Iterable[Any], max_prefetch: int) -> Iterator[Any]:
    if max_prefetch <= 0:
        yield from batches
        return

    queue: Queue[tuple[str, Any]] = Queue(maxsize=max_prefetch)

    def producer() -> None:
        try:
            for batch in batches:
                queue.put(('item', batch))
        except BaseException as exc:
            queue.put(('error', exc))
        finally:
            queue.put(('done', None))

    thread = threading.Thread(target=producer, name='update-h5-prefetch', daemon=True)
    thread.start()
    while True:
        kind, value = queue.get()
        if kind == 'item':
            yield value
        elif kind == 'error':
            raise value
        else:
            return


def run_update_operations(
    batches: Iterable[Any],
    *,
    total_units: int,
    concurrency: int,
    worker: Callable[[Any], UpdateOperationResult],
    unit_count: Callable[[Any], int],
    description: str,
    prefetch_batches: int = 0,
    executor_kind: str = 'thread',
    executor_initializer: Callable[..., None] | None = None,
    executor_initargs: tuple[Any, ...] = (),
) -> UpdateBenchmarkResult:
    ok_count = 0
    fail_count = 0
    operation_count = 0
    successful_operations = 0
    failed_operations = 0
    latencies: list[float] = []
    read_latencies: list[float] = []
    prep_latencies: list[float] = []
    query_latencies: list[float] = []
    delete_latencies: list[float] = []
    insert_latencies: list[float] = []
    rpc_latencies: list[float] = []
    query_count = 0
    delete_count = 0
    insert_count = 0
    deleted_rows = 0
    pending: set[Any] = set()
    max_pending = max(1, concurrency * 2)
    wall_start = time.perf_counter()

    def collect(done: Iterable[Any], pbar: tqdm) -> None:
        nonlocal ok_count, fail_count, operation_count, successful_operations, failed_operations, query_count, delete_count, insert_count, deleted_rows
        for future in done:
            try:
                result = future.result()
            except Exception as exc:
                result = UpdateOperationResult(0, 0, None, repr(exc))
            operation_count += 1
            ok_count += result.ok_count
            fail_count += result.fail_count
            query_count += result.query_count
            delete_count += result.delete_count
            insert_count += result.insert_count
            deleted_rows += result.deleted_rows
            if result.elapsed is not None and result.ok_count > 0:
                latencies.append(result.elapsed)
            if result.read_elapsed is not None and result.ok_count > 0:
                read_latencies.append(result.read_elapsed)
            if result.prep_elapsed is not None and result.ok_count > 0:
                prep_latencies.append(result.prep_elapsed)
            if result.query_elapsed is not None and result.ok_count > 0:
                query_latencies.append(result.query_elapsed)
            if result.delete_elapsed is not None and result.ok_count > 0:
                delete_latencies.append(result.delete_elapsed)
            if result.insert_elapsed is not None and result.ok_count > 0:
                insert_latencies.append(result.insert_elapsed)
            if result.rpc_elapsed is not None and result.ok_count > 0:
                rpc_latencies.append(result.rpc_elapsed)
            if result.fail_count > 0 or result.error:
                failed_operations += 1
            else:
                successful_operations += 1
            if result.error:
                tqdm.write(result.error)
            pbar.update(result.ok_count + result.fail_count)

    executor_cls: type[ThreadPoolExecutor] | type[ProcessPoolExecutor]
    executor_kwargs: dict[str, Any] = {'max_workers': max(1, concurrency)}
    if executor_kind == 'process':
        import multiprocessing as mp

        executor_cls = ProcessPoolExecutor
        executor_kwargs['mp_context'] = mp.get_context('spawn')
    else:
        executor_cls = ThreadPoolExecutor
    if executor_initializer is not None:
        executor_kwargs['initializer'] = executor_initializer
        if executor_initargs:
            executor_kwargs['initargs'] = executor_initargs

    with executor_cls(**executor_kwargs) as executor:
        with tqdm(total=total_units, desc=description, unit='row') as pbar:
            for batch in prefetch_iterable(batches, prefetch_batches):
                if unit_count(batch) <= 0:
                    continue
                while len(pending) >= max_pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    collect(done, pbar)
                pending.add(executor.submit(worker, batch))
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                collect(done, pbar)

    return UpdateBenchmarkResult(
        ok_count=ok_count,
        fail_count=fail_count,
        latencies=latencies,
        read_latencies=read_latencies,
        prep_latencies=prep_latencies,
        query_latencies=query_latencies,
        delete_latencies=delete_latencies,
        insert_latencies=insert_latencies,
        rpc_latencies=rpc_latencies,
        wall_elapsed=time.perf_counter() - wall_start,
        operation_count=operation_count,
        successful_operations=successful_operations,
        failed_operations=failed_operations,
        query_count=query_count,
        delete_count=delete_count,
        insert_count=insert_count,
        deleted_rows=deleted_rows,
    )


def print_update_result_table(title: str, result: UpdateBenchmarkResult, *, metadata: dict[str, Any] | None = None) -> None:
    rows: list[tuple[str, str]] = []
    if metadata:
        rows.extend((key, str(value)) for key, value in metadata.items())
        rows.append(('-', '-'))
    overall = percentile_stats(result.latencies)
    rows.extend(
        [
            ('success_rows', str(result.ok_count)),
            ('failed_rows', str(result.fail_count)),
            ('success_rate', f'{result.success_rate * 100:.2f}%'),
            ('batch_operations', str(result.operation_count)),
            ('query_operations', str(result.query_count)),
            ('delete_operations', str(result.delete_count)),
            ('insert_operations', str(result.insert_count)),
            ('deleted_rows', str(result.deleted_rows)),
            ('successful_operations', str(result.successful_operations)),
            ('failed_operations', str(result.failed_operations)),
            ('wall_elapsed(s)', f'{result.wall_elapsed:.6f}'),
            ('success_rows/sec', f'{result.success_rows_per_second:.2f}'),
            ('total_rows/sec', f'{result.total_rows_per_second:.2f}'),
            ('operations/sec', f'{result.operations_per_second:.2f}'),
            ('latency_samples', str(len(result.latencies))),
            ('TP50(s)', f'{overall["TP50"]:.6f}'),
            ('TP90(s)', f'{overall["TP90"]:.6f}'),
            ('TP99(s)', f'{overall["TP99"]:.6f}'),
        ]
    )
    rows.append(('-', '以下为 update 阶段分项指标'))
    for label, latencies in (
        ('query', result.query_latencies),
        ('delete', result.delete_latencies),
        ('insert', result.insert_latencies),
    ):
        stats = percentile_stats(latencies)
        rows.extend(
            [
                (f'{label}_latency_samples', str(len(latencies))),
                (f'{label}_TP50(s)', f'{stats["TP50"]:.6f}'),
                (f'{label}_TP90(s)', f'{stats["TP90"]:.6f}'),
                (f'{label}_TP99(s)', f'{stats["TP99"]:.6f}'),
            ]
        )
    if result.read_latencies or result.prep_latencies or result.rpc_latencies:
        rows.append(('-', '以下为辅助验证脚本指标'))
    for label, latencies in (('read', result.read_latencies), ('prep', result.prep_latencies), ('rpc', result.rpc_latencies)):
        if not latencies:
            continue
        stats = percentile_stats(latencies)
        rows.extend(
            [
                (f'{label}_latency_samples', str(len(latencies))),
                (f'{label}_TP50(s)', f'{stats["TP50"]:.6f}'),
                (f'{label}_TP90(s)', f'{stats["TP90"]:.6f}'),
                (f'{label}_TP99(s)', f'{stats["TP99"]:.6f}'),
            ]
        )
    width = max(len(k) for k, _ in rows + [(title, '')])
    print('\n' + title)
    print('-' * (width + 22))
    for key, value in rows:
        print(f'{key:<{width}} | {value}')
    print()
    print('结果解读提示')
    print('- TP50/TP90/TP99 统计的是一次 update batch operation 的耗时。')
    print('- query_TP*/delete_TP*/insert_TP* 分别表示一次 batch query、batch delete、batch insert 的耗时，不是单条记录平均耗时。')
    print('- rpc_TP* 表示同一个 batch 内 query + delete + insert 的总 RPC 耗时。')
    print('- query_operations/delete_operations/insert_operations 是实际 RPC 次数；deleted_rows 才是实际删除的旧记录数。')
    print('- 如果需要估算单条记录耗时，可以结合 insert_batch_size、success_rows、deleted_rows 和各阶段 TP 粗略折算。')


def load_collection_if_needed(client: Any, collection_name: str, replica_number: int, timeout: float, no_load: bool) -> None:
    if no_load:
        return
    client.load_collection(collection_name, replica_number=replica_number, timeout=timeout)


def flush_collection(config: MilvusConfig, collection_name: str, timeout: float) -> None:
    from milvus_bench_common import make_client

    client = make_client(config)
    try:
        maybe_flush(client, collection_name, timeout)
    finally:
        client.close()
