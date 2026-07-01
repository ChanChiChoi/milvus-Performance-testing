#!/usr/bin/env python3
from __future__ import annotations

import os
for _key in ('http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'all_proxy', 'ALL_PROXY', 'no_proxy', 'NO_PROXY', 'htp_proxy'):
    os.environ[_key] = ''

import argparse
import time
from pathlib import Path

import h5py
import numpy as np

from milvus_bench_common import (
    COMMON_DATA_VERSION,
    H5SliceSpec,
    add_connection_args,
    config_from_args,
    ensure_database,
    get_thread_client,
    make_client,
    remove_tmp_and_replace,
    shard_row_count,
)
from update_common import (
    UpdateOperationResult,
    decode_record_common,
    expected_shard_paths,
    flush_collection,
    get_process_state,
    h5_rows,
    init_update_process_worker,
    iter_update_h5_slice_specs,
    iter_update_h5_slices,
    load_collection_if_needed,
    print_update_result_table,
    query_delete_insert_records,
    run_update_operations,
    update_shards_are_complete,
    validate_ready_shards,
    write_mixed_update_shard,
)

DEFAULT_COLLECTION = 'test_vector'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Update benchmark for single FLOAT16 vectors')
    add_connection_args(parser)
    parser.add_argument('--total-rows', type=int, default=1_000_000)
    parser.add_argument('--shard-rows', type=int, default=250_000)
    parser.add_argument('--ready-data-dir', required=True)
    parser.add_argument('--data-dir', default='data/update_test_vector')
    parser.add_argument('--generation-batch-size', type=int, default=2048)
    parser.add_argument('--force-regenerate', action='store_true')
    parser.add_argument('--skip-generate', action='store_true')
    parser.add_argument('--generate-only', action='store_true')
    parser.add_argument('--collection-name', default=DEFAULT_COLLECTION)
    parser.add_argument('--replica-number', type=int, default=2)
    parser.add_argument('--vector-dim', type=int, default=1024)
    parser.add_argument('--insert-batch-size', type=int, default=1000)
    parser.add_argument('--prefetch-batches', type=int, default=0)
    parser.add_argument('--executor-kind', choices=['thread', 'process'], default='process')
    parser.add_argument('--concurrency', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20260610)
    parser.add_argument('--random-ratio', type=int, required=True, help='Random row percentage in each 100-row window, 1-100')
    parser.add_argument('--no-load', action='store_true', help='Do not call load_collection before update')
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    errors = []
    if args.total_rows <= 0:
        errors.append(f'--total-rows must be positive, got {args.total_rows}')
    if args.shard_rows <= 0:
        errors.append(f'--shard-rows must be positive, got {args.shard_rows}')
    if args.vector_dim <= 0:
        errors.append(f'--vector-dim must be positive, got {args.vector_dim}')
    if args.replica_number <= 0:
        errors.append(f'--replica-number must be positive, got {args.replica_number}')
    if args.insert_batch_size <= 0:
        errors.append(f'--insert-batch-size must be positive, got {args.insert_batch_size}')
    if args.concurrency <= 0:
        errors.append(f'--concurrency must be positive, got {args.concurrency}')
    if args.prefetch_batches < 0:
        errors.append(f'--prefetch-batches must be >= 0, got {args.prefetch_batches}')
    if args.generation_batch_size <= 0:
        errors.append(f'--generation-batch-size must be positive, got {args.generation_batch_size}')
    if args.random_ratio < 1 or args.random_ratio > 100:
        errors.append(f'--random-ratio must be between 1 and 100, got {args.random_ratio}')
    if errors:
        raise SystemExit('Invalid arguments:\n  ' + '\n  '.join(errors))


def generate_shards(args: argparse.Namespace) -> None:
    ready_paths = validate_ready_shards(
        args.ready_data_dir,
        args.total_rows,
        args.shard_rows,
        random_ratio=args.random_ratio,
        kind='single_vector',
        attrs={'vector_dim': args.vector_dim, 'common_data_version': COMMON_DATA_VERSION},
        required_datasets=('id', 'uuid1', 'uuid2', 'platform', 'text', 'vector'),
    )
    attrs = {
        'kind': 'single_vector_update',
        'rows': 0,
        'shard_rows': args.shard_rows,
        'total_rows': args.total_rows,
        'vector_dim': args.vector_dim,
        'common_data_version': COMMON_DATA_VERSION,
        'random_ratio': args.random_ratio,
    }
    complete_attrs = {k: v for k, v in attrs.items() if k != 'rows'}
    if not args.force_regenerate and update_shards_are_complete(
        args.data_dir, args.total_rows, args.shard_rows, required_attrs=complete_attrs
    ):
        print(f'update shards already complete: {args.data_dir}')
        return

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    paths = expected_shard_paths(data_dir, args.total_rows, args.shard_rows)
    for shard_idx, path in enumerate(paths):
        rows = shard_row_count(args.total_rows, args.shard_rows, shard_idx)
        tmp_path = path.with_suffix('.h5.tmp')
        if tmp_path.exists():
            tmp_path.unlink()
        print(f'generating {path} rows={rows}')
        shard_attrs = dict(attrs)
        shard_attrs['rows'] = rows
        write_mixed_update_shard(
            tmp_path=tmp_path,
            ready_path=ready_paths[shard_idx],
            rows=rows,
            global_start_id=shard_idx * args.shard_rows + 1,
            random_start_id=args.total_rows + shard_idx * args.shard_rows + 1,
            random_ratio=args.random_ratio,
            rng=rng,
            attrs=shard_attrs,
            vector_shape_tail=(args.vector_dim,),
            vector_chunks=(min(max(1, args.generation_batch_size), rows), args.vector_dim),
        )
        remove_tmp_and_replace(tmp_path, path)


def batch_to_records(batch: dict[str, np.ndarray]) -> list[dict[str, object]]:
    vectors = batch['vector']
    records: list[dict[str, object]] = []
    for idx in range(len(batch['id'])):
        record = decode_record_common(batch, idx)
        record['vector'] = vectors[idx]
        records.append(record)
    return records


def process_update_worker(spec: H5SliceSpec) -> UpdateOperationResult:
    state = get_process_state()
    worker_start = time.perf_counter()
    read_start = worker_start
    with h5py.File(spec.path, 'r') as h5:
        batch = {name: h5[name][spec.start:spec.end] for name in ('id', 'uuid1', 'uuid2', 'platform', 'text', 'vector', 'source')}
    read_elapsed = time.perf_counter() - read_start
    prep_start = time.perf_counter()
    records = batch_to_records(batch)
    prep_elapsed = time.perf_counter() - prep_start
    try:
        client = get_thread_client(state['config'])  # type: ignore[arg-type]
        query_elapsed, delete_elapsed, insert_elapsed, deleted_rows, query_count, delete_count, insert_count = query_delete_insert_records(
            client,
            collection_name=state['collection_name'],  # type: ignore[arg-type]
            records=records,
            timeout=state['timeout'],  # type: ignore[arg-type]
        )
        return UpdateOperationResult(
            len(records),
            0,
            time.perf_counter() - worker_start,
            read_elapsed=read_elapsed,
            prep_elapsed=prep_elapsed,
            query_elapsed=query_elapsed,
            delete_elapsed=delete_elapsed,
            insert_elapsed=insert_elapsed,
            rpc_elapsed=query_elapsed + delete_elapsed + insert_elapsed,
            query_count=query_count,
            delete_count=delete_count,
            insert_count=insert_count,
            deleted_rows=deleted_rows,
        )
    except Exception as exc:
        return UpdateOperationResult(0, len(records), time.perf_counter() - worker_start, repr(exc), read_elapsed, prep_elapsed)


def update_records(args: argparse.Namespace) -> None:
    config = config_from_args(args)
    ensure_database(config, args.create_db)
    client = make_client(config)
    try:
        if not client.has_collection(args.collection_name):
            raise RuntimeError(f'collection {args.collection_name!r} does not exist; run insert script first')
        load_collection_if_needed(client, args.collection_name, args.replica_number, args.timeout, args.no_load)
    finally:
        client.close()

    paths = expected_shard_paths(args.data_dir, args.total_rows, args.shard_rows)
    total_rows = sum(h5_rows(path) for path in paths)
    prefetch_batches = args.prefetch_batches if args.prefetch_batches > 0 else args.concurrency * 2

    if args.executor_kind == 'process':
        batches = iter_update_h5_slice_specs(paths, args.insert_batch_size)
        result = run_update_operations(
            batches,
            total_units=total_rows,
            concurrency=args.concurrency,
            worker=process_update_worker,
            unit_count=lambda spec: spec.rows,
            description='update single',
            prefetch_batches=prefetch_batches,
            executor_kind='process',
            executor_initializer=init_update_process_worker,
            executor_initargs=(config, args.collection_name, args.timeout, 'single'),
        )
    else:
        def worker(batch: dict[str, np.ndarray]) -> UpdateOperationResult:
            worker_start = time.perf_counter()
            read_elapsed = float(batch.pop('_read_elapsed', 0.0))
            prep_start = time.perf_counter()
            records = batch_to_records(batch)
            prep_elapsed = time.perf_counter() - prep_start
            try:
                client = get_thread_client(config)
                query_elapsed, delete_elapsed, insert_elapsed, deleted_rows, query_count, delete_count, insert_count = query_delete_insert_records(
                    client,
                    collection_name=args.collection_name,
                    records=records,
                    timeout=args.timeout,
                )
                return UpdateOperationResult(
                    len(records),
                    0,
                    time.perf_counter() - worker_start,
                    read_elapsed=read_elapsed,
                    prep_elapsed=prep_elapsed,
                    query_elapsed=query_elapsed,
                    delete_elapsed=delete_elapsed,
                    insert_elapsed=insert_elapsed,
                    rpc_elapsed=query_elapsed + delete_elapsed + insert_elapsed,
                    query_count=query_count,
                    delete_count=delete_count,
                    insert_count=insert_count,
                    deleted_rows=deleted_rows,
                )
            except Exception as exc:
                return UpdateOperationResult(0, len(records), time.perf_counter() - worker_start, repr(exc), read_elapsed, prep_elapsed)

        result = run_update_operations(
            iter_update_h5_slices(paths, args.insert_batch_size),
            total_units=total_rows,
            concurrency=args.concurrency,
            worker=worker,
            unit_count=lambda batch: len(batch['id']),
            description='update single',
            prefetch_batches=prefetch_batches,
        )

    flush_collection(config, args.collection_name, args.timeout)
    print_update_result_table(
        'single vector update result',
        result,
        metadata={
            'uri': config.uri,
            'db_name': config.db_name,
            'collection': args.collection_name,
            'replica_number': args.replica_number,
            'ready_data_dir': args.ready_data_dir,
            'data_dir': args.data_dir,
            'total_rows': total_rows,
            'shard_rows': args.shard_rows,
            'vector_dim': args.vector_dim,
            'insert_batch_size': args.insert_batch_size,
            'random_ratio': args.random_ratio,
            'concurrency': args.concurrency,
            'prefetch_batches': prefetch_batches,
            'executor_kind': args.executor_kind,
        },
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    if not args.skip_generate:
        generate_shards(args)
    if args.generate_only:
        return
    update_records(args)


if __name__ == '__main__':
    main()
