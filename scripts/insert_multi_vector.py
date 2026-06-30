#!/usr/bin/env python3
from __future__ import annotations

import os
for _key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY", "no_proxy", "NO_PROXY", "htp_proxy"):
    os.environ[_key] = ""

import argparse
import time
from pathlib import Path

import h5py
import numpy as np
from pymilvus import DataType

from milvus_bench_common import (
    COMMON_DATA_VERSION,
    H5SliceSpec,
    MilvusConfig,
    OperationResult,
    delete_ratio_candidates,
    add_connection_args,
    add_data_args,
    config_from_args,
    ensure_database,
    expected_shard_paths,
    fill_common_datasets,
    get_thread_client,
    h5_rows,
    init_common_datasets,
    iter_h5_slice_specs,
    iter_h5_slices,
    make_client,
    maybe_flush,
    remove_tmp_and_replace,
    run_concurrent_operations,
    shard_row_count,
    shards_are_complete,
    print_result_table,
    query_delete_then_insert_records,
)

DEFAULT_COLLECTION = "test_multi_vector"
MODE_STRUCT_FLOAT32 = "struct-float32"
MODE_FLAT_FP16 = "flat-fp16"


_PROCESS_STATE: dict[str, object] = {}


def init_process_worker(config: MilvusConfig, collection_name: str, delete_ratio: int, timeout: float, mode: str) -> None:
    _PROCESS_STATE.clear()
    _PROCESS_STATE.update(
        {
            "config": config,
            "collection_name": collection_name,
            "delete_ratio": delete_ratio,
            "timeout": timeout,
            "mode": mode,
        }
    )


def process_insert_worker(spec: H5SliceSpec) -> OperationResult:
    state = _PROCESS_STATE
    worker_start = time.perf_counter()
    read_start = worker_start
    with h5py.File(spec.path, "r") as h5:
        batch = {name: h5[name][spec.start:spec.end] for name in ("id", "uuid1", "uuid2", "platform", "text", "vector")}
    read_elapsed = time.perf_counter() - read_start
    prep_start = time.perf_counter()
    records = batch_to_records(batch, state["mode"])  # type: ignore[arg-type]
    prep_elapsed = time.perf_counter() - prep_start
    try:
        client = get_thread_client(state["config"])  # type: ignore[arg-type]
        rpc_start = time.perf_counter()
        client.insert(collection_name=state["collection_name"], data=records, timeout=state["timeout"])  # type: ignore[index]
        delete_records = delete_ratio_candidates(records, state["delete_ratio"])  # type: ignore[arg-type]
        if delete_records:
            query_delete_then_insert_records(
                client,
                collection_name=state["collection_name"],  # type: ignore[index]
                records=delete_records,
                timeout=state["timeout"],  # type: ignore[index]
            )
        rpc_elapsed = time.perf_counter() - rpc_start
        elapsed = time.perf_counter() - worker_start
        return OperationResult(len(records), 0, elapsed, read_elapsed=read_elapsed, prep_elapsed=prep_elapsed, rpc_elapsed=rpc_elapsed)
    except Exception as exc:
        elapsed = time.perf_counter() - worker_start
        return OperationResult(0, len(records), elapsed, repr(exc), read_elapsed=read_elapsed, prep_elapsed=prep_elapsed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Insert benchmark for ColBERT-style multi vectors")
    add_connection_args(parser)
    add_data_args(parser, "data/test_multi_vector")
    parser.add_argument("--collection-name", default=DEFAULT_COLLECTION)
    parser.add_argument("--num-shards", type=int, default=8, help="Milvus collection shard count")
    parser.add_argument("--replica-number", type=int, default=2, help="Replica count used when loading collection")
    parser.add_argument("--vector-dim", type=int, default=128)
    parser.add_argument("--token-count", type=int, default=300)
    parser.add_argument("--insert-batch-size", type=int, default=8)
    parser.add_argument("--executor-kind", choices=["thread", "process"], default="thread", help="Use threads or processes for the worker pool")
    parser.add_argument("--prefetch-batches", type=int, default=0, help="Read this many HDF5 batches ahead; 0 uses concurrency * 2")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--multi-vector-mode", choices=[MODE_STRUCT_FLOAT32, MODE_FLAT_FP16], default=MODE_STRUCT_FLOAT32)
    parser.add_argument("--metric-type", default="MAX_SIM_COSINE", help="Metric for struct-float32 mode")
    parser.add_argument("--flat-metric-type", default="COSINE", help="Metric for flat-fp16 mode")
    parser.add_argument("--index-type", default="HNSW")
    parser.add_argument("--hnsw-m", type=int, default=16)
    parser.add_argument("--hnsw-ef-construction", type=int, default=200)
    parser.add_argument(
        "--vector-chunk-rows",
        type=int,
        default=4,
        help="HDF5 chunk row count for the multi-vector dataset; affects local shard IO only",
    )
    parser.add_argument("--seed", type=int, default=20260610)
    parser.add_argument("--delete-ratio", type=int, default=0, help="Delete the last N rows in each 100 inserted rows; 0 disables delete simulation")
    parser.add_argument("--no-drop", action="store_true", help="Do not drop an existing collection")
    return parser.parse_args()


def generate_shards(args: argparse.Namespace) -> None:
    attrs = {"kind": "multi_vector", "vector_dim": args.vector_dim, "token_count": args.token_count, "common_data_version": COMMON_DATA_VERSION}
    if not args.force_regenerate and shards_are_complete(
        args.data_dir, args.total_rows, args.shard_rows, required_attrs=attrs
    ):
        print(f"data shards already complete: {args.data_dir}")
        return

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    paths = expected_shard_paths(data_dir, args.total_rows, args.shard_rows)

    for shard_idx, path in enumerate(paths):
        rows = shard_row_count(args.total_rows, args.shard_rows, shard_idx)
        global_start_id = shard_idx * args.shard_rows + 1
        tmp_path = path.with_suffix(".h5.tmp")
        if tmp_path.exists():
            tmp_path.unlink()
        print(f"generating {path} rows={rows}")
        with h5py.File(tmp_path, "w") as h5:
            h5.attrs["kind"] = "multi_vector"
            h5.attrs["rows"] = rows
            h5.attrs["shard_rows"] = args.shard_rows
            h5.attrs["total_rows"] = args.total_rows
            h5.attrs["vector_dim"] = args.vector_dim
            h5.attrs["common_data_version"] = COMMON_DATA_VERSION
            h5.attrs["token_count"] = args.token_count
            init_common_datasets(h5, rows, global_start_id)
            h5.create_dataset(
                "vector",
                shape=(rows, args.token_count, args.vector_dim),
                dtype="float16",
                chunks=(min(max(1, args.vector_chunk_rows), rows), args.token_count, args.vector_dim),
            )
            for start in range(0, rows, args.generation_batch_size):
                end = min(start + args.generation_batch_size, rows)
                size = end - start
                fill_common_datasets(h5, start, end, global_start_id, rng)
                h5["vector"][start:end] = rng.random(
                    (size, args.token_count, args.vector_dim), dtype=np.float32
                ).astype(np.float16)
        remove_tmp_and_replace(tmp_path, path)


def create_collection(args: argparse.Namespace) -> None:
    config = config_from_args(args)
    ensure_database(config, args.create_db)
    client = make_client(config)
    try:
        if client.has_collection(args.collection_name):
            if args.no_drop:
                raise RuntimeError(f"collection {args.collection_name!r} already exists")
            client.drop_collection(args.collection_name, timeout=args.timeout)

        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("pk", DataType.INT64, is_primary=True, auto_id=False)
        schema.add_field("id", DataType.INT64)
        schema.add_field("uuid1", DataType.VARCHAR, max_length=36)
        schema.add_field("uuid2", DataType.VARCHAR, max_length=36)
        schema.add_field("platform", DataType.VARCHAR, max_length=16)
        schema.add_field("text", DataType.VARCHAR, max_length=64)

        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="uuid1",
            index_name="uuid1_inverted_idx",
            index_type="INVERTED",
        )
        index_params.add_index(
            field_name="uuid2",
            index_name="uuid2_inverted_idx",
            index_type="INVERTED",
        )
        if args.multi_vector_mode == MODE_STRUCT_FLOAT32:
            struct_schema = client.create_struct_field_schema()
            struct_schema.add_field("token_vector", DataType.FLOAT_VECTOR, dim=args.vector_dim)
            schema.add_field(
                "tokens",
                DataType.ARRAY,
                element_type=DataType.STRUCT,
                struct_schema=struct_schema,
                max_capacity=args.token_count,
            )
            index_params.add_index(
                field_name="tokens[token_vector]",
                index_name="tokens_token_vector_hnsw_idx",
                index_type=args.index_type,
                metric_type=args.metric_type,
                params={"M": args.hnsw_m, "efConstruction": args.hnsw_ef_construction},
            )
        else:
            schema.add_field("vector", DataType.FLOAT16_VECTOR, dim=args.token_count * args.vector_dim)
            index_params.add_index(
                field_name="vector",
                index_name="flat_vector_hnsw_idx",
                index_type=args.index_type,
                metric_type=args.flat_metric_type,
                params={"M": args.hnsw_m, "efConstruction": args.hnsw_ef_construction},
            )

        client.create_collection(
            collection_name=args.collection_name,
            schema=schema,
            num_shards=args.num_shards,
            timeout=args.timeout,
        )
        client.create_index(args.collection_name, index_params, timeout=args.timeout)
        client.load_collection(
            args.collection_name,
            replica_number=args.replica_number,
            timeout=args.timeout,
        )
        print(
            f"created collection {args.collection_name} mode={args.multi_vector_mode} "
            f"num_shards={args.num_shards} replica_number={args.replica_number}"
        )
    finally:
        client.close()


def batch_to_records(batch: dict[str, np.ndarray], mode: str) -> list[dict[str, object]]:
    ids = batch["id"]
    uuid1 = batch["uuid1"]
    uuid2 = batch["uuid2"]
    platform = batch["platform"]
    text = batch["text"]
    vectors = batch["vector"]
    size = len(ids)
    records: list[dict[str, object]] = [None] * size  # type: ignore[list-item]
    for idx in range(size):
        row_id = int(ids[idx])
        record: dict[str, object] = {
            "pk": row_id,
            "id": row_id,
            "uuid1": uuid1[idx].decode("utf-8"),
            "uuid2": uuid2[idx].decode("utf-8"),
            "platform": platform[idx].decode("utf-8"),
            "text": text[idx].decode("utf-8"),
        }
        matrix = vectors[idx]
        if mode == MODE_STRUCT_FLOAT32:
            matrix32 = matrix.astype(np.float32, copy=False)
            record["tokens"] = [{"token_vector": token} for token in matrix32]
        else:
            record["vector"] = matrix.reshape(-1)
        records[idx] = record
    return records


def insert_records(args: argparse.Namespace) -> None:
    config = config_from_args(args)
    paths = expected_shard_paths(args.data_dir, args.total_rows, args.shard_rows)
    total_rows = sum(h5_rows(path) for path in paths)
    prefetch_batches = args.prefetch_batches if args.prefetch_batches > 0 else args.concurrency * 2

    if args.executor_kind == "process":
        batches = iter_h5_slice_specs(paths, args.insert_batch_size)
        result = run_concurrent_operations(
            batches,
            total_units=total_rows,
            concurrency=args.concurrency,
            worker=process_insert_worker,
            unit_count=lambda spec: spec.rows,
            description="insert multi",
            prefetch_batches=prefetch_batches,
            executor_kind="process",
            executor_initializer=init_process_worker,
            executor_initargs=(config, args.collection_name, args.delete_ratio, args.timeout, args.multi_vector_mode),
        )
    else:
        def batches() -> object:
            yield from iter_h5_slices(paths, args.insert_batch_size, ("id", "uuid1", "uuid2", "platform", "text", "vector"))

        def worker(batch: dict[str, np.ndarray]) -> OperationResult:
            worker_start = time.perf_counter()
            records = batch_to_records(batch, args.multi_vector_mode)
            prep_elapsed = time.perf_counter() - worker_start
            try:
                client = get_thread_client(config)
                rpc_start = time.perf_counter()
                client.insert(collection_name=args.collection_name, data=records, timeout=args.timeout)
                delete_records = delete_ratio_candidates(records, args.delete_ratio)
                if delete_records:
                    query_delete_then_insert_records(
                        client,
                        collection_name=args.collection_name,
                        records=delete_records,
                        timeout=args.timeout,
                    )
                rpc_elapsed = time.perf_counter() - rpc_start
                elapsed = time.perf_counter() - worker_start
                return OperationResult(len(records), 0, elapsed, prep_elapsed=prep_elapsed, rpc_elapsed=rpc_elapsed)
            except Exception as exc:
                elapsed = time.perf_counter() - worker_start
                return OperationResult(0, len(records), elapsed, repr(exc), prep_elapsed=prep_elapsed)

        result = run_concurrent_operations(
            batches(),
            total_units=total_rows,
            concurrency=args.concurrency,
            worker=worker,
            unit_count=lambda batch: len(batch["id"]),
            description="insert multi",
            prefetch_batches=prefetch_batches,
        )

    client = make_client(config)
    try:
        maybe_flush(client, args.collection_name, args.timeout)
    finally:
        client.close()
    print_result_table(
        "multi vector insert result",
        result,
        metadata={
            "uri": config.uri,
            "db_name": config.db_name,
            "collection": args.collection_name,
            "num_shards": args.num_shards,
            "replica_number": args.replica_number,
            "data_dir": args.data_dir,
            "total_rows": total_rows,
            "shard_rows": args.shard_rows,
            "vector_dim": args.vector_dim,
            "token_count": args.token_count,
            "stored_dtype": "float16",
            "milvus_mode": args.multi_vector_mode,
            "insert_batch_size": args.insert_batch_size,
            "delete_ratio": args.delete_ratio,
            "concurrency": args.concurrency,
            "prefetch_batches": prefetch_batches,
            "executor_kind": args.executor_kind,
            "index_type": args.index_type,
            "metric_type": args.metric_type if args.multi_vector_mode == MODE_STRUCT_FLOAT32 else args.flat_metric_type,
        },
    )


def validate_args(args: argparse.Namespace) -> None:
    errors = []
    if args.total_rows <= 0:
        errors.append(f"--total-rows must be positive, got {args.total_rows}")
    if args.shard_rows <= 0:
        errors.append(f"--shard-rows must be positive, got {args.shard_rows}")
    if args.vector_dim <= 0:
        errors.append(f"--vector-dim must be positive, got {args.vector_dim}")
    if args.num_shards <= 0:
        errors.append(f"--num-shards must be positive, got {args.num_shards}")
    if args.replica_number <= 0:
        errors.append(f"--replica-number must be positive, got {args.replica_number}")
    if args.token_count <= 0:
        errors.append(f"--token-count must be positive, got {args.token_count}")
    if args.insert_batch_size <= 0:
        errors.append(f"--insert-batch-size must be positive, got {args.insert_batch_size}")
    if args.delete_ratio < 0 or args.delete_ratio > 100:
        errors.append(f"--delete-ratio must be between 0 and 100, got {args.delete_ratio}")
    if args.concurrency <= 0:
        errors.append(f"--concurrency must be positive, got {args.concurrency}")
    if args.prefetch_batches < 0:
        errors.append(f"--prefetch-batches must be >= 0, got {args.prefetch_batches}")
    if args.hnsw_m <= 0:
        errors.append(f"--hnsw-m must be positive, got {args.hnsw_m}")
    if args.hnsw_ef_construction <= 0:
        errors.append(f"--hnsw-ef-construction must be positive, got {args.hnsw_ef_construction}")
    if args.generation_batch_size <= 0:
        errors.append(f"--generation-batch-size must be positive, got {args.generation_batch_size}")
    if args.vector_chunk_rows <= 0:
        errors.append(f"--vector-chunk-rows must be positive, got {args.vector_chunk_rows}")
    if args.multi_vector_mode == MODE_FLAT_FP16 and args.token_count * args.vector_dim > 32768:
        errors.append(
            f"flat-fp16 mode: token_count({args.token_count}) * vector_dim({args.vector_dim}) = "
            f"{args.token_count * args.vector_dim} exceeds 32768 limit"
        )
    if errors:
        raise SystemExit("Invalid arguments:\n  " + "\n  ".join(errors))


def main() -> None:
    args = parse_args()
    validate_args(args)
    if not args.skip_generate:
        generate_shards(args)
    if args.generate_only:
        return
    create_collection(args)
    insert_records(args)


if __name__ == "__main__":
    main()
