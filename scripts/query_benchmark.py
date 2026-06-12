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
from pymilvus.client.embedding_list import EmbeddingList

from milvus_bench_common import (
    OperationResult,
    add_connection_args,
    config_from_args,
    ensure_database,
    expected_shard_paths,
    get_thread_client,
    h5_rows,
    make_client,
    remove_tmp_and_replace,
    run_concurrent_operations,
    shard_row_count,
    shards_are_complete,
    print_result_table,
)

DEFAULT_SINGLE_COLLECTION = "test_vector"
DEFAULT_MULTI_COLLECTION = "test_multi_vector"
MODE_STRUCT_FLOAT32 = "struct-float32"
MODE_FLAT_FP16 = "flat-fp16"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Milvus search benchmark for single and multi vectors")
    add_connection_args(parser)
    parser.add_argument("--total-rows", type=int, default=1_000_000)
    parser.add_argument("--shard-rows", type=int, default=250_000)
    parser.add_argument("--data-dir", default="data/query_samples")
    parser.add_argument("--generation-batch-size", type=int, default=512)
    parser.add_argument("--force-regenerate", action="store_true")
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--single-collection", default=DEFAULT_SINGLE_COLLECTION)
    parser.add_argument("--multi-collection", default=DEFAULT_MULTI_COLLECTION)
    parser.add_argument("--target", choices=["single", "multi", "both"], default="both")
    parser.add_argument("--multi-vector-mode", choices=[MODE_STRUCT_FLOAT32, MODE_FLAT_FP16], default=MODE_STRUCT_FLOAT32)
    parser.add_argument("--vector-dim", type=int, default=1024)
    parser.add_argument("--multi-vector-dim", type=int, default=128)
    parser.add_argument("--query-token-count", type=int, default=30)
    parser.add_argument("--data-id-max", type=int, default=1_000_000)
    parser.add_argument("--query-read-batch-size", type=int, default=512)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--limit", type=int, default=4000)
    parser.add_argument("--single-metric-type", default="COSINE")
    parser.add_argument("--multi-metric-type", default="MAX_SIM_COSINE")
    parser.add_argument("--flat-metric-type", default="COSINE")
    parser.add_argument("--search-ef", type=int, default=4000)
    parser.add_argument("--id-filter", choices=["none", "eq", "gte"], default="none")
    parser.add_argument("--seed", type=int, default=20260610)
    parser.add_argument("--no-load", action="store_true", help="Do not call load_collection before searching")
    return parser.parse_args()


def generate_shards(args: argparse.Namespace) -> None:
    attrs = {
        "kind": "query_samples",
        "vector_dim": args.vector_dim,
        "multi_vector_dim": args.multi_vector_dim,
        "query_token_count": args.query_token_count,
        "data_id_max": args.data_id_max,
    }
    if not args.force_regenerate and shards_are_complete(
        args.data_dir, args.total_rows, args.shard_rows, required_attrs=attrs
    ):
        print(f"query shards already complete: {args.data_dir}")
        return

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    paths = expected_shard_paths(data_dir, args.total_rows, args.shard_rows)

    for shard_idx, path in enumerate(paths):
        rows = shard_row_count(args.total_rows, args.shard_rows, shard_idx)
        tmp_path = path.with_suffix(".h5.tmp")
        if tmp_path.exists():
            tmp_path.unlink()
        print(f"generating {path} rows={rows}")
        with h5py.File(tmp_path, "w") as h5:
            h5.attrs["kind"] = "query_samples"
            h5.attrs["rows"] = rows
            h5.attrs["shard_rows"] = args.shard_rows
            h5.attrs["total_rows"] = args.total_rows
            h5.attrs["vector_dim"] = args.vector_dim
            h5.attrs["multi_vector_dim"] = args.multi_vector_dim
            h5.attrs["query_token_count"] = args.query_token_count
            h5.attrs["data_id_max"] = args.data_id_max
            h5.create_dataset("id", shape=(rows,), dtype="int64")
            h5.create_dataset(
                "single_vector",
                shape=(rows, args.vector_dim),
                dtype="float16",
                chunks=(min(max(1, args.generation_batch_size), rows), args.vector_dim),
            )
            h5.create_dataset(
                "multi_vector",
                shape=(rows, args.query_token_count, args.multi_vector_dim),
                dtype="float16",
                chunks=(min(max(1, min(args.generation_batch_size, 16)), rows), args.query_token_count, args.multi_vector_dim),
            )
            for start in range(0, rows, args.generation_batch_size):
                end = min(start + args.generation_batch_size, rows)
                size = end - start
                h5["id"][start:end] = rng.integers(1, args.data_id_max + 1, size=size, dtype=np.int64)
                h5["single_vector"][start:end] = rng.random((size, args.vector_dim), dtype=np.float32).astype(np.float16)
                h5["multi_vector"][start:end] = rng.random(
                    (size, args.query_token_count, args.multi_vector_dim), dtype=np.float32
                ).astype(np.float16)
        remove_tmp_and_replace(tmp_path, path)


def filter_expr(sample_id: int, mode: str) -> str:
    if mode == "eq":
        return f"id == {sample_id}"
    if mode == "gte":
        return f"id >= {sample_id}"
    return ""


def iter_query_samples(paths: list[Path], batch_size: int):
    for path in paths:
        with h5py.File(path, "r") as h5:
            rows = int(h5.attrs["rows"])
            for start in range(0, rows, batch_size):
                end = min(start + batch_size, rows)
                ids = h5["id"][start:end]
                single = h5["single_vector"][start:end]
                multi = h5["multi_vector"][start:end]
                for idx in range(end - start):
                    yield {
                        "id": int(ids[idx]),
                        "single_vector": single[idx].copy(),
                        "multi_vector": multi[idx].copy(),
                    }


def validate_query_shards(args: argparse.Namespace, paths: list[Path]) -> None:
    errors: list[str] = []
    for path in paths:
        if not path.exists():
            errors.append(f"missing query shard: {path}")
            continue
        try:
            with h5py.File(path, "r") as h5:
                rows = int(h5.attrs.get("rows", -1))
                if rows < 0:
                    errors.append(f"{path}: missing rows attr")
                for name in ("id", "single_vector", "multi_vector"):
                    if name not in h5:
                        errors.append(f"{path}: missing dataset {name!r}")
                if "single_vector" in h5 and h5["single_vector"].shape != (rows, args.vector_dim):
                    errors.append(
                        f"{path}: single_vector shape {h5['single_vector'].shape} "
                        f"!= ({rows}, {args.vector_dim})"
                    )
                expected_multi_shape = (rows, args.query_token_count, args.multi_vector_dim)
                if "multi_vector" in h5 and h5["multi_vector"].shape != expected_multi_shape:
                    errors.append(
                        f"{path}: multi_vector shape {h5['multi_vector'].shape} "
                        f"!= {expected_multi_shape}"
                    )
                attr_checks = {
                    "kind": "query_samples",
                    "vector_dim": args.vector_dim,
                    "multi_vector_dim": args.multi_vector_dim,
                    "query_token_count": args.query_token_count,
                    "data_id_max": args.data_id_max,
                }
                for key, expected in attr_checks.items():
                    actual = h5.attrs.get(key)
                    if actual != expected:
                        errors.append(f"{path}: attr {key}={actual!r} != {expected!r}")
        except Exception as exc:
            errors.append(f"{path}: failed to inspect shard: {exc!r}")
    if errors:
        raise SystemExit("Invalid query shards:\n  " + "\n  ".join(errors))

def load_collections(args: argparse.Namespace) -> None:
    config = config_from_args(args)
    client = make_client(config)
    try:
        if args.target in ("single", "both"):
            client.load_collection(args.single_collection, timeout=args.timeout)
        if args.target in ("multi", "both"):
            client.load_collection(args.multi_collection, timeout=args.timeout)
    finally:
        client.close()


def run_queries(args: argparse.Namespace) -> None:
    config = config_from_args(args)
    ensure_database(config, args.create_db)
    if not args.no_load:
        load_collections(args)

    paths = expected_shard_paths(args.data_dir, args.total_rows, args.shard_rows)
    validate_query_shards(args, paths)
    total_rows = sum(h5_rows(path) for path in paths)

    def worker(sample: dict[str, object]) -> OperationResult:
        client = get_thread_client(config)
        sample_id = int(sample["id"])
        expr = filter_expr(sample_id, args.id_filter)
        try:
            start = time.perf_counter()
            if args.target in ("single", "both"):
                client.search(
                    collection_name=args.single_collection,
                    data=[sample["single_vector"]],
                    anns_field="vector",
                    filter=expr,
                    limit=args.limit,
                    output_fields=["id", "platform"],
                    search_params={"metric_type": args.single_metric_type, "params": {"ef": args.search_ef}},
                    timeout=args.timeout,
                )
            if args.target in ("multi", "both"):
                matrix = sample["multi_vector"]
                if args.multi_vector_mode == MODE_STRUCT_FLOAT32:
                    query = EmbeddingList(matrix.astype(np.float32, copy=False), dim=args.multi_vector_dim, dtype="float32")
                    client.search(
                        collection_name=args.multi_collection,
                        data=[query],
                        anns_field="tokens[token_vector]",
                        filter=expr,
                        limit=args.limit,
                        output_fields=["id", "platform"],
                        search_params={"metric_type": args.multi_metric_type, "params": {"ef": args.search_ef}},
                        timeout=args.timeout,
                    )
                else:
                    client.search(
                        collection_name=args.multi_collection,
                        data=[matrix.reshape(-1)],
                        anns_field="vector",
                        filter=expr,
                        limit=args.limit,
                        output_fields=["id", "platform"],
                        search_params={"metric_type": args.flat_metric_type, "params": {"ef": args.search_ef}},
                        timeout=args.timeout,
                    )
            elapsed = time.perf_counter() - start
            return OperationResult(1, 0, elapsed)
        except Exception as exc:
            return OperationResult(0, 1, None, repr(exc))

    result = run_concurrent_operations(
        iter_query_samples(paths, args.query_read_batch_size),
        total_units=total_rows,
        concurrency=args.concurrency,
        worker=worker,
        unit_count=lambda _sample: 1,
        description="query",
    )
    print_result_table(
        "query result",
        result,
        metadata={
            "uri": config.uri,
            "db_name": config.db_name,
            "target": args.target,
            "single_collection": args.single_collection,
            "multi_collection": args.multi_collection,
            "data_dir": args.data_dir,
            "total_queries": total_rows,
            "shard_rows": args.shard_rows,
            "vector_dim": args.vector_dim,
            "multi_vector_dim": args.multi_vector_dim,
            "query_token_count": args.query_token_count,
            "multi_vector_mode": args.multi_vector_mode,
            "query_read_batch_size": args.query_read_batch_size,
            "concurrency": args.concurrency,
            "limit": args.limit,
            "search_ef": args.search_ef,
            "id_filter": args.id_filter,
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
    if args.multi_vector_dim <= 0:
        errors.append(f"--multi-vector-dim must be positive, got {args.multi_vector_dim}")
    if args.query_token_count <= 0:
        errors.append(f"--query-token-count must be positive, got {args.query_token_count}")
    if args.concurrency <= 0:
        errors.append(f"--concurrency must be positive, got {args.concurrency}")
    if args.limit <= 0:
        errors.append(f"--limit must be positive, got {args.limit}")
    if not args.generate_only and args.search_ef < args.limit:
        errors.append(f"--search-ef({args.search_ef}) must be >= --limit({args.limit})")
    if args.data_id_max <= 0:
        errors.append(f"--data-id-max must be positive, got {args.data_id_max}")
    if args.query_read_batch_size <= 0:
        errors.append(f"--query-read-batch-size must be positive, got {args.query_read_batch_size}")
    if args.generation_batch_size <= 0:
        errors.append(f"--generation-batch-size must be positive, got {args.generation_batch_size}")
    if errors:
        raise SystemExit("Invalid arguments:\n  " + "\n  ".join(errors))


def main() -> None:
    args = parse_args()
    validate_args(args)
    if not args.skip_generate:
        generate_shards(args)
    if args.generate_only:
        return
    run_queries(args)


if __name__ == "__main__":
    main()
