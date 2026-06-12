#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from pymilvus import DataType

from milvus_bench_common import (
    add_connection_args,
    config_from_args,
    ensure_database,
    make_client,
)

DEFAULT_COLLECTIONS = ("test_vector", "test_multi_vector")
DEFAULT_METRICS_FILTERS = ("querynode", "size")

SCALAR_BYTES = {
    DataType.BOOL: 1,
    DataType.INT8: 1,
    DataType.INT16: 2,
    DataType.INT32: 4,
    DataType.INT64: 8,
    DataType.FLOAT: 4,
    DataType.DOUBLE: 8,
}

VECTOR_BYTES = {
    DataType.FLOAT_VECTOR: 4,
    DataType.FLOAT16_VECTOR: 2,
    DataType.BFLOAT16_VECTOR: 2,
    DataType.INT8_VECTOR: 1,
}


@dataclass
class MetricsMatch:
    metric_name: str
    value: float | None
    line: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report loaded segment memory for Milvus collections"
    )
    add_connection_args(parser)
    parser.add_argument(
        "collections",
        nargs="*",
        default=list(DEFAULT_COLLECTIONS),
        help="Collection names to inspect. Defaults to test_vector and test_multi_vector.",
    )
    parser.add_argument(
        "--load",
        action="store_true",
        help="Load each collection before reading loaded segment memory.",
    )
    parser.add_argument(
        "--release-after",
        action="store_true",
        help="Release each successfully inspected collection after reporting.",
    )
    parser.add_argument(
        "--show-segments",
        action="store_true",
        help="Print per-segment mem_size details.",
    )
    parser.add_argument(
        "--metrics-url",
        default="",
        help="Optional Prometheus /metrics URL, for example http://127.0.0.1:9091/metrics inside a Milvus container.",
    )
    parser.add_argument(
        "--metrics-timeout",
        type=float,
        default=5.0,
        help="HTTP timeout in seconds for --metrics-url.",
    )
    parser.add_argument(
        "--metrics-filter",
        action="append",
        default=None,
        help="Case-insensitive substring filter for metrics lines. Can be repeated. Defaults to querynode and size.",
    )
    return parser.parse_args()


def human_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(size) < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PiB"


def human_metric_value(value: float) -> str:
    if not math.isfinite(value):
        return str(value)
    if value >= 0 and value.is_integer():
        return f"{int(value)} / {human_bytes(int(value))}"
    return str(value)


def state_label(load_state: dict[str, Any]) -> str:
    state = load_state.get("state", "unknown")
    label = getattr(state, "name", str(state))
    progress = load_state.get("progress")
    if progress is not None:
        return f"{label} ({progress})"
    return label


def print_table(title: str, rows: list[tuple[str, str]]) -> None:
    width = max(len(key) for key, _ in rows + [(title, "")])
    print("\n" + title)
    print("-" * (width + 22))
    for key, value in rows:
        print(f"{key:<{width}} | {value}")


def field_type_name(field_type: Any) -> str:
    return getattr(field_type, "name", str(field_type))


def int_param(params: dict[str, Any], key: str) -> int:
    value = params.get(key, 0)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def estimate_field_bytes_per_row(field: dict[str, Any]) -> tuple[int, str]:
    field_type = field.get("type")
    params = field.get("params") or {}

    if field_type in SCALAR_BYTES:
        return SCALAR_BYTES[field_type], field_type_name(field_type)

    if field_type == DataType.VARCHAR:
        return int_param(params, "max_length"), "VARCHAR max_length"

    if field_type in VECTOR_BYTES:
        dim = int_param(params, "dim")
        return dim * VECTOR_BYTES[field_type], f"{field_type_name(field_type)} dim={dim}"

    if field_type == DataType.ARRAY:
        max_capacity = int_param(params, "max_capacity")
        element_type = field.get("element_type")
        if element_type == DataType.STRUCT:
            struct_total = 0
            parts: list[str] = []
            for struct_field in field.get("struct_fields") or []:
                size, note = estimate_field_bytes_per_row(struct_field)
                struct_total += size
                parts.append(f"{struct_field.get('name')}:{note}")
            return (
                max_capacity * struct_total,
                f"ARRAY<STRUCT> max_capacity={max_capacity} ({', '.join(parts)})",
            )
        if element_type in SCALAR_BYTES:
            return (
                max_capacity * SCALAR_BYTES[element_type],
                f"ARRAY<{field_type_name(element_type)}> max_capacity={max_capacity}",
            )

    return 0, f"{field_type_name(field_type)} unsupported_estimate"


def estimate_payload_per_row(description: dict[str, Any]) -> tuple[int, list[tuple[str, int, str]]]:
    fields: list[tuple[str, int, str]] = []
    total = 0
    for field in description.get("fields") or []:
        size, note = estimate_field_bytes_per_row(field)
        fields.append((str(field.get("name")), size, note))
        total += size
    return total, fields


def fetch_metrics(metrics_url: str, timeout: float) -> tuple[str | None, str | None]:
    if not metrics_url:
        return None, None
    try:
        with urllib.request.urlopen(metrics_url, timeout=timeout) as response:
            data = response.read()
        return data.decode("utf-8", errors="replace"), None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return None, repr(exc)


def metric_name_from_line(line: str) -> str:
    sample = line.split(None, 1)[0]
    if "{" in sample:
        return sample.split("{", 1)[0]
    return sample


def metric_value_from_line(line: str) -> float | None:
    parts = line.split()
    if len(parts) < 2:
        return None
    try:
        return float(parts[1])
    except ValueError:
        return None


def find_metrics_matches(metrics_text: str, collection_id: str, filters: list[str]) -> list[MetricsMatch]:
    collection_id_text = str(collection_id)
    lowered_filters = [item.lower() for item in filters]
    matches: list[MetricsMatch] = []
    for raw_line in metrics_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        line_lower = line.lower()
        if collection_id_text not in line:
            continue
        if any(filter_text not in line_lower for filter_text in lowered_filters):
            continue
        matches.append(
            MetricsMatch(
                metric_name=metric_name_from_line(line),
                value=metric_value_from_line(line),
                line=line,
            )
        )
    return matches


def print_metrics_matches(collection_id: str, matches: list[MetricsMatch], filters: list[str]) -> None:
    print("metrics_source: Prometheus /metrics")
    print(f"metrics_collection_id: {collection_id}")
    print(f"metrics_filters: {','.join(filters)}")
    print(f"metrics_match_count: {len(matches)}")
    if not matches:
        return

    grouped: dict[str, list[float]] = defaultdict(list)
    for match in matches:
        if match.value is not None:
            grouped[match.metric_name].append(match.value)

    if grouped:
        print("metrics_grouped_values")
        for metric_name, values in sorted(grouped.items()):
            total = sum(values)
            print(
                f"  {metric_name}: samples={len(values)} "
                f"sum={human_metric_value(total)} values={[human_metric_value(value) for value in values]}"
            )

    print("metrics_raw_lines")
    for match in matches:
        print(f"  {match.line}")


def inspect_collection(
    client: Any,
    collection_name: str,
    args: argparse.Namespace,
    metrics_text: str | None,
    metrics_error: str | None,
) -> None:
    if not client.has_collection(collection_name, timeout=args.timeout):
        print_table(
            collection_name,
            [
                ("status", "not_found"),
                ("loaded_mem", "0 B"),
                ("loaded_mem_bytes", "0"),
            ],
        )
        return

    if args.load:
        client.load_collection(collection_name, timeout=args.timeout)

    stats = client.get_collection_stats(collection_name, timeout=args.timeout)
    description = client.describe_collection(collection_name, timeout=args.timeout)
    load_state = client.get_load_state(collection_name, timeout=args.timeout)
    segments = client.list_loaded_segments(collection_name, timeout=args.timeout)
    persistent_segments = client.list_persistent_segments(collection_name, timeout=args.timeout)

    collection_id = str(description.get("collection_id", "unknown"))
    loaded_mem = sum(int(segment.mem_size) for segment in segments)
    loaded_rows = sum(int(segment.num_rows) for segment in segments)
    persistent_rows = sum(int(segment.num_rows) for segment in persistent_segments)
    segment_ids = {int(segment.segment_id) for segment in segments}
    mem_size_available = bool(segments) and any(int(segment.mem_size) > 0 for segment in segments)
    payload_per_row, field_estimates = estimate_payload_per_row(description)
    node_ids = sorted(
        {
            int(node_id)
            for segment in segments
            for node_id in getattr(segment, "node_ids", [])
        }
    )
    node_copy_count = sum(
        max(1, len(getattr(segment, "node_ids", []) or [])) for segment in segments
    )

    node_mem: dict[int, int] = defaultdict(int)
    node_rows: dict[int, int] = defaultdict(int)
    for segment in segments:
        ids = list(getattr(segment, "node_ids", []) or [])
        if not ids:
            continue
        for node_id in ids:
            node_mem[int(node_id)] += int(segment.mem_size)
            node_rows[int(node_id)] += int(segment.num_rows)

    row_count = stats.get("row_count", "unknown")
    row_count_int = row_count if isinstance(row_count, int) else None
    loaded_row_gap = row_count_int - loaded_rows if row_count_int is not None else "unknown"
    metrics_filters = args.metrics_filter or list(DEFAULT_METRICS_FILTERS)
    metrics_matches = (
        find_metrics_matches(metrics_text, collection_id, metrics_filters)
        if metrics_text is not None and collection_id != "unknown"
        else []
    )

    rows = [
        ("status", "ok"),
        ("collection_id", collection_id),
        ("load_state", state_label(load_state)),
        ("row_count", str(row_count)),
        ("persistent_rows", str(persistent_rows)),
        ("persistent_segments", str(len(persistent_segments))),
        ("loaded_rows", str(loaded_rows)),
        ("loaded_row_gap", str(loaded_row_gap)),
        ("loaded_segments", str(len(segments))),
        ("unique_segments", str(len(segment_ids))),
        ("node_ids", ",".join(str(node_id) for node_id in node_ids) or "-"),
        ("node_copy_count", str(node_copy_count)),
        (
            "mem_size_status",
            "available" if mem_size_available else "unavailable_server_returned_zero",
        ),
        (
            "loaded_mem_status",
            "actual_pymilvus_mem_size"
            if mem_size_available
            else "not_actual_memory_server_returned_zero",
        ),
        ("loaded_mem", human_bytes(loaded_mem)),
        ("loaded_mem_bytes", str(loaded_mem)),
        ("metrics_status", "available" if metrics_text is not None else "not_requested_or_unavailable"),
        ("metrics_match_count", str(len(metrics_matches))),
        ("payload_estimate_status", "estimate_only_not_actual_memory"),
        ("payload_per_row_estimate", human_bytes(payload_per_row)),
        ("payload_per_row_bytes", str(payload_per_row)),
        ("loaded_payload_estimate", human_bytes(payload_per_row * loaded_rows)),
        ("loaded_payload_estimate_bytes", str(payload_per_row * loaded_rows)),
        (
            "row_count_payload_estimate",
            human_bytes(payload_per_row * row_count_int) if row_count_int is not None else "unknown",
        ),
    ]
    print_table(collection_name, rows)

    if mem_size_available:
        print(
            "memory_note: loaded_mem is the PyMilvus segment mem_size returned by Milvus; "
            "all payload_* values below are estimates only, not actual Milvus memory."
        )
    else:
        print(
            "memory_note: loaded_mem is NOT actual memory because Milvus returned 0 for every "
            "loaded segment mem_size; all payload_* values below are estimates only, not actual Milvus memory."
        )

    if metrics_error:
        print(f"metrics_error: {metrics_error}")
    elif metrics_text is not None:
        print_metrics_matches(collection_id, metrics_matches, metrics_filters)

    if mem_size_available and node_mem:
        print("node_mem_estimate")
        for node_id, mem_size in sorted(node_mem.items()):
            print(f"  node {node_id}: {human_bytes(mem_size)} / {mem_size} bytes")
    elif node_rows and payload_per_row > 0:
        print("node_payload_estimate")
        for node_id, rows_on_node in sorted(node_rows.items()):
            mem_size = rows_on_node * payload_per_row
            print(
                f"  node {node_id}: rows={rows_on_node} "
                f"payload={human_bytes(mem_size)} / {mem_size} bytes"
            )

    if field_estimates:
        print("field_payload_estimate_per_row")
        for field_name, size, note in field_estimates:
            print(f"  {field_name}: {human_bytes(size)} / {size} bytes ({note})")
        print(
            "estimate_note: payload estimates exclude index, segment metadata, "
            "mmap/cache, and query runtime overhead"
        )

    if args.show_segments:
        print("segments")
        for segment in sorted(segments, key=lambda item: int(item.segment_id)):
            print(
                "  "
                f"segment_id={segment.segment_id} "
                f"rows={segment.num_rows} "
                f"mem={human_bytes(int(segment.mem_size))} "
                f"mem_bytes={int(segment.mem_size)} "
                f"node_ids={list(getattr(segment, 'node_ids', []) or [])}"
            )

    if args.release_after:
        client.release_collection(collection_name, timeout=args.timeout)
        print(f"released {collection_name}")


def main() -> None:
    args = parse_args()
    config = config_from_args(args)
    ensure_database(config, args.create_db)
    metrics_text, metrics_error = fetch_metrics(args.metrics_url, args.metrics_timeout)
    client = make_client(config)
    try:
        for collection_name in args.collections:
            inspect_collection(client, collection_name, args, metrics_text, metrics_error)
    finally:
        client.close()


if __name__ == "__main__":
    main()
