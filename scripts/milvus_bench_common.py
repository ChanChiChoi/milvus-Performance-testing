from __future__ import annotations

import math
import multiprocessing as mp
import os
import threading
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from collections.abc import Sequence
from dataclasses import dataclass
from queue import Queue
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import h5py
import numpy as np
from pymilvus import MilvusClient
from tqdm import tqdm

PROXY_ENV_KEYS = (
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "all_proxy",
    "ALL_PROXY",
    "no_proxy",
    "NO_PROXY",
    "htp_proxy",
)

DEFAULT_URI_ENV = "MILVUS_URI"
DEFAULT_URI = os.environ.get(DEFAULT_URI_ENV) or "http://localhost:19530"
DEFAULT_DB_NAME = "llmbp"
DEFAULT_TOTAL_ROWS = 1_000_000
DEFAULT_SHARD_ROWS = 250_000
DEFAULT_TIMEOUT = 60.0
COMMON_DATA_VERSION = 2
PLATFORMS = np.asarray(["苏宁".encode("utf-8"), "淘宝".encode("utf-8"), "京东".encode("utf-8")], dtype="S16")
TEXT_CHARS = np.asarray(list("的一是在不了有和人这中大为上个国我以要他时来用们生到作地于出就分对成会可主发年动同工也能下过子说产种面而方后多定行学法所民得经十三之进着等部度家电力里如水化高自二理起小物现实加量都两体制机当使点从业本去把性好应开它合还因由其些然前外天政四日那社义事平形相全表间样与关各重新线内数正心反你明看原又么利比或但质气第向道命此变条只没结解问意建月公无系军很情者最立代想已通并提直题党程展五果料象员革位入常文总次品式活设及管特件长求老头基资边流路级少图山统接知较将组见计别她手角期根论运农指几九区强放决西被干做必战先回则任取据处理世受领"))
UUID_NAMESPACE = uuid.UUID("7fd54d72-4a67-42df-9ff7-02bf352f7b16")


def clear_proxy_env() -> None:
    for key in PROXY_ENV_KEYS:
        os.environ[key] = ""


clear_proxy_env()


@dataclass(frozen=True)
class MilvusConfig:
    uri: str
    db_name: str
    user: str = ""
    password: str = ""
    token: str = ""
    timeout: float = DEFAULT_TIMEOUT


@dataclass
class OperationResult:
    ok_count: int
    fail_count: int
    elapsed: float | None = None
    error: str | None = None
    read_elapsed: float | None = None
    prep_elapsed: float | None = None
    rpc_elapsed: float | None = None


@dataclass
class BenchmarkResult:
    ok_count: int
    fail_count: int
    latencies: list[float]
    prep_latencies: list[float]
    rpc_latencies: list[float]
    read_latencies: list[float]
    wall_elapsed: float
    operation_count: int
    successful_operations: int
    failed_operations: int

    @property
    def total_count(self) -> int:
        return self.ok_count + self.fail_count

    @property
    def success_rate(self) -> float:
        total = self.total_count
        return self.ok_count / total if total else 0.0

    @property
    def success_rows_per_second(self) -> float:
        return self.ok_count / self.wall_elapsed if self.wall_elapsed > 0 else 0.0

    @property
    def total_rows_per_second(self) -> float:
        return self.total_count / self.wall_elapsed if self.wall_elapsed > 0 else 0.0

    @property
    def operations_per_second(self) -> float:
        return self.operation_count / self.wall_elapsed if self.wall_elapsed > 0 else 0.0


@dataclass(frozen=True)
class H5SliceSpec:
    path: Path
    start: int
    end: int
    rows: int


_thread_local = threading.local()


def normalize_uri(uri: str) -> str:
    uri = uri.strip()
    if not uri:
        raise ValueError(f"Milvus URI is required. Set ${DEFAULT_URI_ENV} or pass --uri.")
    if "://" in uri:
        return uri
    return f"http://{uri}"


def add_connection_args(parser: Any) -> None:
    parser.add_argument(
        "--uri",
        default=DEFAULT_URI,
        help=f"Milvus URI. Defaults to ${DEFAULT_URI_ENV} or http://localhost:19530.",
    )
    parser.add_argument("--db-name", default=DEFAULT_DB_NAME, help="Milvus database name")
    parser.add_argument("--user", default="", help="Milvus user")
    parser.add_argument("--password", default="", help="Milvus password")
    parser.add_argument("--token", default="", help="Milvus token")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="Milvus RPC timeout")
    parser.add_argument(
        "--create-db",
        action="store_true",
        help="Create --db-name if it does not exist",
    )


def config_from_args(args: Any) -> MilvusConfig:
    return MilvusConfig(
        uri=normalize_uri(args.uri),
        db_name=args.db_name,
        user=args.user,
        password=args.password,
        token=args.token,
        timeout=args.timeout,
    )


def make_client(config: MilvusConfig, *, use_database: bool = True) -> MilvusClient:
    clear_proxy_env()
    kwargs: dict[str, Any] = {
        "uri": config.uri,
        "user": config.user,
        "password": config.password,
        "token": config.token,
        "timeout": config.timeout,
    }
    if use_database and config.db_name:
        kwargs["db_name"] = config.db_name
    return MilvusClient(**kwargs)


def ensure_database(config: MilvusConfig, create_db: bool) -> None:
    if not config.db_name:
        return
    root = make_client(config, use_database=False)
    try:
        databases = root.list_databases(timeout=config.timeout)
        if config.db_name in databases:
            return
        if not create_db:
            raise RuntimeError(
                f"Milvus database {config.db_name!r} does not exist. "
                f"Existing databases: {databases}. Pass --db-name or --create-db."
            )
        root.create_database(config.db_name, timeout=config.timeout)
    finally:
        root.close()


def get_thread_client(config: MilvusConfig) -> MilvusClient:
    client = getattr(_thread_local, "milvus_client", None)
    client_key = getattr(_thread_local, "milvus_client_key", None)
    key = (config.uri, config.db_name, config.user, config.password, config.token, config.timeout)
    if client is None or client_key != key:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass
        client = make_client(config)
        _thread_local.milvus_client = client
        _thread_local.milvus_client_key = key
    return client


def add_data_args(parser: Any, default_data_dir: str) -> None:
    parser.add_argument("--total-rows", type=int, default=DEFAULT_TOTAL_ROWS)
    parser.add_argument("--shard-rows", type=int, default=DEFAULT_SHARD_ROWS)
    parser.add_argument("--data-dir", default=default_data_dir)
    parser.add_argument("--generation-batch-size", type=int, default=2048)
    parser.add_argument("--force-regenerate", action="store_true")
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--generate-only", action="store_true")


def expected_shard_paths(data_dir: str | Path, total_rows: int, shard_rows: int) -> list[Path]:
    data_path = Path(data_dir)
    shard_count = math.ceil(total_rows / shard_rows)
    return [data_path / f"part-{idx:04d}.h5" for idx in range(shard_count)]


def shard_row_count(total_rows: int, shard_rows: int, shard_idx: int) -> int:
    start = shard_idx * shard_rows
    return max(0, min(shard_rows, total_rows - start))


def h5_rows(path: Path) -> int:
    with h5py.File(path, "r") as h5:
        return int(h5.attrs["rows"])


def shards_are_complete(
    data_dir: str | Path,
    total_rows: int,
    shard_rows: int,
    *,
    required_attrs: dict[str, Any] | None = None,
) -> bool:
    paths = expected_shard_paths(data_dir, total_rows, shard_rows)
    if not paths or not all(path.exists() for path in paths):
        return False

    row_sum = 0
    required_attrs = required_attrs or {}
    for idx, path in enumerate(paths):
        expected_rows = shard_row_count(total_rows, shard_rows, idx)
        try:
            with h5py.File(path, "r") as h5:
                rows = int(h5.attrs["rows"])
                if rows != expected_rows:
                    return False
                for key, expected_value in required_attrs.items():
                    actual = h5.attrs.get(key)
                    if isinstance(actual, bytes):
                        actual = actual.decode("utf-8")
                    if actual != expected_value:
                        return False
                row_sum += rows
        except Exception:
            return False
    return row_sum == total_rows


def init_common_datasets(h5: h5py.File, rows: int, global_start_id: int) -> None:
    h5.create_dataset("id", data=np.arange(global_start_id, global_start_id + rows, dtype=np.int64))
    h5.create_dataset("uuid1", shape=(rows,), dtype="S36")
    h5.create_dataset("uuid2", shape=(rows,), dtype="S36")
    h5.create_dataset("platform", shape=(rows,), dtype="S16")
    h5.create_dataset("text", shape=(rows,), dtype="S64")


def fill_common_datasets(
    h5: h5py.File,
    start: int,
    end: int,
    global_start_id: int,
    rng: np.random.Generator,
) -> None:
    size = end - start
    ids = np.arange(global_start_id + start, global_start_id + end, dtype=np.int64)
    h5["uuid1"][start:end] = make_uuid_array(ids, salt="uuid1")
    h5["uuid2"][start:end] = make_uuid_array(ids, salt="uuid2")
    h5["platform"][start:end] = rng.choice(PLATFORMS, size=size)
    h5["text"][start:end] = make_text_array(rng, size, 16)


def make_uuid_array(ids: np.ndarray, *, salt: str) -> np.ndarray:
    values: list[bytes] = []
    for value in ids:
        values.append(str(uuid.uuid5(UUID_NAMESPACE, f"{salt}:{int(value)}")).encode("ascii"))
    return np.asarray(values, dtype="S36")


def make_text_array(rng: np.random.Generator, size: int, length: int) -> np.ndarray:
    indexes = rng.integers(0, len(TEXT_CHARS), size=(size, length), dtype=np.int16)
    return np.asarray(
        ["".join(TEXT_CHARS[row]).encode("utf-8") for row in indexes],
        dtype="S64",
    )


def decode_bytes(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "decode"):
        return value.decode("utf-8")
    return str(value)




def milvus_string_literal(value: object) -> str:
    text = str(value)
    return chr(34) + text.replace(chr(92), chr(92) + chr(92)).replace(chr(34), chr(92) + chr(34)) + chr(34)


def record_delete_filter(record: dict[str, object]) -> str:
    uuid1 = milvus_string_literal(record["uuid1"])
    uuid2 = milvus_string_literal(record["uuid2"])
    return f"uuid1 == {uuid1} and uuid2 == {uuid2}"


def delete_ratio_candidates(records: list[dict[str, object]], delete_ratio: int) -> list[dict[str, object]]:
    if delete_ratio <= 0:
        return []
    return [record for record in records if (int(record["id"]) - 1) % 100 >= 100 - delete_ratio]


def query_delete_then_insert_records(
    client: MilvusClient,
    *,
    collection_name: str,
    records: list[dict[str, object]],
    timeout: float,
) -> tuple[int, int]:
    found_count = 0
    deleted_count = 0
    for record in records:
        expr = record_delete_filter(record)
        rows = client.query(collection_name=collection_name, filter=expr, output_fields=["uuid1", "uuid2"], timeout=timeout)
        found_count += len(rows or [])
        client.delete(collection_name=collection_name, filter=expr, timeout=timeout)
        deleted_count += 1
    if records:
        client.insert(collection_name=collection_name, data=records, timeout=timeout)
    return found_count, deleted_count

def iter_h5_slice_specs(paths: Iterable[Path], batch_size: int) -> Iterator[H5SliceSpec]:
    for path in paths:
        with h5py.File(path, "r") as h5:
            rows = int(h5.attrs["rows"])
            for start in range(0, rows, batch_size):
                end = min(start + batch_size, rows)
                yield H5SliceSpec(path=path, start=start, end=end, rows=end - start)


def iter_h5_slices(paths: Iterable[Path], batch_size: int, dataset_names: tuple[str, ...]) -> Iterator[dict[str, Any]]:
    for path in paths:
        with h5py.File(path, "r") as h5:
            rows = int(h5.attrs["rows"])
            for start in range(0, rows, batch_size):
                end = min(start + batch_size, rows)
                read_start = time.perf_counter()
                batch = {name: h5[name][start:end] for name in dataset_names}
                batch["_read_elapsed"] = time.perf_counter() - read_start
                yield batch


def remove_tmp_and_replace(tmp_path: Path, final_path: Path) -> None:
    if final_path.exists():
        final_path.unlink()
    tmp_path.replace(final_path)


def prefetch_iterable(batches: Iterable[Any], max_prefetch: int) -> Iterator[Any]:
    if max_prefetch <= 0:
        yield from batches
        return

    queue: Queue[tuple[str, Any]] = Queue(maxsize=max_prefetch)

    def producer() -> None:
        try:
            for batch in batches:
                queue.put(("item", batch))
        except BaseException as exc:
            queue.put(("error", exc))
        finally:
            queue.put(("done", None))

    thread = threading.Thread(target=producer, name="h5-prefetch", daemon=True)
    thread.start()
    while True:
        kind, value = queue.get()
        if kind == "item":
            yield value
        elif kind == "error":
            raise value
        else:
            return


def run_concurrent_operations(
    batches: Iterable[Any],
    *,
    total_units: int,
    concurrency: int,
    worker: Callable[[Any], OperationResult],
    unit_count: Callable[[Any], int],
    description: str,
    prefetch_batches: int = 0,
    executor_kind: str = "thread",
    executor_initializer: Callable[..., None] | None = None,
    executor_initargs: tuple[Any, ...] = (),
) -> BenchmarkResult:
    ok_count = 0
    fail_count = 0
    operation_count = 0
    successful_operations = 0
    failed_operations = 0
    latencies: list[float] = []
    prep_latencies: list[float] = []
    rpc_latencies: list[float] = []
    read_latencies: list[float] = []
    pending: set[Any] = set()
    max_pending = max(1, concurrency * 2)
    wall_start = time.perf_counter()

    def collect(done: Iterable[Any], pbar: tqdm) -> None:
        nonlocal ok_count, fail_count, operation_count, successful_operations, failed_operations
        for future in done:
            try:
                future_result = future.result()
            except Exception as exc:
                future_result = OperationResult(0, 0, None, repr(exc))
            if isinstance(future_result, Sequence) and not isinstance(future_result, (bytes, str)):
                results = list(future_result)
            else:
                results = [future_result]
            for result in results:
                operation_count += 1
                ok_count += result.ok_count
                fail_count += result.fail_count
                if result.elapsed is not None and result.ok_count > 0:
                    latencies.append(result.elapsed)
                if result.read_elapsed is not None and result.ok_count > 0:
                    read_latencies.append(result.read_elapsed)
                if result.prep_elapsed is not None and result.ok_count > 0:
                    prep_latencies.append(result.prep_elapsed)
                if result.rpc_elapsed is not None and result.ok_count > 0:
                    rpc_latencies.append(result.rpc_elapsed)
                if result.fail_count > 0 or result.error:
                    failed_operations += 1
                else:
                    successful_operations += 1
                if result.error:
                    tqdm.write(result.error)
            pbar.update(sum(r.ok_count + r.fail_count for r in results))

    executor_cls: type[ThreadPoolExecutor] | type[ProcessPoolExecutor]
    executor_kwargs: dict[str, Any] = {"max_workers": max(1, concurrency)}
    if executor_kind == "process":
        executor_cls = ProcessPoolExecutor
        executor_kwargs["mp_context"] = mp.get_context("spawn")
    else:
        executor_cls = ThreadPoolExecutor
    if executor_initializer is not None:
        executor_kwargs["initializer"] = executor_initializer
        if executor_initargs:
            executor_kwargs["initargs"] = executor_initargs
    with executor_cls(**executor_kwargs) as executor:
        with tqdm(total=total_units, desc=description, unit="row") as pbar:
            for batch in prefetch_iterable(batches, prefetch_batches):
                if unit_count(batch) <= 0:
                    continue
                if isinstance(batch, dict):
                    read_elapsed = batch.get("_read_elapsed")
                    if read_elapsed is not None:
                        read_latencies.append(float(read_elapsed))
                while len(pending) >= max_pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    collect(done, pbar)
                pending.add(executor.submit(worker, batch))
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                collect(done, pbar)

    wall_elapsed = time.perf_counter() - wall_start
    return BenchmarkResult(
        ok_count=ok_count,
        fail_count=fail_count,
        latencies=latencies,
        prep_latencies=prep_latencies,
        rpc_latencies=rpc_latencies,
        read_latencies=read_latencies,
        wall_elapsed=wall_elapsed,
        operation_count=operation_count,
        successful_operations=successful_operations,
        failed_operations=failed_operations,
    )


def percentile_stats(latencies: list[float]) -> dict[str, float]:
    if not latencies:
        return {"TP50": 0.0, "TP90": 0.0, "TP99": 0.0}
    arr = np.asarray(latencies, dtype=np.float64)
    return {
        "TP50": float(np.percentile(arr, 50)),
        "TP90": float(np.percentile(arr, 90)),
        "TP99": float(np.percentile(arr, 99)),
    }


def print_result_table(
    title: str,
    result: BenchmarkResult,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    stats = percentile_stats(result.latencies)
    rows: list[tuple[str, str]] = []
    if metadata:
        rows.extend((key, str(value)) for key, value in metadata.items())
        rows.append(("-", "-"))
    rows.extend(
        [
            ("success_rows", str(result.ok_count)),
            ("failed_rows", str(result.fail_count)),
            ("success_rate", f"{result.success_rate * 100:.2f}%"),
            ("operations", str(result.operation_count)),
            ("successful_operations", str(result.successful_operations)),
            ("failed_operations", str(result.failed_operations)),
            ("wall_elapsed(s)", f"{result.wall_elapsed:.6f}"),
            ("success_rows/sec", f"{result.success_rows_per_second:.2f}"),
            ("total_rows/sec", f"{result.total_rows_per_second:.2f}"),
            ("operations/sec", f"{result.operations_per_second:.2f}"),
            ("latency_samples", str(len(result.latencies))),
            ("TP50(s)", f"{stats['TP50']:.6f}"),
            ("TP90(s)", f"{stats['TP90']:.6f}"),
            ("TP99(s)", f"{stats['TP99']:.6f}"),
        ]
    )
    has_helper_metrics = bool(result.read_latencies or result.prep_latencies or result.rpc_latencies)
    if has_helper_metrics:
        rows.append(("-", "以下为辅助验证脚本指标"))
    if result.read_latencies:
        read_stats = percentile_stats(result.read_latencies)
        rows.extend(
            [
                ("read_latency_samples", str(len(result.read_latencies))),
                ("read_TP50(s)", f"{read_stats['TP50']:.6f}"),
                ("read_TP90(s)", f"{read_stats['TP90']:.6f}"),
                ("read_TP99(s)", f"{read_stats['TP99']:.6f}"),
            ]
        )
    if result.prep_latencies:
        prep_stats = percentile_stats(result.prep_latencies)
        rows.extend(
            [
                ("prep_latency_samples", str(len(result.prep_latencies))),
                ("prep_TP50(s)", f"{prep_stats['TP50']:.6f}"),
                ("prep_TP90(s)", f"{prep_stats['TP90']:.6f}"),
                ("prep_TP99(s)", f"{prep_stats['TP99']:.6f}"),
            ]
        )
    if result.rpc_latencies:
        rpc_stats = percentile_stats(result.rpc_latencies)
        rows.extend(
            [
                ("rpc_latency_samples", str(len(result.rpc_latencies))),
                ("rpc_TP50(s)", f"{rpc_stats['TP50']:.6f}"),
                ("rpc_TP90(s)", f"{rpc_stats['TP90']:.6f}"),
                ("rpc_TP99(s)", f"{rpc_stats['TP99']:.6f}"),
            ]
        )
    width = max(len(k) for k, _ in rows + [(title, "")])
    print("\n" + title)
    print("-" * (width + 22))
    for key, value in rows:
        print(f"{key:<{width}} | {value}")
    print()
    print("结果解读提示")
    print("- 如果 read_* 较高，且 wall_elapsed 对 concurrency 变化不敏感，多半是本地磁盘或 HDF5 串行读取成为瓶颈。")
    print("- 如果 prep_* 较高，尤其是多向量写入，说明 Python 构造 token 对象和 float32 转换可能受 CPU 或 GIL 限制。")
    print("- 如果 rpc_* 较高，并且 concurrency 增加后 rpc_TP* 上升但吞吐不涨，说明 Milvus、网络、Proxy 或 DataNode 可能已经打满。")
    print("- wall_elapsed(s) 表示并发执行阶段的墙钟耗时，包含 HDF5 读取、records 构造、Milvus 请求、线程等待和进度条更新；不包含数据生成、建库建表、建索引、load_collection 和最终 flush。")


def timed_call(fn: Callable[[], Any]) -> tuple[Any, float]:
    start = time.perf_counter()
    value = fn()
    return value, time.perf_counter() - start


def maybe_flush(client: MilvusClient, collection_name: str, timeout: float) -> None:
    try:
        client.flush(collection_name=collection_name, timeout=timeout)
    except Exception as exc:
        print(f"flush skipped/failed: {exc!r}")
