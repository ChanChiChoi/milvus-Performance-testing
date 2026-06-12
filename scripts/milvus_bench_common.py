from __future__ import annotations

import math
import os
import random
import string
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
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
PLATFORMS = np.asarray([b"suning", b"taobao", b"jingdong"], dtype="S8")
TEXT_ALPHABET = np.frombuffer(
    (string.ascii_letters + string.digits).encode("ascii"), dtype=np.uint8
)


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


@dataclass
class BenchmarkResult:
    ok_count: int
    fail_count: int
    latencies: list[float]
    wall_elapsed: float
    operation_count: int
    successful_operations: int
    failed_operations: int

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
    h5.create_dataset("platform", shape=(rows,), dtype="S8")
    h5.create_dataset("text", shape=(rows,), dtype="S50")


def fill_common_datasets(
    h5: h5py.File,
    start: int,
    end: int,
    global_start_id: int,
    rng: np.random.Generator,
) -> None:
    size = end - start
    ids = np.arange(global_start_id + start, global_start_id + end, dtype=np.int64)
    h5["uuid1"][start:end] = make_uuid_array(ids, salt=0)
    h5["uuid2"][start:end] = make_uuid_array(ids, salt=10_000_000_000_000)
    h5["platform"][start:end] = rng.choice(PLATFORMS, size=size)
    h5["text"][start:end] = make_text_array(rng, size, 50)


def make_uuid_array(ids: np.ndarray, *, salt: int) -> np.ndarray:
    values: list[bytes] = []
    for value in ids:
        raw = f"{(int(value) + salt) % (1 << 128):032x}"
        values.append(
            f"{raw[:8]}-{raw[8:12]}-{raw[12:16]}-{raw[16:20]}-{raw[20:]}".encode(
                "ascii"
            )
        )
    return np.asarray(values, dtype="S36")


def make_text_array(rng: np.random.Generator, size: int, length: int) -> np.ndarray:
    indexes = rng.integers(0, len(TEXT_ALPHABET), size=(size, length), dtype=np.int16)
    chars = TEXT_ALPHABET[indexes]
    return np.asarray([row.tobytes() for row in chars], dtype=f"S{length}")


def decode_bytes(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("ascii")
    if hasattr(value, "decode"):
        return value.decode("ascii")
    return str(value)


def iter_h5_slices(paths: Iterable[Path], batch_size: int, dataset_names: tuple[str, ...]) -> Iterator[dict[str, Any]]:
    for path in paths:
        with h5py.File(path, "r") as h5:
            rows = int(h5.attrs["rows"])
            for start in range(0, rows, batch_size):
                end = min(start + batch_size, rows)
                yield {name: h5[name][start:end] for name in dataset_names}


def remove_tmp_and_replace(tmp_path: Path, final_path: Path) -> None:
    if final_path.exists():
        final_path.unlink()
    tmp_path.replace(final_path)


def run_concurrent_operations(
    batches: Iterable[Any],
    *,
    total_units: int,
    concurrency: int,
    worker: Callable[[Any], OperationResult],
    unit_count: Callable[[Any], int],
    description: str,
) -> BenchmarkResult:
    ok_count = 0
    fail_count = 0
    operation_count = 0
    successful_operations = 0
    failed_operations = 0
    latencies: list[float] = []
    pending: set[Any] = set()
    max_pending = max(1, concurrency * 2)
    wall_start = time.perf_counter()

    def collect(done: Iterable[Any], pbar: tqdm) -> None:
        nonlocal ok_count, fail_count, operation_count, successful_operations, failed_operations
        for future in done:
            try:
                result = future.result()
            except Exception as exc:
                result = OperationResult(0, 0, None, repr(exc))
            operation_count += 1
            ok_count += result.ok_count
            fail_count += result.fail_count
            if result.elapsed is not None and result.ok_count > 0:
                latencies.append(result.elapsed)
            if result.fail_count > 0 or result.error:
                failed_operations += 1
            else:
                successful_operations += 1
            pbar.update(result.ok_count + result.fail_count)
            if result.error:
                tqdm.write(result.error)

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
        with tqdm(total=total_units, desc=description, unit="row") as pbar:
            for batch in batches:
                if unit_count(batch) <= 0:
                    continue
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
    width = max(len(k) for k, _ in rows + [(title, "")])
    print("\n" + title)
    print("-" * (width + 22))
    for key, value in rows:
        print(f"{key:<{width}} | {value}")


def timed_call(fn: Callable[[], Any]) -> tuple[Any, float]:
    start = time.perf_counter()
    value = fn()
    return value, time.perf_counter() - start


def maybe_flush(client: MilvusClient, collection_name: str, timeout: float) -> None:
    try:
        client.flush(collection_name=collection_name, timeout=timeout)
    except Exception as exc:
        print(f"flush skipped/failed: {exc!r}")
