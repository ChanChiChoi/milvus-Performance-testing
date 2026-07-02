# my-milvus-yace

Milvus 压测脚本，使用 `uv` 管理依赖。

## 关键结论

- Milvus 服务端版本曾在测试环境探测为 `2.6.16`。
- 普通 dense `FLOAT16_VECTOR` 已通过极小写入测试。
- 官方 `v2.6.x` StructArray 文档声明 Struct 内向量支持 `FLOAT16_VECTOR` 等类型，但当前服务端实测仍只接受 `FLOAT_VECTOR` 子向量；`FLOAT16_VECTOR` 子向量在建表阶段会失败并返回 `now only float vector is supported`。
- 使用临时 `pymilvus 3.0.0` 重测也失败，因此当前判断不是项目里的 `pymilvus==2.6.15` 过低导致，而是这台服务端实际能力与文档声明不一致。
- 因此多向量写入和查询默认使用 `struct-float32` 模式：HDF5 落盘保存 fp16，插入/查询时按批次转成 float32 StructArray。

## 脚本清单

- `scripts/insert_single_vector.py`：生成单向量 HDF5 分片，重建并写入 `test_vector` collection，支持 `--executor-kind thread|process` 和 `--prefetch-batches`，默认使用 `process`。
- `scripts/insert_multi_vector.py`：生成多向量 HDF5 分片，重建并写入 `test_multi_vector` collection，支持 `--executor-kind thread|process` 和 `--prefetch-batches`，默认使用 `process`。
- `scripts/update_single_vector.py`：基于单向量 ready 分片生成 ready/random 混合 update 分片，对已有 `test_vector` 执行 `query -> delete(if found) -> insert`，默认使用 `process`。
- `scripts/update_multi_vector.py`：基于多向量 ready 分片生成 ready/random 混合 update 分片，对已有 `test_multi_vector` 执行 `query -> delete(if found) -> insert`，默认使用 `process`。
- `scripts/query_benchmark.py`：生成查询样本，执行单向量、多向量或组合查询压测，支持 `--executor-kind thread|process`、`--prefetch-batches` 和 `--replica-number`，默认使用 `process`。
- `scripts/collection_memory.py`：查看 collection 加载状态、segment 行数，并可结合 QueryNode `/metrics` 过滤 collection 相关 size 指标。

## 环境准备

项目要求 Python `>=3.13`，依赖由 `uv` 安装和隔离。首次执行前建议先同步依赖：

```bash
uv sync
```

设置 Milvus 连接地址（不要把内部地址写入仓库）：

```bash
export MILVUS_URI="http://localhost:19530"
```

所有脚本默认读取 `MILVUS_URI`；也可以临时用 `--uri` 覆盖。

默认依赖版本里固定了 `pymilvus==2.6.15`；当前 Milvus 服务端版本探测为 `2.6.16`。如果后续服务端升级或更换集群，建议先用小规模功能测试确认 StructArray 多向量能力和脚本参数仍然匹配。

## 数据分片

会生成 HDF5 的脚本分别使用独立目录：

- `data/test_vector`
- `data/test_multi_vector`
- `data/update_test_vector`
- `data/update_test_multi_vector`
- `data/query_samples`

默认每个目录生成 4 个 `.h5` 分片，每个分片 25 万条，总计 100 万条。脚本重跑时会按分片数量、行数、关键维度参数和公共标量数据版本判断是否跳过生成。

默认数据量很大：单向量 HDF5 原始向量 payload 约 1.91 GiB；多向量按 `1000000 x 300 x 128 x fp16` 计算，仅向量 payload 约 71.53 GiB，实际文件还包含标量字段和 HDF5 元数据。标量字段中 `id` 是查询和 update 匹配字段，并在写入脚本建 `INVERTED` 索引；`uuid1` / `uuid2` 使用标准 UUID 字符串，作为业务字段保留但不建索引；`platform` / `text` 使用 UTF-8 中文内容，`text` 为随机中文字符串。

## 执行前注意事项

- 写入脚本默认会 drop 已存在的目标 collection 后重建；如果只是检查现有 collection，不要直接运行写入命令，或加 `--no-drop` 让脚本在 collection 已存在时报错退出。
- update 脚本不会重建 collection，要求先用对应 insert 脚本写入 ready 数据；`--ready-data-dir` 必须指向 insert 生成的 HDF5 分片目录；如果 ready 行数不足以按 `--random-ratio` 生成目标 update 分片，脚本会在生成前报错，提示先重新运行 insert 脚本生成足够数据。
- 写入脚本支持 `--num-shards`，传给 Milvus `create_collection(num_shards=...)`，只在新建 collection 时生效；写入脚本和查询脚本都支持 `--replica-number`，会在脚本主动 `load_collection(replica_number=...)` 时生效，需要集群有足够 QueryNode 或 resource group 容量。
- 写入脚本支持 `--delete-ratio` 模拟 update 压力：默认 `0`，不执行额外操作；取值 `1-100` 时，每 100 条插入数据中选最后 N 条执行 `query -> delete -> insert`，其中 query/delete 使用同一批命中记录的 `id in [...]` / `pk in [...]` 批量 filter，最后再把同一批记录原样插回去。
- 参数必须保持一致：单向量写入的 `--vector-dim` 要等于查询脚本的 `--vector-dim`；多向量写入的 `--vector-dim` 要等于查询脚本的 `--multi-vector-dim`。
- 多向量写入的 `--token-count` 是 collection 每条样本最大 token vector 数量，查询脚本的 `--query-token-count` 是每条查询使用的 token vector 数量；当前默认写入 300、查询 30。
- 修改维度、token 数、`data_id_max` 或数据规模后，旧 HDF5 分片可能不再匹配；建议加 `--force-regenerate` 重新生成。查询脚本会在读取分片前校验 attrs 和 dataset shape。
- `single_limit=4000`、`multi_limit=4000` 对多向量查询非常重；如果显式设置 `search_ef`，正式压测前先用较小 `total_rows`、`single_limit`、`multi_limit` 和 `search_ef` 做冒烟与容量曲线。
- `--timeout` 默认 60 秒；多向量高 topK 查询可能触发 `DEADLINE_EXCEEDED`，需要结合服务端负载决定是否提高到 120/180 秒。

## 推荐执行顺序

1. 先跑“常用变体”里的小规模功能测试，确认客户端依赖、数据库权限、建表、写入和查询链路都正常。
2. 再分别跑单向量写入和多向量写入；写入完成后确认脚本摘要里的 `success_rate`、`success_rows`、collection 名称、维度、`num_shards` 和 `replica_number`。
3. 查询压测先跑 `--target single`，再跑 `--target multi`，最后再跑 `--target both`；多向量建议先降低 `--total-rows`、`--single-limit` / `--multi-limit` 和 `--search-ef` 做容量曲线。
4. 需要看 collection 内存时，先用 `scripts/collection_memory.py` 拿到 `collection_id`；如果 PyMilvus `mem_size` 返回 0，再进 QueryNode 容器或端口转发 `/metrics` 按 `collection_id` 过滤真实 size 指标。
5. 每轮正式压测建议用 `tee` 保存完整输出，后续对照 Milvus QueryNode、DataNode、Proxy、磁盘 IO 和网络监控一起看。

## 输出指标

写入和查询压测脚本都会在最后输出两类信息：本轮运行参数和结果摘要。建议每轮压测都保留完整输出，运维可以据此对照机器 CPU、内存、磁盘 IO、网络和 Milvus 组件指标。结果表里会先输出业务指标，再用分隔线标出“以下为辅助验证脚本指标”，方便区分主结果和阶段耗时。

### 统计口径

- `wall_elapsed(s)` 只统计并发执行阶段，不包含数据生成、建表、load collection 和 flush 时间。
- 写入脚本里的 `operation` 是一次 `client.insert()` 调用，通常等于一个 batch。
- 查询脚本里的 `operation` 是一个查询样本；当 `--target both` 时，一个 operation 内会先执行单向量查询，再执行多向量查询。`--multi-id-filter in` 在 `both` 下会把 single 返回的 `--single-limit` 个 id 作为 multi 查询的 `id in [...]` 过滤条件；在 `multi` 下会基于当前查询样本 id 确定性模拟 `--single-limit` 个 id。
- update 脚本里的 `operation` 是一个 update batch；batch 内会批量 query 一次、命中时批量 delete 一次，然后批量 insert 一次。
- `TP50(s)` / `TP90(s)` / `TP99(s)` 只统计成功 operation 的耗时。
- `read_TP*` / `prep_TP*` / `rpc_TP*` 是辅助验证指标，分别对应 HDF5 读取、Python 对象构造和 Milvus RPC。
- update 脚本会额外输出 `query_TP*` / `delete_TP*` / `insert_TP*`，分别对应 update 阶段的查询、删除和插入耗时；`rpc_TP*` 表示每个 update batch 内 query + delete + insert 的总 RPC 耗时。
- 写入脚本的 TP 是 batch 延迟，不是单条样本延迟；查询脚本的 TP 是单条查询样本延迟。
- `operations/sec` 在写入脚本中是 batch QPS，在查询脚本中基本等同于查询 QPS。
- `success_rows/sec` 更适合看样本吞吐；写入脚本表示成功写入样本数每秒，查询脚本表示成功查询样本数每秒。

### 运行参数字段

这些字段用于标识本轮压测的输入条件：

- `uri`：Milvus 服务地址。
- `db_name`：Milvus database 名称。
- `collection`：写入脚本当前重建并写入的 collection。
- `num_shards`：Milvus collection 写入分片数，写入脚本默认 `8`；这是远端 collection 的 shard 数，不是本地 HDF5 分片行数。
- `replica_number`：脚本主动加载 collection 时使用的副本数，写入和查询脚本默认 `2`。
- `single_collection` / `multi_collection`：查询脚本使用的单向量和多向量 collection。
- `target`：查询目标，取值为 `single`、`multi` 或 `both`。
- `data_dir`：本轮读取或生成 HDF5 分片的目录。
- `total_rows` / `total_queries`：本轮计划处理的写入样本数或查询样本数。
- `shard_rows`：每个 HDF5 分片的样本数，默认 25 万。
- `vector_dim`：单向量维度默认 1024；在多向量写入脚本中表示每个 token vector 的维度，默认 128。
- `multi_vector_dim`：查询脚本中的多向量 token vector 维度，默认 128。
- `token_count`：写入多向量样本的 token 向量数量，默认 300。
- `query_token_count`：查询样本里的多向量 token 数量，默认 30。
- `stored_dtype`：HDF5 分片中的向量存储精度；多向量脚本当前为 `float16`。
- `milvus_mode` / `multi_vector_mode`：多向量在 Milvus 里的写入/查询模式；当前默认 `struct-float32`。
- `insert_batch_size`：每次 `client.insert()` 提交的样本数。
- `delete_ratio`：写入脚本的 update 模拟比例，默认 `0` 表示关闭；例如 `1` 表示每 100 条插入后对最后 1 条执行 `query -> delete -> insert`。
- `ready_data_dir`：update 脚本读取既有 ready 数据的目录，通常指向对应 insert 脚本生成的 `data/test_vector` 或 `data/test_multi_vector`。
- `random_ratio`：update 脚本每 100 行窗口内随机新数据比例，范围 `1-100`；例如 `10` 表示 90 行 ready + 10 行 random，打散后写入 `data_dir`。
- `query_read_batch_size`：查询脚本每次从 HDF5 分片读入内存的样本数。
- `concurrency`：worker 池并发数；`executor_kind` 默认为 `process`，也可显式改为 `thread`。
- `index_type`：向量索引类型，默认 `HNSW`。
- `metric_type`：当前向量检索 metric；多向量默认 `MAX_SIM_COSINE`。
- `single_limit`：single 查询 topK，正式压测默认 4000。
- `multi_limit`：multi 查询 topK；`--target multi` 和 `--target both` 都使用该参数，默认等于 `single_limit`。
- `search_ef`：可选 HNSW 查询参数 `ef`；不传或传 `none` 时不会写入 Milvus search API，由 Milvus 使用默认值。传数字时应不小于实际使用的 topK，即 `single_limit` 和 `multi_limit` 中较大的值。
- `single_id_filter`：single 查询是否附带 id 过滤，默认 `none`，取值 `none|eq|gte`。
- `multi_id_filter`：multi 查询是否附带 id 过滤，默认 `eq`，取值 `none|eq|gte|in`；`in` 支持 `--target multi` 和 `--target both`；`both` 使用 single 实际返回的 id，`multi` 基于当前查询样本 id 确定性模拟 `single_limit` 个 id。

### CLI 参数补充

以下参数存在于三个压测脚本的命令行里，但不一定会出现在最终结果表的运行参数字段中：

- `--user` / `--password` / `--token`：Milvus 认证参数；未开启认证时保持默认空字符串。
- `--timeout`：Milvus RPC 超时时间，单位秒，默认 `60`。
- `--create-db`：当 `--db-name` 不存在时自动创建 database；不加时 database 不存在会报错退出。
- `--collection-name`：写入脚本重建并写入的 collection 名称；查询脚本使用 `--single-collection` 和 `--multi-collection` 指定查询目标。
- `--num-shards`：仅写入脚本支持，创建 Milvus collection 时传入的 shard 数，默认 `8`；修改该值需要重建 collection。
- `--replica-number`：写入和查询脚本均支持，传给 `load_collection()` 的副本数，默认 `2`；副本数过大时 load 可能因 QueryNode 资源不足失败。查询脚本加 `--no-load` 时不会重新 load，也不会应用该参数。
- `--generation-batch-size`：生成 HDF5 数据时每批写入内存数组的行数；单向量和查询单向量数据也用它作为 HDF5 chunk 行数，多向量写入的 `vector` chunk 由 `--vector-chunk-rows` 单独控制。
- `--force-regenerate`：忽略已有完整分片，重新生成 HDF5 数据。
- `--skip-generate`：跳过数据生成，直接读取现有 HDF5 分片执行建表、写入或查询。
- `--generate-only`：只生成 HDF5 分片，生成完成后退出。
- `--seed`：随机数据生成种子，默认 `20260610`。
- `--delete-ratio`：仅写入脚本支持，范围 `0-100`，默认 `0`。为 `0` 时只执行原始插入；大于 0 时每 100 条中选最后 N 条模拟 update，流程为先按同一批记录的 `id in [...]` 批量查询，再按命中的 `pk in [...]` 批量删除旧记录，最后将同一批记录插回去。
- `--executor-kind`：写入、查询和 update 脚本均支持 `thread|process`，默认 `process`。
- `--ready-data-dir`：仅 update 脚本支持，指定对应 insert 脚本之前生成的 ready 分片目录；ready 行数不足时会直接报错。
- `--random-ratio`：仅 update 脚本支持，范围 `1-100`；每 100 行窗口内按该比例生成随机新数据，其余从 `--ready-data-dir` 读取，并在窗口内打散。
- `--no-drop`：仅写入脚本支持；collection 已存在时不删除重建，而是报错退出。
- `--no-load`：仅查询脚本支持；查询前不调用 `load_collection`，适合 collection 已经由外部按目标副本数加载的场景。
- `--hnsw-m` / `--hnsw-ef-construction`：建 HNSW 索引时的 `M` 和 `efConstruction` 参数。
- `--metric-type`：写入脚本建索引用的 metric；单向量默认 `COSINE`，多向量 `struct-float32` 默认 `MAX_SIM_COSINE`。
- `--single-metric-type` / `--multi-metric-type` / `--flat-metric-type`：查询脚本搜索参数 metric；分别用于单向量、`struct-float32` 多向量和 `flat-fp16` 多向量。
- `--single-limit` / `--multi-limit`：查询脚本 topK 参数；single 查询使用 `--single-limit`，multi 查询使用 `--multi-limit`，`--multi-limit` 不传时默认等于 `--single-limit`。
- `--single-id-filter` / `--multi-id-filter`：查询脚本 id 过滤参数；single 默认 `none`，multi 默认 `eq`，`--multi-id-filter in` 支持 `--target multi` 和 `--target both`；`multi` 会基于当前查询样本 id 确定性模拟 `--single-limit` 个 id，`both` 使用 single 实际返回 id。
- `--search-ef`：可选 HNSW `ef` 参数；不传或传 `none` 时不会写入 Milvus search API。
- `--data-id-max`：查询样本随机生成 id 的最大值，默认 `1000000`；应与已写入数据的 id 范围一致。
- `--vector-chunk-rows`：多向量写入脚本生成 HDF5 `vector` dataset 时使用的 chunk 行数，默认 `4`；只影响本地分片文件的读写块大小，不影响 Milvus schema、向量维度、token 数或写入批次大小。

### 维度参数对照

| 场景 | 写入参数 | 查询参数 | 当前默认 |
| --- | --- | --- | --- |
| 单向量 | `insert_single_vector.py --vector-dim` | `query_benchmark.py --vector-dim` | `1024` |
| 多向量 token vector | `insert_multi_vector.py --vector-dim` | `query_benchmark.py --multi-vector-dim` | `128` |
| 多向量 token 数 | `insert_multi_vector.py --token-count` | `query_benchmark.py --query-token-count` | 写入 `300`，查询 `30` |

### 结果字段

这些字段用于判断压测结果和瓶颈拐点：

- `success_rows`：成功写入或查询的样本数。
- `failed_rows`：失败的样本数。
- `success_rate`：样本级成功率，等于 `success_rows / (success_rows + failed_rows)`。
- `operations`：实际提交到 worker 池执行的 operation 数；写入通常是 batch 数，查询通常是查询样本数。
- `batch_operations`：仅 update 脚本输出，表示实际提交到 worker 池执行的 update batch 数。
- `query_operations` / `delete_operations` / `insert_operations`：仅 update 脚本输出，分别表示实际执行的 query、delete、insert RPC 次数；batch 模式下每个成功 batch 通常各 1 次 query 和 insert，命中旧数据时 1 次 delete。
- `deleted_rows`：仅 update 脚本输出，表示 query 命中并被 delete 删除的旧记录数。
- `successful_operations`：完全成功的 operation 数。
- `failed_operations`：失败或部分失败的 operation 数。
- `wall_elapsed(s)`：并发执行阶段墙钟耗时，包含 HDF5 读取、records 构造、Milvus 请求、线程/进程等待和进度条更新；不包含数据生成、建库建表、建索引、`load_collection` 和最终 `flush`。
- `success_rows/sec`：成功样本吞吐，建议作为主要吞吐指标。
- `total_rows/sec`：成功和失败样本合计吞吐，用于观察失败较多时客户端实际推进速度。
- `operations/sec`：操作级 QPS；写入是 batch QPS，查询是查询 QPS。
- `read_latency_samples` / `prep_latency_samples` / `rpc_latency_samples`：辅助验证指标的样本数。
- `latency_samples`：参与 TP 统计的成功 operation 数。
- `TP50(s)`：成功 operation 的 50 分位耗时。
- `TP90(s)`：成功 operation 的 90 分位耗时。
- `TP99(s)`：成功 operation 的 99 分位耗时，建议重点用于判断服务抖动和容量拐点。

### Collection 内存口径

- `scripts/collection_memory.py` 优先读取 Milvus QueryNode 已加载 segment 的 `mem_size`；如果服务端返回 0，会明确标记 `mem_size_status=unavailable_server_returned_zero` 和 `loaded_mem_status=not_actual_memory_server_returned_zero`。
- 如果能访问 Milvus 组件容器内的 Prometheus `/metrics`，可以给内存脚本传 `--metrics-url http://127.0.0.1:9091/metrics`；脚本会用 `describe_collection()` 返回的 `collection_id` 过滤 `querynode` + `size` 指标，并打印匹配指标原文和按 metric name 汇总的值。
- 当前脚本里的 `payload_*`、`node_payload_estimate`、`field_payload_estimate_per_row` 都是按 schema 和行数推算的估算值，不是 Milvus 实际内存占用；脚本输出中会用 `payload_estimate_status=estimate_only_not_actual_memory` 和 `memory_note` 明确提示。
- `payload_*` 估算只覆盖原始字段 payload，不包含 HNSW/向量索引、segment 元数据、mmap/cache 和查询运行时开销；真实 QueryNode 进程内存应以服务端 Prometheus、容器 RSS 或运维监控为准。
- 如果 collection 没有 load，内存脚本可能显示 0 或无 loaded segment；需要主动加载时加 `--load`。

### Collection 内存字段

`scripts/collection_memory.py` 会按 collection 输出以下字段：

- `collection_id`：Milvus collection id，来自 `describe_collection()`；容器内查 `/metrics` 时可以用这个 id 过滤指标。
- `load_state`：Milvus 返回的 collection 加载状态。
- `row_count`：collection 统计行数，来自 `get_collection_stats()`。
- `persistent_rows` / `persistent_segments`：持久化 segment 的行数和 segment 数，来自 `list_persistent_segments()`。
- `loaded_rows` / `loaded_row_gap`：当前 loaded segments 覆盖的行数，以及 `row_count - loaded_rows` 的差值。
- `loaded_segments` / `unique_segments`：当前 QueryNode 已加载的 segment 记录数和按 `segment_id` 去重后的 segment 数。
- `node_ids`：承载这些 loaded segments 的 QueryNode id。
- `node_copy_count`：按 segment 所在 node 数量估算的副本拷贝数。
- `mem_size_status`：`available` 表示服务端返回了可用 `mem_size`；`unavailable_server_returned_zero` 表示 collection 已加载但服务端返回的 segment `mem_size` 全是 0。
- `loaded_mem_status`：`actual_pymilvus_mem_size` 表示 `loaded_mem` 可作为 PyMilvus 返回的 loaded segment 内存口径；`not_actual_memory_server_returned_zero` 表示 `loaded_mem` 不是实际内存，只是服务端返回 0。
- `loaded_mem` / `loaded_mem_bytes`：对 `list_loaded_segments()` 返回的 `mem_size` 求和；只有 `loaded_mem_status=actual_pymilvus_mem_size` 时才可作为 PyMilvus loaded segment 内存口径。
- `metrics_status` / `metrics_match_count`：是否成功读取 `--metrics-url`，以及按 `collection_id`、`querynode`、`size` 过滤后的 Prometheus 指标行数。
- `metrics_grouped_values` / `metrics_raw_lines`：传入 `--metrics-url` 时输出；前者按 metric name 汇总数值，后者保留 Prometheus 原始行，方便确认具体指标含义。
- `payload_estimate_status`：固定为 `estimate_only_not_actual_memory`，表示所有 `payload_*` 值都是估算值，不是实际 Milvus 内存。
- `payload_per_row_estimate` / `payload_per_row_bytes`：根据 schema 估算的单行原始字段 payload。
- `loaded_payload_estimate` / `row_count_payload_estimate`：根据 loaded 行数或 collection 统计行数推算的原始字段 payload；不包含索引、segment 元数据、mmap/cache 和查询运行时开销。
- `node_mem_estimate`：当 `mem_size` 可用时按 node 聚合真实返回值。
- `node_payload_estimate`：当 `mem_size` 不可用时按 node 行数聚合 payload 估算，不是实际 node 内存。
- `field_payload_estimate_per_row`：按字段拆分的单行 payload 估算，用于确认向量维度、token 数和字段类型是否符合预期。
- `memory_note`：脚本每个 collection 输出后都会打印该提示，说明当前 `loaded_mem` 是否可信，以及 `payload_*` 是否只是估算。

## 执行命令

脚本开头会清空 proxy 环境变量。真实压测建议在无 proxy shell 中执行。

### 1. 单向量写入

该命令会生成 `data/test_vector` 下 4 个 HDF5 分片，重建 `test_vector` collection，然后写入 100 万条单向量数据。

```bash
uv run python scripts/insert_single_vector.py \
  --uri "$MILVUS_URI" \
  --db-name llmbp \
  --timeout 60 \
  --total-rows 1000000 \
  --shard-rows 250000 \
  --data-dir data/test_vector \
  --generation-batch-size 2048 \
  --collection-name test_vector \
  --num-shards 8 \
  --replica-number 2 \
  --vector-dim 1024 \
  --insert-batch-size 1000 \
  --concurrency 4 \
  --executor-kind process \
  --metric-type COSINE \
  --index-type HNSW \
  --hnsw-m 16 \
  --hnsw-ef-construction 200 \
  --seed 20260610 \
  --delete-ratio 0
```

### 2. 多向量写入

该命令会生成 `data/test_multi_vector` 下 4 个 HDF5 分片，重建 `test_multi_vector` collection，然后写入 100 万条多向量数据。

当前服务端实测 StructArray 内 `FLOAT16_VECTOR` 建表失败，所以默认使用 `struct-float32`：落盘保存 fp16，写入 Milvus 前按批次转 float32。

```bash
uv run python scripts/insert_multi_vector.py \
  --uri "$MILVUS_URI" \
  --db-name llmbp \
  --timeout 60 \
  --total-rows 1000000 \
  --shard-rows 250000 \
  --data-dir data/test_multi_vector \
  --generation-batch-size 2048 \
  --collection-name test_multi_vector \
  --num-shards 8 \
  --replica-number 2 \
  --vector-dim 128 \
  --token-count 300 \
  --insert-batch-size 8 \
  --concurrency 2 \
  --prefetch-batches 4 \
  --executor-kind process \
  --multi-vector-mode struct-float32 \
  --metric-type MAX_SIM_COSINE \
  --flat-metric-type COSINE \
  --index-type HNSW \
  --hnsw-m 16 \
  --hnsw-ef-construction 200 \
  --vector-chunk-rows 4 \
  --seed 20260610 \
  --delete-ratio 0
```

### 3. 单向量 update

该命令会基于 `data/test_vector` 生成 `data/update_test_vector` 分片。`--random-ratio 10` 表示每 100 行中读取 90 行 ready 数据、生成 10 行 random 数据，打散后执行 update。脚本要求 `test_vector` 已经存在，不会 drop 或重建 collection。

```bash
uv run python scripts/update_single_vector.py \
  --uri "$MILVUS_URI" \
  --db-name llmbp \
  --timeout 60 \
  --total-rows 1000000 \
  --shard-rows 250000 \
  --ready-data-dir data/test_vector \
  --data-dir data/update_test_vector \
  --generation-batch-size 2048 \
  --collection-name test_vector \
  --replica-number 2 \
  --vector-dim 1024 \
  --insert-batch-size 1000 \
  --concurrency 4 \
  --executor-kind process \
  --random-ratio 10 \
  --seed 20260610
```

### 4. 多向量 update

该命令会基于 `data/test_multi_vector` 生成 `data/update_test_multi_vector` 分片，并对已有 `test_multi_vector` 执行 update。

```bash
uv run python scripts/update_multi_vector.py \
  --uri "$MILVUS_URI" \
  --db-name llmbp \
  --timeout 60 \
  --total-rows 1000000 \
  --shard-rows 250000 \
  --ready-data-dir data/test_multi_vector \
  --data-dir data/update_test_multi_vector \
  --generation-batch-size 2048 \
  --vector-chunk-rows 4 \
  --collection-name test_multi_vector \
  --replica-number 2 \
  --vector-dim 128 \
  --token-count 300 \
  --insert-batch-size 8 \
  --concurrency 2 \
  --prefetch-batches 4 \
  --executor-kind process \
  --multi-vector-mode struct-float32 \
  --random-ratio 10 \
  --seed 20260610
```

### 5. 查询压测

该命令会生成 `data/query_samples` 下 4 个 HDF5 查询分片，按 `--replica-number` 加载 `test_vector` 和 `test_multi_vector`，然后对 100 万条查询样本做 topK=4000 查询。查询脚本默认使用 `--executor-kind process`，并支持 `--prefetch-batches`，适合把客户端预处理和服务端 RPC 分开观察。若 collection 已由外部按目标副本数加载，可加 `--no-load` 避免脚本重新加载。

```bash
uv run python scripts/query_benchmark.py \
  --uri "$MILVUS_URI" \
  --db-name llmbp \
  --timeout 60 \
  --total-rows 1000000 \
  --shard-rows 250000 \
  --data-dir data/query_samples \
  --generation-batch-size 512 \
  --single-collection test_vector \
  --multi-collection test_multi_vector \
  --replica-number 2 \
  --target both \
  --multi-vector-mode struct-float32 \
  --vector-dim 1024 \
  --multi-vector-dim 128 \
  --query-token-count 30 \
  --data-id-max 1000000 \
  --query-read-batch-size 512 \
  --concurrency 8 \
  --prefetch-batches 16 \
  --executor-kind process \
  --single-limit 4000 \
  --multi-limit 4000 \
  --single-metric-type COSINE \
  --multi-metric-type MAX_SIM_COSINE \
  --flat-metric-type COSINE \
  --single-id-filter none \
  --multi-id-filter eq \
  --seed 20260610
```

### 2026-06-12 查询冒烟结论

本轮验证使用当前 collection：`test_vector` 为单向量 `FLOAT16_VECTOR dim=1024`，`test_multi_vector` 为多向量 `tokens[token_vector] FLOAT_VECTOR dim=128`；查询样本为单向量 1024 维、多向量 `30 x 128`。

- `target=single`：60 条查询全部成功，`success_rate=100.00%`，`wall_elapsed=2.14s`，`TP50=0.129s`，`TP90=0.171s`，`TP99=0.221s`。
- `target=multi`：60 条查询成功 59 条、失败 1 条，失败原因为 `DEADLINE_EXCEEDED` 60 秒超时；`wall_elapsed=613.69s`，`TP50=40.985s`，`TP90=50.410s`，`TP99=54.857s`。
- `target=both`：6 条查询全部成功，`success_rate=100.00%`，`wall_elapsed=56.33s`，`TP50=18.503s`，`TP90=20.885s`，`TP99=22.287s`。

结论：维度拆分链路正常，没有发现单向量 1024 维和多向量 128 维的 shape/EmbeddingList 维度错配；当前主要瓶颈是多向量在 `single_limit=4000`、`multi_limit=4000`、`query_token_count=30` 下查询非常重，并可能触发 60 秒 RPC timeout。正式压测多向量时建议先降低 `single_limit/multi_limit` 做容量曲线；如果需要手动控制 HNSW `ef`，再显式设置 `--search-ef`，或提高 `--timeout` 到 120/180 秒后观察 timeout 情况。

## FAQ

如果你在对 `concurrency`、`insert_batch_size`、`prefetch_batches` 或 `executor_kind` 做对比，建议先看根目录的 [faq.md](faq.md)。里面把“单窗口不变、双窗口变快、`wall_elapsed` 变化不大”等现象整理成了可直接复用的问答。

### 6. Collection 内存查看

该命令查看 `test_vector` 和 `test_multi_vector` 当前已加载 segment 的内存占用。默认不会主动 load collection，避免无意中把大集合加载进 QueryNode。

```bash
uv run python scripts/collection_memory.py \
  --uri "$MILVUS_URI" \
  --db-name llmbp \
  --timeout 60 \
  test_vector \
  test_multi_vector
```

如果需要脚本先执行 `load_collection()` 再查看，加 `--load`：

```bash
uv run python scripts/collection_memory.py \
  --uri "$MILVUS_URI" \
  --db-name llmbp \
  --timeout 60 \
  --load \
  test_vector \
  test_multi_vector
```

需要排查 segment 级别明细时，加 `--show-segments`。如果只是临时查看并希望查看后释放 collection，可以加 `--release-after`。

如果能进入 Milvus QueryNode 容器或通过端口转发访问组件 metrics，可以直接用 collection id 过滤 Prometheus 指标。先用脚本输出里的 `collection_id`，再在容器内执行：

```bash
curl -s http://127.0.0.1:9091/metrics \
  | grep -i 'querynode' \
  | grep -i 'size' \
  | grep 466899299346216185
```

也可以让脚本自动读取 metrics 并按当前 collection 的 `collection_id` 过滤：

```bash
uv run python scripts/collection_memory.py \
  --uri "$MILVUS_URI" \
  --db-name llmbp \
  --timeout 60 \
  --metrics-url http://127.0.0.1:9091/metrics \
  test_vector \
  test_multi_vector
```

`127.0.0.1:9091` 通常是容器或 Pod 内视角；如果在普通客户端机器执行脚本，需要先做端口转发或换成可访问的 metrics 地址。

## 常用变体

### 只生成数据分片

```bash
uv run python scripts/insert_single_vector.py \
  --generate-only \
  --uri "$MILVUS_URI" \
  --db-name llmbp \
  --total-rows 1000000 \
  --shard-rows 250000 \
  --data-dir data/test_vector \
  --vector-dim 1024

uv run python scripts/insert_multi_vector.py \
  --generate-only \
  --uri "$MILVUS_URI" \
  --db-name llmbp \
  --total-rows 1000000 \
  --shard-rows 250000 \
  --data-dir data/test_multi_vector \
  --vector-dim 128 \
  --token-count 300 \
  --multi-vector-mode struct-float32

uv run python scripts/query_benchmark.py \
  --generate-only \
  --uri "$MILVUS_URI" \
  --db-name llmbp \
  --total-rows 1000000 \
  --shard-rows 250000 \
  --data-dir data/query_samples \
  --vector-dim 1024 \
  --multi-vector-dim 128 \
  --query-token-count 30 \
  --data-id-max 1000000

uv run python scripts/update_single_vector.py \
  --generate-only \
  --uri "$MILVUS_URI" \
  --db-name llmbp \
  --total-rows 1000000 \
  --shard-rows 250000 \
  --ready-data-dir data/test_vector \
  --data-dir data/update_test_vector \
  --vector-dim 1024 \
  --random-ratio 10

uv run python scripts/update_multi_vector.py \
  --generate-only \
  --uri "$MILVUS_URI" \
  --db-name llmbp \
  --total-rows 1000000 \
  --shard-rows 250000 \
  --ready-data-dir data/test_multi_vector \
  --data-dir data/update_test_multi_vector \
  --vector-dim 128 \
  --token-count 300 \
  --multi-vector-mode struct-float32 \
  --random-ratio 10
```

### 已有分片时跳过生成

```bash
uv run python scripts/insert_single_vector.py \
  --skip-generate \
  --uri "$MILVUS_URI" \
  --db-name llmbp \
  --timeout 60 \
  --total-rows 1000000 \
  --shard-rows 250000 \
  --data-dir data/test_vector \
  --collection-name test_vector \
  --num-shards 8 \
  --replica-number 2 \
  --vector-dim 1024 \
  --insert-batch-size 1000 \
  --concurrency 4

uv run python scripts/insert_multi_vector.py \
  --skip-generate \
  --uri "$MILVUS_URI" \
  --db-name llmbp \
  --timeout 60 \
  --total-rows 1000000 \
  --shard-rows 250000 \
  --data-dir data/test_multi_vector \
  --collection-name test_multi_vector \
  --num-shards 8 \
  --replica-number 2 \
  --vector-dim 128 \
  --token-count 300 \
  --insert-batch-size 8 \
  --concurrency 2 \
  --prefetch-batches 4 \
  --executor-kind process \
  --multi-vector-mode struct-float32

uv run python scripts/query_benchmark.py \
  --skip-generate \
  --uri "$MILVUS_URI" \
  --db-name llmbp \
  --timeout 60 \
  --total-rows 1000000 \
  --shard-rows 250000 \
  --data-dir data/query_samples \
  --single-collection test_vector \
  --multi-collection test_multi_vector \
  --replica-number 2 \
  --target both \
  --multi-vector-mode struct-float32 \
  --vector-dim 1024 \
  --multi-vector-dim 128 \
  --query-token-count 30 \
  --data-id-max 1000000 \
  --query-read-batch-size 512 \
  --concurrency 8 \
  --prefetch-batches 16 \
  --executor-kind process \
  --single-limit 4000 \
  --multi-limit 4000 \
```

### 小规模功能测试

以下命令只生成少量数据并使用 `/tmp` 目录，适合验证脚本链路。

```bash
uv run python scripts/insert_single_vector.py \
  --force-regenerate \
  --total-rows 2 \
  --shard-rows 2 \
  --vector-dim 4 \
  --generation-batch-size 2 \
  --data-dir /tmp/my_script_single_func \
  --collection-name tmp_codex_script_single \
  --num-shards 8 \
  --replica-number 2 \
  --db-name default \
  --insert-batch-size 2 \
  --concurrency 1 \
  --timeout 20

uv run python scripts/insert_multi_vector.py \
  --force-regenerate \
  --total-rows 2 \
  --shard-rows 2 \
  --vector-dim 4 \
  --token-count 2 \
  --generation-batch-size 2 \
  --vector-chunk-rows 1 \
  --data-dir /tmp/my_script_multi_func \
  --collection-name tmp_codex_script_multi \
  --num-shards 8 \
  --replica-number 2 \
  --db-name default \
  --insert-batch-size 1 \
  --concurrency 1 \
  --timeout 20

uv run python scripts/query_benchmark.py \
  --force-regenerate \
  --total-rows 2 \
  --shard-rows 2 \
  --vector-dim 4 \
  --multi-vector-dim 4 \
  --query-token-count 2 \
  --generation-batch-size 2 \
  --data-id-max 2 \
  --data-dir /tmp/my_script_query_func \
  --single-collection tmp_codex_script_single \
  --multi-collection tmp_codex_script_multi \
  --replica-number 2 \
  --db-name default \
  --target both \
  --concurrency 1 \
  --query-read-batch-size 2 \
  --single-limit 2 \
  --multi-limit 2 \
  --timeout 20
```

功能测试后删除临时 collection：

```bash
uv run python - <<'PY'
import os
for key in (
    'http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY',
    'all_proxy', 'ALL_PROXY', 'no_proxy', 'NO_PROXY', 'htp_proxy',
):
    os.environ[key] = ''
from pymilvus import MilvusClient
client = MilvusClient(uri=os.environ["MILVUS_URI"], db_name="default", timeout=20)
for name in ['tmp_codex_script_single', 'tmp_codex_script_multi']:
    if client.has_collection(name):
        client.drop_collection(name, timeout=20)
        print('dropped', name)
client.close()
PY
```

## 常见排查

- `Invalid query shards`：查询分片和当前参数不匹配，常见于改过 `--vector-dim`、`--multi-vector-dim`、`--query-token-count` 或 `--data-id-max` 后继续复用旧 HDF5；加 `--force-regenerate` 重新生成。
- `--search-ef must be >= max search limit`：显式设置的 HNSW 查询参数太小；要么降低 `--single-limit` / `--multi-limit`，要么提高 `--search-ef`，或不传 `--search-ef` 让 Milvus 使用默认值。
- `DEADLINE_EXCEEDED`：服务端在 `--timeout` 内没有完成 RPC；多向量高 topK 场景先降低 `--single-limit/multi-limit/query-token-count`，或按需要显式设置 `--search-ef` 做曲线，再考虑把 `--timeout` 提高到 120/180 秒。
- `now only float vector is supported`：当前服务端不支持 Struct 内 `FLOAT16_VECTOR`；保持默认 `--multi-vector-mode struct-float32`。
- `127.0.0.1:9091` 访问失败：这个地址通常只在 Milvus 容器或 Pod 内可见；在客户端机器执行时需要端口转发，或改成实际可访问的 metrics 地址。
- `collection_memory.py` 里 `loaded_mem=0`：当前服务端可能没有返回可用的 segment `mem_size`；此时只能把 `payload_*` 当估算值，真实内存以 `/metrics`、容器 RSS 或运维监控为准。
- `replica_number` load 失败：通常是 QueryNode 数量、resource group 或内存容量不足；先临时降低副本数验证链路，再结合集群资源调整。
