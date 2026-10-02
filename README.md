# NInfer-4090

NInfer-4090 是把 NInfer 推理引擎跑在 **48 GiB NVIDIA RTX 4090(sm_89)** 上的下游分支,原生
Linux 直接部署(**不需要 Docker**)。引擎是 C++20/CUDA 自研实现,加载官方 `.ninfer` 权重容器,
通过 CLI 或 HTTP 服务(OpenAI Responses/Chat Completions、Anthropic Messages)提供推理。

主要负载是 **Qwen3.8-27B** 长上下文与投机解码。当前生产配置为 **NInfer v3 artifact +
bound-instance Engine**(2026-10-01 起),已经完整编译并通过发布门禁;上一代
**v2 artifact + 旧 Engine** 仍以 tag `v2-sm89-production-2026-09-30` 保留,可回退。

这份 README 面向第一次使用的人:按顺序做完就能构建、拿到权重、启动服务并发一次请求。想深挖再看
文末的文档链接。项目历史与维护边界的说明在
[docs/maintainer/downstream-maintenance.md](docs/maintainer/downstream-maintenance.md)。

---

## 这个仓库支持什么 / 不支持什么

**支持(在本机 4090 + Linux 上验证过)**

- 文本推理:CLI([docs/cli.md](docs/cli.md))与 HTTP 服务([docs/serving.md](docs/serving.md))。
- 上下文:`--max-context` 可开到模型原生上限;长上下文门禁见
  [发布记录](docs/maintainer/2026-10-01-v3-sm89-release.md)。
- 投机解码:`--spec mtp|dflash|dflash2 --draft-tokens N`;Qwen3.8-27B 上 MTP 与 DFlash2 都可用。
- KV 精度:`--kv-dtype bf16|int8|fp8|nvfp4|k8v4`。**速度实验中 KV 不得低于 INT8**(见下文性能一节)。
- 服务能力:流式输出、工具调用、token 计数、鉴权、Vision 图像/视频输入、会话槽持久化
  (`--slot-save-path`)、前缀复用、CUDA Graph。

**不支持或未验证**

- 多卡、NVFP4/W4A4 原生加速:**sm_89 没有原生 FP4**,相关路由不可用(发布记录里 13 项 skip 即此类)。
- Windows 路径与 Qwen3.6-35B-A3B 目标在 4090 上未验证(继承自上游,本分支未测)。
- 高并发抢占式调度:本引擎是启动期固定的 1–8 路并发、有界 FIFO,不做请求级抢占。

---

## 环境要求

- NVIDIA RTX 4090(48 GiB)、Linux、较新的 NVIDIA 驱动。
- CUDA 13.1(本分支验证工具链;CUDA 13.4 的对照实验见
  [roadmap](docs/maintainer/sm89-performance-roadmap.md) 的 H10 项)、GCC 14.2、CMake + Ninja。
- 权重需要约 20 GB 磁盘与约 18–19 GB 显存(随配置不同,见发布记录的 host load/bind 数据)。

---

## 1. 获取 artifact

**官方 v3 artifact**(2026-10-02 用 HTTP HEAD 核验固定 revision,`x-linked-etag` 与下表 SHA-256 一致):

| 字段 | 值 |
|---|---|
| 仓库与文件 | `neroued/Qwen3.8-27B-NInfer` → `qwen3_8_27b.ninfer` |
| Revision | `1cbd84e7221e51186bd7f093a149912d2489625b` |
| Size | 20,437,521,664 bytes(19.03 GiB) |
| SHA-256 | `81f924d440c27261d820c19a9f8d45794c5aee410f8a68bd358133fa8c0375da` |
| Container version | 3 |

两种等价做法,都得到仓库根目录下的 `models/qwen3_8_27b.ninfer`(与 `scripts/run-qwen38-c1.sh` 的默认路径一致)。

```bash
# 方式 A:仓库自带脚本(已 pin 上面的 revision,并校验 SHA-256)
# 脚本默认写到 scripts/models,因此显式指定 NINFER_MODEL_DIR 使其落在仓库根的 models/
NINFER_MODEL_DIR="$PWD/models" bash scripts/download-qwen38.sh

# 方式 B:直接用 Hugging Face CLI 并自行校验(--local-dir models 同样是仓库根的 models/)
hf download neroued/Qwen3.8-27B-NInfer qwen3_8_27b.ninfer \
  --revision 1cbd84e7221e51186bd7f093a149912d2489625b \
  --local-dir models
printf '%s  %s\n' \
  '81f924d440c27261d820c19a9f8d45794c5aee410f8a68bd358133fa8c0375da' \
  'models/qwen3_8_27b.ninfer' | sha256sum --check
```

**版本身份(三个都是不同文件,别互相替代)**:

| 文件 | Size | SHA-256 | 说明 |
|---|---|---|---|
| 当前官方 v3 `qwen3_8_27b.ninfer` | 20,437,521,664 | `81f924d4…` | 上面两种方式下载到的就是它;同时也是本仓[发布记录](docs/maintainer/2026-10-01-v3-sm89-release.md)与性能台账使用的产物(同一字节流) |
| 上一版 v3(历史) | 20,437,520,896 | `e91dbf53…` | revision `51630a0c…` 的发布内容;若你手上是这份,属于较旧构建 |
| v2 回退(旧容器) | — | `0634abb0…` | 见发布记录的 v2 回退项 |

张量清单、格式分布与许可见 [model-cards/Qwen3.8-27B-NInfer](model-cards/Qwen3.8-27B-NInfer/README.md);
自己从源权重转换见 [docs/weight-conversion.md](docs/weight-conversion.md)。

---

## 2. 构建

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j
```

产物:

- `build/apps/ninfer-serve` — HTTP 服务(下面用它)。
- `build/apps/ninfer` — 单次调用的 CLI(见 [docs/cli.md](docs/cli.md))。

---

## 3. 启动服务

默认(与 `scripts/run-qwen38-c1.sh` 一致,64K 上下文、MTP3、INT8 KV、单并发):

```bash
./build/apps/ninfer-serve models/qwen3_8_27b.ninfer \
  --host 127.0.0.1 --port 8080 \
  --max-context 65536 --kv-capacity 65536 \
  --max-concurrency 1 --max-pending-requests 16 \
  --prefill-chunk 1024 --kv-dtype int8 \
  --spec mtp --draft-tokens 3 --lm-head-draft
```

本文性能一节的"短码聚焦"结果用的是另一套 workload(8K 上下文、DFlash2 K7),命令是:

```bash
./build/apps/ninfer-serve models/qwen3_8_27b.ninfer \
  --host 127.0.0.1 --port 8080 \
  --max-context 8192 --kv-capacity 8192 \
  --max-concurrency 1 --prefill-chunk 1024 --kv-dtype int8 \
  --spec dflash2 --draft-tokens 7 --lm-head-draft
```

两次首次启动都要建 CUDA Graph:发布记录测到冷建图约 **9.5 分钟**,之后有缓存时 < 8 秒。所有
选项的权威拼写与默认值以 `./build/apps/ninfer-serve --help` 为准;完整服务契约(队列、超时、
状态、鉴权、工具调用、Vision)见 [docs/serving.md](docs/serving.md)。

---

## 4. 最小 API 调用

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [
      {"role": "system", "content": "Answer concisely."},
      {"role": "user", "content": "What is speculative decoding?"}
    ],
    "max_tokens": 128
  }'
```

Anthropic Messages、Responses 接口、流式、图像输入与工具调用的等价示例都在
[docs/serving.md](docs/serving.md)。

---

## 5. v2 → v3 有什么不同

按发布记录([2026-10-01-v3-sm89-release.md](docs/maintainer/2026-10-01-v3-sm89-release.md)):

- **容器与运行时**:改用 **NInfer artifact v3** 与 bound-instance `Qwen3.5` 运行时架构;
  Engine 与 artifact 绑定实例化,不再走旧的全局装配。
- **服务与持久会话**:恢复 **durable session slots** 与完整 Serve 契约(会话槽保存/恢复、
  前缀复用、host 状态恢复);v2 时期这些能力不完整。
- **长上下文**:通过 128K/256K 级别的长上下文正确性矩阵(64K NIAH 与 oracle 逐位一致)。
- **多模态与协议**:Vision 图像/视频输入与 OpenAI Responses/Chat Completions、Anthropic
  Messages 两套协议都在同一服务里。
- **sm_89 路由**:v3 落地时把若干热点切到 Ada 上更合适的实现(见下文"Op 级正向改动"),
  并保留可回退的 v2 生产 tag。

---

## 6. 当前性能研究的真实结果

所有数字都带 workload。**Op 级收益不等于端到端收益**,下面分开写。

### 6.1 单请求(C1)聚焦 workload

workload:V3 artifact、`scenario_code_python` fixture、seed 7632647173703958409、
8192 上下文、INT8 KV、输出 4096 token、单请求、同一二进制。

| 配置 | 解码吞吐 | 每轮设备时间 | 接受率 | 每轮 token |
|---|---|---|---|---|
| **DFlash2 K7(当前聚焦配置)** | **213.17 tok/s** | 24.203 ms | 59.56% | 5.169 |
| MTP3(同 binary / 同 fixture) | 147.56 tok/s | 22.843 ms | 79.10% | 3.373 |

结论:短码 workload 上 **DFlash2 K7 明显优于 MTP3**;K6 为 186.39 tok/s,窗口调大或调小都更差。
注意保存的 V2 基线(153.67 tok/s)用的是 **MTP3**,而当前配置是 DFlash2 K7:这是**跨投机后端**的对比,
不是同一后端的版本 A/B,+38.7% 只表示两端各自最佳配置之间的距离;同一后端(V2 → V3 优化 MTP3)的对照
见下文 6.6 与发布记录。

### 6.2 并发(聚合)workload —— 与 6.1 不是同一个指标

workload:DFlash2 K7、8192 上下文、INT8 KV、decode-saturation,同时跑 N 路请求。

| 并发 | 聚合吞吐 | 稳态 batch |
|---|---|---|
| C=2 | 125.7 tok/s | 2 |
| C=4 | 218.7 tok/s | 4 |
| C=8 | 348.5 tok/s | 8 |

聚合吞吐**不是**单请求延迟:并发翻倍并不带来吞吐翻倍(限制在 GPU 批内 device work)。要比较
"单用户更快",看 6.1;要比较"机器总产出",看 6.2。

### 6.3 64K 长上下文(C1)

workload:NIAH、INT8 KV,v3 与 v2 同条件对照(发布记录)。

| 指标 | v2 | v3 | 差异 |
|---|---|---|---|
| 64K NIAH prefill | 1,884.70 tok/s | 1,884.52 tok/s | −0.01% |
| 64K NIAH decode | 149.59 tok/s | 149.44 tok/s | −0.10% |

长上下文**持平**;两段 64K 回答都与 oracle 逐位一致。

### 6.4 已落地的 Op 级正向改动(实测值,仅代表该算子)

| 算子 / 形状 | 旧路由 | 现路由 | Op 级变化 |
|---|---|---|---|
| attention Q5 gate/value,T=8,split-output | 57.344 µs | 39.936 µs | −30.4% |
| GDN QK Q4,T=8 small-T MMA | 35.840 µs | 31.744 µs | −11.4% |
| GDN value/z Q5,T=8 K-split(生产输出形态) | 81.920 µs | 66.560 µs | −18.75% |

口径:cold-L2(256 MiB flush)、median、单算子(不含端到端)。**这些是 Op 级数字**;端到端是否
受益取决于该算子在每轮里的占比与接受率是否变化。已有反例:attention QKV Q4 的 small-T 候选
Op 级快 30.4%,但同 binary 的 K7 请求反而 −1.14%,因此**已回退生产路由**。

### 6.5 被证否 / 已关闭的方向(负面结果同样记录)

| 方向 | 结果 | 处理 |
|---|---|---|
| DFlash2 K8(窗口 8) | 119.58 tok/s(−43.9%) | 关闭 |
| MTP K3/K4(短码) | 147.56 / 149.28 tok/s,均低于 K7 | 保留 K7 |
| Q5 → Q4 表示层(LinearAdd 两类形状) | Op 级反而慢 7.5–11.6%(合计 +0.918 ms/轮) | 关闭,不做转换 |
| Q4 SwiGLU 行合并 / CTA 内双缓冲 | −2.7% / +0.67%(噪声内) | 已回退原型 |
| 跨 CTA K 切分 | 两点判别显示"固定开销"模型成立,方向为负 | 不做 |
| 七项热点审计(recurrent_record、attention small-T、RMSNorm、gdn_norm_gating、GDN conv prepare、gdn_projected_conv、记账+context/KV) | 合计 1.78 ms/轮,但分散在 ≥7 个互不相关算子,现实可回收 ≈0.24–0.44 ms/轮(推算) | 单项均不可单独达标 |
| GDN conv prepare `Columns=8→16` | Op 级 correctness 逐位一致,但 median ratio 1.0000、saving 0 | 已测关闭并回退 bench |

### 6.6 与目标的关系,以及数据边界

- 目标是把 C1 的 K7 从 213.17 提到 **230.5 tok/s(+8.1%,约需 −1.81 ms/轮)**。
- 按目前证据,这个缺口**没有被任何已知可执行项覆盖**:上面 6.5 列出的残余量与目标差 4–8 倍;
  想达标需要改变前提(例如更激进的权重表示、更好的草案质量/训练,或者改换记分口径到并发聚合)。
- **不给出成功率**:多数性能点是单样本;本机 Nsight Compute 因权限不可用(`ERR_NVGPUCTRPERM`),
  所以"有效带宽/延迟受限"一类判断是从字节与时间推算出来的,不是 DRAM/L2 计数器读数。
- **KV 精度底线**:不得用低于 INT8 的 KV 换速度(FP8 KV 不作为提速手段)。
- 短码 MTP 相比 v2 约慢 6%,与上面"K7 更快"不矛盾——两者 workload 不同,发布记录把它列为
  明确的后续目标,而不是用最好看的数字覆盖。

完整实验台账(含每次 A/B 的命令、门限与结论)在
[docs/maintainer/sm89-performance-roadmap.md](docs/maintainer/sm89-performance-roadmap.md);
每模型基线是上游 **RTX 5090** 的历史测量(本页只把它当来源引用,**不是本机 4090 结果**)见
[docs/performance/qwen3.8-27b.md](docs/performance/qwen3.8-27b.md);
发布数据在 [docs/performance/data/v3-sm89-release-2026-10-01.json](docs/performance/data/v3-sm89-release-2026-10-01.json)。

---

## 7. 从哪里继续读

| 想做的事 | 看这里 |
|---|---|
| 命令行、采样、投机解码、图像输入 | [docs/cli.md](docs/cli.md) |
| HTTP 服务、协议、流式、状态、鉴权、工具调用 | [docs/serving.md](docs/serving.md) |
| 权重转换与自定义格式 | [docs/weight-conversion.md](docs/weight-conversion.md) |
| v3 发布内容、门禁、不可变输入 | [docs/maintainer/2026-10-01-v3-sm89-release.md](docs/maintainer/2026-10-01-v3-sm89-release.md) |
| 性能路线图与实验台账 | [docs/maintainer/sm89-performance-roadmap.md](docs/maintainer/sm89-performance-roadmap.md) |
| 跑测试 | [tests/README.md](tests/README.md) |
| 跑基准 | [bench/README.md](bench/README.md) |
| 全部文档入口 | [docs/README.md](docs/README.md) |

---

## 上游与许可

本分支是 [Neroued/ninfer](https://github.com/Neroued/ninfer) 的 `sm_89` 下游,并参考
[sergiuszm/ninfer-4090](https://github.com/sergiuszm/ninfer-4090) 的 Ada 适配。通用修复应回到
上游;Ada 专属的能力分派、kernel、显存规划与 48 GiB 配置留在这里,并保持可度量。

许可证见 [LICENSE](LICENSE);模型权重遵循其各自发布页面的许可。
