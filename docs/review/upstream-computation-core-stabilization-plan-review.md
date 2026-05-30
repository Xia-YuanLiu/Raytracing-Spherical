# Review: Upstream Computation Core Stabilization Plan

Reviewer: Claude Code (automated codebase audit)
Date: 2026-05-29
Plan reviewed: `docs/plans/upstream-computation-core-stabilization-plan.md`

---

## 1. 总体判断

**这份 plan 方向正确，优先级大体合理，可以执行。** 它的核心判断——先建验证护栏再重构再优化——与这个仓库作为上游科学计算库的角色完全一致。

最大风险不是 plan 本身，而是以下三点：

1. **P0.1 与现有 `test_package.py` 有大量重叠。** Plan 没有意识到已有 25/50 个符号的导入测试，直接新建 `test_public_api_contract.py` 会导致重复维护负担。
2. **P0.6（benchmark 标准化）优先级过高。** 现有 benchmark 已经可以工作，输出字段基本够用。在没有性能基线需求之前，打磨 benchmark 是低回报的。
3. **P1.3（batch tracing）在 P1 中为时过早。** 没有性能基线数据证明批量追迹是瓶颈，过早引入会增加复杂度。

---

## 2. 验证方法

### 读取的文件

- `docs/plans/upstream-computation-core-stabilization-plan.md`（全文）
- `src/spherical_raytracing/__init__.py`（`__all__` 列表）
- `src/spherical_raytracing/solvers.py`（`_make_segment` 及其变体）
- `src/spherical_raytracing/junction_tracing.py`（`_make_segment`）
- `src/spherical_raytracing/_turning.py`
- `src/spherical_raytracing/metrics.py`
- `src/spherical_raytracing/numerical_metrics.py`
- `src/spherical_raytracing/numerical_loader.py`
- `src/spherical_raytracing/junctions.py`
- `src/spherical_raytracing/junction_analysis.py`
- `src/spherical_raytracing/diagnostics.py`
- `src/spherical_raytracing/observers.py`
- `src/spherical_raytracing/policies.py`
- `src/spherical_raytracing/transfer.py`
- `src/spherical_raytracing/sources.py`
- `src/spherical_raytracing/imaging.py`
- `tests/test_package.py`
- `tests/test_numerical_metric_validation.py`
- `tests/test_numerical_metrics.py`
- `tests/test_static_junction_backend_agreement.py`
- `scripts/benchmark_numerical_metric_perf.py`
- `docs/adr/0001-upstream-computation-core-api-contract.md`
- `docs/review/raytracing-spherical-review-roadmap.md`（前 80 行）
- `pyproject.toml`
- `.gitignore`

### 运行的命令

| 命令 | 结果 |
|---|---|
| `python3 -m pytest tests/ -q --tb=no` | 240 passed, 28 warnings, 150s |
| `python3 -c "import spherical_raytracing; print(len(spherical_raytracing.__all__))"` | 50 |
| `grep -rn "pytest.mark.slow" tests/` | 无结果（不存在） |
| `grep -rn "Li-DM-BH" tests/ scripts/` | 确认 4 个文件依赖本地路径 |
| `git ls-files outputs/ \| wc -l` | 429 个文件，48MB |
| `du -sh outputs/` | 48M |

### 未运行的原因

- **L4 网格收敛测试**：不存在，无法运行。
- **L5 后端比较测试**：不存在，无法运行。
- **benchmark script**：依赖本地 Li-DM-BH 路径（可选跳过），未完整运行。

---

## 3. Findings

按严重程度排序。

### F1（中等）：`__all__` 计数准确，但 plan 没有发现已有部分 API 测试

**Plan 声称**：需要新建 `test_public_api_contract.py` 来保护 50 个导出符号。

**实际情况**：

- `__all__` 确实包含 50 个符号（plan 正确）。
- `tests/test_package.py` 已经测试了其中 25 个的导入和基本可用性。
- 剩余 25 个未覆盖的符号是：`CriticalCurve`, `EventType`, `ImageGrid`, `IntersectionRecord`, `JunctionPhysicsDiagnostics`, `ObservedIntensity`, `OutsideTracePolicy`, `RadialProfile`, `RayDiagnostics`, `RayEvent`, `RayResult`, `RaySegment`, `RenderedImage`, `SamplingOptions`, `SegmentConstants`, `ShellCrossingRecord`, `SolverOptions`, `ThroughTracePolicy`, `annotate_junction_intersections`, `compute_intersections`, `junction_redshift_factor`, `match_static_shell`, `observed_intensity`, `render_axisymmetric_image`, `sample_radial_profile`。

**影响**：如果直接新建 `test_public_api_contract.py` 而不整合 `test_package.py`，会有两个测试文件做几乎相同的事，维护负担加倍。

**建议**：在 `test_package.py` 中扩展覆盖，而不是新建文件。或者将 `test_package.py` 重命名为 `test_public_api_contract.py` 并在其中补齐缺失的 25 个符号。

### F2（中等）：P0.6 benchmark 标准化优先级过高

**Plan 声称**：P0.6 需要在优化前"标准化 benchmark 输出"。

**实际情况**：

- `scripts/benchmark_numerical_metric_perf.py` 已经存在并且可以工作。
- 它已经输出：n、median、p95、termination reasons、turning point counts、category tags、fast segment stats（accepted/attempted/fallback/late_fallback）、u_at_probes count。
- 缺少的字段（date、Python/NumPy 版本、grid size）是锦上添花，不影响优化决策。
- Plan 说"P0 does not require hard timing thresholds"，这意味着 benchmark 标准化在 P0 阶段没有实际验收标准。

**影响**：在没有实际性能优化需求之前打磨 benchmark 输出格式，投入产出比低。

**建议**：将 P0.6 降级为 P1.4（与工程 guardrails 一起做）或删除，改为"P1 首次优化前跑一次 benchmark 记录基线"。

### F3（低）：P0.2 synthetic `.npz` fixtures 部分冗余

**Plan 声称**：需要新建 `tests/fixtures/metrics/` 目录和 synthetic `.npz` 文件。

**实际情况**：

- `tests/test_numerical_metric_validation.py` 已经有一个 `manufacture_numerical_metric()` 工厂函数，能在测试中动态生成 synthetic payload，支持 `metadata` 和 `standard-rab` 两种风格。
- `tests/test_numerical_metrics.py` 也已经有动态构造 `.npz` 文件的 fixture（`test_load_metric_npz_handles_li_dm_bh_format` 等，用 `tmp_path`）。
- 仅有的真实 Li-DM-BH 依赖在 `test_numerical_metrics.py` 的 `LI_DATA_DIR` fixture 和 `li_dm_bh_payloads` fixture 中，都已用 `pytest.skip` 保护。

**影响**：新建静态 `.npz` 文件会增加需要维护的二进制 fixture，而现有的动态构造已经覆盖了同样的路径。

**建议**：如果目的是消除 `pytest.skip` 的测试盲区，应该在 `test_numerical_metrics.py` 中为那些 currently-skipping 的测试写 synthetic 替代版本，而不是新建 fixture 目录。静态 `.npz` 文件只在需要验证真实上游格式兼容性时才有价值——这个需求目前不存在。

### F4（低）：P0.3 L4 收敛测试的范围定义模糊

**Plan 声称**："compare thin-disk intersection radius trend or non-worsening envelope"。

**问题**：

- "trend or non-worsening envelope" 是一个不明确的验收标准。收敛测试的核心价值是检测回归，但"非劣化"在没有基线的情况下无法判定。
- Plan 说"three grid densities"，但没有说明哪三个。实际的收敛行为高度依赖网格间距选择。
- 排除项（near-critical, horizon-adjacent, table-boundary, max_phi truncation）虽然合理，但排除了很多实际的困难 ray。

**建议**：L4 smoke test 应该是一个相对回归测试（用固定种子固定参数，断言具体数值在固定容限内），而不是试图断言"收敛趋势"。收敛趋势研究可以是 `@pytest.mark.slow` 的非阻塞测试。

### F5（低）：P0.5 junction 退化测试的价值有限

**Plan 声称**：identical-metric junction 应该退化到单度规结果。

**实际情况**：

- 这是一个有意义的物理不变量测试（能量比 = 1，b 不变）。
- 但 `match_static_shell()` 在内/外度规相同时，`sqrt(A_to/A_from) = 1`，这是一个简单的数学恒等式。测试的是一行 `ratio = math.sqrt(a_to / a_from)` 在 `a_to == a_from` 时返回 1。
- 更有价值的 junction 物理测试应该是：非退化 case 的红移因子是否与单度规有限半径观测者的红移因子一致（这已经部分由 `test_static_junction_sources.py` 覆盖）。

**建议**：保留 P0.5 但缩小范围——只测试 `match_static_shell` 的数学不变量，不做完整成像对比。完整成像对比的价值不大，因为 junction solver 和单度规 solver 走的是完全不同的代码路径，结果精确一致反而不太可能（也不应该期望）。

### F6（信息）：segment 构造重复的真实程度

**Plan 声称**：`solvers.py` 和 `junction_tracing.py` 之间存在 segment 构造重复。

**实际情况**：

- `solvers.py` 有三个 `_make_segment` 变体：`_make_segment`（调度器，42 行）、`_make_segment_tabulated_fast`（快速路径，~145 行）、`_make_segment_quad_reference`（参考路径，~92 行）。
- `junction_tracing.py` 有一个 `_make_segment`（95 行），只实现 quad 路径。
- 两者的核心积分逻辑结构相同（turning point 变量替换 + `quad` + `brentq` 反求），但 junction 版本有一些差异：不需要 fast path、参数传递方式不同、region boundary 逻辑不同。
- 重复程度：约 60-70 行核心积分逻辑是实质性重复。

**影响**：P1.1 提取共享 kernel 是有意义的，但规模比 plan 暗示的要小。实际可提取的是一个纯函数式的 `compute_segment_integral(metric, b, u0, u1, phi0, direction, ...)` helper，而不是一个完整的 segment factory。

### F7（信息）：429 个 git-tracked 生成文件

**Plan 提到**：P2.1 再处理 `outputs/`。

**实际情况**：

- `outputs/` 下有 254 个 PNG、89 个 CSV、80 个 JSON，共 429 个文件，48MB。
- 这些文件全部被 git 跟踪（`git ls-files`）。
- `.gitignore` 没有排除 `outputs/`。

**影响**：这 48MB 的生成物会膨胀 clone 体积。虽然 plan 说"P2 再处理"，但如果 P0 要建 CI，这些文件会拖慢 clone。建议至少在 P0.7 CI 阶段就加一个 `.gitignore` 规则。

### F8（信息）：没有 `@pytest.mark.slow`，没有 `conftest.py`

**Plan 依赖**：P0.3 要注册 `@pytest.mark.slow`。

**实际情况**：

- 当前测试中没有任何 `@pytest.mark.slow`。
- 没有 `tests/conftest.py`。
- `pyproject.toml` 没有 `markers` 配置。

**影响**：P0.3 需要在 `pyproject.toml` 中注册 marker 并创建 `conftest.py`。这是实际工作量，plan 应该明确提到。

---

## 4. P0/P1/P2 顺序建议

### 应该保持的

| 任务 | 理由 |
|---|---|
| P0.1 API 测试 | 核心需求，但应扩展现有 `test_package.py` 而非新建文件 |
| P0.2 synthetic fixtures | 方向对，但应评估是否需要静态文件 vs 动态构造 |
| P0.4 L5 后端比较 | 科学正确性的重要护栏，当前确实不存在 |
| P0.5 junction 退化 | 物理不变量测试，便宜且有意义 |
| P0.7 CI | 核心工程需求 |
| P1.1 segment kernel 提取 | 减少重复，为后续优化铺路 |
| P1.2 TabulatedMetric 快速路径 | 已有基础，只需在验证护栏建好后继续 |

### 应该提前的

| 任务 | 从 → 到 | 理由 |
|---|---|---|
| `.gitignore` 更新（排除 `outputs/`） | P2 → P0.7 的一部分 | CI clone 速度 |

### 应该推后的

| 任务 | 从 → 到 | 理由 |
|---|---|---|
| P0.6 benchmark 标准化 | P0 → P1（或删除） | 现有 benchmark 已够用，标准化没有硬性验收标准 |
| P1.3 batch tracing | P1 → P2 或更后 | 没有性能数据证明它是瓶颈 |

### 应该合并的

| 合并 | 理由 |
|---|---|
| P0.1 + `test_package.py` 扩展 | 避免两个文件做同样的事 |
| P0.3 L4 smoke → 并入 `test_numerical_metric_validation.py` | 主题相同，放一个文件更好维护 |

### 应该删除的

| 任务 | 理由 |
|---|---|
| `tests/fixtures/metrics/generate_fixtures.py` | 动态构造已经够用，静态 fixture 生成器是额外维护成本 |

---

## 5. 精简后的推荐执行计划

### P0（核心护栏）

1. **P0.1 扩展 `test_package.py`**（不是新建文件）
   - 在现有 `test_package.py` 中补齐剩余 25 个符号的导入测试
   - 增加 `__all__` 集合不变量断言
   - 增加两个工作流 smoke test（单度规 + junction）
   - 验证：`pytest tests/test_package.py -q`

2. **P0.2 消除 Li-DM-BH skip 盲区**
   - 为 `test_numerical_metrics.py` 中 currently-skipping 的真实文件测试添加 synthetic 替代
   - 不需要新建 `tests/fixtures/` 目录
   - 验证：`pytest tests/test_numerical_metrics.py -q`（不依赖本地路径）

3. **P0.3 L4 收敛 smoke**
   - 在 `test_numerical_metric_validation.py` 中新增 L4 测试函数
   - 用固定参数、固定种子，断言具体数值在固定容限内
   - 注册 `@pytest.mark.slow`（需更新 `pyproject.toml` 和新建 `tests/conftest.py`）
   - 验证：`pytest tests/test_numerical_metric_validation.py -q`

4. **P0.4 L5 后端比较**
   - 新建 `tests/test_backend_comparator.py`
   - scope 如 plan 所述（Schwarzschild strict + RN/RN-dS warm）
   - 验证：`pytest tests/test_backend_comparator.py -q`

5. **P0.5 junction 退化不变量**
   - 新建 `tests/test_static_junction_degenerate.py`
   - 只测试 `match_static_shell` 在 identical-metric 时的数学不变量
   - 验证：`pytest tests/test_static_junction_degenerate.py -q`

6. **P0.6 CI + gitignore**
   - 新建 `.github/workflows/test.yml`
   - 在 `.gitignore` 中排除 `outputs/`（或至少排除 `*.png`）
   - 验证：CI 运行 `pytest -q`

### P1（简化 + 性能）

7. **P1.1 提取共享 segment 积分 helper**
   - 从 `solvers.py` 和 `junction_tracing.py` 提取公共的 quad 积分 + brentq 反求逻辑
   - 放入 `_segments.py` 或 `_integrals.py`
   - 保留两处不同的调度逻辑不变
   - 验证：全量测试通过

8. **P1.2 TabulatedMetric 快速路径优化**
   - 在 P0 验证护栏之上继续推进
   - benchmark 只需在优化前后各跑一次记录数字
   - 验证：`pytest tests/test_numerical_metric_validation.py tests/test_numerical_metrics.py -q`

9. **P1.3 工程 guardrails**
   - 可选加入 ruff lint、coverage 报告（不设硬阈值）
   - benchmark 输出格式标准化（如果还需要的话）
   - 不做 batch tracing（等性能数据再说）

### P2（治理 + 大决定）

- 按计划执行：artifact 分类、Hamiltonian 决策、文档、typing policy
- batch tracing 在有性能数据后再决定

---

## 6. 未解决问题和需要人工确认的点

1. **`outputs/` 的 429 个 git-tracked 文件是否还需要？** 如果下游或论文依赖这些具体文件，不能直接 `.gitignore`。需要确认。

2. **`test_package.py` 是扩展还是替换？** Plan 说新建 `test_public_api_contract.py`，但 `test_package.py` 已经存在。扩展更省事，但文件名不如 `test_public_api_contract.py` 清晰。这是风格选择。

3. **L4 收敛测试的容限从哪来？** Plan 没有给出具体数值。建议跑一次 manufactured metric 在 n=1000/2000/4000 上的结果，用实际误差量级来定容限。

4. **P0.5 junction 退化测试是否应该对比完整成像？** 我认为不应该（代码路径差异太大），但如果作者认为 junction solver 应该精确复现单度规结果，那就需要对比。这是科学判断。

5. **P1.3 batch tracing 是否有实际的性能瓶颈数据？** Plan 把它放在 P1，但没有证据表明成像管线中的逐 ray 调用开销是实际瓶颈。如果 `sample_radial_profile` 已经足够快，batch tracing 可能是不必要的复杂度。

6. **`docs/adr/` 和 `docs/review/raytracing-spherical-review-roadmap.md` 是否应该提交？** 这些文件似乎由外部审查者生成。如果它们是临时文件，应该 `.gitignore`。如果是长期治理文档，应该保留但归档在 `docs/` 下，而不是仓库根目录。

7. **Plan 引用了 `docs/review/raytracing-spherical-review-roadmap.md` 但没有说明它与本 plan 的关系。** 这个 34KB 的文件包含了大量详细建议。如果本 plan 已经从中提炼了执行项，那个文件是否应该标记为"已采纳"或"已归档"？
