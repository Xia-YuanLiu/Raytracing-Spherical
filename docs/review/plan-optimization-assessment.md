# Plan Optimization Assessment

评估当前 plan（`docs/plans/upstream-computation-core-stabilization-plan.md`）相对于 review 建议（`docs/review/upstream-computation-core-stabilization-plan-review.md`）的优化结果。

Date: 2026-05-29

---

## 1. 逐条比对：Review 的 8 个 Findings，Plan 采纳了几个？

### F1（中等）：P0.1 应扩展现有 `test_package.py` 而非新建文件

**Review 建议**：在 `test_package.py` 中补齐剩余 25 个符号，或将其重命名。

**Plan 当前版本**（P0.1，line 130-215）：
```
Expected files:
- preferred: extend or rename tests/test_package.py
- acceptable: add tests/test_public_api_contract.py only if it does not
  duplicate a second hand-maintained API list
```
```
Acceptance criteria:
- the existing 25-export smoke coverage is extended rather than duplicated
```

**判定：✅ 完全采纳。** Plan 明确优先扩展 `test_package.py`，把"新建文件"降为"可接受但需避免重复"。同时承认了已有 25/50 覆盖这一事实（line 64-65 的 Current Findings 也更新了）。

### F2（中等）：P0.6 benchmark 标准化优先级过高

**Review 建议**：降级到 P1 或删除，现有 benchmark 已够用。

**Plan 当前版本**（P0.6，line 393-429）：
- 标题从 "Standardize Benchmark Baseline Output" 改为 **"Keep Benchmark Baseline Runnable"**
- Required fields 从 10 项缩减到 6 项（metric name, ray count, category counts, median, p95, fallback）
- 明确说 "Nice-to-have fields ... may be added in P1"
- 增加了 "do not spend P0 time on benchmark format polish beyond what is needed for a smoke run"

**判定：✅ 大幅采纳。** 标题改为"Keep Runnable"反映了对 review 的回应——不再追求标准化，而是确保可运行。字段要求也做了分层（P0 必须 vs P1 锦上添花）。这是合理的折中：P0 仍然需要一个可运行的 benchmark（因为后面 P1.2 优化时要用），但不再要求格式打磨。

### F3（低）：P0.2 静态 `.npz` fixture 可能冗余

**Review 建议**：不需要新建 `tests/fixtures/` 目录，用动态构造替代。

**Plan 当前版本**（P0.2，line 217-254）：
```
Expected files:
- preferred: tests/test_numerical_metrics.py
- preferred: tests/test_numerical_metric_validation.py
- optional only if a path-based script/benchmark needs a real file:
  tests/fixtures/metrics/*.npz
- optional if static fixtures are added: tests/fixtures/metrics/README.md

Fixture policy:
- dynamic synthetic payloads are preferred over checked-in binary fixtures
```

**判定：✅ 完全采纳。** 策略从"新建 fixture 目录"变为"优先动态构造"。静态文件降为 optional，只在路径工作流确实需要时才用。这与 review 的建议完全一致。

### F4（低）：L4 收敛测试范围模糊

**Review 建议**：用固定参数断言固定容限，不要用"非劣化"语言；明确网格密度。

**Plan 当前版本**（P0.3，line 256-311）：
```
- fixed grid densities such as 800, 1600, 3200
- fixed non-critical rays such as 0.70*b_crit, 1.10*b_crit, ...
- use explicit tolerances from an initial measured baseline, not vague
  "non-worsening" language
```
```
- preferred: extend tests/test_numerical_metric_validation.py
- acceptable if it grows too large: tests/test_numerical_metric_convergence.py
```

**判定：✅ 完全采纳。** 消除了"non-worsening envelope"的模糊语言，给出了具体的网格密度和 ray 参数示例。文件归属也改为优先扩展现有验证文件。验收标准明确要求 "fixed numerical tolerances on a stable manufactured case"。

### F5（低）：P0.5 junction 退化测试范围过大

**Review 建议**：只测数学不变量，不做完整成像对比。

**Plan 当前版本**（P0.5，line 360-391）：
```
- a minimal direct-intersection smoke may compare single-metric and junction
  intensity only when the selected ray has stable matching intersections
```
```
Acceptance criteria:
- direct intersection strength agrees with single-metric finite-observer result
  only for a stable matched case
- exact event sequence equality is not required
- no broad imaging comparison is required in P0
```

**判定：✅ 完全采纳。** 成像对比从"必须做"变为"可选，只在稳定匹配时"。明确排除了 "broad imaging comparison" 和 "exact event sequence equality"。核心不变量测试（energy ratio=1, b 不变, 红移无额外 shell factor）保留，这是正确的。

### F6（信息）：segment 重复程度比 plan 暗示的小

**Review 观察**：实际重复约 60-70 行核心逻辑。

**Plan 当前版本**（P1.1，line 480-523）：未修改 scope，但 Current Findings（line 72-73）承认了 "Segment construction duplication ... is real"。

**判定：⚠️ 部分采纳。** Plan 没有明确缩小 P1.1 的 scope 或降低预期。不过 P1.1 的 Non-goals 列表（line 505-508）本身就限制了范围（"no performance fast path in this extraction"），所以实际执行时不太可能过度重构。这不是问题，但可以在 P1 执行时更明确地声明 "提取目标是一个 ~60-70 行的纯函数 helper"。

### F7（信息）：429 个 git-tracked 生成文件

**Review 建议**：P0.7 CI 阶段就处理 `.gitignore`。

**Plan 当前版本**：
- 新增了 P0.0（line 86-123）专门处理 commit boundary 和 hygiene 问题
- `.gitignore` 已更新（现在排除了 `outputs/` 和 `docs/*` 但有白名单）
- 但 429 个已跟踪文件仍然存在（gitignore 不影响已跟踪文件）

**判定：✅ 采纳，且有额外改进。** 计划新增了 P0.0（这是 review 没有建议的），主动处理了 gitignore/tracking 问题。`.gitignore` 已更新（`outputs/` 在忽略列表中，`docs/` 也有白名单控制）。但已跟踪的 `outputs/` 文件需要 `git rm --cached` 才能真正从 clone 中消除——plan 在 P0.0 中说 "do not clean up tracked outputs/ in P0"，推迟到 P2.1。这是合理的，因为贸然删除可能破坏下游依赖。

### F8（信息）：没有 `@pytest.mark.slow`，没有 `conftest.py`

**Review 建议**：plan 应明确提到需要注册 marker 和创建 conftest。

**Plan 当前版本**（P0.3，line 284-285）：
```
- marker registration in pyproject.toml is enough unless shared fixtures truly
  need tests/conftest.py
```

**判定：✅ 采纳。** 明确了 marker 注册位置，并正确指出 `conftest.py` 不是必须的（除非需要共享 fixture）。这比 review 的建议更精准。

---

## 2. 优化后 plan 相比原始版本新增了什么？

### 新增 P0.0：Confirm Commit Boundary And Repository Hygiene

**Review 没有建议这个**，但 plan 主动加了。这是回应了 `.gitignore` 中 `docs/*` 被忽略但 `docs/plans/`、`docs/adr/`、`docs/review/` 又被白名单加回来这个现实情况。P0.0 确保 implementation agent 不会在 `docs/` 被忽略的情况下遗漏持久化文档。

**评价：✅ 有价值的新增。** 这是一个实际的工程风险点——如果 agent 不知道 `docs/*` 被 gitignore 但 `docs/plans/*` 被白名单加回来，可能会创建文档但忘记 `git add -f`。

### 新增 Current Findings 更新

Original plan 的 Current Findings 只列了 7 条。当前版本更新了 3 条（line 64-80）：
- 承认已有 25/50 API 测试
- 承认 dynamic synthetic payloads 已存在
- 承认 benchmark 已可运行
- 新增 repository hygiene 风险

**评价：✅ 有价值。** 这些更正让后续 implementation agent 不至于重复已有工作。

### P1.3 合并了 benchmark 和工程 guardrails

Original plan 有 P1.3（batch tracing）和 P1.4（工程 guardrails）两个独立任务。当前版本合并为一个 P1.3（"Light Engineering Guardrails And Benchmark Polish"），batch tracing 被推到 P2.5。

**评价：✅ 与 review 建议一致。**

### 新增 P2.5：Decide Whether Internal Batch Tracing Is Worth It

Original plan 的 batch tracing 在 P1.3。当前版本将其推到 P2 并加了明确的决策门槛。

**评价：✅ 完全采纳了 review 的核心建议。**

### 新增 ADR 0002 和 ADR 0003

两个新 ADR 正式化了 plan 中的决策：
- ADR 0002：`TabulatedMetric + Hamiltonian` 不支持
- ADR 0003：推迟 batch tracing

**评价：✅ 有价值。** 把 plan 中的隐含决策变成显式 ADR，后续 reference 更方便。

### Handoff prompt 更新

Current plan 的 handoff prompt（line 737-764）新增了三条约束：
- "Do not add internal batch tracing in P0/P1 unless benchmark evidence first proves it is the right bottleneck"
- "Before implementation, check ignored/untracked files so durable docs and imported runtime modules are not accidentally omitted"

**评价：✅ 有价值。** 把 P0.0 和 batch tracing 决策直接嵌入 handoff prompt，减少 implementation agent 犯错的可能。

---

## 3. 有没有比之前变差的地方？

### 3.1 P0.6 的定位略显模糊

Original plan 说 "Standardize Benchmark Baseline Output"，scope 清晰（标准化输出格式）。当前版本说 "Keep Benchmark Baseline Runnable"，scope 变模糊了——"runnable" 是什么意思？脚本现在就是 runnable 的。

**实际影响：低。** Acceptance criteria 给出了足够清晰的验收标准（"script runs on manufactured synthetic metrics", "script does not require private Li-DM-BH data"）。标题虽然模糊，但内容不模糊。

### 3.2 P0.0 增加了 P0 的步骤数

Original plan P0 有 7 步。当前版本有 8 步（新增 P0.0）。

**实际影响：低。** P0.0 本质上是一个 pre-flight check（跑 `git status`，确认文件状态），工作量极小（5-10 分钟）。但它防止了 implementation agent 在 `docs/*` 被 gitignore 的情况下创建无法被跟踪的文档——这是一个真实的陷阱。

### 3.3 No-Go List 新增了一条

Line 716: "do not add internal batch tracing until profiling justifies it"

**实际影响：正面。** 这是保护性约束，不是限制。

### 3.4 没有变差的地方

逐项检查后，**没有发现任何比 review 之前更差的改动**。所有变更要么是采纳 review 建议，要么是新增有价值的保护措施。

---

## 4. 总结

### 优化前 vs 优化后对比

| 维度 | 优化前（original plan） | 优化后（current plan） | 变化 |
|---|---|---|---|
| **P0 步骤数** | 7 | 8（+P0.0） | +1 步，但 P0.0 工作量极小 |
| **P1 步骤数** | 4（含 batch tracing） | 3（batch tracing 推到 P2） | 更精简 |
| **P2 步骤数** | 4 | 5（+batch tracing 决策） | 更合理 |
| **API 测试策略** | 新建文件 | 扩展现有文件（优先） | 避免重复 |
| **Fixture 策略** | 新建静态 `.npz` | 优先动态构造 | 减少维护成本 |
| **L4 验收标准** | "non-worsening envelope" | 固定参数固定容限 | 可执行 |
| **Junction 退化范围** | 包含完整成像对比 | 核心不变量 + 可选 smoke | 更务实 |
| **Benchmark P0 定位** | 标准化输出格式 | 确保可运行 | 降低 P0 工作量 |
| **Batch tracing** | P1（直接执行） | P2.5（需要决策门槛） | 避免过早复杂化 |
| **Gitignore/hygiene** | 未提及 | P0.0 专门处理 | 填补实际风险 |
| **ADR 数量** | 1（0001） | 3（+0002, +0003） | 决策显式化 |
| **Handoff prompt** | 6 条约束 | 8 条约束 | 更完整 |

### Review 的 8 个 Findings 采纳情况

| Finding | 严重度 | 采纳情况 |
|---|---|---|
| F1: API 测试重叠 | 中等 | ✅ 完全采纳 |
| F2: benchmark 优先级过高 | 中等 | ✅ 大幅采纳 |
| F3: 静态 fixture 冗余 | 低 | ✅ 完全采纳 |
| F4: L4 范围模糊 | 低 | ✅ 完全采纳 |
| F5: junction 退化范围过大 | 低 | ✅ 完全采纳 |
| F6: segment 重复程度 | 信息 | ⚠️ 部分采纳（P1.1 scope 未明确缩小） |
| F7: 429 tracked 文件 | 信息 | ✅ 采纳（+P0.0，gitignore 已更新） |
| F8: slow marker/conftest | 信息 | ✅ 采纳 |

**采纳率：7/8 完全采纳，1/8 部分采纳。**

### 遗留问题

1. **F6 的 P1.1 scope**：segment 重复的实际规模（~60-70 行）没有在 plan 中明确。P1 执行时应该先测量再决定提取范围。
2. **`outputs/` 的 429 个已跟踪文件**：`.gitignore` 已排除 `outputs/`，但已跟踪的文件不会自动消失。需要 `git rm --cached` 或等 P2.1 处理。
3. **L4 的具体容限**：plan 说 "from an initial measured baseline"，但没给出数值。执行时需要先跑一次基线测量。
4. **`docs/review/raytracing-spherical-review-roadmap.md`（原根目录 roadmap）**：已归档到 review 文档目录；后续仍可标记采纳状态。

### 最终结论

**优化后的 plan 在所有关键维度上都比原始版本更好：**
- 更精确（承认已有测试、动态 fixture 等事实）
- 更精简（batch tracing 推迟、benchmark 要求降低）
- 更务实（固定参数替代模糊语言、缩小 junction 测试范围）
- 更安全（P0.0 处理 gitignore 陷阱、新增 ADR、handoff prompt 更完整）

**没有发现任何比优化前更差的改动。**
