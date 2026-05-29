# 数值度规接入计划 - 实施 prompts

按 [数值度规接入计划.md](数值度规接入计划.md) 的 v3 版本拆成两个独立 PR：
- **Prompt #1**：步骤 3 - solver 边界修复（独立于数值度规，纯解析度规 RNdS 即可回归）
- **Prompt #2**：步骤 1/2/4/5 - 数值度规接入

## 推荐顺序

先 Prompt #1 → 合入 → 再 Prompt #2。理由：Prompt #2 的端到端 / thin-disk smoke 测试依赖 Prompt #1 修复的 outbound 行为。

---

## Prompt #1：solver 边界修复

```
请帮我实施 `docs/plans/数值度规接入计划.md` 的【步骤 3】，仅做 solver 边界修复，**不要**碰数值度规相关代码。

## 必读
1. `docs/plans/数值度规接入计划.md` - 整体设计与决策（你只做步骤 3，但要看决策 5/6/8/10/11 和"已知 solver 缺陷"两段了解上下文与容差）
2. `_prototype_check/prototype.py` - 我已经写好的原型，里面 `_inward_stop_u` / `_outward_stop_u` / `_make_outbound_segment` / `_patched_trace_b` 几个函数就是步骤 3 的参考实现，可以直接参照逻辑（但要按 plan 重写成生产质量代码并整合进 solver）
3. `src/spherical_raytracing/solvers.py` - 当前要修的文件
4. `src/spherical_raytracing/junction_tracing.py` - 边界逻辑也有引用，需要同步

## 方法：TDD
请使用 `tdd` skill 的 red-green-refactor 循环，**vertical slice，一次一个测试**。不要先写完所有测试再写实现。

## 实施顺序（每一步都是一个 RED→GREEN cycle）
1. 新增 `tests/test_solvers_bounded_domain.py`，写第一个测试：RNdS `b=0.5·b_crit` inbound 终止 `HORIZON` 且 `phi_end>0`。运行确认 RED（现在 phi_end=-0.326）。
2. 在 `solvers.py` 抽出 `_inward_stop_u()`（按 plan 决策 6 语义），让该测试 GREEN。同步删除 `_outer_horizon_u`，把其它引用替换。
3. 加第二个测试：`b=0.95·b_crit` inbound 终止 `HORIZON`。
4. 加第三个测试：`b=1.05·b_crit` outbound 终止于 `r_c`（`r_end != r_obs`，event 为 `HORIZON` 或 `OUTER_BOUNDARY`）。此处实现 `_outward_stop_u` + outbound 独立积分（不再用 mirror）。
5. 加第四个测试：`b > b_max = r_obs/sqrt(A(r_obs))` raise `ValueError`。
6. 跑现有 `pytest -q`，确保没有 regression（特别是 Schwarzschild / RN 这些 `r_max=inf` 的解析度规）。
7. 同步 `OdeTrajectorySolver` 和 `HamiltonianTrajectorySolver` 中的 `_outer_horizon_u` / `max(horizons())` 引用。`junction_tracing.py` 中相关位置同步。
8. 跑 `pytest -q`，全绿。

## 不可跑偏的硬约束
- 容差判定基于**原始 `valid_radial_domain()` 端点** vs `horizons()`，不要用 buffered `u` 反推 `r` 来比较（plan 决策 6 已说明，`rel_tol=1e-6, abs_tol=1e-8`）。
- outbound 段在 `r_max < inf` 时**必须独立积分**（用 `_make_segment(direction="outward")`），不要尝试"截断 mirror"——这是 prototype 已经证伪的死胡同。
- `r_max == math.inf` 路径保留现有 `_mirror_escape_segment` 行为不变（Schwarzschild/RN 不能 regress）。
- 不要碰 `numerical_metrics.py` 或 `TabulatedMetric`（这是步骤 1，独立 PR）。

## 完成定义
- [ ] 4 个新测试全绿
- [ ] `pytest -q` 全绿（包括所有现有测试，特别是 `test_static_junction_backend_agreement.py` / `test_junction_atlas_scripts.py` 这些重度依赖 solver 行为的）
- [ ] `_outer_horizon_u` 已从所有 `.py` 文件中消失（`rg _outer_horizon_u` 无结果）
- [ ] 改动只动 `solvers.py` / `junction_tracing.py` / `tests/test_solvers_bounded_domain.py`

完成后告诉我新增了多少行、改了多少行，以及四个新测试在修复前各自的 RED 表现是什么。
```

---

## Prompt #2：数值度规接入（步骤 3 合入后再做）

```
请帮我实施 `docs/plans/数值度规接入计划.md` 的【步骤 1/2/4/5】（数值度规模块、加载器、公共 API、测试）。步骤 3（solver 边界修复）已经在上一个 commit 完成（详见下面 "当前 branch 状态"）。

## 当前 branch 状态（必读，不要误判）

- 当前 branch：`codex/numerical-spherical-bh-raytracing`
- HEAD commit：`2d17239 fix(solvers): support bounded static domains in inward/outbound/b_max`
- 步骤 3 已经在**这个 branch 的本地 commit** 中完成（**未 push、未提 PR、未 merge 到 main**）。`_inward_stop_u` / `_outward_stop_u` / `_classify_radial_event` / `b_max` 校验等 API 在 `src/spherical_raytracing/solvers.py` 已就位，**直接复用**，不要重做。
- Branch 策略：在当前 branch **直接叠加 commit**。**不要** push / 新 branch / 提 PR / `git checkout main`。提交粒度由用户决定，你完成后等用户指示。
- 数据 / 原型路径（绝对路径，本机环境）：
  - `/home/yuanpang/Work/Li-DM-BH/outputs/sanity/data/*.npz`（7 个文件；路径不存在时 `pytest.skip`）
  - `_prototype_check/prototype.py`（untracked 工作底稿，**直接读 file system**，不做 git 状态假设；**不要**修改、不要 commit）
- 新增文件目标位置：
  - `src/spherical_raytracing/numerical_metrics.py`：`TabulatedMetric` + `LiDMBHMetadata` + `CanonicalPayload` + `StaticDomainSelector`
  - `src/spherical_raytracing/numerical_loader.py`：`load_metric_npz` + `ValidationOptions` + `ValidationGate`
  - `tests/test_numerical_metrics.py`：12 个 tracer bullet 测试
- 顶层导出 7 个 symbol：`TabulatedMetric`, `load_metric_npz`, `CanonicalPayload`, `LiDMBHMetadata`, `ValidationOptions`, `StaticDomainSelector`, `ValidationGate`（"完成定义" 里的 5 个 import 是最小集合，另外两个也要在 `__init__.py` 暴露）。
- Bullet 12 选 `burkert_Lneq` 的理由：prototype 已验证过此数据上 thin-disk smoke 能触发 `intersection_count >= 2`；其它数据未必都行，先以此为基线。
- 测试时长基线：完整 pytest 当前 144s / 163 passed。调试阶段请用 `pytest tests/test_numerical_metrics.py -x`；最后一次回归才跑完整 `pytest -q`。
- Workspace path 当前在 `\home\yuanpang\Work\raytracing-spherical`（Linux 路径用 `/home/yuanpang/Work/raytracing-spherical`）。

## 必读
1. `docs/plans/数值度规接入计划.md` - 整体设计；你这次涵盖步骤 1/2/4/5，所有 11 条决策都要读
2. `_prototype_check/prototype.py` - `TabulatedMetric` 类已经在原型里写好，包括 PCHIP + 端点 0 锚点、域外硬失败、`|C|<c_tol` raise、`G(u, b)` 一般两函数公式。你可以参照其逻辑，但要按 plan 步骤 1 拆出 `LiDMBHMetadata` / `CanonicalPayload` / `ValidationOptions` / `ValidationGate` / `StaticDomainSelector` 五个独立组件，而不是揉成一个类。
3. `/home/yuanpang/Work/Li-DM-BH/outputs/sanity/data/*.npz` - 真实输入数据，7 个文件
4. `src/spherical_raytracing/metrics.py` - 协议 `StaticSphericalMetric` 在这里
5. `src/spherical_raytracing/__init__.py` - 公共导出
6. `outputs/lqg_fig3/lqg_fig3_profile.csv` - 目标产物形态参考

## 方法：TDD
使用 `tdd` skill，vertical slice。**每一个 dataclass / 校验项都是独立的 tracer bullet**。

## 实施顺序
1. **Tracer bullet 1**：`LiDMBHMetadata` dataclass + 从 JSON 解析的测试。RED 用一个 `params` 字典 fixture，期望解析出正确字段；GREEN 写最小 dataclass。
2. **Tracer bullet 2**：`CanonicalPayload` dataclass（含 `raw_B: np.ndarray | None`）。
3. **Tracer bullet 3**：`load_metric_npz` 在 Li-DM-BH 格式上能跑通（产出 `CanonicalPayload`）；标准 `r/A/B` 格式同样跑通且 `raw_B is None`。
4. **Tracer bullet 4**：`ValidationGate` 硬失败项（非单调 r / NaN / 缺 key / conventions 冲突）。每一项一个测试。
5. **Tracer bullet 5**：`ValidationGate` 软告警项（`B*C≈1` / metadata roots 残差）。用 `pytest.warns(UserWarning)` 验证。
6. **Tracer bullet 6**：`StaticDomainSelector` 默认 `domain_index=-1` 和 `radial_bounds` 显式覆盖。
7. **Tracer bullet 7**：`TabulatedMetric.A` / `B` / `G` 基本行为 + 域外 raise + `|C|<c_tol` raise。
8. **Tracer bullet 8**：`TabulatedMetric` PCHIP 端点 0 锚点 (`A(r_h) == 0`)。
9. **Tracer bullet 9**：Schwarzschild round-trip：解析 Schwarzschild 构造 payload → 跑 trace → 与 `SchwarzschildMetric` 解析 trace 对比，相对误差 `< 1e-8`。
10. **Tracer bullet 10**：`TabulatedMetric + InfinityObserver` raise；`TabulatedMetric + ThroughTracePolicy` raise。
11. **Tracer bullet 11**：7 个真实 `.npz` 端到端 smoke（`pytest.fixture(scope="session")` 共享加载结果，路径不存在 `pytest.skip`）。每个文件：能加载 → `TabulatedMetric` → `FiniteStaticObserver(r_obs=中点)` → `QuadTransferSolver.trace_b(b_crit·1.05)` → `phi_end > 0`、`r_end != r_obs`。
12. **Tracer bullet 12**：`burkert_Lneq` 上 thin-disk profile smoke：`b ∈ [b_crit·1.01, b_crit·1.1]` 区间至少一个 b 出现 `intersection_count >= 2`。
13. **公共 API 与文档**：更新 `__init__.py` 导出、`README.md` / `README.zh-CN.md` 增加数值度规章节，说明 plan 决策 4/6/10/11 的关键约束。

## 不可跑偏的硬约束
- 不要把 `LiDMBHMetadata` / `CanonicalPayload` / `ValidationOptions` 揉成一个类——plan 决策 3/8 明确要求分离。
- PCHIP 拟合数据必须是**静态域内子数组 + 端点 0 锚点**，不能用全 `r` 数组（`nfw_L0` 跨 5 个量级时 CubicSpline 都会过冲，PCHIP 在全数组上虽不过冲但端点零点对齐弱）。
- `B = 1/C`，**不要**默认插 `B`；`|C| < c_tol` 时 `B / dB_dr` raise。
- 域外查询全部硬 raise，不要返回 NaN 让 solver 静默传播。
- `TabulatedMetric + InfinityObserver` 必须 raise，不要尝试"远场外推 G(u=0)"——plan 决策 11 已经把这放在非目标里。
- Hamiltonian solver 在数值度规下**不保证稳定**，但接口（`B(r)`, `dB_dr(r)`）要能正常 raise 友好错误，不要静默 return inf。

## 完成定义
- [ ] 所有 tracer bullet 测试全绿
- [ ] `pytest -q` 全绿（步骤 3 已合入的 solver bounded-domain 测试也仍然绿）
- [ ] `from spherical_raytracing import TabulatedMetric, load_metric_npz, CanonicalPayload, LiDMBHMetadata, ValidationOptions` 能 import
- [ ] README 章节有 quick-start：`load_metric_npz('hernquist_L0.npz') → FiniteStaticObserver → QuadTransferSolver` 完整代码示例
- [ ] `_prototype_check/prototype.py` 可以删除或移到 `scripts/` 作为参考；如果保留，加 `.gitignore` 条目

完成后告诉我：（1）新增模块的行数、（2）测试覆盖了 plan 步骤 5 列表中的多少项（按编号）、（3）7 个真实 `.npz` 文件 smoke 测试的运行时间。
```
