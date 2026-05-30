# Raytracing Spherical 仓库长期改进方向评估与优先级路线图

**评估视角：** 科学计算库 / 数值方法 / Python 包架构  
**目标：** 在不追求大规模重构、不立即写代码的前提下，判断这个仓库未来最值得改进的方向。  
**主要依据：** `REPO_OVERVIEW.md`，更新时间 2026-05-29。  

---

## 0. 执行摘要

基于目前的仓库全景信息，我的总体判断是：这个项目下一阶段最值得投资的不是“扩展更多功能”，而是把现有的 ray tracing、薄盘成像、数值度规和静态 junction 路径变成一个**可重复、可诊断、可对比、可被下游长期依赖的科学计算库**。

这个仓库已经被其他项目依赖，因此公共 API 的稳定性本身就是科学可信度的一部分。未来的主要改进应优先发生在：

1. **验证体系**：补齐数值度规和后端一致性的验证缺口。
2. **内部数值逻辑复用**：减少单度规 solver 与 junction solver 中重复的 segment / 事件 / 插值逻辑。
3. **性能基准与低风险快路径**：先建立 benchmark 和 scalar oracle，再推进批量 ray tracing、数值度规快路径和 atlas 性能优化。
4. **API 合同测试**：保护下游已有调用方式，避免“看似清理、实际破坏”的接口调整。
5. **可复现 fixture 与 CI**：让真实数值度规路径、脚本路径和 artifact 生成路径不依赖本机私有环境。

简而言之：

> 下一阶段最应该做的是：**用更强的验证矩阵和更少的重复数值实现，把现有功能变成稳定可信的上游科学计算库；性能优化应继续推进，但必须以 scalar oracle、收敛研究和 API contract 为护栏。**

---

## 1. 优先级总览

### 1.1 现在最应该做的

1. **补齐数值正确性验证**
   - 完成 L4 网格收敛研究。
   - 完成 L5 单度规 Quad / ODE / Hamiltonian 后端 sanity comparator。
   - 增加红移权重、junction matching、薄盘交点和 termination event 的物理不变量测试。

2. **冻结并测试公共 API**
   - 增加 `test_public_api_contract.py`。
   - 显式断言 `spherical_raytracing.__all__` 的公共名称集合。
   - 保护关键构造函数、方法名、返回 dataclass 字段和语义。

3. **为数值度规增加可复现 `.npz` fixture**
   - 不再只依赖作者本机路径的真实数据 smoke test。
   - 至少覆盖标准 `r/A/B` 与 Li-DM-BH-like `r/f/g/B/params` 两种输入格式。

4. **建立 benchmark 基线**
   - 在优化前先记录当前 profile、ray tracing、TabulatedMetric、atlas 生成等路径的耗时。
   - 后续优化必须报告速度提升与误差变化。

5. **提取私有共享 segment / ray kernel**
   - 保持公共 API 不变。
   - 减少 `QuadTransferSolver` 与 `StaticJunctionTransferSolver` 中重复的数值 segment 构造逻辑。

### 1.2 有价值但可以稍后做的

1. 明确 Hamiltonian 后端的长期定位：诊断工具还是可支持生产路径。
2. 把 `outputs/`、atlas artifact、脚本生成结果分成示例、回归基准、发布产物几类管理。
3. 分阶段加入 ruff、coverage、mypy、`py.typed`、CHANGELOG 和 deprecation policy。
4. 增加更系统的文档：坐标约定、红移权重、临界曲线、终止事件、near-critical 排除带、数值度规适用边界。
5. 对 `sample_radial_profile()` 和 atlas 管线引入内部批量 tracing，但暂不急着暴露新公共 API。

### 1.3 看起来合理但不建议优先做的

1. 不要直接重命名公共 API。
2. 不要把所有 dataclass 改造成大型继承体系。
3. 不要为了性能立即替换 NumPy/SciPy，或引入 JAX、GPU、C++ 扩展等复杂依赖。
4. 不要马上把 face-on 轴对称成像扩展为完整通用相机，除非这是明确的新需求。
5. 不要删除 `outputs/`、`refs/`、`LQGBHShadow/`，除非先明确它们是否仍有历史、验证或复现用途。
6. 不要为了假想需求增加复杂配置层、插件系统或高度抽象的 solver framework。

---

## 2. 科学性与数值正确性

科学性和数值正确性应当是这个仓库的最高优先级。这个库用于 ray tracing、薄盘交点、红移权重、静态壳匹配、成像 profile 和 atlas 生成。任何局部数值误差都有可能被下游成像和参数扫描放大。

### 2.1 当前验证体系的主要缺口

根据仓库概览，目前数值度规验证已经有：

- **L1**：解析度规采样成 manufactured `TabulatedMetric` 后，比较 `A`、`C`、`dA/dr`、`dB/dr`、`G`、horizon、photon sphere、critical curve。
- **L2**：小规模 b-sweep，对比解析路径和数值表格路径的 ray termination、segment、`phi_end`、`u_end`、`u(phi)`。
- **L3**：薄盘交点和 `observed_intensity` smoke。

但仍缺少：

- **L4**：慢速网格收敛研究。
- **L5**：单度规 Quad / ODE / Hamiltonian 后端 sanity comparator。

这两个缺口非常关键。L1-L3 能说明“基本路径能跑通”，但还不足以说明“数值误差随着分辨率按预期收敛”，也不足以说明“不同后端在非奇异、非临界区域给出一致结果”。

### 2.2 最应该补的验证：L4 网格收敛

#### 为什么重要

数值度规路径的核心风险不是明显崩溃，而是：

- 插值误差悄悄改变 photon sphere 或 critical curve。
- 表格分辨率不足导致 turning point 偏移。
- `u(phi)` 插值误差影响薄盘交点半径。
- near-critical ray 附近的误差被误认为真实物理结构。
- profile 峰位置和 ring edge 被表格误差污染。

如果没有网格收敛研究，就很难区分：

- 真实物理效应；
- 数值表格太稀疏；
- 插值策略问题；
- solver 快路径误差；
- near-critical 排除带内不可稳定追迹。

#### 建议怎么做

建议设计一个慢速但清晰的 convergence test 矩阵，不放入默认快速测试，而是作为 slow test 或 benchmark-like validation。

可以选择几类解析度规作为 manufactured data：

1. Schwarzschild。
2. Reissner-Nordstrom。
3. Reissner-Nordstrom-de Sitter。
4. LQG-like 度规。

对每个解析度规，生成不同径向分辨率的 `TabulatedMetric`，例如：

```text
N = 256, 512, 1024, 2048, 4096
```

比较对象包括：

- `A(r)`、`C(r)`、`dA/dr`、`dB/dr`。
- `G(u,b)`。
- horizon 位置。
- photon sphere 位置。
- critical curve。
- ray termination type。
- segment count。
- `phi_end`。
- `u_end`。
- `u(phi)`。
- 薄盘交点半径。
- `observed_intensity`。
- profile 峰位置和峰值。

需要特别标记或排除：

- near-critical ray。
- 非常靠近 horizon 的 ray。
- max_phi 截断 case。
- 表格边界附近 case。

#### 预期收益

- 可以为数值度规路径建立误差预算。
- 可以判断 PCHIP、Gauss-Legendre 快路径、turning point 搜索是否可信。
- 可以为未来优化提供安全网。
- 可以让下游知道表格分辨率对图像和 profile 的影响。

#### 风险和代价

- 测试会变慢。
- 误差阈值不好设，尤其在 near-critical 区域。
- 需要区分真正错误和可解释的数值敏感性。

#### 如何验证改进有效

每次修改数值度规或 solver 后，至少应能给出：

- 误差随 `N` 增大的下降趋势。
- 快路径与 scalar reference path 的差异。
- profile 峰位误差是否在阈值内。
- near-critical 排除带内是否被正确标记，而不是被错误地当作精确结果。

---

### 2.3 后端一致性验证：L5 sanity comparator

#### 为什么重要

仓库有多个后端：

- `QuadTransferSolver`：主生产路径。
- `OdeTrajectorySolver`：辅助路径。
- `HamiltonianTrajectorySolver`：诊断路径。

表面看，多个后端可以互相验证。但需要注意：`OdeTrajectorySolver` 并不是完全独立的 oracle，因为它入口会先调用 `QuadTransferSolver` 做参考分类，出射段也委托 Quad 的正则化 segment 构造。因此，L5 的目标不应写成“ODE 证明 Quad 正确”，而应写成：

> 在非临界、非边界、短程路径上，Quad / ODE / Hamiltonian 的几何结果、事件分类和守恒量诊断没有明显矛盾。

#### 建议怎么做

选择温和参数区域，避免一开始就覆盖极端 case：

- 不靠近 critical impact parameter。
- 不靠近 horizon。
- 不靠近表格边界。
- 不触发 max_phi。
- 使用有限观测者以兼容 Hamiltonian 后端。

比较内容包括：

- termination type。
- segment count。
- `phi_end`。
- `u_end`。
- 若有 turning point，比较 turning point 位置。
- 薄盘交点半径。
- Hamiltonian constraint drift。
- energy drift。
- angular momentum drift。
- impact parameter drift。

#### 预期收益

- 明确各后端的可信范围。
- 防止某个后端 silently diverge。
- 为 Hamiltonian 后端是否能升级为生产路径提供依据。

#### 风险和代价

- 不同后端的数值轨迹参数化可能不同，不能简单逐点比较。
- near turning point 区域对 ODE 很不友好。
- Hamiltonian 对数值表格度规目前没有稳定承诺，不宜一开始纳入严格矩阵。

#### 如何验证改进有效

- 建立一组固定 case，输出后端比较表。
- 对每个 case 给出可接受误差阈值。
- 对超出阈值的情况，必须分类为：真实错误、后端适用范围外、near-critical、边界敏感、或数值分辨率不足。

---

### 2.4 红移权重与 junction matching 的物理不变量测试

#### 为什么重要

红移权重和 junction matching 是最容易产生“图像看起来合理但物理上错误”的地方。

单度规路径中，观测强度使用 `g^4` 权重：

- 无穷远观测者：`g = sqrt(A(r_em))`。
- 有限静态观测者：`g = sqrt(A(r_em) / A(r_obs))`。

junction 路径中，壳穿越会改变能量，进而改变 impact parameter 和红移权重。静态壳匹配中：

- 角动量 `L` 连续。
- 能量按 `sqrt(A_to/A_from)` 重标定。
- 因此 `b = L/E` 按 `sqrt(A_from/A_to)` 改变。

这些公式跨越多个模块：`junctions.py`、`junction_tracing.py`、`transfer.py`、`sources.py`。如果某处符号、region label、能量比或红移权重处理不一致，最终图像仍可能平滑，但物理解释会错。

#### 建议测试

1. **壳内外度规相同的退化测试**
   - 若 inner metric 与 outer metric 相同，junction tracing 应退化为单度规 tracing。
   - shell crossing 不应改变最终强度。

2. **`A_from == A_to` 的匹配测试**
   - energy ratio 应为 1。
   - `b` 不应跳变。

3. **单次穿壳与往返穿壳测试**
   - energy ratio product 应符合解析乘积。
   - 往返后在对称 case 中应恢复预期能量比例。

4. **预写 `redshift_weight` 与 `observed_intensity()` 一致性测试**
   - 在可比 case 中，对比 junction diagnostics 中预写的权重与源函数重新计算的权重。

5. **region label 一致性测试**
   - `DiskWindow` 的 region 选择必须和 intersection diagnostics 中的 region 一致。
   - region 不匹配时不能静默给出错误强度。

#### 预期收益

- 防止跨模块物理约定漂移。
- 提高 junction 成像结果可信度。
- 为 atlas 中的红移 cusp、V 形结构、峰值差异提供物理保障。

#### 风险和代价

- 需要小心选择参数，避免壳靠近 horizon 或 photon sphere。
- junction case 容易出现多个事件，测试需要足够小而明确。

#### 如何验证改进有效

- 退化 case 与单度规 case 一致。
- energy ratio 和 `b` jump 有解析预期。
- 强度权重在不同路径计算下一致。
- region label 错误能被测试捕获。

---

## 3. 简洁性：减少重复实现，而不是增加抽象

这个仓库规模并不大。根据概览，`src/spherical_raytracing` 约 15 个 Python 文件、5437 行代码，测试 26 个文件。对这个规模来说，最危险的不是“模块太少”，而是“核心数值逻辑在多个地方重复实现”。

### 3.1 最值得做的内部重构：共享 segment 构造与 ray kernel

#### 当前问题

`QuadTransferSolver` 是单度规主路径，已经有针对 `TabulatedMetric` 的 Gauss-Legendre segment 快速路径。`StaticJunctionTransferSolver` 有自己的 segment 积分逻辑，虽然共享 `_turning.py` 的转折点扫描，但没有共享单度规 solver 的 TabulatedMetric 快路径。

这会带来几个问题：

- 一个路径修复了 segment bug，另一个路径可能没有同步。
- 单度规和 junction 对 turning point、outbound segment、bounded outer boundary 的处理可能逐渐漂移。
- 数值度规优化无法自然复用到 junction 路径。
- 测试需要覆盖两套相似但不同的实现。

#### 建议怎么做

不要改公共类名、方法名或返回结构。建议新增私有模块，例如：

```text
_segments.py
_ray_kernels.py
_quadrature_segments.py
```

其中只放小而稳定的共享内部原语：

- inbound segment 构造。
- outbound segment 构造。
- turning point 附近正则化。
- `phi(u)` / `u(phi)` 插值。
- bounded outer segment。
- infinity mirror segment。
- TabulatedMetric 快速积分与 scalar fallback。
- segment diagnostics。

`QuadTransferSolver` 和 `StaticJunctionTransferSolver` 仍保留原有公共 API，只在内部调用共享私有函数。

#### 预期收益

- 减少重复数值逻辑。
- 降低 bug 面积。
- 让数值度规快路径自然复用于 junction。
- 更容易写局部单元测试。
- 更容易验证 `RaySegment.u_at()` 精度。

#### 风险和代价

这是核心路径重构，风险不低。错误可能改变大量 regression 结果。

#### 建议迁移方式

1. 先写 characterization tests，锁定当前行为。
2. 对若干固定 b/alpha/shell case 保存：
   - termination type；
   - segment count；
   - event sequence；
   - turning point；
   - `phi_end`；
   - `u_at()`；
   - intersection radius；
   - observed intensity；
   - profile 峰位。
3. 再提取私有共享函数。
4. 最后要求重构前后数值结果在误差阈值内一致。

---

### 3.2 不建议引入大型 solver 框架

一个看起来合理但实际不建议优先做的方向，是把所有 solver 统一成复杂的继承树或 plugin framework。例如：

```text
BaseSolver
  QuadSolver
  OdeSolver
  HamiltonianSolver
  JunctionQuadSolver
  JunctionHamiltonianSolver
  NumericalMetricSolver
  ...
```

这类设计表面上整洁，但实际可能带来：

- 抽象层过多。
- 调试路径变长。
- 物理语义被隐藏在基类回调里。
- 下游难以理解 solver 的真实适用范围。
- 为假想需求付出复杂度成本。

更好的方向是：

- 公共 API 保持扁平。
- 私有共享函数保持小而纯。
- solver 仍按物理场景区分：单度规、junction、Hamiltonian diagnostic。
- 复杂性放在测试矩阵和 diagnostics 中，而不是放在类型层级里。

---

## 4. 性能：先有基准，再做低风险优化

性能优化应当继续推进，但不应该先从更换技术栈开始。当前最有价值的性能方向是：

1. 建立 benchmark 基线。
2. 推进批量 b-sweep / profile / atlas tracing。
3. 复用 TabulatedMetric 快路径。
4. 缓存纯数值对象。
5. 保留 scalar reference path 作为 correctness oracle。

### 4.1 先建立 benchmark 基线

#### 为什么重要

没有 benchmark，很难判断：

- 慢在哪里。
- 哪个优化真的有效。
- 快路径是否只在特定 case 上有效。
- 优化是否牺牲了 `u_at()` 或 intersection 精度。

#### 建议 benchmark 内容

至少包含：

1. 单度规 Schwarzschild b-sweep。
2. `TabulatedMetric` b-sweep。
3. 薄盘 intersection + intensity。
4. `sample_radial_profile()`。
5. `render_axisymmetric_image()`。
6. junction transfer tracing。
7. atlas quick preset。
8. TabulatedMetric fast path vs scalar fallback。

输出指标包括：

- wall time。
- rays per second。
- fallback ratio。
- segment count distribution。
- unresolved / near-critical count。
- profile 峰位误差。
- scalar vs fast path 最大差异。

#### 如何验证改进有效

性能 PR 不应只报告“快了”，还应报告：

- 快了多少。
- 哪条路径快。
- 哪些 case 没变快。
- 数值误差是否变化。
- fallback 是否增加或减少。
- 是否改变 termination 分类。

---

### 4.2 最值得做的性能方向：内部批量 tracing

#### 当前问题

`sample_radial_profile()` 是逐 ray 调用。atlas 和成像路径本质上都是大量 b 或 screen radius 的 sweep。如果每个 ray 都完整走一遍 Python 层调用、重复构造中间对象，性能会受限。

#### 建议怎么做

第一阶段不要暴露新的公共 API，而是增加内部批量核心，例如：

```python
_trace_many_b(...)
_sample_profile_batch_core(...)
```

公共 `sample_radial_profile()` 仍保持原签名。

内部可以批量处理：

- impact parameter 数组。
- TabulatedMetric 的 `_G_array()`。
- turning point scan 的采样网格。
- Gauss-Legendre nodes/weights。
- fixed observer 下的 `A(r_obs)`。
- profile adaptive refinement 中新增点的 tracing。

#### 预期收益

- 加速 profile。
- 加速二维成像。
- 加速 junction atlas。
- 减少重复计算。
- 不破坏下游 API。

#### 风险和代价

- batch mask 容易隐藏单个 ray 的异常。
- 不同 ray 的 termination type 和 segment count 不一致，批量数据结构设计要谨慎。
- near-critical 和 boundary case 不能被批量路径粗暴吞掉。

#### 如何验证改进有效

对同一组 b-grid，比较 scalar 与 batch 的：

- termination type。
- segment count。
- `phi_end`。
- `u_end`。
- `u_at()`。
- intersection radius。
- observed intensity。
- profile 峰位和峰值。

---

### 4.3 TabulatedMetric 快路径：值得继续，但必须 fail closed

当前设计中，`TabulatedMetric` 已有私有快速路径：

- 向量化 `_G_array()`。
- Gauss-Legendre 32/64 阶对比验收。
- 失败时回退到标量 `quad`。

这个方向是正确的。建议继续推进，但必须遵守以下原则：

1. 标量 quadrature 始终保留为 correctness oracle。
2. 快路径失败时必须显式 fallback，而不是输出可疑结果。
3. 快路径不能牺牲 `RaySegment.u_at()` 精度。
4. 快路径必须覆盖 junction segment 之前，先有单度规充分验证。
5. 每个快路径都应有 scalar-vs-fast 对照测试。

---

### 4.4 不要牺牲 `RaySegment.u_at()` 精度

这是性能优化的红线。

薄盘交点由固定角度 `phi = pi/2 + n*pi` 决定。`compute_intersections()` 依赖 `RaySegment.u_at()` 找到对应半径。m=1、m=2、m>=3 分别对应 direct、lensing ring、photon ring。

如果 `u_at()` 精度下降，会直接导致：

- ring 半径偏移。
- profile 峰位置偏移。
- photon ring contribution 错误。
- atlas 中结构分类不稳定。

因此可以优化积分速度，但不能用低阶插值或粗糙采样替代 `u(phi)` 的精确构造，除非有明确误差估计和回退路径。

---

### 4.5 暂不建议 JIT、GPU 或替换 SciPy

目前不建议优先考虑：

- JAX。
- Numba。
- Cython。
- C++ 扩展。
- GPU tracing。
- 替换 SciPy integrate / interpolate。

原因是：

1. 当前瓶颈更可能是逐 ray 调用、重复积分、缺少批处理和缓存。
2. 新技术栈会增加安装复杂度。
3. 会影响下游库的可用性。
4. 科学验证矩阵还不够强，无法安全支撑大规模底层替换。
5. NumPy/SciPy 对当前规模已经足够合适。

只有在以下条件满足后，才值得重新评估：

- L4/L5 验证完成。
- benchmark 明确显示 Python 层或 SciPy 成为主瓶颈。
- scalar oracle 与 fast path 长期稳定。
- 下游确实需要大规模高分辨率 atlas 或图像生成。
- 新依赖不会显著破坏安装和维护成本。

---

## 5. 架构和耦合

当前模块边界总体是合理的。主要模块大致按物理和数值职责划分：

- `metrics.py`：解析度规。
- `numerical_metrics.py` / `numerical_loader.py`：数值表格度规。
- `observers.py`：观测者。
- `policies.py`：求解策略。
- `solvers.py`：单度规求解器。
- `junctions.py` / `junction_tracing.py`：静态壳与 junction tracing。
- `transfer.py` / `sources.py` / `imaging.py`：薄盘交点、源函数、成像。
- `junction_analysis.py`：atlas 分析。

不建议大规模改变这些文件边界。真正需要关注的是跨模块物理语义是否一致。

### 5.1 高风险耦合点

以下逻辑跨模块出现，是最需要测试保护的地方：

1. `RaySegment.u_at()` 与 `compute_intersections()`。
2. `DiskWindow` region 与 junction region labels。
3. `observed_intensity()` 与 junction diagnostics 中的 `redshift_weight`。
4. `StaticJunctionTransferSolver` 的事件循环与单度规 segment 构造。
5. `TabulatedMetric` 支持域、静态域限制与 solver policy。
6. `ThroughTracePolicy` 与 junction tracing 的语义区分。
7. near-critical ray 的排除策略与 profile adaptive refinement。

### 5.2 建议增加契约测试，而不是马上拆模块

对跨模块语义，最好的改进不是立刻拆文件，而是增加 contract tests：

1. **intersection contract**
   - 给定 `RaySegment`，`compute_intersections()` 找到的 order、phi、radius 应一致。

2. **region contract**
   - region label 不匹配时不应静默计算强度。

3. **TabulatedMetric domain contract**
   - 域外查询应硬失败。
   - `u<=0`、`|C|` 太小、表格支持外查询不能产生 NaN 继续进入 solver。

4. **policy contract**
   - `TabulatedMetric + InfinityObserver` 应被阻断。
   - `TabulatedMetric + ThroughTracePolicy` 应被阻断。
   - 单度规 `ThroughTracePolicy` 不应被误用为 junction。

5. **junction matching contract**
   - shell crossing 后 `E/L/b` 的更新应符合解析预期。

#### 预期收益

- 降低跨模块物理语义漂移风险。
- 比重构文件更直接、更安全。
- 为后续内部重构提供保护。

---

## 6. 上游库接口与 API 稳定性

这个仓库是其他项目依赖的上游计算库。因此，API 稳定性不是工程洁癖，而是核心需求。

### 6.1 当前需要保护的 API 面

顶层包通过 `spherical_raytracing.__all__` 暴露大量公共符号，包括：

- 解析度规。
- 数值度规。
- 观测者和策略。
- 单度规求解器。
- 诊断数据结构。
- 薄盘与成像函数。
- junction 相关类和函数。

需要特别保护的调用形态包括：

```python
metric.A(r)
metric.B(r)
metric.G(u, b)
metric.horizons()
metric.photon_spheres()
metric.critical_curves()
```

```python
QuadTransferSolver(...).trace_b(b)
```

```python
StaticJunctionTransferSolver(...).trace_b(...)
StaticJunctionTransferSolver(...).trace_alpha(...)
StaticJunctionTransferSolver(...).trace_screen_radius(...)
```

```python
compute_intersections(ray, DiskWindow(...), max_order=...)
observed_intensity(intersections, source, metric, observer, region_metrics=None)
```

### 6.2 建议增加 public API contract test

建议新增：

```text
tests/test_public_api_contract.py
```

测试内容包括：

1. `spherical_raytracing.__all__` 包含预期公共符号。
2. 常见 `from spherical_raytracing import ...` 导入成功。
3. 关键类可按当前签名构造。
4. 关键方法存在且返回预期 dataclass。
5. 返回对象字段名和基本语义不变。
6. 常见端到端调用链不变。

#### 预期收益

- 防止“顺手清理 import”破坏下游。
- 防止 dataclass 字段重命名造成隐性破坏。
- 为后续重构提供安全网。

#### 风险和代价

- API contract test 会让接口演进更慢。
- 但对于上游库，这是合理代价。

### 6.3 破坏性调整原则

原则上不建议破坏公共 API。只有在以下情况中，才值得考虑破坏性调整：

1. 当前 API 会系统性诱导科学误用。
2. 参数名或返回语义明显错误。
3. 保持兼容会导致无法修复严重 bug。
4. 下游迁移成本可控。

即便如此，也应采用迁移路径：

1. 新增正确 API。
2. 保留旧 API。
3. 旧 API 发 `DeprecationWarning`。
4. 文档说明物理或数值原因。
5. 增加迁移测试。
6. 至少保留一个 minor release 周期。

---

## 7. Hamiltonian 后端定位

Hamiltonian 后端目前更适合作为诊断工具，而不是生产后端。

### 7.1 当前定位

`HamiltonianTrajectorySolver`：

- 要求 `FiniteStaticObserver`。
- 要求 `OutsideTracePolicy`。
- 诊断 Hamiltonian constraint、能量、角动量和 impact parameter 漂移。
- 不支持 `ThroughTracePolicy`。
- 对 `TabulatedMetric` 没有稳定承诺。

`StaticJunctionHamiltonianSolver`：

- 可在壳处重启状态。
- 使用匹配后的 `E/L/b`。
- 可与 transfer 后端比较。

### 7.2 建议

短期内不要把 Hamiltonian 包装成与 Quad 同等的生产后端。更合理的文档定位是：

- Quad 是主生产路径。
- ODE 是有限辅助验证路径。
- Hamiltonian 是守恒量、事件边界和后端差异诊断工具。
- `TabulatedMetric + Hamiltonian` 属于 experimental 或 unsupported，除非补齐专门验证。

### 7.3 若未来要升级 Hamiltonian 支持，需要补的验证

如果要让 Hamiltonian 支持数值表格度规，需要至少验证：

1. PCHIP 导数与解析导数 / finite difference 的一致性。
2. Hamiltonian constraint drift 随步长收敛。
3. energy drift 与 angular momentum drift 在无壳区域守恒。
4. boundary / horizon 附近没有虚假穿越。
5. 与 Quad 在非临界、非边界 case 上分类一致。
6. shell crossing 后的状态重启满足 matching residual 阈值。

---

## 8. Artifact、fixture 与可复现性

### 8.1 当前问题

仓库中已有大量 `outputs/` 生成文件，也有真实 Li-DM-BH `.npz` 文件 smoke test。但真实文件测试依赖本机外部路径：

```text
/home/yuanpang/Work/Li-DM-BH/outputs/sanity/data
```

路径不存在时会 skip。

这意味着：

- 其他机器无法完整验证真实数值度规路径。
- CI 无法覆盖真实 loader 变体。
- 下游无法知道 `.npz` 输入格式是否长期稳定。

### 8.2 建议增加小型 `.npz` fixture

建议在仓库中加入极小 fixture，例如：

```text
tests/fixtures/metrics/schwarzschild_tabulated_small.npz
tests/fixtures/metrics/li_dm_bh_like_small.npz
```

fixture 不需要大，不需要覆盖完整物理参数空间，只需要覆盖：

1. 标准 `r/A/B` 输入。
2. Li-DM-BH-like `r/f/g/B/params` 输入。
3. metadata roots。
4. ValidationGate。
5. StaticDomainSelector。
6. TabulatedMetric 插值。
7. solver smoke。
8. thin disk intersection smoke。

### 8.3 `outputs/` 管理建议

不要直接删除 `outputs/`。建议先分类：

1. **示例 artifact**
   - README 或文档展示用。

2. **回归基准 artifact**
   - 可用 hash、manifest 或误差阈值管理。

3. **论文/报告 artifact**
   - 保留生成脚本和参数配置。

4. **临时 artifact**
   - 可以从仓库移除或加入 `.gitignore`。

### 8.4 预期收益

- CI 可复现。
- 下游可验证。
- 数值度规接口更稳定。
- 脚本和 artifact 的职责更清楚。

---

## 9. 工程化改进

当前 `pyproject.toml` 比较轻量，没有明显 CI、lint、type-check 或 coverage 配置。这是可以改进的，但应分阶段推进，不能和核心数值重构混成一个大改。

### 9.1 第一阶段：最小 CI 与测试护栏

建议先加入：

- GitHub Actions 或等价 CI。
- Python 3.11 / 3.12 测试矩阵。
- `pip install -e ".[test]"`。
- `pytest -q`。
- public API contract test。
- 小型 `.npz` fixture 测试。
- coverage 只采集，不强制高阈值。

### 9.2 第二阶段：轻量 lint 与 coverage

再加入：

- ruff。
- 低争议规则：unused imports、pyflakes、import order。
- coverage gate，但阈值不要一开始太高。
- slow test marker。
- benchmark 脚本可运行性 smoke。

### 9.3 第三阶段：类型与发布策略

最后再考虑：

- mypy 或 pyright。
- 只先检查 public API 和 dataclass 边界。
- `py.typed`。
- CHANGELOG。
- deprecation policy。
- release checklist。

### 9.4 不建议一次性强推严格类型

科学计算代码中，数值数组、scalar/array dual behavior、SciPy callable、diagnostic dict 很容易让类型系统变复杂。如果过早强推严格 mypy，可能导致大量为了类型而类型的代码，反而降低简洁性。

---

## 10. 分阶段路线图

### P0：立即优先，1–2 个迭代内完成

#### P0.1 Public API contract test

**为什么重要：** 保护下游依赖。  
**收益：** 防止 import、类名、字段名被无意破坏。  
**风险/代价：** 降低随意改 API 的自由度。  
**验证：** CI 中固定运行 `test_public_api_contract.py`。

#### P0.2 L4 manufactured grid convergence 最小版本

**为什么重要：** 判断 `TabulatedMetric` 的误差是否随分辨率收敛。  
**收益：** 数值度规可信度大幅提升。  
**风险/代价：** 慢，需要标记 slow。  
**验证：** 输出误差随 `N` 下降的表格。

#### P0.3 L5 后端 sanity comparator 最小版本

**为什么重要：** 明确 Quad / ODE / Hamiltonian 的一致性与适用边界。  
**收益：** 防止后端长期漂移。  
**风险/代价：** 不能把 ODE 当完全独立 oracle。  
**验证：** 非临界、非边界 case 中 termination、segment、守恒量一致。

#### P0.4 小型 `.npz` fixture

**为什么重要：** 让数值度规 loader 和 validation 可在 CI 中复现。  
**收益：** 不再依赖本机私有路径。  
**风险/代价：** 需要维护 fixture 格式。  
**验证：** 标准格式和 Li-DM-BH-like 格式都能跑通。

#### P0.5 Benchmark baseline

**为什么重要：** 没有基准就无法判断优化价值。  
**收益：** 后续性能优化可量化。  
**风险/代价：** benchmark 本身需要稳定环境。  
**验证：** 固定命令输出稳定指标。

---

### P1：验证护栏稳定后推进

#### P1.1 提取私有 segment kernel

**为什么重要：** 减少单度规和 junction 的重复数值逻辑。  
**收益：** 降低 bug 面积，让快路径复用。  
**风险/代价：** 核心路径重构风险高。  
**验证：** 重构前后 characterization tests 一致。

#### P1.2 TabulatedMetric 快路径复用到 junction

**为什么重要：** junction atlas 也需要数值度规性能。  
**收益：** profile 和 atlas 加速。  
**风险/代价：** junction 事件循环更复杂。  
**验证：** transfer/junction scalar vs fast 一致。

#### P1.3 `sample_radial_profile()` 内部批处理

**为什么重要：** profile 是大量 ray tracing 的自然批量任务。  
**收益：** 提升成像和 atlas 速度。  
**风险/代价：** batch mask 可能隐藏异常。  
**验证：** scalar/batch 对照。

#### P1.4 Artifact manifest 与代表 case regression

**为什么重要：** atlas 和示例输出需要可追踪。  
**收益：** 防止图像和 CSV 无意漂移。  
**风险/代价：** artifact 管理成本增加。  
**验证：** manifest/hash/数值阈值回归。

---

### P2：长期增强

#### P2.1 Hamiltonian 数值度规支持评估

**为什么重要：** 如果未来要把 Hamiltonian 升级成生产后端，必须先验证。  
**收益：** 增加独立诊断能力。  
**风险/代价：** 需要大量守恒量和导数验证。  
**验证：** constraint drift、energy drift、angular momentum drift 收敛。

#### P2.2 更完整的用户文档和科学约定文档

**为什么重要：** 下游需要理解适用边界。  
**收益：** 减少误用。  
**风险/代价：** 需要持续维护。  
**验证：** 文档中的示例可运行，公式有测试对应。

#### P2.3 类型系统与发布流程

**为什么重要：** 作为上游库，发布流程需要稳定。  
**收益：** 下游升级更安全。  
**风险/代价：** 类型注解维护成本。  
**验证：** release checklist、CHANGELOG、deprecation tests。

---

## 11. 不建议优先投入的方向

### 11.1 大规模重命名或重排公共 API

不建议为了美观重命名类、函数或 dataclass 字段。这个仓库已经被下游依赖，公共 API 的破坏会直接影响下游。

### 11.2 大型继承体系或插件系统

当前仓库规模不需要复杂 framework。过多抽象会掩盖物理语义，降低可调试性。

### 11.3 过早引入 JIT/GPU/C++

在验证矩阵和 benchmark 不完善前，引入新技术栈只会增加复杂度。当前更值得做的是批处理、缓存和复用。

### 11.4 过早扩展完整相机模型

当前成像是 face-on 轴对称薄盘。扩展到倾斜盘、非轴对称发射、完整相机模型是新项目级别的功能，不应和现有库稳定化混在一起。

### 11.5 删除历史 artifact

`outputs/`、`refs/`、`LQGBHShadow/` 可能有复现、论文、参考或历史价值。删除前应先分类和确认用途。

---

## 12. 建议的验收标准

未来每个重要改进 PR 都应尽量回答以下问题：

1. 是否改变公共 API？如果改变，是否有兼容层和迁移测试？
2. 是否改变 ray tracing 数值结果？如果改变，是否有物理或数值解释？
3. 是否影响 near-critical ray 的处理？
4. 是否影响 `RaySegment.u_at()` 精度？
5. 是否影响薄盘交点半径？
6. 是否影响 redshift weight？
7. 是否影响 junction shell crossing 的 energy / b matching？
8. 是否通过 scalar vs fast path 对照？
9. 是否通过 manufactured convergence？
10. 是否有 benchmark 证明性能提升？
11. 是否有 regression test 防止未来退化？

---

## 13. 最终建议

这个仓库目前最应该走的路线不是“功能膨胀”，而是“可信上游库化”。

具体来说：

1. **先补验证**，尤其是 L4/L5、红移权重、junction matching、薄盘交点和 public API contract。
2. **再做内部复用**，优先提取私有 segment / ray kernel，减少重复数值逻辑。
3. **然后做性能优化**，重点是批量 tracing、TabulatedMetric 快路径复用、profile / atlas benchmark。
4. **最后再考虑更大功能**，例如 Hamiltonian 生产化、完整相机模型、更多物理源模型等。

如果只能选一个最核心的方向，那就是：

> **把现有 solver、数值度规、junction 和成像管线放进一个强验证、低重复、API 稳定的框架中。**

这会比增加新功能更能提高这个代码库的长期质量，也更符合它作为下游依赖的上游科学计算库定位。
