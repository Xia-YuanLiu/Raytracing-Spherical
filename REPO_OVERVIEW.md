# Raytracing Spherical 仓库全景文档

更新时间：2026-05-29

本文档面向后续代码审查者，例如 GPT Pro。目标不是让审查者重写仓库，而是让审查者在理解现有物理约定、数值边界和公共接口兼容性之后，提出真正有价值、可以落地、且不破坏下游仓库的改进建议。

## 0. 给审查者的约束

这个仓库已经被其他仓库依赖。任何建议都必须优先保护现有接口。

请把以下内容视为硬约束：

- 保留 `spherical_raytracing.__all__` 暴露的公共名称、构造参数、主要方法名和返回数据结构语义。
- 如确实需要调整接口，优先给出兼容层、弃用路径和迁移测试，而不是直接改名或删除。
- 内部私有函数和私有模块可以建议重构，但要说明为什么能降低复杂度、提高测试性或减少数值风险。
- 不要建议大规模重写、换语言、换掉 NumPy/SciPy、或引入复杂框架，除非能明确证明收益大于兼容和验证成本。
- 对物理公式、坐标约定、红移权重、临界曲线和终止事件的建议必须配套测试策略。
- 生成输出和参考资料不应被当作核心运行时代码来重构。

优先希望审查者回答：

1. 哪些改进能在保留公共 API 的前提下提高正确性、可维护性或性能？
2. 哪些模块边界当前让数值逻辑难以验证？
3. 哪些测试缺口最可能隐藏真实错误？
4. 哪些性能优化值得做，哪些只是复杂化？
5. 哪些文档、打包或 CI 改进能让下游使用更稳？

## 1. 项目概要

`spherical-raytracing` 是一个 Python 包，用于在静态球对称时空中做零测地线追迹，并生成面朝观测的轴对称薄盘图像。

核心场景：

- 单一度规时空：Schwarzschild、Reissner-Nordstrom、Reissner-Nordstrom-de Sitter、LQG-like 度规。
- 数值表格度规：从 `.npz` 表格构造 `TabulatedMetric`，支持标准 `r/A/B` 和 Li-DM-BH 风格 `r/f/g/B/params`。
- 静态薄壳拼接时空：用一个静态壳连接同族内外度规，光线可以穿过壳并按匹配条件更新能量和冲量参数。
- 成像管线：追迹每条光线，记录薄盘交点，计算红移权重，采样一维强度剖面，再旋转成二维轴对称图像。

主要科学输出：

- 一维径向强度 profile。
- 二维轴对称图像。
- 静态 junction 参数空间 atlas、phase map 和代表 case artifact。

## 2. 技术栈与仓库规模

实测代码规模来自 `wc -l`：

| 项目 | 当前状态 |
| --- | --- |
| Python | `>=3.11` |
| 运行依赖 | `numpy>=1.26`, `scipy>=1.11` |
| 测试依赖 | `pytest>=8.0` |
| 绘图脚本依赖 | README 建议额外安装 `matplotlib` |
| 构建 | `setuptools>=69`, `wheel` |
| 许可证 | BSD 3-Clause |
| `src/spherical_raytracing` | 15 个 Python 文件，5437 行 |
| `tests` | 26 个测试文件 |
| `scripts` | 11 个脚本 |

当前 `pyproject.toml` 比较轻量：没有 console scripts、ruff/mypy/coverage 配置，也没有 `py.typed`。这可以作为工程化改进方向，但不能以破坏现有导入路径为代价。

## 3. 目录结构

```text
raytracing-spherical/
  src/spherical_raytracing/
    __init__.py                  # 顶层公共 API 导出
    diagnostics.py               # RayResult, RaySegment, RayDiagnostics, EventType
    metrics.py                   # 解析静态球对称度规
    numerical_metrics.py         # TabulatedMetric 与数值 payload 数据结构
    numerical_loader.py          # .npz 加载与 ValidationGate
    observers.py                 # InfinityObserver, FiniteStaticObserver
    policies.py                  # SolverOptions, Outside/Through trace policy
    solvers.py                   # 单一度规 Quad, ODE, Hamiltonian 后端
    _turning.py                  # 转折点搜索共享实现
    transfer.py                  # 薄盘窗口与交点
    sources.py                   # 薄盘源、红移权重、观测强度
    imaging.py                   # profile 采样与轴对称渲染
    junctions.py                 # 静态壳、匹配条件、junction 诊断
    junction_tracing.py          # junction transfer/Hamiltonian 后端
    junction_analysis.py         # atlas 参数过滤、分类、profile 诊断
  tests/                         # 单元、回归、脚本和验证测试
  scripts/                       # artifact 生成、atlas、benchmark 脚本
  docs/plans/                    # 数值度规接入、验证、性能计划
  notebooks/                     # RN-dS walkthrough notebook
  refs/                          # 参考论文 PDF
  LQGBHShadow/                   # 遗留 Mathematica 参考代码
  outputs/                       # 已生成的图像、CSV、JSON artifact
```

`outputs/` 中有大量已跟踪的生成文件。审查者可以关注它们是否应作为回归基准或发布 artifact 管理，但不要把它们当作核心库实现。

## 4. 公共 API 面

顶层包当前通过 `src/spherical_raytracing/__init__.py` 暴露约 50 个公共符号。下游仓库很可能直接从 `spherical_raytracing` 导入这些名称。

公共 API 大致分组如下：

| 类别 | 符号 |
| --- | --- |
| 解析度规 | `SchwarzschildMetric`, `ReissnerNordstromMetric`, `ReissnerNordstromDeSitterMetric`, `LQGMetric` |
| 数值度规 | `CanonicalPayload`, `LiDMBHMetadata`, `StaticDomainSelector`, `TabulatedMetric`, `ValidationOptions`, `ValidationGate`, `load_metric_npz` |
| 观测者与策略 | `InfinityObserver`, `FiniteStaticObserver`, `SolverOptions`, `OutsideTracePolicy`, `ThroughTracePolicy` |
| 单度规求解器 | `QuadTransferSolver`, `OdeTrajectorySolver`, `HamiltonianTrajectorySolver` |
| 诊断数据 | `CriticalCurve`, `EventType`, `RayDiagnostics`, `RayEvent`, `RayResult`, `RaySegment` |
| 薄盘与成像 | `DiskWindow`, `IntersectionRecord`, `ThinDiskSource`, `ObservedIntensity`, `ImageGrid`, `RadialProfile`, `RenderedImage`, `SamplingOptions`, `compute_intersections`, `observed_intensity`, `sample_radial_profile`, `render_axisymmetric_image` |
| junction | `StaticShell`, `StaticJunctionSpacetime`, `SegmentConstants`, `ShellCrossingRecord`, `JunctionRayResult`, `JunctionPhysicsDiagnostics`, `StaticJunctionTransferSolver`, `StaticJunctionHamiltonianSolver`, `match_static_shell`, `junction_redshift_factor`, `annotate_junction_intersections`, `compare_junction_results`, `trace_with_backend_compare` |

需要特别保护的调用形态：

- `metric.A(r)`, `metric.B(r)`, `metric.G(u, b)`, `metric.horizons()`, `metric.photon_spheres()`, `metric.critical_curves()`。
- `QuadTransferSolver(...).trace_b(b)` 返回 `RayResult`。
- `StaticJunctionTransferSolver(...).trace_b(...)`, `.trace_alpha(...)`, `.trace_screen_radius(...)` 返回 `JunctionRayResult`。
- `compute_intersections(ray, DiskWindow(...), max_order=...)`。
- `observed_intensity(intersections, source, metric, observer, region_metrics=None)`。

## 5. 核心架构

### 5.1 度规层

`metrics.py` 定义 `StaticSphericalMetric` 协议。实现者需要提供：

```python
region: str
A(r)
B(r)
dA_dr(r)
dB_dr(r)
G(u, b)
horizons()
photon_spheres()
critical_curves()
valid_radial_domain()
static_domains()
```

解析度规包括：

| 度规 | 参数 | 说明 |
| --- | --- | --- |
| `SchwarzschildMetric` | `mass` | `A=1-2M/r`, 外部静态域为事件视界外 |
| `ReissnerNordstromMetric` | `mass`, `charge` | 支持有符号 charge，但几何依赖 `Q^2` |
| `ReissnerNordstromDeSitterMetric` | `mass`, `charge`, `cosmological_constant` | 支持 `Lambda>=0`，可能有有界静态 patch |
| `LQGMetric` | `mass`, `alpha` | LQG-like `f(r)=1-2M/r+alpha*M^2/r^4` |

解析度规当前都满足 `B(r)=1/A(r)`，但协议本身允许一般两函数度规。`G(u,b)` 用 `u=1/r` 表示零测地线径向势。

### 5.2 数值表格度规

数值管线是：

```text
.npz -> load_metric_npz -> CanonicalPayload
     -> ValidationGate
     -> StaticDomainSelector
     -> TabulatedMetric
```

当前支持两种输入：

- 标准 `r/A/B`：内部计算 `C=1/B`，无上游 metadata roots。
- Li-DM-BH `r/f/g/B/params`：内部映射 `A=f`, `C=g`，`raw_B` 用于 `B*C≈1` 软校验。

`TabulatedMetric` 的关键设计：

- 插值 `A(r)` 和 `C(r)=1/B(r)`，不是直接插值 `B(r)`，以避免视界附近 `B` 发散。
- 使用 `PchipInterpolator(extrapolate=False)`。
- 静态域端点只有在匹配 metadata horizon 时才注入 `(r_boundary, 0)` 锚点，避免把普通表格边界伪造成视界。
- 没有 metadata roots 时，会从插值表格推断 horizon 和 photon sphere roots，让标准 `r/A/B` payload 仍可用。
- 域外查询、表格支持外查询、`u<=0`、`|C|` 太小等情况会硬失败，而不是把 NaN/inf 传给求解器。
- `TabulatedMetric + InfinityObserver` 和 `TabulatedMetric + ThroughTracePolicy` 在 solver 入口被硬阻断。
- Hamiltonian 后端对 `TabulatedMetric` 不作为稳定承诺。文档和计划建议数值度规主路径使用 `QuadTransferSolver`，必要时用 `OdeTrajectorySolver` 做有限辅助验证。

### 5.3 观测者层

`InfinityObserver`：

- 屏幕坐标就是 impact parameter `b`。
- `u_start=0`。
- 适合解析无穷远场，不适合有限表格度规。

`FiniteStaticObserver`：

- 位于有限半径 `r_obs`。
- 要求 `A(r_obs)>0`。
- `impact_parameter(alpha)=r_obs*sin(alpha)/sqrt(A(r_obs))`。
- `screen_radius(alpha)=r_obs*tan(alpha)`。
- `u_start=1/r_obs`。

### 5.4 单一度规求解器

`QuadTransferSolver` 是主生产路径。

核心行为：

- near-critical ray 直接返回带 `near_critical=True` 的空 ray，而不是声称精确追迹。
- 从观测者位置向内追迹，搜索第一个转折点。
- 无转折时终止于 horizon、inner boundary 或其他边界。
- 有转折时构造 inbound 和 outbound segment。
- 无穷远逃逸段使用镜像构造。
- 有界外边界，例如 RN-dS cosmological horizon，用实际 outward segment。
- `TabulatedMetric` 有私有快速路径：向量化 `_G_array()`，Gauss-Legendre 32/64 阶对比验收，失败时回退到标量 `quad` 参考实现。

`OdeTrajectorySolver` 是辅助后端：

- 入口先调用 `QuadTransferSolver` 获得参考分类和基本结构。
- 入射段用 `solve_ivp`。
- 从转折点出射时仍委托 Quad 的正则化 segment 构造，因为 ODE 在 `du/dphi=0` 的转折点难以直接启动。
- 因此它不是完全独立的 oracle。

`HamiltonianTrajectorySolver` 是诊断后端：

- 要求 `FiniteStaticObserver` 和 `OutsideTracePolicy`。
- 状态变量为 `(r, p_r, phi, p_t, p_phi)`。
- 诊断 Hamiltonian 约束、能量、角动量和 impact parameter 漂移。
- 不支持 `ThroughTracePolicy`。

单度规 `ThroughTracePolicy` 只是在同一个 metric 实例上做 horizon continuation 和 region label 切换。它不是静态壳 junction，也不是不同度规之间的拼接。

### 5.5 转折点搜索

`_turning.py` 提供共享转折点扫描：

- `_first_turning_point_scalar_scan()`：标量扫描 `G(u,b)`。
- `_first_turning_point_scan()`：在 `TabulatedMetric` 且扫描区间受支持时使用 `_G_array()` 向量化，否则回退标量扫描。

扫描语义是找第一个 `G` 从正到非正的区间，然后用 `brentq` 精化。

### 5.6 静态 thin-shell junction

`StaticJunctionSpacetime` 目前只支持同族静态 junction：

- Schwarzschild-Schwarzschild。
- RN-RN。
- RN-dS-RN-dS。

不支持 LQG junction，也不支持混合族 junction。

核心匹配：

- 壳半径 `R`。
- 角动量 `L` 连续。
- 能量按 `sqrt(A_to/A_from)` 重标定。
- 因此 `b=L/E` 会按 `sqrt(A_from/A_to)` 改变。
- 计算 surface energy density 和 surface pressure。
- 诊断 warning 包括非静态壳/观测者、裸奇点、壳靠近视界、charge mismatch、lambda mismatch、负面能量密度等。

`StaticJunctionTransferSolver` 是 junction 主生产路径：

- 要求有限半径静态观测者，且观测者在壳外。
- 循环最多 64 次事件。
- 事件包括转折点、壳穿越、horizon、inner/outer boundary、escape、max_phi。
- 壳穿越时记录 `ShellCrossingRecord` 和匹配 residual。

`StaticJunctionHamiltonianSolver` 是 junction 诊断路径：

- 独立进行 Hamiltonian 积分。
- 在壳处重启状态，并使用匹配后的 `E/L/b`。
- `trace_with_backend_compare()` 会运行 transfer 和 Hamiltonian 两个后端，并把差异写入 `backend_comparison` 和 diagnostics。

### 5.7 薄盘、源与成像

`transfer.py`：

- `DiskWindow` 定义薄盘半径窗口和启用 region。
- `compute_intersections()` 在固定角度 `phi = pi/2 + n*pi` 找 ray 与薄盘交点。
- `m=1` 是 direct，`m=2` 是 lensing ring，`m>=3` 是 photon ring。

`sources.py`：

- `ThinDiskSource` 接收 `intensity_profile(r, region)`。
- 单度规红移权重为 `g^4`，其中无穷远观测者 `g=sqrt(A(r_em))`，有限观测者 `g=sqrt(A(r_em)/A(r_obs))`。
- junction 管线可在 intersection diagnostics 中预先写入 `redshift_weight`，从而包含壳穿越的 energy ratio product。

`imaging.py`：

- `sample_radial_profile()` 从初始均匀网格开始，遇到强度梯度、相对跳变、termination/path class 变化、shell crossing count 变化、backend disagreement 或 unresolved 标记时加密采样。
- `RadialProfile.evaluate()` 用 PCHIP 插值，域外返回 0。
- `render_axisymmetric_image()` 把一维 profile 旋转成二维图像。

### 5.8 Atlas 系统

`junction_analysis.py` 和 `scripts/generate_junction_atlas.py` 组成静态 junction atlas 管线：

- `AtlasParameters` 定义内外质量、电荷、宇宙学常数、壳半径、观测半径。
- `admit_parameters()` 做物理过滤。
- `classify_photon_spheres()` 分类 visible photon sphere：`inner_only`, `outer_only`, `double`, `none`。
- `select_representative_cases()` 从候选里按 category 选代表 case。
- `diagnose_profile()` 从 profile 中提取峰、红移 cusp、V 形结构、壳穿越数量等。

脚本会生成 phase map、代表 case 的 CSV/JSON/PNG，以及可选报告图像。

## 6. 脚本与生成输出

主要脚本：

| 脚本 | 用途 |
| --- | --- |
| `generate_validation_tables.py` | Schwarzschild transfer 验证表 |
| `generate_fig5_profiles.py` | Schwarzschild Fig. 5 风格 profile/image |
| `generate_lqg_fig3_profiles.py` | LQG Fig. 3 风格 profile 和 ring-edge 对比 |
| `generate_static_junction_validation.py` | 静态 junction 验证数据 |
| `generate_static_junction_fig3_fig4.py` | 静态 junction Fig. 3/Fig. 4 复现 artifact |
| `generate_static_junction_profiles.py` | 通用静态 junction profile |
| `generate_rn_junction_images.py` | RN junction 参数 sweep |
| `generate_junction_atlas.py` | RN/RN-dS atlas 生成 |
| `write_junction_atlas_report.py` | 从 atlas manifest 写中文报告 |
| `generate_numerical_metric_images.py` | 数值度规图像 smoke/sanity |
| `benchmark_numerical_metric_perf.py` | 数值度规性能基准 |

`outputs/README.md` 说明了生成 artifact 分组。大目录已经在仓库中存在，后续改进可以考虑 artifact 管理策略，但不应在接口兼容评审中随意删除。

## 7. 测试体系

测试覆盖层次：

| 层次 | 文件 |
| --- | --- |
| 包导入与 API | `test_package.py` |
| 度规 | `test_metrics.py`, `test_numerical_metrics.py`, `test_numerical_metric_validation.py` |
| 观测者和策略 | `test_observers.py`, `test_policies.py` |
| 单度规求解器 | `test_quad_solver.py`, `test_ode_solver.py`, `test_hamiltonian_solver.py`, `test_solvers_bounded_domain.py` |
| 薄盘、源、成像 | `test_transfer.py`, `test_sources.py`, `test_imaging.py` |
| junction 物理和追迹 | `test_static_junctions.py`, `test_static_junction_transfer_solver.py`, `test_static_junction_hamiltonian_solver.py`, `test_static_junction_backend_agreement.py` |
| junction 成像和源 | `test_static_junction_imaging.py`, `test_static_junction_sources.py` |
| atlas | `test_junction_analysis.py`, `test_junction_atlas_scripts.py` |
| 脚本 | `test_fig5_profile_script.py`, `test_lqg_fig3_profile_script.py`, `test_static_junction_scripts.py`, `test_validation_script.py` |

数值度规验证的当前实际状态：

- L1 已实现：解析度规采样成 manufactured `TabulatedMetric` 后比较 `A`, `C`, `dA/dr`, `dB/dr`, `G`, horizon/photon sphere/critical curve。
- L2 已实现：小规模 b-sweep 比较解析路径和数值表格路径的 ray 终止、segment、`phi_end`, `u_end`, `u(phi)`。
- L3 已实现：一个薄盘交点和 `observed_intensity` smoke。
- L4 仍是 TODO：慢速网格收敛研究。
- L5 仍是 TODO：单度规 Quad/ODE/Hamiltonian 后端 sanity comparator。

`test_numerical_metrics.py` 还包含真实 Li-DM-BH `.npz` 文件 smoke，但这些测试依赖本机外部路径 `/home/yuanpang/Work/Li-DM-BH/outputs/sanity/data`；路径不存在时会 skip。

## 8. 已文档化设计决策

重要设计文档：

- `docs/plans/数值度规接入计划.md`
- `docs/plans/numerical-metric-validation-plan.md`
- `docs/plans/numerical-metric-perf-plan.md`
- `docs/plans/implementation-prompts.md`
- `CONTEXT.md`

当前代码中已经落实的关键决策：

- 数值度规插值 `C=1/B`，避免视界附近直接插值 `B`。
- 静态域端点只在匹配 metadata horizon 时零锚定。
- 默认静态域选择可由 metadata、用户指定或 `A>0 & C>0` 推断。
- `TabulatedMetric + InfinityObserver` 首期不支持。
- `TabulatedMetric + ThroughTracePolicy` 首期不支持。
- 数值度规主路径是 `FiniteStaticObserver + QuadTransferSolver`。
- `OdeTrajectorySolver` 可作辅助，但它依赖 Quad 的部分逻辑。
- Hamiltonian 对数值度规不保证稳定，计划中属于后续验证方向。
- 性能优化保留标量 quadrature 作为 correctness oracle，并让快速路径 fail closed。

## 9. 已知限制

这些限制不一定是 bug，但会影响后续改进优先级：

1. `StaticJunctionTransferSolver` 有自己的 segment 积分逻辑。它共享 `_turning.py` 的转折点扫描，但没有共享单度规 `QuadTransferSolver` 的 TabulatedMetric Gauss-Legendre segment 快速路径。
2. `OdeTrajectorySolver` 不是完全独立后端，因为它用 Quad 做预分类和转折点出射段。
3. `HamiltonianTrajectorySolver` 对数值表格度规没有稳定性保证和专门验证矩阵。
4. `sample_radial_profile()` 是逐 ray 调用，没有批量追迹 API。
5. 成像当前面向 face-on 轴对称薄盘，不支持倾斜盘、非轴对称发射或完整相机模型。
6. 数值度规真实文件测试依赖本机外部数据目录，仓库本身不包含 `.npz` fixture。
7. `LQGBHShadow/` 是遗留 Mathematica 参考代码，不属于 Python 包运行时。
8. 没有明显的 CI、lint、type-check 或 coverage 配置。

## 10. 值得 GPT Pro 重点评估的方向

下面是建议审查者优先考虑的方向。每个方向都应以“保持公共 API 不变”为前提。

1. 数值正确性测试增强  
   优先补 L4 网格收敛和 L5 单度规后端 sanity comparator。尤其要区分真正错误、near-critical 排除带、`max_phi` 截断和表格稀疏导致的可解释误差。

2. 求解器内部复用  
   评估 `QuadTransferSolver` 与 `StaticJunctionTransferSolver` 的 segment 构造是否能提取共享内部实现。目标是减少重复数值逻辑，而不是改变公共类。

3. 数值度规性能路径  
   继续沿 `numerical-metric-perf-plan.md` 的 fail-closed 原则推进。不要为了速度牺牲 `RaySegment.u_at()` 精度，因为薄盘交点依赖它。

4. Hamiltonian 后端定位  
   明确它是诊断工具还是可支持生产路径。如果要支持数值度规，先补守恒量、边界和表格导数验证。

5. API 稳定性机制  
   可以增加更明确的 public API 测试、导入兼容测试、CHANGELOG 或 deprecation policy。不要直接重命名现有公共类和函数。

6. 工程化配置  
   可以建议添加 CI、coverage、ruff、mypy 或 `py.typed`，但要分阶段，不要和数值逻辑重构混在一个大改里。

7. Artifact 与 fixture 策略  
   评估 `outputs/` 是否应作为基准 artifact、示例 artifact 或发布产物管理。真实 `.npz` fixture 的缺失会限制可复现验证。

## 11. 不建议优先做的事

以下建议大概率会破坏下游或浪费精力，除非同时给出严谨兼容方案：

- 直接改顶层公共 API 名称。
- 把所有 dataclass 改成大型继承体系。
- 把 NumPy/SciPy 数值核心替换成不成熟依赖。
- 把 face-on 轴对称成像扩展为完整通用相机，除非这是明确新需求。
- 删除 `outputs/`、`refs/`、`LQGBHShadow/`，除非先确定它们不是用户需要的历史 artifact。
- 为单次使用场景增加复杂配置层。

## 12. 运行与验证

常用命令：

```bash
pip install -e ".[test]"
pytest -q
```

生成示例：

```bash
python scripts/generate_fig5_profiles.py \
  --profile all \
  --output outputs/schwarzschild_fig5/fig5_profiles.csv \
  --plot-output outputs/schwarzschild_fig5/fig5_profiles.png \
  --image-output outputs/schwarzschild_fig5/fig5_images.png
```

```bash
python scripts/generate_junction_atlas.py \
  --preset quick \
  --families rn rnds \
  --output-dir outputs/junction_atlas
```

数值度规 benchmark：

```bash
python scripts/benchmark_numerical_metric_perf.py
```

## 13. 审查时的推荐阅读顺序

1. `README.md` 或 `README.zh-CN.md`：用户级功能和示例。
2. `src/spherical_raytracing/__init__.py`：公共 API 面。
3. `src/spherical_raytracing/diagnostics.py`：核心返回数据结构。
4. `src/spherical_raytracing/metrics.py`：解析度规协议与实现。
5. `src/spherical_raytracing/solvers.py`：单度规追迹主逻辑。
6. `src/spherical_raytracing/numerical_metrics.py` 和 `numerical_loader.py`：数值表格度规。
7. `src/spherical_raytracing/junctions.py` 和 `junction_tracing.py`：静态壳物理和追迹。
8. `src/spherical_raytracing/transfer.py`, `sources.py`, `imaging.py`：成像管线。
9. `tests/test_numerical_metric_validation.py`：最能代表数值正确性目标的测试。
10. `docs/plans/`：理解已有计划，避免重新提出已经接受或拒绝的方向。
