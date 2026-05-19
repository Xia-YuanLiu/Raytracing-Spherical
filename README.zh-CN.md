# Raytracing Spherical

[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](LICENSE)

[English](README.md) | 中文

Raytracing Spherical 是一个用于静态球对称时空中类光线追踪的 Python 包。它同时支持单一度规下的薄盘成像，以及静态薄壳 junction 时空；在后者中，光线可以穿过壳层，在匹配的内外度规之间传播。

这个包目前主要面向正面观测的轴对称图像生成。对每个屏幕坐标或冲量参数，代码会追踪光线、记录薄盘交点、计算红移权重，并采样得到一维强度剖面，最后渲染成二维图像。

## 功能

- 支持 areal-radius 坐标下的静态球对称度规接口：
  `ds^2 = -A(r)dt^2 + B(r)dr^2 + r^2 dOmega^2`
- 内置 Schwarzschild、LQG-like、Reissner-Nordstrom 和
  Reissner-Nordstrom-de Sitter 度规
- 支持无穷远观测者和有限半径静态观测者
- 提供求积转移、采样 ODE 和 Hamiltonian 轨迹求解器
- 支持薄盘交点、路径分类、红移权重、径向剖面采样和轴对称图像渲染
- 支持静态薄壳 junction 追踪，包括壳层穿越记录、能量与冲量参数匹配，以及 junction 物理诊断
- 支持 junction 情形下 transfer 与 Hamiltonian 后端的交叉比较
- 提供 Schwarzschild、LQG-like、静态 junction、RN、RN-dS 和 atlas 风格输出的生成脚本

## 安装

需要 Python 3.11 或更新版本。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
```

核心包依赖 NumPy 和 SciPy。如果需要运行生成脚本中的绘图和图像输出选项，还需要安装 Matplotlib：

```bash
pip install matplotlib
```

## 快速开始：Schwarzschild 薄盘

```python
from spherical_raytracing import (
    DiskWindow,
    InfinityObserver,
    QuadTransferSolver,
    SchwarzschildMetric,
    ThinDiskSource,
    compute_intersections,
    observed_intensity,
)

metric = SchwarzschildMetric(mass=1.0)
observer = InfinityObserver()
solver = QuadTransferSolver(metric=metric, observer=observer)

ray = solver.trace_b(8.0)
disk = DiskWindow(r_min=6.0, r_max=100.0)
intersections = compute_intersections(ray, disk, max_order=3)

source = ThinDiskSource(lambda r, region=None: 1.0 / r**2)
intensity = observed_intensity(intersections, source, metric, observer)

print(ray.diagnostics.termination_reason)
print(intensity.total)
```

## 快速开始：数值度规（`.npz` 表格）

对于以 `.npz` 文件形式提供的表格度规（Li-DM-BH 的 `r/f/g/B/params` 布局，或标准的
`r/A/B` 表格），请使用 `load_metric_npz` 配合 `TabulatedMetric` 与有限半径静态观测者：

```python
from spherical_raytracing import (
    DiskWindow,
    FiniteStaticObserver,
    QuadTransferSolver,
    StaticDomainSelector,
    TabulatedMetric,
    ThinDiskSource,
    compute_intersections,
    load_metric_npz,
    observed_intensity,
)

payload = load_metric_npz("hernquist_L0.npz")
r_lo, r_hi = StaticDomainSelector().choose(payload)
metric = TabulatedMetric(payload=payload, static_domain=(r_lo, r_hi))

r_ph = metric.photon_spheres()[-1]
r_obs = 5.0 * r_ph
observer = FiniteStaticObserver(r_obs=r_obs, metric=metric)
solver = QuadTransferSolver(metric=metric, observer=observer)

b_crit = metric.critical_curves()[0].b_crit
ray = solver.trace_b(b_crit * 1.05)

disk = DiskWindow(r_min=r_ph * 1.05, r_max=r_obs - 0.5)
intersections = compute_intersections(ray, disk, max_order=4)
source = ThinDiskSource(lambda r, region=None: 1.0 / r**2)
intensity = observed_intensity(intersections, source, metric, observer)
```

关键约束（完整设计见 `docs/plans/数值度规接入计划.md`）：

- Li-DM-BH 表格中 `A=f`、`C=g`；包内部用
  `scipy.interpolate.PchipInterpolator` 在静态域子数组上同时插值 `A(r)`
  和 `C(r)=1/B(r)`（决策 1/4）。
- `(r_boundary, 0.0)` 锚点**仅在静态域端点与 metadata horizon 在
  决策 6 容差内匹配时**才注入。非 horizon 端点——例如标准 `r/A/B` 表格的
  支持上下界，或 `StaticDomainSelector(radial_bounds=...)` 在静态域内部
  截取的子区间——**不会**被零锚点，避免在普通支持边界处人为造出假视界
  扭曲附近的数值。
- 也接受标准 `r/A/B` 布局；此时 `payload.metadata.raw_B is None`，
  `B*C≈1` 软检查自动跳过（决策 7）。这类 payload 没有 metadata roots，
  所以 `TabulatedMetric.horizons()` / `photon_spheres()` /
  `critical_curves()` 会从插值后的表格数值推断 `A=0` 与
  `r*A'-2*A = 0` 的根，保证 `b_crit` 与 near-critical 保护仍可用。
- 默认选择最后一个静态域；可通过
  `StaticDomainSelector(domain_index=..., radial_bounds=...)` 覆盖（决策 5）。
- `TabulatedMetric` 仅支持 `FiniteStaticObserver` + `OutsideTracePolicy` 组合；
  与 `InfinityObserver` 或 `ThroughTracePolicy` 组合在 solver 入口
  直接 raise `NotImplementedError`（决策 10/11），同时
  `r_obs` 必须落在表格的 grid 覆盖范围内。
- 域外查询、越过 `[r_grid_min, r_grid_max]` 的查询，以及
  `|C| < c_tol` 的查询都硬 raise `ValueError`，
  不会让 NaN/inf 静默传到 solver（决策 8）。
- `TabulatedMetric` 的 per-ray 性能预期比解析度规慢 10–100×（PCHIP 评估开销）。
  Hamiltonian 后端对数值度规的稳定性首期不保证——请使用
  `QuadTransferSolver`（或 `OdeTrajectorySolver`）。

## 快速开始：静态 Junction

```python
from spherical_raytracing import (
    FiniteStaticObserver,
    ReissnerNordstromMetric,
    StaticJunctionSpacetime,
    StaticJunctionTransferSolver,
    StaticShell,
)

inner = ReissnerNordstromMetric(mass=0.8, charge=0.2, region="inner")
outer = ReissnerNordstromMetric(mass=1.0, charge=0.2, region="outer")
junction = StaticJunctionSpacetime(
    inner_metric=inner,
    outer_metric=outer,
    shell=StaticShell(radius=4.0),
)
observer = FiniteStaticObserver(r_obs=50.0, metric=outer)
solver = StaticJunctionTransferSolver(junction=junction, observer=observer)

ray = solver.trace_b(4.0)

print(ray.termination_reason)
print(len(ray.shell_crossings))
print(ray.physics_warnings)
```

`StaticJunctionSpacetime` 目前支持同族静态 junction：Schwarzschild-Schwarzschild、RN-RN 或 RN-dS-RN-dS。

## 项目结构

```text
src/spherical_raytracing/
  diagnostics.py          # 光线事件、光线段、诊断信息和临界曲线
  imaging.py              # 径向剖面采样和图像渲染
  junction_analysis.py    # Junction 案例摘要和分类辅助函数
  junction_tracing.py     # 静态 junction transfer 和 Hamiltonian 求解器
  junctions.py            # 壳层匹配、壳层穿越和 junction 诊断
  metrics.py              # 静态球对称度规
  numerical_loader.py     # 数值度规 .npz 加载器与校验门禁
  numerical_metrics.py    # CanonicalPayload、LiDMBHMetadata、ValidationOptions、
                          # StaticDomainSelector、TabulatedMetric
  observers.py            # 无穷远和有限半径静态观测者
  policies.py             # 追踪策略和求解器选项
  solvers.py              # 单一度规的求积、ODE 和 Hamiltonian 求解器
  sources.py              # 薄盘源和观测强度辅助函数
  transfer.py             # 薄盘窗口和交点记录
```

## 脚本

生成 Schwarzschild 验证表：

```bash
python scripts/generate_validation_tables.py \
  --output outputs/schwarzschild_transfer.json \
  --samples 32
```

生成 Gralla-Holz-Wald Fig. 5 风格的 Schwarzschild 薄盘强度剖面：

```bash
python scripts/generate_fig5_profiles.py \
  --profile all \
  --output outputs/schwarzschild_fig5/fig5_profiles.csv \
  --plot-output outputs/schwarzschild_fig5/fig5_profiles.png \
  --image-output outputs/schwarzschild_fig5/fig5_images.png
```

生成 LQG Fig. 3 风格的剖面和 ring-edge 对比表：

```bash
python scripts/generate_lqg_fig3_profiles.py \
  --output outputs/lqg_fig3/lqg_fig3_profile.csv \
  --plot-output outputs/lqg_fig3/lqg_fig3_profile.png \
  --image-output outputs/lqg_fig3/lqg_fig3_image.png \
  --comparison-output outputs/lqg_fig3/lqg_fig3_ring_edges.csv
```

生成静态 junction 验证数据：

```bash
python scripts/generate_static_junction_validation.py \
  --output outputs/static_junction_validation.json \
  --samples 8
```

生成静态 junction Fig. 3/Fig. 4 复现输出：

```bash
python scripts/generate_static_junction_fig3_fig4.py \
  --output outputs/static_junction/fig3_fig4/static_junction_fig3_fig4.json \
  --fig3-output outputs/static_junction/fig3_fig4/static_junction_fig3.png \
  --fig4-image-output outputs/static_junction/fig3_fig4/static_junction_fig4_image.png \
  --fig4-trajectories-output outputs/static_junction/fig3_fig4/static_junction_fig4_trajectories.png
```

生成精选 RN 静态 junction 图像：

```bash
python scripts/generate_rn_junction_images.py \
  --output-dir outputs/rn_junction_sweep \
  --compare-backends
```

生成 RN/RN-dS 静态 junction atlas：

```bash
python scripts/generate_junction_atlas.py \
  --preset quick \
  --output-dir outputs/junction_atlas \
  --compare-backends \
  --include-schwarzschild-reference \
  --emissivity both
```

根据生成的 manifest 写出 Markdown atlas 报告：

```bash
python scripts/write_junction_atlas_report.py \
  --manifest outputs/junction_atlas/manifest.json \
  --output docs/junction-atlas/rn-rnds-static-junction-atlas.md
```

## 生成输出

生成产物按实验分组放在 `outputs/` 下：

- `outputs/junction_atlas/`：RN/RN-dS atlas 的 manifest、phase map、代表案例剖面、图像、transfer-redshift 图，以及可选的 Schwarzschild reference 产物
- `outputs/junction_atlas_schwarzschild_reference/`：由 atlas 生成器单独生成的 Schwarzschild reference 运行结果
- `outputs/rn_junction_sweep/`：精选 RN 静态 junction 扫描图像和每个案例的 JSON 摘要
- `outputs/static_junction/fig3_fig4/`：静态 junction Fig. 3/Fig. 4 复现数据和图像
- `outputs/schwarzschild_fig5/`：Schwarzschild Fig. 5 剖面和图像输出
- `outputs/lqg_fig3/`：LQG Fig. 3 剖面、图像和 ring-edge 输出

仓库中仍保留了一些早期运行产生的根目录输出文件，用于兼容旧路径。新的生成结果建议放入对应的分组输出目录。

## 测试

```bash
pytest -q
```

测试覆盖度规量、观测者映射、求解器事件、Hamiltonian 与 transfer 后端一致性、薄盘交点、红移权重、图像采样、静态 junction 壳层匹配、junction 诊断，以及公开生成脚本。

## 说明

- 默认使用无量纲几何单位，多数示例以 `M = 1` 作为默认尺度。
- `QuadTransferSolver` 是单一度规正面薄盘图像的主要生产路径。
- `StaticJunctionTransferSolver` 是静态 junction 图像的主要生产路径；`StaticJunctionHamiltonianSolver` 用于诊断和后端交叉比较。
- RN-dS junction 必须位于干净的静态区域内。诊断会报告壳层或观测者不处于静态区域、靠近视界、电荷或宇宙学常数跳变、负表面能量密度等情况。
- Atlas 和图像生成脚本可能产生较大的输出目录。

## 引用

如果你在学术工作中使用本代码，请使用 [CITATION.cff](CITATION.cff) 中的元数据引用本仓库。若你的分析使用了其中的时空模型、辐射模型或论文图像对比脚本，也请同时引用对应的原始科学论文。

## 许可证

本项目使用 BSD 3-Clause License。详见 [LICENSE](LICENSE)。
