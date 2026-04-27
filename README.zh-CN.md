# Raytracing Spherical

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[English](README.md) | 中文

Raytracing Spherical 是一个用于静态球对称时空中类光线追踪的 Python 包。它利用球对称时空中的守恒量，把正面观测图像的计算降维为一维冲量参数剖面 `I(b)`，再将该径向剖面旋转成二维图像。

## 功能

- 支持 areal-radius 坐标下的一般静态球对称度规：
  `ds^2 = -A(r)dt^2 + B(r)dr^2 + r^2 dOmega^2`
- 提供 Schwarzschild 和 LQG-like 度规示例
- 支持无穷远观测者和有限半径静态观测者
- 提供求积、采样 ODE、外部静态坐标 Hamiltonian 三类光线求解器
- 支持薄盘转移交点和观测强度分解
- 支持轴对称径向剖面采样和二维图像渲染
- 提供验证表和论文图像风格的生成脚本

## 安装

需要 Python 3.11 或更新版本。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
```

核心包依赖 NumPy 和 SciPy。如果需要运行脚本中的绘图输出选项，还需要安装 Matplotlib：

```bash
pip install matplotlib
```

## 快速开始

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

## 项目结构

```text
src/spherical_raytracing/
  diagnostics.py    # 事件、光线段、诊断数据结构
  imaging.py        # 径向剖面采样和轴对称图像渲染
  metrics.py        # 度规协议，以及 Schwarzschild/LQG-like 度规
  observers.py      # 无穷远和有限半径静态观测者映射
  policies.py       # 外部追踪和 through-trace 策略
  solvers.py        # 求积、ODE、Hamiltonian 光线求解器
  sources.py        # 薄盘源和观测强度计算
  transfer.py       # 薄盘交点记录
```

## 脚本

生成 Schwarzschild 验证表：

```bash
python scripts/generate_validation_tables.py \
  --output outputs/schwarzschild_transfer.json \
  --samples 32
```

生成 Gralla-Holz-Wald Fig. 5 风格的薄盘强度剖面：

```bash
python scripts/generate_fig5_profiles.py \
  --profile all \
  --output outputs/fig5_profiles.csv \
  --plot-output outputs/fig5_profiles.png \
  --image-output outputs/fig5_images.png
```

生成 LQG Fig. 3 风格的剖面和 ring-edge 对比表：

```bash
python scripts/generate_lqg_fig3_profiles.py \
  --output outputs/lqg_fig3_profile.csv \
  --plot-output outputs/lqg_fig3_profile.png \
  --image-output outputs/lqg_fig3_image.png \
  --comparison-output outputs/lqg_fig3_ring_edges.csv
```

仓库中的 `outputs/` 目录已经包含预生成的 CSV 和 PNG 结果。

## 测试

```bash
pytest -q
```

测试覆盖 Schwarzschild 解析量、观测者映射、求解器事件、薄盘交点、红移权重、图像采样，以及公开的生成脚本。

## 说明

- 默认使用无量纲几何单位，通常取 `M = 1`。
- 正面薄盘图像的主要生产路径是 `QuadTransferSolver`。
- `OdeTrajectorySolver` 和 `HamiltonianTrajectorySolver` 主要用于轨迹诊断和交叉验证。

## 许可证

本项目使用 MIT License。详见 [LICENSE](LICENSE)。
