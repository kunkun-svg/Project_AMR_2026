"""读取两列 I/Q 样本，绘制时域波形和复数信号的双边 FFT 幅度谱。

采样率默认 4096 Hz；配套合成样本含 +300 Hz、-700 Hz 分量。
运行时加 --check-sample，可把实际测得的两个最大频点与预设分量核对。
"""

import argparse
import csv
from pathlib import Path
import sys

import matplotlib

# 使用非交互后端，终端运行即可保存图片，不需要弹出图形窗口。
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# 本文件位于 <项目>/Docs/Stage2_Outputs/code/，路径不依赖终端工作目录。
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = PROJECT_ROOT / "Docs/Stage2_Outputs/iq_fft.png"
AMPLITUDE_FLOOR = 1e-12  # 对数幅度下限，对应 -240 dB re 1。


def default_input() -> Path:
    """优先读项目根目录 Data；兼容当前资料包放在 Docs/Stage2 的布局。"""
    candidates = (
        PROJECT_ROOT / "Data/Stage2/sample_iq.csv",
        PROJECT_ROOT / "Docs/Stage2/Data/Stage2/sample_iq.csv",
    )
    return next((path for path in candidates if path.is_file()), candidates[0])


def load_iq(path: Path) -> np.ndarray:
    """检查 CSV 表头、列数和数值，并组合 x[n] = I[n] + j Q[n]。"""
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream), [])
        if [name.strip().upper() for name in header] != ["I", "Q"]:
            raise ValueError("CSV 表头必须依次为 I,Q。")
        data = np.loadtxt(stream, delimiter=",", dtype=float, ndmin=2)
    if data.shape[1] != 2 or data.shape[0] < 2:
        raise ValueError("CSV 必须包含两列 I、Q，且至少有两个采样点。")
    if not np.isfinite(data).all():
        raise ValueError("CSV 中存在 NaN 或无穷值。")

    # I 是同相分量（实部），Q 是正交分量（虚部）；二者共同保留幅相信息。
    return data[:, 0] + 1j * data[:, 1]


def compute_fft(iq: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """返回同时经过 fftshift 的频率轴和归一化复数频谱。"""
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("采样率必须是有限正数。")
    iq = np.asarray(iq, dtype=complex)
    if iq.ndim != 1 or iq.size < 2 or not np.isfinite(iq).all():
        raise ValueError("IQ 信号必须是一维、有限值且至少有两个采样点。")
    n = iq.size

    # 复数 IQ 必须使用完整 FFT，不能用只保留实信号单边谱的 rfft。
    # exp(+j*2*pi*f*t) 在复平面逆时针旋转，exp(-j*2*pi*f*t) 顺时针；
    # 这就是本约定下正、负频率的含义。复数谱一般不具有共轭对称性。
    # 默认 FFT 未归一化；除以 N 后，落在频点上的 A*exp(j*2*pi*f*t)
    # 对应峰值幅度为 |A|。双边复数幅度谱不再乘 2，也不按最大值缩放。
    spectrum = np.fft.fft(iq) / n
    frequency = np.fft.fftfreq(n, d=1.0 / fs)

    # FFT 原始顺序为零频、正频、负频；两者一起 shift 才能保持对应。
    # 偶数 N 时频率范围为 [-fs/2, fs/2)，零频移动到图中间。
    return np.fft.fftshift(frequency), np.fft.fftshift(spectrum)


def amplitude_db(amplitude: np.ndarray) -> np.ndarray:
    """幅度转换为 dB re 1；先设下限，避免 log10(0) 产生 -inf。"""
    return 20.0 * np.log10(np.maximum(amplitude, AMPLITUDE_FLOOR))


def plot_result(iq, fs, frequency, magnitude_db, top_bins, output):
    """保存 I/Q 时域波形和保留正、负频率的双边幅度谱。"""
    time_ms = np.arange(iq.size) / fs * 1000.0
    visible = min(96, iq.size)  # 放大前 96 点，使 I、Q 的振荡清晰可见。
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    axes[0].plot(time_ms[:visible], iq.real[:visible], label="I (real)")
    axes[0].plot(time_ms[:visible], iq.imag[:visible], label="Q (imag)", alpha=0.8)
    axes[0].set(
        xlabel="Time (ms)", ylabel="Amplitude",
        title=f"Complex I/Q waveform: first {visible} of {iq.size} samples",
    )
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.25)

    axes[1].plot(frequency, magnitude_db, linewidth=0.85, color="tab:blue")
    axes[1].scatter(frequency[top_bins], magnitude_db[top_bins],
                    color="tab:red", s=25, zorder=3)
    for index in top_bins:
        axes[1].annotate(
            f"{frequency[index]:+.1f} Hz\n{magnitude_db[index]:.2f} dB",
            (frequency[index], magnitude_db[index]),
            xytext=(0, 10), textcoords="offset points", ha="center", fontsize=9,
        )
    axes[1].set(
        xlabel="Frequency (Hz)", ylabel="Magnitude (dB re 1)",
        title=f"Two-sided FFT: |FFT(I + jQ)| / N, df = {fs / iq.size:g} Hz",
        xlim=(-fs / 2, fs / 2),
    )
    # 给峰值标注留出空间；全零输入也能得到正常坐标范围。
    lower = max(-240.0, float(np.min(magnitude_db)) - 5.0)
    axes[1].set_ylim(lower, float(np.max(magnitude_db)) + 28.0)
    axes[1].grid(alpha=0.25)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=default_input(),
                        help="输入 CSV，包含表头 I,Q；默认自动寻找配套样本。")
    parser.add_argument("--fs", type=float, default=4096.0, help="采样率，单位 Hz。")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="保存 PNG 的路径。")
    parser.add_argument("--check-sample", action="store_true",
                        help="核对两个最大频点是否匹配配套样本的 +300、-700 Hz。")
    args = parser.parse_args()

    try:
        iq = load_iq(args.input)
        frequency, spectrum = compute_fft(iq, args.fs)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    amplitude = np.abs(spectrum)
    magnitude_db = amplitude_db(amplitude)

    # 按实际计算幅度排序；预设频率不参与峰值选择或绘图。
    # 配套双音恰好落在 FFT 频点上，因此两个最大频点就是两个主峰。
    # 若换成非整周期信号，泄漏可能使相邻最大频点属于同一个主瓣。
    top_bins = np.argsort(amplitude)[-2:][::-1]
    plot_result(iq, args.fs, frequency, magnitude_db, top_bins, args.output)

    print(f"Python 解释器：{sys.executable}")
    print(f"Python 版本：{sys.version.split()[0]}")
    print(f"NumPy：{np.__version__}；Matplotlib：{matplotlib.__version__}")
    print(f"实际输入：{args.input.resolve()}")
    print(f"I/Q 数据形状：({iq.size}, 2)；复数信号形状：{iq.shape}")
    print(f"采样率：{args.fs:g} Hz；N={iq.size}；记录时长：{iq.size / args.fs:g} s")
    print(f"频率分辨率 df = fs/N = {args.fs / iq.size:g} Hz")
    print("实际计算的两个最高频谱峰（按幅度降序）：")
    for rank, index in enumerate(top_bins, start=1):
        print(f"  峰 {rank}：{frequency[index]:+.6f} Hz，"
              f"归一化幅度={amplitude[index]:.9f}，"
              f"幅度={magnitude_db[index]:+.6f} dB re 1")
    print(f"图片已保存：{args.output.resolve()}")

    if args.check_sample:
        # 预设值只用于事后核对；半个频点为容差，且不使用相对容差。
        measured = np.sort(frequency[top_bins])
        expected = np.array([-700.0, 300.0])
        tolerance = args.fs / iq.size / 2.0
        passed = bool(np.all(amplitude[top_bins] > AMPLITUDE_FLOOR)
                      and np.allclose(measured, expected, rtol=0, atol=tolerance))
        status = "通过" if passed else "失败"
        print(f"预设频率核对：{status}（预期 +300 Hz、-700 Hz；"
              f"容差 {tolerance:g} Hz）")
        return 0 if passed else 1
    return 0


# FFT 对整段信号给出一张频谱，不直接标明分量出现的时间；
# STFT 在滑动短窗内反复计算 FFT，输出“时间 × 频率”信息。
# STFT 短窗时间定位较好，长窗频率分辨能力较好，二者需要权衡。
if __name__ == "__main__":
    raise SystemExit(main())
