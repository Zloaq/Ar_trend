#!/opt/anaconda3/envs/p11/bin/python3
"""sawtooth（テント型）× Newton 法によるピークセンタリング用モジュール

このファイルは、将来実装を埋められるように **型付きのきれいなインターフェイス**だけを定義します。
提供する主な要素は次のとおりです：
- 線形補間 I(x) を返すラッパー
- テント型 sawtooth カーネルの生成
- ゼロ交差で中心を求める目的関数 F(x) の構築
- Newton 法（安全策として二分法フォールバック）での中心探索

"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple, Optional, List

import numpy as np



# ---------------------------------------------------------------------------
# データ構造
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Kernel:
    """鋸歯状（テント型）カーネルの **連続関数**表現。

    属性
    ------
    a : float
        半幅（a = width / 2）。
    f : Callable[[float], float]
        連続カーネル f(u)。u は相対座標（ピクセル単位）。
    """

    a: float
    f: Callable[[float], float]


@dataclass(frozen=True)
class CenterParams:
    """ピークセンタリングを制御するパラメータ群。

    fwidth : float
        特徴（ライン）の全幅［ピクセル］。典型的には 3–5。
    tol : float
        収束判定の許容誤差［ピクセル］（|Δx| < tol で収束）。
    max_iter : int
        Newton の最大反復回数。
    min_range : float
        窓内ダイナミックレンジがこの値未満ならセンタリングを棄却。
    """

    fwidth: float = 4.0
    tol: float = 1e-2
    max_iter: int = 20
    max_shift: float = 1.0
    min_range: float = 0.0
    A: float = 1.5 * fwidth
    B: float = 0.75 * fwidth


@dataclass(frozen=True)
class CenterResult:
    """センタリングの結果（推定値・収束状況など）。

    属性
    ------
    xc : float
        推定されたサブピクセル中心（ピクセル単位）。
    converged : bool
        収束判定に成功したか。
    n_iter : int
        実行した反復回数。
    """

    xc: float
    converged: bool
    n_iter: int




def interp_data(y: np.ndarray) -> Callable[[float], float]:
    """離散スペクトル配列 y から、実数座標 x で評価できる I(x) を返す。

    パラメータ
    ----------
    y : np.ndarray
        整数ピクセル座標 [0, 1, ..., N-1] にサンプリングされた 1D スペクトル。

    戻り値
    ------
    Callable[[float], float]
        端をクランプした **線形補間**の I(x) を返す関数。
    """
    y = np.asarray(y, dtype=float)
    n = y.size

    def I(x: float) -> float:
        if x <= 0:
            return float(y[0])
        if x >= n - 1:
            return float(y[-1])
        i = int(np.floor(x))
        t = x - i
        return float((1.0 - t) * y[i] + t * y[i + 1])

    return I


def create_sawtooth(width: float) -> Kernel:
    """テント型の sawtooth カーネル f(u) を連続関数として生成する。

    形状は基準点: -a→0, -a/2→-1, 0→0, a/2→+1, a→0（スケール任意）。

    パラメータ
    ----------
    width : float
        カーネルの全幅（ピクセル）。

    戻り値
    ------
    Kernel
        連続カーネルのラッパー（a, f）。
    """
    a = width / 2.0
    def f(u: float) -> float:
        if u <= -a or u >= a:
            return 0.0
        elif -a <= u < -a/2:
            return -(2.0 / a) * (u + a)
        elif -a/2 <= u < a/2:
            return (2.0 / a) * u
        else:  # a/2 <= u < a
            return -(2.0 / a) * (u - a)

    return Kernel(a=a, f=f)



def make_objective(
    I: Callable[[float], float],
    kernel: Kernel,
    x0: int,#これは 0-start
    params: CenterParams,
    ) -> Callable[[float], float]:
    """目的関数 F(x) を構築して返す：

    F(x) = ∫ (I(ξ) - I0) * f(ξ - x) dξ

    基線 I0 は [x0 - window, x0 + window] でロバスト推定。発光/吸収の扱いもここで適用。
    外部インターフェイスは連続関数のみを扱い、**刻み幅は公開しない**（内部実装で自動選択）。
    """
    a = kernel.a
    
    # ロバスト基線推定（窓内メディアン）
    L_A = x0 - params.A
    R_A = x0 + params.A
    grid = np.linspace(L_A, R_A, int(2 * params.A) + 1)
    vals = np.array([I(xx) for xx in grid])
    I0 = float(vals.min())


    # 連続カーネル f(u)
    h = params.tol / 10.0  # 例: tol=1e-3 → h=1e-4
    nseg = int(np.ceil((2 * a) / h))
    u = np.linspace(-a, a, nseg + 1)
    f_vals = np.array([kernel.f(uu) for uu in u])
    

    # 内部数値積分（自動で十分細かい分割数を選ぶ）
    def F(x: float) -> float:
        # u ∈ [-a, a] を自動分割（幅に比例して増やす）。
        # 外部 API には刻みを見せない。
        I_vals = np.array([I(x + uu) for uu in u])
        integrand = (I_vals - I0) * f_vals
        #integrand = np.array([I(x + uu) * f(uu) for uu in u])
        return float(np.trapz(integrand, u))

    return F


def create_sawtooth_gradient(width: float) -> Kernel:
    """テント型カーネルの **勾配（x 微分）** g(u) を生成する。

    元のテント型カーネル f(u) の傾きは区間ごとに一定なので、
    その勾配 g(u) は

        u ∈ [-a,   -a/2) → -2/a
        u ∈ [-a/2,  a/2) →  2/a
        u ∈ [ a/2,  a   ) → -2/a
        それ以外           →  0

    パラメータ
    ----------
    width : float
        カーネルの全幅（ピクセル）。a = width / 2.

    戻り値
    ------
    Kernel
        連続カーネルのラッパー（a, g）。
    """
    a = width / 2.0

    def f(u: float) -> float:
        if u <= -a or u >= a:
            return 0.0
        elif -a <= u < -a / 2.0:
            return -(2.0 / a)
        elif -a / 2.0 <= u < a / 2.0:
            return 2.0 / a
        else:  # a/2 <= u < a
            return -(2.0 / a)

    return Kernel(a=a, f=f)


def make_objective_gradient(
    I: Callable[[float], float],
    kernel: Kernel,
    x0: int,  # これは 0-start
    params: CenterParams,
) -> Callable[[float], float]:
    """目的関数の勾配に相当する F_grad(x) を構築して返す。

    F_grad(x) = F'(x) = -∫ (I(ξ) - I0) * g(ξ - x) dξ

    ここで g はテント型カーネルの勾配（create_sawtooth_gradient で生成）。

    基線 I0 は [x0 - window, x0 + window] でロバスト推定。発光/吸収の扱いも
    make_objective と同様にここで適用する。
    外部インターフェイスは連続関数のみを扱い、刻み幅は公開しない。
    """

    # ロバスト基線推定（窓内メディアン）
    L_A = x0 - params.A
    R_A = x0 + params.A
    grid = np.linspace(L_A, R_A, int(2 * params.A) + 1)
    vals = np.array([I(xx) for xx in grid])
    I0 = float(vals.min())

    a = kernel.a
    h = params.tol / 10.0 
    nseg = int(np.ceil((2 * a) / h))
    u = np.linspace(-a, a, nseg + 1)
    f_vals = np.array([kernel.f(uu) for uu in u])

    # 内部数値積分（自動で十分細かい分割数を選ぶ）
    def F_grad(x: float) -> float:
        # u ∈ [-a, a] を自動分割（幅に比例して増やす）。
         # 例: tol=1e-3 → h=1e-4
        I_vals = np.array([I(x + uu) for uu in u])
        integrand = (I_vals - I0) * f_vals
        #integrand = np.array([I(x + uu) * f(uu) for uu in u])
        return float(np.trapz(integrand, u))

    return F_grad



def newton_method(
    y: np.ndarray,
    x0: int,#0-start
    params: Optional[CenterParams] = None,
    ) -> CenterResult:
    """初期値 x0 近傍で Newton 法により中心を求める。

    パラメータ
    ----------
    y : np.ndarray
        整数ピクセルにサンプリングされた 1D スペクトル。
    x0 : float
        初期推定（ピクセル単位）。
    params : Optional[CenterParams]
        カーネル/窓/許容誤差などの制御パラメータ（未指定なら既定値）。

    戻り値
    ------
    CenterResult
        推定中心と診断情報。
    """
    if params is None:
        params = CenterParams()

    I = interp_data(y)

    # ロバスト基線推定（窓内メディアン）
    L_B = x0 - params.B
    R_B = x0 + params.B

    #y = y - params.mu_noise
    kernel = create_sawtooth(params.fwidth)
    F = make_objective(I, kernel, x0, params)
    grad_kernel = create_sawtooth_gradient(params.fwidth)
    F_grad = make_objective_gradient(I, grad_kernel, x0, params)


    #xs = np.linspace(L_B, R_B, 41)
    #Fs = np.array([F(xx) for xx in xs])

    bracket = None
    """
    #意味ない
    # 近傍最小 |F| の格子点（フォールバック用シード）
    kmin = int(np.argmin(np.abs(Fs)))
    # 即時ゼロにはしない：局所解の誤認や粗格子のアーチファクトを避ける

    # 符号反転の検出
    bracket = None
    for i in range(len(xs) - 1):
        # 格子点での厳密ゼロは即時返さず、微小幅ブラケットを作る
        if Fs[i] == 0.0:
            xi = float(xs[i])
            epsx = max(params.tol, 1e-3 * kernel.a)
            bracket = (max(L, xi - epsx), min(R, xi + epsx))
            break
        if Fs[i] * Fs[i + 1] < 0.0:
            bracket = (float(xs[i]), float(xs[i + 1]))
            break
    """

    # Newton 本体
    x = float(x0)
    n_iter = 0
    converged = False
    method = "newton"

    for n_iter in range(1, params.max_iter + 1):
        #print(n_iter)
        Fx = F(x)
        # 勾配カーネルから直接 F'(x) を評価
        #newton 法とは異なる設計だが、これでうまくいく。
        #方向は Fx に押し付け、Fpはスケーリングのみ
        Fp = abs(F_grad(x))
        if Fp < 1e-12:
            # 勾配がほぼゼロのときは、符号だけで小さい固定ステップ
            dx = np.sign(Fx) * min(params.tol, params.max_shift)
        else:
            dx = Fx / Fp
        #print(f"dx: {dx} Fx: {Fx} Fp: {Fp}")
        # ダンプ（大ジャンプ抑制）
        if abs(dx) > params.B or abs(dx) > params.max_shift:
            dx = np.sign(dx) * params.max_shift
        x_new = x + dx
        x_new = float(np.clip(x_new, L_B, R_B))
        if abs(x_new - x) < params.tol:
            x = x_new
            converged = True
            break
        x = x_new
        #fig, ax1 = plt.subplots(figsize=(12, 8))
        #test_plot(fig, ax1, x, n_iter, y)

    Fx_final = F(x)

    sigma_x = None

    """
    #理想的にsolidな輝線でないので意味ない
    if converged and params.sigma > 0.0:
        # 最終位置 x での傾き F'(x) を「勾配カーネル」で評価する
        Fp_final = F_grad(x)

        if np.isfinite(Fp_final) and abs(Fp_final) > 1e-12:
            # 正しいセンタリング誤差 Var(dx) = σ² Σ g_i² / (F')², g は勾配カーネル
            a_grad = grad_kernel.a
            nseg = max(32, int(np.ceil(20 * a_grad)))
            u = np.linspace(-a_grad, a_grad, nseg + 1)
            g_vals = np.array([grad_kernel.f(uu) for uu in u])
            sum_g2 = float(np.sum(g_vals ** 2))
            var_x = (params.sigma ** 2) * sum_g2 / (Fp_final ** 2)
            sigma_x = float(np.sqrt(var_x))
    """


    return CenterResult(
        xc=float(x),
        converged=bool(converged),
        n_iter=int(n_iter),
    )


__all__ = [
    "Kernel",
    "CenterParams",
    "CenterResult",
    "interp_data",
    "create_sawtooth",
    "make_objective",
    "create_sawtooth_gradient",
    "make_objective_gradient",
    "newton_method",
]


if __name__ == "__main__":
    # このモジュールはテストやノートブックからのインポート利用を想定。
    pass