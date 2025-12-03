#!/opt/anaconda3/envs/p11/bin/python3

import os
import sys
import csv
import glob
import logging
import tempfile
import subprocess
import sqlite3
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, List, Set, Tuple
from collections import defaultdict

import numpy as np
from astropy.io import fits
import h5py

import sawtooth_newton as snt
from concurrent.futures import ProcessPoolExecutor, as_completed
import re


def worker_init() -> None:
    """ProcessPoolExecutor 用の初期化関数。ワーカープロセスごとにロガーをクリアする。"""
    root_logger = logging.getLogger()
    root_logger.handlers.clear()


def setup_object_logger(object_name: str, date_label: str) -> None:
    """object ごとに専用のログファイルをセットアップする。"""
    root_logger = logging.getLogger()

    # 既存の FileHandler を外してクローズ（多重追加・多重書き込み防止）
    for h in list(root_logger.handlers):
        if isinstance(h, logging.FileHandler):
            root_logger.removeHandler(h)
            h.close()

    log_dir = Path(WORK_DIR) / "logs" / date_label
    log_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / f"cross_corr_{object_name}.log"

    # mode="a" で毎回追記
    file_handler = logging.FileHandler(log_path, mode="a")
    formatter = logging.Formatter("%(asctime)s [PID %(process)d] %(levelname)s: %(message)s")
    file_handler.setFormatter(formatter)

    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)


load_dotenv("config.env")

DB_PATH = os.getenv("DB_PATH")
RAID_PC = os.getenv("RAID_PC")
RAID_DIR = os.getenv("RAID_DIR")
RAWDATA_DIR= os.getenv("RAWDATA_DIR")
WORK_DIR = os.getenv("WORK_DIR_ave")
KERNEL_CONFIG_DIR = os.getenv("KERNEL_CONFIG_DIR")

# Required local files and kernel configurations
REQUIRED_LOCAL_FILES = {
    "ar_name_csv": Path(__file__).parent / "ar_name.csv",
}

KERNEL_CONFIG = [
    ((0, 128), Path(KERNEL_CONFIG_DIR) / "kernel_with_ranges_64.npz"),
    ((128, 256), Path(KERNEL_CONFIG_DIR) / "kernel_with_ranges_192.npz"),
    ((256, 384), Path(KERNEL_CONFIG_DIR) / "kernel_with_ranges_320.npz"),
    ((384, 512), Path(KERNEL_CONFIG_DIR) / "kernel_with_ranges_448.npz"),
]

KERNEL_CACHE: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]] = {}


def _ensure_kernel_cache_loaded() -> None:
    """
    KERNEL_CONFIG で指定された npz を一度だけ読み込み、メモリ上にキャッシュする。
    各ワーカープロセスごとに最初に呼ばれたタイミングでロードされる。
    """
    global KERNEL_CACHE
    if KERNEL_CACHE:
        return

    cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
    for (ymin, ymax), kernel_path in KERNEL_CONFIG:
        kernel, fit_ranges, ar_features, mu_wavelength_pairs, meta = read_kernel_npz(kernel_path)
        cache[(ymin, ymax)] = (kernel, mu_wavelength_pairs)
    KERNEL_CACHE = cache


def get_kernel_for_raw_idx(raw_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    raw_idx に対応する y 範囲のカーネルを KERNEL_CACHE から取得する。
    対応する範囲がなければエラーとしてログを出して終了する。
    """
    _ensure_kernel_cache_loaded()
    for (ymin, ymax), _ in KERNEL_CONFIG:
        if ymin <= raw_idx < ymax:
            return KERNEL_CACHE[(ymin, ymax)]
    logging.error(f"raw_idx={raw_idx} is out of KERNEL_CONFIG range")
    sys.exit(1)


def validate_environment() -> None:
    """
    必要な環境変数・ディレクトリ・ローカルファイルがそろっているかをチェックする。
    問題があればエラーメッセージを出して終了する。
    """
    errors = []

    # 環境変数のチェック
    env_vars = [
        ("DB_PATH", DB_PATH),
        ("RAID_PC", RAID_PC),
        ("RAID_DIR", RAID_DIR),
        ("RAWDATA_DIR", RAWDATA_DIR),
        ("WORK_DIR", WORK_DIR),
        ("KERNEL_CONFIG_DIR", KERNEL_CONFIG_DIR),
    ]

    for name, value in env_vars:
        if not value:
            errors.append(f"環境変数 {name} が設定されていません。")

    # DB の存在確認
    if DB_PATH:
        db_path = Path(DB_PATH)
        if not db_path.exists():
            errors.append(f"DB_PATH で指定されたデータベースファイルが存在しません: {db_path}")

    # ローカル必須ファイルの存在確認
    for desc, path in REQUIRED_LOCAL_FILES.items():
        if not path.exists():
            errors.append(f"必須ファイルが見つかりません ({desc}): {path}")

    # 作業ディレクトリの存在確認（存在しなければエラーとする）
    for name, value in [("RAWDATA_DIR", RAWDATA_DIR), ("WORK_DIR", WORK_DIR)]:
        if value:
            d = Path(value)
            if not d.exists():
                errors.append(f"{name} で指定されたディレクトリが存在しません: {d}")

    # カーネル npz ファイルの存在確認
    for (ymin, ymax), kernel_path in KERNEL_CONFIG:
        if not kernel_path.exists():
            errors.append(f"kernel npz が見つかりません (y-range [{ymin}, {ymax})): {kernel_path}")

    if errors:
        for msg in errors:
            logging.error(msg)
        # どれか 1 つでも問題があれば即終了
        sys.exit(1)


def db_search(conn: sqlite3.Connection, object_name, date_label=None) -> Dict[str, List[str]]:
    
    """
    framesテーブルからAr系のframeを抽出し、
    {date_label: [base_name, ...]} の辞書を返す。
    """
    if date_label is None:
        query = (
            "SELECT date_label, base_name "
            "FROM frames "
            f"WHERE object = '{object_name}' "
            f"AND filepath COLLATE NOCASE LIKE '%/spec%'"
        )
    else:
        query = (
            "SELECT date_label, base_name "
            "FROM frames "
            f"WHERE object = '{object_name}' "
            f"AND date_label = '{date_label}' "
            f"AND filepath COLLATE NOCASE LIKE '%/spec%'"
        )
    cur = conn.cursor()
    logging.debug(f"db_search SQL for object={object_name}, date_label={date_label}: {query}")
    cur.execute(query)

    filepath_dict: Dict[str, List[str]] = {}
    rows = cur.fetchall()
    #logging.info(f"db_search: object={object_name}, date_label={date_label}, hits={len(rows)}")
    for date_label, base_name in rows:
        filepath_dict.setdefault(date_label, []).append(base_name)
    return filepath_dict


def do_scp_raw_fits(date_label: str, object_name: str, base_name_list: List[str]) -> None:
    # Extract Num1 from basenames
    num1_set = set()
    for bn in base_name_list:
        m = re.match(r"spec\d{6}-(\d{4})\.fits", bn)
        if m:
            num1_set.add(m.group(1))

    # If nothing matched, do nothing
    if not num1_set:
        logging.warning(f"No valid Num1 found in base_name_list for {date_label}")
        return

    num1_list = sorted(num1_set)
    num_min = int(num1_list[0])
    num_max = int(num1_list[-1])

    dst_dir = Path(RAWDATA_DIR) / object_name / date_label
    dst_dir.mkdir(parents=True, exist_ok=True)

    # シェルスクリプトを即席で作って、bash で実行する   
    script_content = f"""#!/bin/bash
set -euo pipefail
src="{RAID_PC}:{RAID_DIR}/{date_label}/spec/spec{date_label}-{{{num_min:04d}..{num_max:04d}}}.fits"
dst="{dst_dir}"
mkdir -p "$dst"
echo "scp $src $dst"
scp {RAID_PC}:{RAID_DIR}/{date_label}/spec/spec{date_label}-{{{num_min:04d}..{num_max:04d}}}.fits "$dst"
"""

    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".sh") as tmp:
        script_path = tmp.name
        tmp.write(script_content)
    result = subprocess.run(["bash", script_path], capture_output=True, text=True)

    logging.info(f"scp_raw_fits: scp command end")


def do_remove_raw_fits(date_label: str, object_name: str):
    target_dir = Path(RAWDATA_DIR) / object_name / date_label
    cmd = ["rm", "-f", str(target_dir / "*.fits")]
    subprocess.run(" ".join(cmd), shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)



def center_row_cut_ar_fits(fits_path, ymin, ymax):
    hdul = fits.open(fits_path)
    data = hdul[0].data
    hdul.close()
    data = data[ymin:ymax, :]
    center_row = data[data.shape[0] // 2, :]
    return center_row


def chose_row_cut_ar_fits(fits_path, raw_idx):
    hdul = fits.open(fits_path)
    data = hdul[0].data
    hdul.close()
    center_row = data[raw_idx, :]
    return center_row


def read_kernel_npz(npz_path):
    with np.load(npz_path, allow_pickle=True) as data:
        kernel = data["kernel"]
        fit_ranges = data["fit_ranges"]
        ar_features = data["ar_features"]
        mu_wavelength_pairs = data["mu_wavelength_pairs"]
        meta = data["meta"]
    return kernel, fit_ranges, ar_features, mu_wavelength_pairs, meta


def get_cross_corr(ar_data, kernel):
    # 平均を引いて正規化
    ar_data = (ar_data - np.mean(ar_data)) / np.std(ar_data)
    kernel = (kernel - np.mean(kernel)) / np.std(kernel)

    # 相互相関を計算（full モードで全シフトを考慮）
    corr = np.correlate(ar_data, kernel, mode='valid')

    # シフト量を計算
    lags = np.arange(0, len(ar_data) - len(kernel) + 1)

    # 最大相関の位置を求める
    max_idx = np.argmax(corr)
    best_lag = max_idx
    max_corr = corr[max_idx]

    # プロット（確認用）
    # plt.figure()
    # plt.plot(lags, corr)
    # plt.title(f"Cross-correlation (best lag = {best_lag})")
    # plt.xlabel("Lag")
    # plt.ylabel("Correlation")
    # plt.show()

    return corr, best_lag, max_corr

    


def get_sawtooth_center(center_row, mu_wavelength_pairs, best_lag):
    """
    def newton_method(
    y: np.ndarray,
    x0: float,
    params: Optional[CenterParams] = None,
    ) -> CenterResult:
    """
    snt_results = []
    lambdas = []
    for mu_wavelength_pair in mu_wavelength_pairs:
        #print(f"mu_wavelength_pair: {mu_wavelength_pair}")
        mu_pix, wl = mu_wavelength_pair
        mu_pix_crossed_for_python = mu_pix - 1 + best_lag
        snt_result = snt.newton_method(center_row, mu_pix_crossed_for_python)
        logging.debug(f"sawtooth center result at mu_pix={mu_pix}, wl={wl}: {snt_result}")
        snt_results.append(snt_result)
        lambdas.append(wl)


    return snt_results, lambdas


def write_h5py(h5py_path, header, lambdas, pixpos, converged, pix_vals):
    object_name = header.get("OBJECT", "")
    mjd = header.get("MJD", "")
    offra = header.get("OFFSETRA", "")
    offde = header.get("OFFSETDE", "")
    offro = header.get("OFFSETRO", "")
    azi   = header.get("AZIMUTH", "")
    alt   = header.get("ALTITUDE", "")
    rot   = header.get("ROTATOR", "")

    # collect IMCMB*** cards and NCOMBINE from header
    imcmb_items = []
    ncombine = None
    for key in header:
        if key.startswith("IMCMB"):
            imcmb_items.append((key, header[key]))
        elif key == "NCOMBINE":
            ncombine = header[key]

    # sort IMCMB by key (IMCMB001, IMCMB002, ...)
    imcmb_items.sort(key=lambda kv: kv[0])
    imcmb_values = [v for _, v in imcmb_items]

    with h5py.File(h5py_path, "a") as f:
        header_grp = f.require_group("header")

        # store main header values as attributes on /header
        header_grp.attrs["object_name"] = object_name
        header_grp.attrs["mjd"] = mjd
        header_grp.attrs["offra"] = offra
        header_grp.attrs["offde"] = offde
        header_grp.attrs["offro"] = offro
        header_grp.attrs["azi"] = azi
        header_grp.attrs["alt"] = alt
        header_grp.attrs["rot"] = rot

        # IMCMB dataset as a single string (newline-separated), always recreate
        imcmb_str = "\n".join(str(v) for v in imcmb_values) if imcmb_values else ""
        if "IMCMB" in header_grp:
            del header_grp["IMCMB"]
        dt = h5py.string_dtype(encoding="utf-8")
        header_grp.create_dataset("IMCMB", data=imcmb_str, dtype=dt)

        # NCOMBINE as an attribute on /header (always present)
        if ncombine is not None:
            header_grp.attrs["NCOMBINE"] = int(ncombine)
        else:
            header_grp.attrs["NCOMBINE"] = 1

        # keep lambdas as a root attribute
        f.attrs["lambdas"] = lambdas

        # save 8 x N_y array of xc values
        if "pixpos" in f:
            del f["pixpos"]
        f.create_dataset("pixpos", data=pixpos)
        if "converged" in f:
            del f["converged"]
        f.create_dataset("converged", data=converged)
        if "pix_vals" in f:
            del f["pix_vals"]
        f.create_dataset("pix_vals", data=pix_vals)


def crosscorr_roop(fits_path, h5py_path, window_size=31):

    # ヘッダと画像データを一度だけメモリ上に展開する
    header = fits.getheader(fits_path)
    with fits.open(fits_path, memmap=False) as hdul:
        data = hdul[0].data

    y_indices = list(range(10, 500))
    pix_vals = np.empty((len(y_indices), 8 * window_size), dtype=np.float32)
    pixpos = np.empty((8, len(y_indices)), dtype=np.float32)
    converged = np.empty((8, len(y_indices)), dtype=bool)
    lambdas = None  # ループ内で毎回更新されるが、最終的に最後の値を保存する点は従来実装と同じ

    for j, raw_idx in enumerate(y_indices):
        #logging.info(f"processing raw_idx={raw_idx} in {fits_path}")
        # data から直接必要な行を取り出す（FITS を毎回開き直さない）
        center_row = data[raw_idx, :]

        # raw_idx に対応するカーネルをメモリキャッシュから取得
        kernel, mu_wavelength_pairs = get_kernel_for_raw_idx(raw_idx)

        corr, best_lag, max_corr = get_cross_corr(center_row, kernel)
        snt_result, lambdas = get_sawtooth_center(center_row, mu_wavelength_pairs, best_lag)

        xcs = np.array([r.xc for r in snt_result])  # 8 個（0始まりの float とみなす）
        xcs_int = np.rint(xcs).astype(int)

        # 窓の開始インデックス（中心から window_size//2 分ずらす）
        half = window_size // 2
        starts = xcs_int - half

        # 先頭側(start<0)・けつ側(start+window_size>len(center_row))のはみ出しをまとめてクリップ
        max_start = center_row.size - window_size
        starts = np.clip(starts, 0, max_start)
        # shape = (8, window_size) のインデックスを作って一気にスライス
        idx = starts[:, None] + np.arange(window_size)
        win_segments = center_row[idx]  # (8, window_size)

        # 1start で保存（従来どおり）
        pixpos[:, j] = np.array([r.xc + 1 for r in snt_result], dtype=np.float32)
        converged[:, j] = np.array([r.converged for r in snt_result], dtype=bool)

        # C案: 8本分を1次元に連結して N_y x (8*window_size)
        pix_vals[j, :] = win_segments.ravel()
    write_h5py(h5py_path, header, lambdas, pixpos, converged, pix_vals)



def _hdr_rounded_int(hdr, key: str):
    """FITS ヘッダから key を取り出し、数値なら四捨五入した整数にして返す。
    数値に変換できなければ、そのままの値を返す。
    """
    val = hdr.get(key, "")
    try:
        return int(round(float(val)))
    except (TypeError, ValueError):
        return val

def work_per_date_label(object_name: str, date_label: str, base_name_list: List[str]) -> None:
    """date_label ごとに処理を行うワーカー関数。

    base_name_list に含まれるすべての FITS のヘッダーを読み込み、
    (OBJECT, OFFSETRA, OFFSETDE, OFFSETRO, AZIMUTH, ALTITUDE, ROTATOR)
    が同じもの同士でグループ分けして平均画像を作る。

    グループごとに 1 枚の合成 FITS を作成し、そのヘッダに
    IMCMB001, IMCMB002, ... と NCOMBINE を追加してから、
    その合成 FITS を crosscorr_roop に渡して .h5 を 1 つ作る。
    """
    setup_object_logger(object_name, date_label)

    # 必要な生 FITS をいったんローカルにコピー
    do_scp_raw_fits(date_label, object_name, base_name_list)

    # ヘッダーのキーでグループ分け
    groups = defaultdict(list)  # key: header tuple, value: list[(base_name, fits_path)]
    for base_name in base_name_list:
        fits_path = Path(RAWDATA_DIR) / object_name / date_label / base_name
        if not fits_path.exists():
            logging.warning(f"{fits_path} does not exist. skipping.")
            continue
        try:
            with fits.open(fits_path, memmap=False) as hdul:
                hdr = hdul[0].header
        except OSError as e:
            logging.warning(f"failed to open {fits_path}: {e}. skipping.")
            continue
        key = (
            hdr.get("OBJECT", ""),
            _hdr_rounded_int(hdr, "OFFSETRA"),
            _hdr_rounded_int(hdr, "OFFSETDE"),
            _hdr_rounded_int(hdr, "OFFSETRO"),
            _hdr_rounded_int(hdr, "AZIMUTH"),
            _hdr_rounded_int(hdr, "ALTITUDE"),
            _hdr_rounded_int(hdr, "ROTATOR"),
        )
        groups[key].append((base_name, fits_path))

    # グループごとに平均画像を作成し、合成 FITS を crosscorr_roop に渡す
    for key, members in groups.items():
        if not members:
            continue

        # 代表名: グループ内の最初の base_name をもとに合成 FITS 名を決める
        first_base_name, _ = members[0]
        # 例: specYYYYMMDD-0001.fits -> specYYYYMMDD-0001_cmbNN.fits
        cmb_suffix = f"_cmb{len(members):02d}.fits"
        if first_base_name.lower().endswith(".fits"):
            cmb_base_name = first_base_name[:-5] + cmb_suffix
        else:
            cmb_base_name = first_base_name + cmb_suffix

        cmb_fits_path = Path(RAWDATA_DIR) / object_name / date_label / cmb_base_name

        if not cmb_fits_path.exists():
            data_list = []
            header_ref = None
            imcmb_values = []

            # メンバーを名前順にそろえておく
            for idx, (base_name, fits_path) in enumerate(sorted(members, key=lambda x: x[0]), start=1):
                with fits.open(fits_path, memmap=False) as hdul:
                    if header_ref is None:
                        header_ref = hdul[0].header.copy()
                    data_list.append(hdul[0].data.astype(np.float64))
                imcmb_values.append(base_name.rsplit(".fits", 1)[0])

            if not data_list:
                logging.warning(f"no data to combine for group key={key}. skipping.")
                continue

            stack = np.stack(data_list, axis=0)
            combined = np.mean(stack, axis=0)

            # ヘッダーに IMCMBnnn と NCOMBINE を追加
            for idx, name_no_ext in enumerate(imcmb_values, start=1):
                card_key = f"IMCMB{idx:03d}"
                header_ref[card_key] = (name_no_ext, "combined frame")
            header_ref["NCOMBINE"] = (len(imcmb_values), "number of frames combined")
            header_ref["BITPIX"] = (-32, "data type")

            hdu = fits.PrimaryHDU(data=combined.astype(np.float32), header=header_ref)
            cmb_fits_path.parent.mkdir(parents=True, exist_ok=True)
            hdu.writeto(cmb_fits_path, overwrite=True)
            logging.info(
                f"created combined FITS {cmb_fits_path} from {len(imcmb_values)} frames"
            )
        else:
            logging.info(f"combined FITS {cmb_fits_path} already exists. reuse it.")

        # 合成 FITS に対して crosscorr_roop を実行し、グループごとに 1 つの h5 を作る
        h5py_filename = cmb_base_name.replace(".fits", ".h5")
        h5py_path = Path(WORK_DIR) / date_label / object_name / h5py_filename
        h5py_path.parent.mkdir(parents=True, exist_ok=True)

        if h5py_path.exists():
            logging.warning(f"{h5py_path} already exists. skipping.")
            continue

        logging.info(
            f"processing combined FITS {cmb_fits_path} (group size={len(members)})"
        )
        crosscorr_roop(cmb_fits_path, h5py_path)

    # 生 FITS は最後にまとめて削除
    do_remove_raw_fits(date_label, object_name)



def get_object_list(path: Path):
    objects = []
    with open(path, newline='') as f:
        reader = csv.reader(f)
        first = next(reader, None)

        # BOM / 空ヘッダー処理
        if first and first[0].startswith("\ufeff"):
            first[0] = first[0].lstrip("\ufeff")
        if first and first[0] == "":
            first[0] = "index"

        for row in reader:
            if not row:
                continue
            if row[0].startswith("#"):
                continue
            # 2列目に object がある前提で取得
            if len(row) >= 2:
                objects.append(row[1])
    return objects



def main():
    # 必須ファイル・環境のチェック
    validate_environment()

    csv_path = REQUIRED_LOCAL_FILES["ar_name_csv"]
    conn = sqlite3.connect(DB_PATH)

    logging.info(f"reading object list from {csv_path}")
    objects = get_object_list(csv_path)

    jobs = []
    total_objects = len(objects)
    logging.info(f"number of objects from csv: {total_objects}")
    logging.info(f"starting db_search for {total_objects} objects")
    for idx, object_name in enumerate(objects, 1):
        fits_dict = db_search(conn, object_name)
        for date_label, base_name_list in fits_dict.items():
            jobs.append((object_name, date_label, base_name_list))
    conn.close()

    with ProcessPoolExecutor(max_workers=10, initializer=worker_init) as ex:
        future_to_job = {
            ex.submit(work_per_date_label, object_name, date_label, base_name_list): (object_name, date_label)
            for object_name, date_label, base_name_list in jobs
        }
        total = len(future_to_job)
        done = 0
        for fut in as_completed(future_to_job):
            object_name, date_label = future_to_job[fut]
            done += 1
            sys.stdout.write(f"[{done}/{total}] finished {object_name} {date_label}\n")
            sys.stdout.flush()
            fut.result()

    


if __name__ == "__main__":
    print(f"Start {sys.argv[0]}", flush=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
        force=True,  # ← これを追加
    )
    main()