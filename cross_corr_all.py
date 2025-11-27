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

import numpy as np
from astropy.io import fits
import h5py

import sawtooth_newton as snt
from concurrent.futures import ProcessPoolExecutor, as_completed
import re

def setup_worker_logger() -> None:
    """各ワーカープロセスごとに専用のログファイルをセットアップする。"""
    root_logger = logging.getLogger()
    # すでにファイルハンドラが付いている場合は何もしない（多重追加防止）
    has_file_handler = any(isinstance(h, logging.FileHandler) for h in root_logger.handlers)
    if has_file_handler:
        return

    # ログ出力ディレクトリとファイルパス
    log_dir = Path(WORK_DIR) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    pid = os.getpid()
    log_path = log_dir / f"cross_corr_worker_{pid}.log"

    file_handler = logging.FileHandler(log_path)
    formatter = logging.Formatter("%(asctime)s [PID %(process)d] %(levelname)s: %(message)s")
    file_handler.setFormatter(formatter)

    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)


def worker_init() -> None:
    """ProcessPoolExecutor 用の初期化関数。ワーカープロセスごとにロガーを設定する。"""
    root_logger = logging.getLogger()
    # 親プロセスから継承したハンドラをクリアしてから専用ログを付ける
    root_logger.handlers.clear()
    setup_worker_logger()

load_dotenv("config.env")

DB_PATH = os.getenv("DB_PATH")
RAID_PC = os.getenv("RAID_PC")
RAID_DIR = os.getenv("RAID_DIR")
RAWDATA_DIR= os.getenv("RAWDATA_DIR")
WORK_DIR = os.getenv("WORK_DIR")
KERNEL_CONFIG_DIR = os.getenv("KERNEL_CONFIG_DIR")

# Required local files and kernel configurations
REQUIRED_LOCAL_FILES = {
    "ar_name_csv": Path(__file__).parent / "ar_name.csv",
}

KERNEL_CONFIG = [
    ((0, 128), Path(KERNEL_CONFIG_DIR) / "kernel_average_result_y63_x140-410_20251002-020351.npz"),
    ((128, 256), Path(KERNEL_CONFIG_DIR) / "kernel_average_result_y191_x180-458_20251002-021431.npz"),
    ((256, 384), Path(KERNEL_CONFIG_DIR) / "kernel_average_result_y319_x231-512_20251002-154330.npz"),
    ((384, 512), Path(KERNEL_CONFIG_DIR) / "kernel_average_result_y447_x280-562_20251002-154827.npz"),
]


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
        )
    else:
        query = (
            "SELECT date_label, base_name "
            "FROM frames "
            f"WHERE object = '{object_name}' "
            f"AND date_label = '{date_label}' "
        )
    cur = conn.cursor()
    logging.debug(f"db_search SQL for object={object_name}, date_label={date_label}: {query}")
    cur.execute(query)

    filepath_dict: Dict[str, List[str]] = {}
    rows = cur.fetchall()
    logging.info(f"db_search: object={object_name}, date_label={date_label}, hits={len(rows)}")
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
    script_content = f"""
    #!/bin/bash

    src="{RAID_PC}:{RAID_DIR}/{date_label}/spec/spec{date_label}\*-{{{num_min:04d}..{num_max:04d}}}.fits"
    dst="{dst_dir}"
    mkdir -p "$dst"
    echo "scp $src $dst"
    scp $src "$dst"
    """

    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".sh") as tmp:
        script_path = tmp.name
        tmp.write(script_content)
    result = subprocess.run(["bash", script_path], capture_output=True, text=True)


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


def write_h5py(h5py_path, header, lambdas, pixpos):
    object_name = header.get("OBJECT", "")
    mjd = header.get("MJD", "")
    offra = header.get("OFFSETRA", "")
    offde = header.get("OFFSEDE", "")
    offro = header.get("OFFSETRO", "")
    azi = header.get("AZIMUTH", "")
    alt = header.get("ALTITUDE", "")
    rot = header.get("ROTATOR", "")

    with h5py.File(h5py_path, "a") as f:
        f.attrs["object_name"] = object_name
        f.attrs["mjd"] = mjd
        f.attrs["offra"] = offra
        f.attrs["offde"] = offde
        f.attrs["offro"] = offro
        f.attrs["azi"] = azi
        f.attrs["alt"] = alt
        f.attrs["rot"] = rot
        f.attrs["lambdas"] = lambdas
        # save 8 x N_y array of xc values
        if "pixpos" in f:
            del f["pixpos"]
        f.create_dataset("pixpos", data=pixpos)


def crosscorr_roop(fits_path, h5py_path):

    header = fits.getheader(fits_path)
    y_indices = list(range(10, 501))
    pixpos = np.empty((8, len(y_indices)), dtype=np.float32)

    for j, raw_idx in enumerate(y_indices):
        logging.info(f"processing raw_idx={raw_idx} in {fits_path}")
        center_row = chose_row_cut_ar_fits(fits_path, raw_idx)
        kernel_path = None
        for (ymin, ymax), path in KERNEL_CONFIG:
            if ymin <= raw_idx < ymax:
                kernel_path = path
                break

        if kernel_path is None:
            logging.error(f"raw_idx={raw_idx} is out of KERNEL_CONFIG range for {fits_path}")
            sys.exit(1)

        kernel, fit_ranges, ar_features, mu_wavelength_pairs, meta = read_kernel_npz(kernel_path)
        corr, best_lag, max_corr = get_cross_corr(center_row, kernel)
        snt_result, lambdas = get_sawtooth_center(center_row, mu_wavelength_pairs, best_lag)
        pixpos[:, j] = np.array([r.xc for r in snt_result], dtype=np.float32)

    write_h5py(h5py_path, header, lambdas, pixpos)



def work_per_object(object_name, fits_dict):
    for date_label, base_name_list in fits_dict.items():
        do_scp_raw_fits(date_label, object_name, base_name_list)

        for base_name in base_name_list:
            fits_path = Path(RAWDATA_DIR) / object_name / date_label / base_name
            h5py_filename = base_name.replace(".fits", ".h5")
            h5py_path = Path(WORK_DIR) / object_name / date_label / h5py_filename
            crosscorr_roop(fits_path, h5py_path)
        
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
            # 2列目に object がある前提で取得
            if len(row) >= 2:
                objects.append(row[1])
    return objects

def get_work_dir_path(object_name: str, date_label: str) -> Path:
    """WORK_DIR/object_name/date_label に対応する出力ディレクトリを返す。"""
    d = Path(WORK_DIR) / object_name / date_label
    d.mkdir(parents=True, exist_ok=True)
    return d



def main():
    # 必須ファイル・環境のチェック
    validate_environment()

    csv_path = REQUIRED_LOCAL_FILES["ar_name_csv"]
    conn = sqlite3.connect(DB_PATH)

    logging.info(f"reading object list from {csv_path}")
    objects = get_object_list(csv_path)

    fits_dict_list = []
    total_objects = len(objects)
    logging.info(f"number of objects from csv: {total_objects}")
    for idx, object_name in enumerate(objects, 1):
        logging.info(f"[{idx}/{total_objects}] start db_search for object={object_name}")
        fits_dict = db_search(conn, object_name)
        logging.info(f"[{idx}/{total_objects}] finished db_search for object={object_name}, date_labels={len(fits_dict)}")
        fits_dict_list.append((object_name, fits_dict))
    conn.close()


    with ProcessPoolExecutor(max_workers=5, initializer=worker_init) as ex:
        future_to_object = {
            ex.submit(work_per_object, object_name, fits_dict): object_name
            for object_name, fits_dict in fits_dict_list
        }
        total = len(future_to_object)
        done = 0
        for fut in as_completed(future_to_object):
            object_name = future_to_object[fut]
            done += 1
            print(f"[{done}/{total}] finished {object_name}")
            fut.result()


    #for object_name, fits_dict in fits_dict_list:
    #    work_per_object(object_name, fits_dict)

    


if __name__ == "__main__":
    print(f"Start {sys.argv[0]}")
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
        force=True,  # ← これを追加
    )
    main()