import os
import glob
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu

# =========================================================
# [설정] 경로 및 파라미터
# =========================================================
# GNDVI와 LWIR 파일이 함께 있는 통합 폴더 경로
DIR_INPUT = r"input_data"
DIR_OUTPUT = r"result_cwsi_final"
PATH_TEMP_CSV = r"temp_data.csv"  # 기온 데이터 파일 경로

# 기온 데이터가 CSV에 없을 경우 사용할 기본값
DEFAULT_AIR_TEMP = 32.0

# 식생(작물) 마스킹 사용 여부
# 생육 초기(토양 포함): False / 생육 후기(작물만 분리): True
USE_VEGETATION_MASK = True


# =========================================================

def load_temp_data(csv_path):
    if not os.path.exists(csv_path):
        print(f"[경고] 기온 데이터 파일이 없습니다: {csv_path}")
        return {}
    try:
        df = pd.read_csv(csv_path)
        temp_dict = dict(zip(df['date'], df['temperature']))
        print(f">>> 기온 데이터 로드 완료 ({len(temp_dict)}개)")
        return temp_dict
    except Exception as e:
        print(f"[오류] 기온 데이터 로드 실패: {e}")
        return {}


def parse_filename(filename):
    try:
        base = os.path.splitext(filename)[0]
        parts = base.split('_')
        if len(parts) >= 3:
            code = parts[0]
            session = parts[1]
            date = int(parts[2])
            return code, session, date
    except Exception:
        pass
    return None, None, None


def find_processing_pairs(dir_input):
    """
    단일 폴더 내에서 GNDVI 파일과 매칭되는 LWIR 파일을 찾습니다.
    """
    pairs = []
    # 입력 폴더에서 GNDVI 파일만 먼저 스캔
    gndvi_files = glob.glob(os.path.join(dir_input, "*_GNDVI.tif"))

    for g_path in gndvi_files:
        fname = os.path.basename(g_path)
        code, session, date = parse_filename(fname)

        if code is None:
            continue

        # 동일한 폴더 내에서 일치하는 LWIR 또는 THERMAL 파일 검색
        search_pattern = f"{code}_{session}_{date}*.tif"
        lwir_candidates = glob.glob(os.path.join(dir_input, search_pattern))
        l_path = next(
            (f for f in lwir_candidates if any(x in os.path.basename(f).upper() for x in ["LWIR", "THERMAL"])), None)

        if l_path:
            pairs.append({
                'code': code,
                'session': session,
                'date': date,
                'gndvi_path': g_path,
                'lwir_path': l_path
            })
    return pairs


def read_and_resample_lwir(lwir_path, ref_profile):
    with rasterio.open(lwir_path) as src:
        dst_height = ref_profile['height']
        dst_width = ref_profile['width']
        destination = np.zeros((dst_height, dst_width), dtype=np.float32)

        reproject(
            source=rasterio.band(src, 1),
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_profile['transform'],
            dst_crs=ref_profile['crs'],
            resampling=Resampling.bilinear,
            dst_nodata=np.nan
        )

        destination[destination < -50] = np.nan
        return destination


def calculate_cwsi_auto_threshold(lwir_data, gndvi_data, air_temp, use_mask):
    valid_lwir_mask = ~np.isnan(lwir_data)
    valid_gndvi_pixels = gndvi_data[~np.isnan(gndvi_data)]

    if use_mask and len(valid_gndvi_pixels) > 0:
        try:
            otsu_thresh = threshold_otsu(valid_gndvi_pixels)
            print(f"      └ [마스킹 적용] 자동 임계치: {otsu_thresh:.4f} (작물만 분석)")
        except:
            otsu_thresh = 0.3
            print(f"      └ [경고] 오츠 실패. 기본값 {otsu_thresh} 사용")

        vegetation_mask = (gndvi_data > otsu_thresh)
        valid_mask = vegetation_mask & valid_lwir_mask
    else:
        print("      └ [마스킹 해제] 토양을 포함한 전체 영역 분석")
        valid_mask = (~np.isnan(gndvi_data)) & valid_lwir_mask

    if np.sum(valid_mask) == 0:
        return None, None, None, None

    target_temps = lwir_data[valid_mask]
    t_wet = np.percentile(target_temps, 0.5)
    t_dry_stat = np.percentile(target_temps, 99.5)

    if air_temp is not None:
        t_dry = max(t_dry_stat, air_temp + 5.0)
        used_temp_source = f"AirTemp({air_temp}) + 5.0"
    else:
        t_dry = t_dry_stat
        used_temp_source = "Image Stat(99.5%)"

    if t_dry <= t_wet:
        t_dry = t_wet + 1.0

    print(f"      └ [기준온도] Twet: {t_wet:.2f}°C, Tdry: {t_dry:.2f}°C (Source: {used_temp_source})")

    cwsi_map = np.full_like(lwir_data, np.nan)
    numerator = lwir_data - t_wet
    denominator = t_dry - t_wet

    with np.errstate(divide='ignore', invalid='ignore'):
        calculated = numerator / denominator

    calculated = np.clip(calculated, 0, 1)
    cwsi_map[valid_mask] = calculated[valid_mask]

    return cwsi_map, t_wet, t_dry, valid_mask


def save_cwsi_map(cwsi_map, t_wet, t_dry, title_info, save_path):
    plt.figure(figsize=(10, 8))
    cmap = plt.cm.jet
    cmap.set_bad(color='white')

    plt.imshow(cwsi_map, cmap=cmap, vmin=0, vmax=1)
    cbar = plt.colorbar(label='CWSI (0: Healthy ~ 1: Stressed)')

    plt.title(f"{title_info}\nTwet: {t_wet:.1f}°C, Tdry: {t_dry:.1f}°C")
    plt.axis('off')

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"      └ [CWSI 맵 저장완료] {os.path.basename(save_path)}")


def save_lwir_map(lwir_data, valid_mask, title_info, save_path):
    plt.figure(figsize=(10, 8))

    display_data = np.full_like(lwir_data, np.nan)
    display_data[valid_mask] = lwir_data[valid_mask]

    cmap = plt.cm.inferno
    cmap.set_bad(color='white')

    plt.imshow(display_data, cmap=cmap)
    cbar = plt.colorbar(label='Temperature (°C)')

    plt.title(f"{title_info} (Thermal Map)")
    plt.axis('off')

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"      └ [LWIR 맵 저장완료] {os.path.basename(save_path)}")


def process_pair(pair_info, temp_data):
    code = pair_info['code']
    session = pair_info['session']
    date = pair_info['date']

    air_temp = temp_data.get(date, DEFAULT_AIR_TEMP)
    temp_status = "CSV Data" if date in temp_data else "Default"

    print(f"\n--- [{code}] {session}회차 ({date}) 분석 시작 ---")
    print(f"   Target: {os.path.basename(pair_info['gndvi_path'])}")
    print(f"   Temp: {air_temp}°C ({temp_status})")

    try:
        with rasterio.open(pair_info['gndvi_path']) as src:
            gndvi_data = src.read(1)
            ref_profile = src.profile
            if src.nodata is not None:
                gndvi_data[gndvi_data == src.nodata] = np.nan
            gndvi_data[(gndvi_data < -1) | (gndvi_data > 1)] = np.nan
    except Exception as e:
        print(f"   [오류] GNDVI 로드 실패: {e}")
        return

    try:
        lwir_data = read_and_resample_lwir(pair_info['lwir_path'], ref_profile)
    except Exception as e:
        print(f"   [오류] LWIR 처리 실패: {e}")
        return

    cwsi_map, t_wet, t_dry, valid_mask = calculate_cwsi_auto_threshold(lwir_data, gndvi_data, air_temp,
                                                                       USE_VEGETATION_MASK)

    if cwsi_map is not None:
        if not os.path.exists(DIR_OUTPUT): os.makedirs(DIR_OUTPUT)

        # 저장 파일명 명명 규칙 적용
        lwir_filename = f"{code}_{date}_LWIR_modify.png"
        lwir_path = os.path.join(DIR_OUTPUT, lwir_filename)
        save_lwir_map(lwir_data, valid_mask, f"{code} #{session} ({date})", lwir_path)

        cwsi_filename = f"{code}_{date}_CWSI_modify.png"
        cwsi_path = os.path.join(DIR_OUTPUT, cwsi_filename)
        save_cwsi_map(cwsi_map, t_wet, t_dry, f"{code} #{session} ({date}) CWSI", cwsi_path)

        mean_cwsi = np.nanmean(cwsi_map)
        print(f"      └ [결과] 평균 CWSI: {mean_cwsi:.3f}")
    else:
        print("      └ [경고] 유효한 분석 영역이 없습니다.")


if __name__ == "__main__":
    temp_data_dict = load_temp_data(PATH_TEMP_CSV)

    if os.path.exists(DIR_INPUT):
        processing_pairs = find_processing_pairs(DIR_INPUT)
        processing_pairs.sort(key=lambda x: (x['date'], x['code'], x['session']))

        print(f">>> 총 {len(processing_pairs)}개의 파일 쌍을 찾았습니다.")

        for pair in processing_pairs:
            process_pair(pair, temp_data_dict)
    else:
        print(f"[오류] 폴더를 찾을 수 없습니다: {DIR_INPUT}")