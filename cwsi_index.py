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
DIR_GNDVI = r"C:\Users\user\Desktop\분석프로젝트\thermal-analysis\index_data"
DIR_LWIR = r"C:\Users\user\Desktop\분석프로젝트\thermal-analysis\thermal_data"
DIR_OUTPUT = r"C:\Users\user\Desktop\분석프로젝트\thermal-analysis\result_cwsi_final"
PATH_TEMP_CSV = r"temp_data.csv"  # 기온 데이터 파일 경로

# 기온 데이터가 CSV에 없을 경우 사용할 기본값
DEFAULT_AIR_TEMP = 32.0


def load_temp_data(csv_path):
    """
    CSV 파일에서 날짜별 기온 데이터를 읽어 딕셔너리로 반환
    Format: {250818: 32.1, ...}
    """
    if not os.path.exists(csv_path):
        print(f"[경고] 기온 데이터 파일이 없습니다: {csv_path}")
        return {}

    try:
        df = pd.read_csv(csv_path)
        # date 컬럼을 문자열이나 정수로 통일 (여기서는 정수 가정)
        temp_dict = dict(zip(df['date'], df['temperature']))
        print(f">>> 기온 데이터 로드 완료 ({len(temp_dict)}개)")
        return temp_dict
    except Exception as e:
        print(f"[오류] 기온 데이터 로드 실패: {e}")
        return {}


def parse_filename(filename):
    """
    파일명에서 메타데이터 추출
    예: SM01_02_250818_GNDVI.tif -> (SM01, 02, 250818)
    """
    try:
        # 확장자 제거 및 분리
        base = os.path.splitext(filename)[0]
        parts = base.split('_')

        # 최소 3개 요소 (Code, Session, Date)가 있어야 함
        if len(parts) >= 3:
            code = parts[0]
            session = parts[1]
            date = int(parts[2])  # 날짜는 정수로 변환하여 매칭
            return code, session, date
    except Exception:
        pass
    return None, None, None


def find_processing_pairs(dir_gndvi, dir_lwir):
    """
    GNDVI 폴더를 기준으로 LWIR 파일과 매칭되는 쌍을 찾음
    매칭 키: Code, Session, Date
    """
    pairs = []

    # GNDVI 파일 목록 스캔
    gndvi_files = glob.glob(os.path.join(dir_gndvi, "*.tif"))

    for g_path in gndvi_files:
        fname = os.path.basename(g_path)
        if "GNDVI" not in fname.upper():
            continue

        # 메타데이터 추출
        code, session, date = parse_filename(fname)
        if code is None:
            continue

        # 매칭되는 LWIR 파일 찾기 (Code_Session_Date_ 패턴)
        # 예: SM01_02_250818_LWIR.tif 또는 ..._THERMAL.tif
        search_pattern = f"{code}_{session}_{date}*.tif"
        lwir_candidates = glob.glob(os.path.join(dir_lwir, search_pattern))

        # LWIR/THERMAL이 포함된 파일 선정
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
    """LWIR 이미지를 GNDVI 이미지 크기(해상도)에 맞춰 리샘플링"""
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


def calculate_cwsi_auto_threshold(lwir_data, gndvi_data, air_temp):
    """
    CWSI 계산 함수 (Otsu 알고리즘으로 임계치 자동 설정)
    """
    # 1. 유효 데이터 추출 (NaN 및 NoData 제거) for Otsu
    valid_pixels = gndvi_data[~np.isnan(gndvi_data)]

    if len(valid_pixels) == 0:
        return None, None, None

    # 2. 오츠 알고리즘으로 임계치 자동 계산
    try:
        otsu_thresh = threshold_otsu(valid_pixels)
        print(f"      └ [자동 임계치] {otsu_thresh:.4f} (Otsu)")
    except:
        otsu_thresh = 0.3
        print(f"      └ [경고] 오츠 실패. 기본값 {otsu_thresh} 사용")

    # 3. 식생 마스크 생성 (임계치보다 큰 곳만 식물)
    vegetation_mask = (gndvi_data > otsu_thresh)

    # LWIR 데이터와 교차 검증
    valid_mask = vegetation_mask & (~np.isnan(lwir_data))

    if np.sum(valid_mask) == 0:
        return None, None, None

    # 식물 영역의 온도 데이터만 추출
    canopy_temps = lwir_data[valid_mask]

    # 4. 기준 온도(Twet, Tdry) 산출
    t_wet = np.percentile(canopy_temps, 0.5)
    t_dry_stat = np.percentile(canopy_temps, 99.5)

    # Air Temp 사용 (없으면 통계값 사용)
    if air_temp is not None:
        t_dry = max(t_dry_stat, air_temp + 5.0)
        used_temp_source = f"AirTemp({air_temp}) + 5.0"
    else:
        t_dry = t_dry_stat
        used_temp_source = "Image Stat(99.5%)"

    if t_dry <= t_wet:
        t_dry = t_wet + 1.0

    print(f"      └ [기준온도] Twet: {t_wet:.2f}°C, Tdry: {t_dry:.2f}°C (Source: {used_temp_source})")

    # 5. CWSI 계산
    cwsi_map = np.full_like(lwir_data, np.nan)

    numerator = lwir_data - t_wet
    denominator = t_dry - t_wet

    with np.errstate(divide='ignore', invalid='ignore'):
        calculated = numerator / denominator

    calculated = np.clip(calculated, 0, 1)
    cwsi_map[valid_mask] = calculated[valid_mask]

    return cwsi_map, t_wet, t_dry


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
    print(f"      └ [저장완료] {os.path.basename(save_path)}")


def process_pair(pair_info, temp_data):
    code = pair_info['code']
    session = pair_info['session']
    date = pair_info['date']

    # 해당 날짜의 기온 가져오기
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

    cwsi_map, t_wet, t_dry = calculate_cwsi_auto_threshold(lwir_data, gndvi_data, air_temp)

    if cwsi_map is not None:
        if not os.path.exists(DIR_OUTPUT): os.makedirs(DIR_OUTPUT)

        # 파일명 생성 (Code_Session_Date_CWSI_Auto.png)
        save_filename = f"{code}_{session}_{date}_CWSI_Auto.png"
        save_path = os.path.join(DIR_OUTPUT, save_filename)
        title_info = f"{code} #{session} ({date}) CWSI Map"

        save_cwsi_map(cwsi_map, t_wet, t_dry, title_info, save_path)

        mean_cwsi = np.nanmean(cwsi_map)
        print(f"      └ [결과] 평균 CWSI: {mean_cwsi:.3f}")
    else:
        print("      └ [경고] 유효한 식생 영역이 없습니다.")


if __name__ == "__main__":
    # 1. 기온 데이터 로드
    temp_data_dict = load_temp_data(PATH_TEMP_CSV)

    # 2. 처리할 파일 쌍 찾기
    if os.path.exists(DIR_GNDVI):
        processing_pairs = find_processing_pairs(DIR_GNDVI, DIR_LWIR)

        # 날짜순, 코드순 정렬
        processing_pairs.sort(key=lambda x: (x['date'], x['code'], x['session']))

        print(f">>> 총 {len(processing_pairs)}개의 파일 쌍을 찾았습니다.")

        for pair in processing_pairs:
            process_pair(pair, temp_data_dict)
    else:
        print(f"[오류] 폴더를 찾을 수 없습니다: {DIR_GNDVI}")