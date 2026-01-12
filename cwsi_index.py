import os
import glob
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu

# =========================================================
# [설정] 경로 및 파라미터
# =========================================================
DIR_GNDVI = r"D:\회사관련\thermal-analysis\index_data"
DIR_LWIR = r"D:\회사관련\thermal-analysis\thermal_data"
DIR_OUTPUT = r"D:\회사관련\thermal-analysis\result_cwsi_final"

# [옵션] 촬영 당시 기온 (섭씨)
# 기온 데이터가 있으면 고온 스트레스 기준(Tdry)을 더 정확히 잡을 수 있습니다.
# 없으면 None으로 설정 (이미지 내 통계값 사용)
AIR_TEMP = 32


def find_pair_files(code):
    """필지 코드로 GNDVI와 LWIR 파일 쌍 찾기"""
    gndvi_files = glob.glob(os.path.join(DIR_GNDVI, f"*{code}*.tif"))
    path_gndvi = next((f for f in gndvi_files if "GNDVI" in os.path.basename(f).upper()), None)

    lwir_files = glob.glob(os.path.join(DIR_LWIR, f"*{code}*.tif"))
    path_lwir = next((f for f in lwir_files if any(x in os.path.basename(f).upper() for x in ["LWIR", "THERMAL"])),
                     None)

    return path_gndvi, path_lwir


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


def calculate_cwsi_auto_threshold(lwir_data, gndvi_data, field_code):
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
        print(f"   [자동 임계치] {otsu_thresh:.4f} (Otsu Algorithm)")
    except:
        otsu_thresh = 0.3
        print(f"   [경고] 오츠 실패. 기본값 {otsu_thresh} 사용")

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

    if AIR_TEMP is not None:
        t_dry = max(t_dry_stat, AIR_TEMP + 5.0)
    else:
        t_dry = t_dry_stat

    if t_dry <= t_wet:
        t_dry = t_wet + 1.0

    print(f"   [기준온도] Twet: {t_wet:.2f}°C, Tdry: {t_dry:.2f}°C")

    # 5. CWSI 계산
    cwsi_map = np.full_like(lwir_data, np.nan)

    numerator = lwir_data - t_wet
    denominator = t_dry - t_wet

    with np.errstate(divide='ignore', invalid='ignore'):
        calculated = numerator / denominator

    calculated = np.clip(calculated, 0, 1)
    cwsi_map[valid_mask] = calculated[valid_mask]

    return cwsi_map, t_wet, t_dry


def save_cwsi_map(cwsi_map, t_wet, t_dry, field_code, save_path):
    plt.figure(figsize=(10, 8))
    cmap = plt.cm.jet
    cmap.set_bad(color='white')

    plt.imshow(cwsi_map, cmap=cmap, vmin=0, vmax=1)
    cbar = plt.colorbar(label='CWSI (0: Healthy ~ 1: Stressed)')

    plt.title(f"{field_code} CWSI Map (Auto Threshold)\nTwet: {t_wet:.1f}°C, Tdry: {t_dry:.1f}°C")
    plt.axis('off')

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   [저장완료] {os.path.basename(save_path)}")


def process_cwsi(field_code):
    print(f"\n--- [{field_code}] CWSI 분석 시작 ---")

    path_gndvi, path_lwir = find_pair_files(field_code)

    if not path_gndvi or not path_lwir:
        print(f"   [오류] 파일 쌍 누락")
        return

    try:
        with rasterio.open(path_gndvi) as src:
            gndvi_data = src.read(1)
            ref_profile = src.profile
            if src.nodata is not None:
                gndvi_data[gndvi_data == src.nodata] = np.nan
            # 0~1 사이 값만 유효하므로 노이즈 제거
            gndvi_data[(gndvi_data < -1) | (gndvi_data > 1)] = np.nan
    except Exception as e:
        print(f"   [오류] GNDVI 로드 실패: {e}")
        return

    try:
        lwir_data = read_and_resample_lwir(path_lwir, ref_profile)
    except Exception as e:
        print(f"   [오류] LWIR 처리 실패: {e}")
        return

    cwsi_map, t_wet, t_dry = calculate_cwsi_auto_threshold(lwir_data, gndvi_data, field_code)

    if cwsi_map is not None:
        if not os.path.exists(DIR_OUTPUT): os.makedirs(DIR_OUTPUT)
        save_path = os.path.join(DIR_OUTPUT, f"{field_code}_CWSI_Auto.png")
        save_cwsi_map(cwsi_map, t_wet, t_dry, field_code, save_path)

        mean_cwsi = np.nanmean(cwsi_map)
        print(f"   [결과] 평균 CWSI: {mean_cwsi:.3f}")
    else:
        print("   [경고] 유효한 식생 영역이 없습니다.")


if __name__ == "__main__":
    if os.path.exists(DIR_GNDVI):
        files = glob.glob(os.path.join(DIR_GNDVI, "*.tif"))
        codes = set()
        for f in files:
            fname = os.path.basename(f)
            if '_' in fname:
                codes.add(fname.split('_')[0])
            else:
                codes.add(os.path.splitext(fname)[0])

        sorted_codes = sorted(list(codes))
        print(f">>> 감지된 필지 목록: {sorted_codes}")

        for code in sorted_codes:
            process_cwsi(code)
    else:
        print(f"[오류] 폴더를 찾을 수 없습니다: {DIR_GNDVI}")