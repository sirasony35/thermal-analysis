# -*- coding: utf-8 -*-
"""
CWSI (Crop Water Stress Index) 산출 파이프라인 v2
=================================================
참고논문 기반 로직 업그레이드 (2026-08-31):

1) 식생/토양 분리
   - RGB 정사영상이 있으면 ExG(Excess Green, 정규화 rgb 기반 2g-r-b)로 분류
     [나상일 외 2020, 봄배추 드론 CWSI — ExG + 자연분류(Jenks)]
   - RGB가 없으면 GNDVI 기반으로 폴백
     [Liu et al. 2024, 겨울밀 — NDVI + Otsu 구조와 동일]
   - 임계값 알고리즘: 'otsu'(기본) 또는 'jenks'(2-class 자연분류, 1D k-means)

2) 기준온도 (Twet/Tdry)
   - 식생 화소 온도 히스토그램의 백분위수 (기본 5% / 95%)
     [Liu et al. 2024 — 하위/상위 5%, 가뭄판별 정확도 92.6%]
   - (0.5, 99.5)로 바꾸면 구버전 및 Zhang 계열 논문 방식
   - 구버전의 Tdry = max(통계값, 기온+5) 하한은 논문 근거가 없어 기본 해제
     (TDRY_AIRTEMP_FLOOR = True 로 복원 가능)

3) 품질 검사(QC)
   - 식생 비율이 비정상(<2% 또는 >98%)이면 경고
   - Tdry-Twet < 2도(장면 내 온도 대비 부족)이면 경고
     [Combining UAV 논문 — 백분위 방식은 장면에 스트레스/비스트레스가
      공존할 때만 유효하다는 한계 지적 반영]

4) 산출물: CWSI 맵 PNG + 온도 맵 PNG + CWSI GeoTIFF + 요약 CSV(cwsi_summary.csv)
"""
import os
import csv
import gc
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
DIR_GNDVI = r"index_data"      # GNDVI GeoTIFF 폴더 ({코드}_{회차}_{날짜}_GNDVI.tif)
DIR_LWIR = r"thermal_data"     # LWIR GeoTIFF 폴더 ({코드}_{회차}_{날짜}_LWIR.tif)
# RGB 정사영상 탐색 폴더 (앞선 폴더 우선). 로컬 rgb_data에 파일을 추가하면 즉시 인식됨
DIRS_RGB = [
    r"rgb_data",
    r"C:\Users\user\Desktop\분석프로젝트\geotiff_processing\data",
]
DIR_OUTPUT = r"result_cwsi_v2"
PATH_TEMP_CSV = r"temp_data.csv"   # 날짜(YYMMDD)별 기온

DEFAULT_AIR_TEMP = None        # CSV에 기온이 없을 때 값 (None이면 dT 미산출)

USE_VEGETATION_MASK = True     # 식생(작물) 마스킹 사용 여부
MASK_METHOD = "auto"           # "auto": RGB 있으면 ExG, 없으면 GNDVI / "exg" / "gndvi"
THRESH_ALGO = "otsu"           # "otsu" 또는 "jenks" (2-class 자연분류)

BASELINE_PERCENTILES = (5.0, 95.0)  # (Twet, Tdry) 백분위 [Liu et al. 2024]
TDRY_AIRTEMP_FLOOR = False     # True면 Tdry = max(통계값, 기온+5.0) (구버전 방식)
AIRTEMP_FLOOR_OFFSET = 5.0

SAVE_GEOTIFF = True            # CWSI GeoTIFF 저장 여부
SAVE_LWIR_PNG = True           # 온도 맵 PNG 저장 여부

# QC 경고 기준
QC_VEG_MIN, QC_VEG_MAX = 0.02, 0.98
QC_MIN_BASELINE_RANGE = 2.0    # Tdry-Twet 최소 폭(도)


# =========================================================
# 데이터 로드/탐색
# =========================================================
def load_temp_data(csv_path):
    if not os.path.exists(csv_path):
        print(f"[경고] 기온 데이터 파일이 없습니다: {csv_path}")
        return {}
    try:
        df = pd.read_csv(csv_path)
        temp_dict = dict(zip(df['date'].astype(int), df['temperature'].astype(float)))
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
            return parts[0], parts[1], int(parts[2])
    except Exception:
        pass
    return None, None, None


def find_rgb(tag, dirs_rgb):
    """탐색 폴더 목록에서 {tag}_RGB.tif를 순서대로 찾는다 (앞선 폴더 우선)."""
    for d in dirs_rgb or []:
        cand = os.path.join(d, f"{tag}_RGB.tif")
        if os.path.exists(cand):
            return cand
    return None


def find_processing_pairs(dir_gndvi, dir_lwir, dirs_rgb):
    """GNDVI 폴더 기준으로 동일 태그의 LWIR(필수), RGB(선택) 파일을 매칭한다."""
    pairs = []
    for g_path in glob.glob(os.path.join(dir_gndvi, "*_GNDVI.tif")):
        code, session, date = parse_filename(os.path.basename(g_path))
        if code is None:
            continue
        tag = f"{code}_{session}_{date}"

        l_candidates = [f for f in glob.glob(os.path.join(dir_lwir, f"{tag}*.tif"))
                        if any(x in os.path.basename(f).upper() for x in ["LWIR", "THERMAL"])]
        if not l_candidates:
            print(f"[경고] LWIR 파일 없음, 건너뜀: {tag}")
            continue

        rgb_path = find_rgb(tag, dirs_rgb)

        pairs.append({'code': code, 'session': session, 'date': date, 'tag': tag,
                      'gndvi_path': g_path, 'lwir_path': l_candidates[0], 'rgb_path': rgb_path})
    return pairs


# =========================================================
# 래스터 처리
# =========================================================
def same_grid(profile_a, src_b):
    return (profile_a['width'] == src_b.width and profile_a['height'] == src_b.height
            and profile_a['transform'] == src_b.transform and profile_a['crs'] == src_b.crs)


def read_and_resample_lwir(lwir_path, ref_profile):
    with rasterio.open(lwir_path) as src:
        if same_grid(ref_profile, src):
            data = src.read(1).astype(np.float32)
            if src.nodata is not None:
                data[data == src.nodata] = np.nan
        else:
            data = np.zeros((ref_profile['height'], ref_profile['width']), dtype=np.float32)
            reproject(source=rasterio.band(src, 1), destination=data,
                      src_transform=src.transform, src_crs=src.crs,
                      dst_transform=ref_profile['transform'], dst_crs=ref_profile['crs'],
                      resampling=Resampling.bilinear, dst_nodata=np.nan)
        data[data < -50] = np.nan
        return data


def compute_exg_on_grid(rgb_path, ref_profile):
    """RGB 정사영상을 기준 그리드로 재투영하며 정규화 ExG = (2G-R-B)/(R+G+B)를 계산.

    나상일 외(2020) 방식: 정규화 색좌표(r,g,b)에 대해 ExG = 2g - r - b.
    밴드를 한 장씩 처리해 메모리를 아낀다. 4밴드째(alpha)가 있으면 유효영역 마스크로 사용.
    """
    h, w = ref_profile['height'], ref_profile['width']
    band_sum = np.zeros((h, w), dtype=np.float32)
    numer = np.zeros((h, w), dtype=np.float32)
    coef = {1: -1.0, 2: 2.0, 3: -1.0}  # R, G, B

    with rasterio.open(rgb_path) as src:
        dst = np.zeros((h, w), dtype=np.float32)
        for b in (1, 2, 3):
            dst[:] = 0
            reproject(source=rasterio.band(src, b), destination=dst,
                      src_transform=src.transform, src_crs=src.crs,
                      dst_transform=ref_profile['transform'], dst_crs=ref_profile['crs'],
                      resampling=Resampling.average, dst_nodata=0)
            band_sum += dst
            numer += coef[b] * dst
        alpha_invalid = None
        if src.count >= 4:
            dst[:] = 0
            reproject(source=rasterio.band(src, 4), destination=dst,
                      src_transform=src.transform, src_crs=src.crs,
                      dst_transform=ref_profile['transform'], dst_crs=ref_profile['crs'],
                      resampling=Resampling.nearest, dst_nodata=0)
            alpha_invalid = dst < 128
        del dst

    with np.errstate(divide='ignore', invalid='ignore'):
        exg = numer / band_sum
    exg[band_sum <= 0] = np.nan
    if alpha_invalid is not None:
        exg[alpha_invalid] = np.nan
    del band_sum, numer
    gc.collect()
    return exg


# =========================================================
# 임계값 알고리즘
# =========================================================
def jenks_break_2class(values, max_sample=500_000, n_iter=30):
    """1차원 2-class 자연분류(Jenks) 근사 — 1D k-means로 분리 경계를 찾는다."""
    v = np.asarray(values, dtype=np.float64)
    if v.size > max_sample:
        rng = np.random.default_rng(42)
        v = rng.choice(v, max_sample, replace=False)
    c_lo, c_hi = np.percentile(v, [10, 90])
    for _ in range(n_iter):
        mid = (c_lo + c_hi) / 2.0
        lo, hi = v[v <= mid], v[v > mid]
        if lo.size == 0 or hi.size == 0:
            break
        new_lo, new_hi = lo.mean(), hi.mean()
        if abs(new_lo - c_lo) < 1e-6 and abs(new_hi - c_hi) < 1e-6:
            c_lo, c_hi = new_lo, new_hi
            break
        c_lo, c_hi = new_lo, new_hi
    return (c_lo + c_hi) / 2.0


def auto_threshold(values, algo):
    if algo == "jenks":
        return jenks_break_2class(values)
    return threshold_otsu(values)


# =========================================================
# 식생 마스크
# =========================================================
def build_vegetation_mask(gndvi_data, exg_data, method, algo):
    """식생 마스크 생성. 반환: (mask, 방법 문자열, 임계값)"""
    valid_gndvi = ~np.isnan(gndvi_data)

    use_exg = (method == "exg") or (method == "auto" and exg_data is not None)
    if use_exg and exg_data is None:
        print("      └ [경고] ExG 마스킹 지정됐으나 RGB 없음. GNDVI로 폴백")
        use_exg = False

    if use_exg:
        index_data, name = exg_data, "ExG"
        valid = valid_gndvi & ~np.isnan(exg_data)
    else:
        index_data, name = gndvi_data, "GNDVI"
        valid = valid_gndvi

    pix = index_data[valid]
    if pix.size == 0:
        return None, name, None
    try:
        thr = auto_threshold(pix, algo)
    except Exception as e:
        print(f"      └ [경고] 임계값 산출 실패({e}). 기본값 사용")
        thr = 0.05 if name == "ExG" else 0.3
    mask = valid & (index_data > thr)
    return mask, f"{name}-{algo}", float(thr)


# =========================================================
# CWSI 산출
# =========================================================
def calculate_cwsi(lwir_data, valid_mask, air_temp):
    """식생 화소 온도 백분위수로 Twet/Tdry를 정하고 CWSI 맵을 만든다."""
    target_mask = valid_mask & ~np.isnan(lwir_data)
    n = int(np.sum(target_mask))
    if n == 0:
        return None

    temps = lwir_data[target_mask]
    p_lo, p_hi = BASELINE_PERCENTILES
    t_wet = float(np.percentile(temps, p_lo))
    t_dry = float(np.percentile(temps, p_hi))
    source = f"percentile {p_lo}/{p_hi}"

    if TDRY_AIRTEMP_FLOOR and air_temp is not None:
        floored = max(t_dry, air_temp + AIRTEMP_FLOOR_OFFSET)
        if floored != t_dry:
            source += f" + floor(Tair+{AIRTEMP_FLOOR_OFFSET})"
        t_dry = floored

    qc_notes = []
    if (t_dry - t_wet) < QC_MIN_BASELINE_RANGE:
        qc_notes.append(f"baseline_range<{QC_MIN_BASELINE_RANGE}C")
        print(f"      └ [QC경고] Tdry-Twet={t_dry - t_wet:.2f}도. 장면 온도 대비가 작아 "
              f"백분위 기준선 신뢰도가 낮습니다 (스트레스/비스트레스 공존 필요)")
    if t_dry <= t_wet:
        t_dry = t_wet + 1.0
        qc_notes.append("degenerate_baseline")

    print(f"      └ [기준온도] Twet: {t_wet:.2f}도, Tdry: {t_dry:.2f}도 (Source: {source})")

    cwsi_map = np.full_like(lwir_data, np.nan)
    with np.errstate(divide='ignore', invalid='ignore'):
        calc = (lwir_data - t_wet) / (t_dry - t_wet)
    calc = np.clip(calc, 0, 1)
    cwsi_map[target_mask] = calc[target_mask]
    del calc

    veg_temps_mean = float(np.mean(temps))
    stats = {
        't_wet': t_wet, 't_dry': t_dry, 'n_pixels': n,
        'canopy_temp_mean': veg_temps_mean,
        'delta_t': (veg_temps_mean - air_temp) if air_temp is not None else None,
        'cwsi_mean': float(np.nanmean(cwsi_map)),
        'cwsi_median': float(np.nanmedian(cwsi_map)),
        'cwsi_p90': float(np.nanpercentile(cwsi_map[target_mask], 90)),
        'frac_gt_05': float(np.mean(cwsi_map[target_mask] > 0.5)),
        'qc': ";".join(qc_notes),
    }
    return cwsi_map, stats


# =========================================================
# 저장
# =========================================================
def save_cwsi_map(cwsi_map, t_wet, t_dry, title_info, save_path):
    plt.figure(figsize=(10, 8))
    cmap = plt.get_cmap('jet').copy()
    cmap.set_bad(color='white')
    plt.imshow(cwsi_map, cmap=cmap, vmin=0, vmax=1)
    plt.colorbar(label='CWSI (0: Healthy ~ 1: Stressed)')
    plt.title(f"{title_info}\nTwet: {t_wet:.1f}C, Tdry: {t_dry:.1f}C")
    plt.axis('off')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"      └ [CWSI 맵 저장완료] {os.path.basename(save_path)}")


def save_lwir_map(lwir_data, valid_mask, title_info, save_path):
    plt.figure(figsize=(10, 8))
    display = np.full_like(lwir_data, np.nan)
    display[valid_mask] = lwir_data[valid_mask]
    cmap = plt.get_cmap('inferno').copy()
    cmap.set_bad(color='white')
    plt.imshow(display, cmap=cmap)
    plt.colorbar(label='Temperature (C)')
    plt.title(f"{title_info} (Thermal Map)")
    plt.axis('off')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    del display
    print(f"      └ [LWIR 맵 저장완료] {os.path.basename(save_path)}")


def save_cwsi_geotiff(cwsi_map, ref_profile, save_path):
    profile = dict(ref_profile)
    profile.update(count=1, dtype='float32', nodata=-9999.0, compress='deflate',
                   tiled=True, blockxsize=512, blockysize=512, BIGTIFF='IF_SAFER')
    out = np.where(np.isnan(cwsi_map), -9999.0, cwsi_map).astype(np.float32)
    with rasterio.open(save_path, 'w', **profile) as dst:
        dst.write(out, 1)
    del out
    print(f"      └ [CWSI GeoTIFF 저장완료] {os.path.basename(save_path)}")


SUMMARY_FIELDS = ['tag', 'code', 'session', 'date', 'air_temp', 'mask_method', 'threshold',
                  'veg_fraction', 'n_pixels', 't_wet', 't_dry', 'canopy_temp_mean', 'delta_t',
                  'cwsi_mean', 'cwsi_median', 'cwsi_p90', 'frac_gt_05', 'qc']


def append_summary(row, csv_path):
    exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


# =========================================================
# 메인 처리
# =========================================================
def process_pair(pair, temp_data, summary_name="cwsi_summary.csv"):
    tag = pair['tag']
    air_temp = temp_data.get(pair['date'], DEFAULT_AIR_TEMP)
    temp_status = "CSV" if pair['date'] in temp_data else "Default"

    print(f"\n--- [{pair['code']}] {pair['session']}회차 ({pair['date']}) 분석 시작 ---")
    print(f"   GNDVI: {os.path.basename(pair['gndvi_path'])}")
    print(f"   RGB: {os.path.basename(pair['rgb_path']) if pair['rgb_path'] else '없음 (GNDVI 마스킹 폴백)'}")
    print(f"   기온: {air_temp} ({temp_status})")

    try:
        with rasterio.open(pair['gndvi_path']) as src:
            gndvi = src.read(1).astype(np.float32)
            ref_profile = src.profile
            if src.nodata is not None:
                gndvi[gndvi == src.nodata] = np.nan
            gndvi[(gndvi < -1) | (gndvi > 1)] = np.nan
    except Exception as e:
        print(f"   [오류] GNDVI 로드 실패: {e}")
        return

    try:
        lwir = read_and_resample_lwir(pair['lwir_path'], ref_profile)
    except Exception as e:
        print(f"   [오류] LWIR 처리 실패: {e}")
        return

    exg = None
    if USE_VEGETATION_MASK and MASK_METHOD in ("auto", "exg") and pair['rgb_path']:
        try:
            exg = compute_exg_on_grid(pair['rgb_path'], ref_profile)
        except Exception as e:
            print(f"   [경고] ExG 계산 실패({e}). GNDVI 마스킹으로 폴백")

    if USE_VEGETATION_MASK:
        mask, mask_method, thr = build_vegetation_mask(gndvi, exg, MASK_METHOD, THRESH_ALGO)
        if mask is None:
            print("      └ [경고] 유효 화소가 없습니다.")
            return
        valid_total = ~np.isnan(gndvi)
        veg_frac = float(np.sum(mask)) / max(1, int(np.sum(valid_total)))
        print(f"      └ [마스킹] {mask_method}, 임계값 {thr:.4f}, 식생 비율 {veg_frac * 100:.1f}%")
        veg_qc = None
        if not (QC_VEG_MIN <= veg_frac <= QC_VEG_MAX):
            veg_qc = f"veg_ratio_{veg_frac:.3f}"
            print(f"      └ [QC경고] 식생 비율이 비정상 범위입니다 ({veg_frac * 100:.1f}%). 마스크 확인 필요")
    else:
        mask = ~np.isnan(gndvi)
        mask_method, thr, veg_frac, veg_qc = "none", None, 1.0, None
        print("      └ [마스킹 해제] 토양 포함 전체 영역 분석")
    del exg
    gc.collect()

    result = calculate_cwsi(lwir, mask, air_temp)
    if result is None:
        print("      └ [경고] 유효한 분석 영역이 없습니다.")
        return
    cwsi_map, stats = result
    if veg_qc:
        stats['qc'] = ";".join(x for x in [stats['qc'], veg_qc] if x)

    os.makedirs(DIR_OUTPUT, exist_ok=True)
    title = f"{pair['code']} #{pair['session']} ({pair['date']})"

    if SAVE_LWIR_PNG:
        save_lwir_map(lwir, mask & ~np.isnan(lwir), title,
                      os.path.join(DIR_OUTPUT, f"{tag}_LWIR.png"))
    save_cwsi_map(cwsi_map, stats['t_wet'], stats['t_dry'], f"{title} CWSI",
                  os.path.join(DIR_OUTPUT, f"{tag}_CWSI.png"))
    if SAVE_GEOTIFF:
        save_cwsi_geotiff(cwsi_map, ref_profile, os.path.join(DIR_OUTPUT, f"{tag}_CWSI.tif"))

    row = {'tag': tag, 'code': pair['code'], 'session': pair['session'], 'date': pair['date'],
           'air_temp': air_temp, 'mask_method': mask_method,
           'threshold': None if thr is None else round(thr, 4),
           'veg_fraction': round(veg_frac, 4), 'n_pixels': stats['n_pixels'],
           't_wet': round(stats['t_wet'], 2), 't_dry': round(stats['t_dry'], 2),
           'canopy_temp_mean': round(stats['canopy_temp_mean'], 2),
           'delta_t': None if stats['delta_t'] is None else round(stats['delta_t'], 2),
           'cwsi_mean': round(stats['cwsi_mean'], 3), 'cwsi_median': round(stats['cwsi_median'], 3),
           'cwsi_p90': round(stats['cwsi_p90'], 3), 'frac_gt_05': round(stats['frac_gt_05'], 3),
           'qc': stats['qc']}
    append_summary(row, os.path.join(DIR_OUTPUT, summary_name))

    print(f"      └ [결과] 평균 CWSI: {stats['cwsi_mean']:.3f} (중앙값 {stats['cwsi_median']:.3f}, "
          f"CWSI>0.5 비율 {stats['frac_gt_05'] * 100:.1f}%)")

    del gndvi, lwir, cwsi_map, mask
    gc.collect()


if __name__ == "__main__":
    import sys

    # 사용법: python cwsi_index.py [필지코드 ...]
    #   인자 없음: 전체 처리, 요약은 cwsi_summary.csv
    #   인자 있음(예: SM01 SM02): 해당 필지만 처리 (병렬 분할 실행용).
    #   병렬 실행 시 환경변수 CWSI_PART를 주면 요약이 cwsi_summary_{PART}.csv로 분리 저장됨
    code_filter = set(sys.argv[1:])
    part = os.environ.get("CWSI_PART")
    summary_name = f"cwsi_summary_{part}.csv" if part else "cwsi_summary.csv"

    temp_data_dict = load_temp_data(PATH_TEMP_CSV)

    if not os.path.isdir(DIR_GNDVI) or not os.path.isdir(DIR_LWIR):
        print(f"[오류] 입력 폴더를 찾을 수 없습니다: {DIR_GNDVI} / {DIR_LWIR}")
    else:
        pairs = find_processing_pairs(DIR_GNDVI, DIR_LWIR, DIRS_RGB)
        if code_filter:
            pairs = [p for p in pairs if p['code'] in code_filter or p['tag'] in code_filter]
        pairs.sort(key=lambda x: (x['date'], x['code'], x['session']))
        n_rgb = sum(1 for p in pairs if p['rgb_path'])
        print(f">>> 총 {len(pairs)}개 파일 쌍 (RGB 보유 {n_rgb}개: ExG 마스킹 / "
              f"미보유 {len(pairs) - n_rgb}개: GNDVI 폴백)")

        for p in pairs:
            try:
                process_pair(p, temp_data_dict, summary_name)
            except Exception as e:
                print(f"   [오류] {p['tag']} 처리 실패: {e}")
