# -*- coding: utf-8 -*-
"""v1(GNDVI-Otsu + 0.5/99.5 + Tair+5 하한) vs v2(ExG-Otsu + 5/95) CWSI 정량 비교.

보고서용. 4배 다운샘플로 대표 필지들을 두 로직으로 계산해 차이를 수치화한다.
"""
import os, sys, csv
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.enums import Resampling as ResEnum
from skimage.filters import threshold_otsu

BASE = r"C:\Users\user\Desktop\분석프로젝트\thermal-analysis"
OUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "v1_v2_compare.csv")
DOWN = 4
TEMPS = {250818: 32.1, 250922: 25.5, 251112: 12.0, 260707: 26.6, 260715: 27.9, 260814: 28.4}

TAGS = ["SM01_02_250818", "SM05_02_250818", "SM01_03_250922", "SM01_04_251112",
        "SM13_01_260707", "SM01_01_260715", "SM01_02_260814", "SM13_02_260814"]


def read_down(path):
    with rasterio.open(path) as src:
        h, w = src.height // DOWN, src.width // DOWN
        data = src.read(1, out_shape=(h, w), resampling=ResEnum.average).astype(np.float32)
        if src.nodata is not None:
            data[data == src.nodata] = np.nan
        prof = src.profile.copy()
        prof.update(height=h, width=w,
                    transform=src.transform * src.transform.scale(src.width / w, src.height / h))
        return data, prof


def exg_on_grid(rgb_path, prof):
    h, w = prof['height'], prof['width']
    s = np.zeros((h, w), np.float32); n = np.zeros((h, w), np.float32)
    coef = {1: -1.0, 2: 2.0, 3: -1.0}
    with rasterio.open(rgb_path) as src:
        dst = np.zeros((h, w), np.float32)
        for b in (1, 2, 3):
            dst[:] = 0
            reproject(rasterio.band(src, b), dst, src_transform=src.transform, src_crs=src.crs,
                      dst_transform=prof['transform'], dst_crs=prof['crs'],
                      resampling=Resampling.average, dst_nodata=0)
            s += dst; n += coef[b] * dst
        inval = None
        if src.count >= 4:
            dst[:] = 0
            reproject(rasterio.band(src, 4), dst, src_transform=src.transform, src_crs=src.crs,
                      dst_transform=prof['transform'], dst_crs=prof['crs'],
                      resampling=Resampling.nearest, dst_nodata=0)
            inval = dst < 128
    with np.errstate(divide='ignore', invalid='ignore'):
        exg = n / s
    exg[s <= 0] = np.nan
    if inval is not None:
        exg[inval] = np.nan
    return exg


def cwsi_stats(lwir, mask, p_lo, p_hi, tair=None, floor=False):
    m = mask & ~np.isnan(lwir)
    if m.sum() == 0:
        return None
    t = lwir[m]
    tw, td = np.percentile(t, p_lo), np.percentile(t, p_hi)
    if floor and tair is not None:
        td = max(td, tair + 5.0)
    if td <= tw:
        td = tw + 1.0
    c = np.clip((t - tw) / (td - tw), 0, 1)
    return dict(t_wet=round(float(tw), 2), t_dry=round(float(td), 2),
                cwsi_mean=round(float(c.mean()), 3), cwsi_med=round(float(np.median(c)), 3),
                frac05=round(float((c > 0.5).mean()), 3), n=int(m.sum()))


rows = []
for tag in TAGS:
    g_path = os.path.join(BASE, "index_data", f"{tag}_GNDVI.tif")
    l_path = os.path.join(BASE, "thermal_data", f"{tag}_LWIR.tif")
    r_path = os.path.join(BASE, "rgb_data", f"{tag}_RGB.tif")
    if not all(os.path.exists(p) for p in (g_path, l_path, r_path)):
        print(f"[skip] {tag}")
        continue
    date = int(tag.split("_")[2])
    tair = TEMPS.get(date)

    gndvi, prof = read_down(g_path)
    gndvi[(gndvi < -1) | (gndvi > 1)] = np.nan
    lwir, _ = read_down(l_path)
    lwir[lwir < -50] = np.nan
    exg = exg_on_grid(r_path, prof)
    valid = ~np.isnan(gndvi)

    thr_g = threshold_otsu(gndvi[valid])
    mask_v1 = valid & (gndvi > thr_g)
    v2valid = valid & ~np.isnan(exg)
    thr_e = threshold_otsu(exg[v2valid])
    mask_v2 = v2valid & (exg > thr_e)

    s1 = cwsi_stats(lwir, mask_v1, 0.5, 99.5, tair, floor=True)   # v1 로직
    s2 = cwsi_stats(lwir, mask_v2, 5.0, 95.0)                      # v2 로직
    s2_v1mask = cwsi_stats(lwir, mask_v1, 5.0, 95.0)               # 마스크 효과 분리용
    inter = np.sum(mask_v1 & mask_v2); union = np.sum(mask_v1 | mask_v2)

    row = dict(tag=tag, tair=tair,
               veg_v1=round(float(mask_v1[valid].mean()), 3),
               veg_v2=round(float(mask_v2[valid].mean()), 3),
               iou=round(float(inter / union), 3) if union else None,
               v1_twet=s1['t_wet'], v1_tdry=s1['t_dry'], v1_cwsi=s1['cwsi_mean'], v1_frac05=s1['frac05'],
               v2_twet=s2['t_wet'], v2_tdry=s2['t_dry'], v2_cwsi=s2['cwsi_mean'], v2_frac05=s2['frac05'],
               v2pct_v1mask_cwsi=s2_v1mask['cwsi_mean'])
    rows.append(row)
    print(row)

with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print("saved:", OUT_CSV)
