# thermal-analysis 프로젝트 메모리

> 최종 갱신: 2026-08-31 (v2 전체 84쌍 배치 완료, 종합검토보고서 작성)

## 최신 상태 (2026-08-31)
- 데이터 84쌍으로 확장: 2025 SM01~12×3회차 + 2026 SM01~12(01_260715, 02_260814) + SM13~24(01_260707, 02_260814). RGB도 84개 전부 rgb_data/에 확보
- v2 전체 배치 완료: **83/84 성공** (`result_cwsi_v2/`, 필지별 요약 `cwsi_summary.csv`)
- **손상 파일 2건 (재확보 필요)**: `thermal_data/SM16_02_260814_LWIR.tif`(처리 불가), `rgb_data/SM18_02_260814_RGB.tif`(GNDVI 폴백으로 처리됨). 재복사 후 `python cwsi_index.py SM16 SM18`로 재처리
- 핵심 결과: 2025-09-22 회차는 11/12필지 QC 경고(온도대비 1.4°C, 판정 보류), 스트레스 상위 필지 SM21·SM23(2026-07-07), 260715 회차는 식생 3.4%라 대표성 낮음
- **`종합검토보고서_CWSI_v2.md`** — 로직 검토 + v1/v2 정량비교 + 전체 결과 + 상용 활용성 평가 (최종 산출 문서)
- **`플랫폼연동_고도화로드맵.md`** — Phase 1(무기상 플랫폼 연동, 현재)→2(기상대 연동, ’27)→3(NWSB 절대화, ’27)→4(로직 확장, ’28~). 사용자 보유: 새만금 간이기상대(시간별 기온·습도·강수·풍속·일사·ET). **사용자 결정(’26.9.7): TDR 토양수분 실측·절대 임계값은 로드맵에서 제외 — CWSI 로직 고도화(NWSB)만 반영.** 사용자가 PPT를 직접 수정하기도 하므로 재생성 대신 제자리 패치 우선(patch_roadmap.py 방식)
- **`CWSI_v2_기술이관_대동양식.pptx`** — 개발사 전달용 **유일본** (15슬라이드: 표지+목차+분석/코드/인수스펙/로드맵 13섹션). 구판 2개(대동애그테크 13장, 로직_플랫폼연동 14장 사본)는 2026-09-07 사용자 요청으로 삭제. 생성 스크립트는 세션 스크래치패드 gen_deck_daedong.py (재생성 필요 시 memory 참조)
- PPT 제작 환경 메모: 이 PC엔 node/LibreOffice 없음 → python-pptx(base conda) + PowerPoint COM 렌더 QA. **python-pptx build_freeform은 PowerPoint가 손상 파일로 거부** → 대각선은 PARALLELOGRAM+adjustment로 구현
- **대동 PPT 양식**: `12_PPT양식/_대동_PPT_양식_가이드.md` 필독 (10.83"×7.5", 네이비 #002060+레드 #EF4023, 거버닝 메시지, 고밀도 원칙). 로고는 티타임 pptx의 ppt/media/image2.png(672×126)
- 병렬 실행 지원: `CWSI_PART=p1 python cwsi_index.py SM01 SM02 ...` (워커별 요약 CSV 분리, 4워커로 84쌍 ~1시간)

## 프로젝트 개요
- 드론 열화상(LWIR) + GNDVI(+RGB) 기반 **CWSI(작물 수분 스트레스 지수)** 산출 파이프라인
- 대상: 새만금(SM) 필지 — satelite 프로젝트와 동일한 SM 필지 코드 체계, 길고 좁은 간척지 필지
  - 2025년: SM01~SM12 × 3회차 (02: 250818/32.1°C, 03: 250922/25.5°C, 04: 251112/12°C)
  - 2026년: SM13~SM24 × 2회차 (01: 260707/26.6°C, 02: 260814/28.4°C) — 총 60쌍
- 파일명 규칙: `{필지코드}_{회차}_{날짜YYMMDD}_{GNDVI|LWIR|RGB}.tif`
- 래스터: 단일밴드 float32, ~2cm GSD, EPSG:5179, GNDVI/LWIR **동일 그리드**(재투영 불필요), nodata -10000

## 폴더 구조
- `cwsi_index.py` — 메인 스크립트 v2 (2026-08-31 업그레이드)
- `index_data/` — GNDVI 60개 / `thermal_data/` — LWIR 60개
- `result_cwsi_final/` — **구버전(v1) 결과** PNG 36개 (`*_CWSI_Auto.png`, 비교용 보존)
- `result_cwsi_v2/` — v2 결과 (CWSI PNG + LWIR PNG + CWSI GeoTIFF + `cwsi_summary.csv`)
- `temp_data.csv` — 날짜별 기온 (date=YYMMDD int)
- `참고논문/` — CWSI 논문 8편 (PDF 6 + DOCX 2)
- `논문비교_CWSI로직업그레이드.md` — **논문 방법론 비교·검증·업그레이드 근거 문서 (필독)**
- `참고논문_요약집.md` — 논문 8편 상세 요약 (논문별 서지·설계·방법·수치·채택여부 + v2 대응표)
- RGB 정사영상: **`rgb_data/`** ({태그}_RGB.tif, 4밴드 uint8 RGBA, 별도 그리드). 2025년 36개(19GB)를 2026-08-31 사용자 요청으로 geotiff_processing/data에서 **이동**해 옴(원본 위치엔 더 이상 없음). 스크립트 탐색 순서: rgb_data → geotiff_processing/data. **2026년 SM13~24 RGB는 아직 없음 — 확보 시 rgb_data/에 넣으면 ExG 마스킹 자동 적용**

## v2 로직 (cwsi_index.py)
1. **식생/토양 분리**: RGB 있으면 정규화 ExG=(2G−R−B)/(R+G+B) [나상일 외 2020 봄배추], 없으면 GNDVI 폴백 [Liu et al. 2024 NDVI-Otsu 구조]. 임계값: Otsu(기본) 또는 Jenks(1D k-means 2-class)
2. **Twet/Tdry = 식생 화소 온도 5%/95% 백분위** [Liu et al. 2024, 가뭄판별 92.6%]. 설정으로 (0.5, 99.5) 가능. 구버전의 `Tdry=max(통계, 기온+5)` 하한은 논문 근거 없어 기본 해제(`TDRY_AIRTEMP_FLOOR`)
3. 기온은 ΔT(캐노피 평균온도−기온) 보조지표로만 사용
4. QC 경고: 식생 비율 <2%/>98%, Tdry−Twet<2°C(장면 온도대비 부족 시 백분위 기준선 신뢰 낮음 — SM13_01에서 실제 발동)
5. RGB는 밴드별 순차 재투영(average)으로 메모리 절약, alpha(4밴드)로 유효영역 마스크

## 핵심 검증 결과 (마스킹 방식 비교)
- 생육 최성기(8월): GNDVI-Otsu ≈ ExG-Otsu (IoU 0.84~0.86)
- **노화/수확기(11월): GNDVI-Otsu 식생 70% 과대분류 vs ExG 10~12% 정확** — 갈변 식생·잔사의 GNDVI가 높게 유지되어 Otsu 경계가 왜곡됨. **ExG 마스킹 채택의 결정적 근거**
- 고정 임계(ExG>0.05)는 최성기 과대(78~96%) → 자동 임계(Otsu) 채택
- v2 샘플 결과: SM01_02 평균 CWSI 0.279 / SM01_04 0.519(노화기 — 해석주의) / SM13_01 0.557(QC경고: 온도대비 1.83°C)

## 참고논문 핵심 (상세는 논문비교 md)
- **나상일 외 2020 (KJRS, 봄배추)**: ExG(정규화 rgb)+Jenks로 식생 분리, 식생 min/max=Twet/Tdry — 사용자가 언급한 "ExG 논문"
- **Liu et al. 2024 (RS, 겨울밀)**: NDVI+Otsu, 5%/95% 백분위, 가뭄 임계 정상<0.30~0.41 / 심함>0.48~0.59
- Combining UAV(옥수수): 0.5/99.5 백분위 + SAVI-Otsu, **백분위 방식은 장면에 스트레스/비스트레스 공존 필요**, 촬영 12~16시 최적, 생육말기 노화를 스트레스로 오진 주의
- Kapari 2025(옥수수): NWSB(VPD 회귀) — 시간별 기상자료 필요해 미채택
- 나머지 3편(Ma 2024 실물기준면, Shi 2024 NRCT, 문현동 2022, Soybean Flooding)은 직접 채택 요소 없음

## 실행 방법
- 환경: `C:/Users/user/miniconda3/envs/python312/python.exe` (rasterio 1.4.3 + skimage) — geo_env는 skimage 없음
- `cd thermal-analysis && python cwsi_index.py` — 60쌍 전체 처리(장시간). RAM 128GB로 풀해상도 처리 무리 없음
- git 이력: first commit → cwsi 지수 → CWSI_index 수정 → (v2 미커밋 상태였음, 2026-08-31 기준)
