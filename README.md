# 거시 분석 플랫폼 (macro-lab)

산업연관표와 거시 모형으로 **시나리오를 넣어 파급효과를 계산하고 엑셀로 내려받는** 웹 분석 도구입니다.
왼쪽 메뉴에서 분석을 고르고, 옵션과 충격을 넣으면 요약 지표·차트·상세 표가 바로 나옵니다.
엑셀 파일에는 요약, 입력조건, 결과표, 엑셀 차트, 방법론 시트가 들어갑니다.

| 메뉴 | 상태 | 내용 |
|---|---|---|
| 산업연관표 분석 | 운영 | 한국은행 산업연관표 2015–2024년, 대분류 33·중분류 83부문. 수요 파급 · 가격 파급 · 공급 차질 · 연관 구조 |
| 국제 산업연관표 분석 | 운영 | OECD ICIO 2023년판 2011–2020년, 76개국+기타 지역 × 45개 산업. 국가 간 수요 파급 · 관세 · 공급망 차질 · 가격 파급 · GVC |
| 잠재성장률 추정 | 운영 | 한국 실질 GDP·고용·근로시간·자본서비스·물가·금리(분기) 또는 사용자 자료. 한국은행 방법론을 따른 3분류 12가지 방법 — 통계모형(선형·HP·실시간 HP·CF·Hamilton·BN·UC), 준구조모형(LW형·다변량 HP·구조 VAR), 생산함수모형(생산함수 접근·생산함수 다변량 필터) — 과 모형군 평균, 요인 기여도·자연실업률·중립금리. 한국은행 경제전망(반기·분기 경로)으로 끝점 보정 |
| 중립금리 추정 | 운영 | 기준금리·기대인플레이션·국고채 5·10년·잠재성장률. 6가지 방법 — HP 추세, 비관측 요인 모형, LW형 준구조모형, VAR 장기 예측(Lubik-Matthes형), 시장금리(5년 후 5년 선도), 잠재성장률 연계(오일러) — 실질·명목 중립금리와 통화정책 기조 |
| DSGE 모형 | 준비 중 | Dynare 7.1/Octave로 추정, 웹은 Python 선형 해법 |
| HANK 모형 | 준비 중 | `D:\Claude\Projects\HANK\hank2_soe` 연결(SSJ) |

## 로컬 실행

```bash
cd D:\Claude\Projects\macro-lab
.venv\Scripts\python.exe -m streamlit run app.py
```

브라우저에서 <http://localhost:8501> 을 엽니다. 가상환경을 새로 만들 때는 `python -m venv .venv` 다음 `pip install -r requirements-dev.txt` 를 실행합니다.

테스트(63건: 한국은행 공표 유발계수 대조, 항등식, 잠재성장률 필터 성질, 엑셀 출력):

```bash
.venv\Scripts\python.exe -m pytest -q tests
```

## 폴더 구조

```
app.py                     진입점 — modules/ 를 읽어 왼쪽 메뉴 자동 구성
core/                      공통 플랫폼 (모든 모듈이 같이 씀)
  registry.py              메뉴 자동 등록 (modules/*/module.py 의 META·render)
  result.py                공통 결과 형식 Result (inputs·summary·tables·notes·sources·charts)
  excel.py                 Result → 서식·차트 갖춘 xlsx
  ui.py, charts.py         결과 카드·표·다운로드 버튼, Plotly 차트
engines/io/                계산 엔진 (웹과 무관, 단독 테스트 가능)
  domestic.py              국내: 수량·가격·Ghosh·가설추출·유발계수
  icio.py                  국제: LU 분해 기반 L·b, 관세·공급망·외생가격·GVC
  classify.py, icio_meta.py  부문 연계표, 국가·산업 한글명, 국가 묶음
modules/                   메뉴 하나 = 폴더 하나
  a_home, b_io_domestic, c_io_international, d_potential_growth, e_dsge, f_hank
build/                     원자료 → 표준화 자료 변환 (오프라인)
  build_io_kr.py           ECOS 원자료 → data/processed/io_kr/*.npz
  build_icio.py            OECD zip → data/processed/icio/*.npz
data/processed/            웹이 읽는 자료 (배포에 포함)
data/raw/                  원자료 (배포 제외, 다시 받을 수 있음)
tests/                     엔진·엑셀 검증
```

### 새 분석 모듈 붙이기 (잠재성장률·DSGE·HANK)

1. `engines/<분야>/` 에 계산 엔진을 만들고 `tests/` 에 검증 테스트를 둡니다.
2. `modules/<순서>_<이름>/module.py` 에 다음 두 가지를 둡니다.
   - `META = {"title", "group", "order", "icon", "status": "ready", "slug"}`
   - `render()` 함수
3. `render()` 안에서 계산 결과를 `core.result.Result` 에 담아 `core.ui.summary_cards`, `table`, `excel_button` 을 호출합니다. 엑셀·방법론 상자는 자동으로 붙습니다.
4. 무거운 계산(DSGE 추정, HANK 야코비안)은 `build/` 에서 미리 돌려 `data/processed/` 에 저장하고, 웹에서는 읽기만 합니다.

## 자료 갱신

**한국은행 산업연관표 (ECOS Open API)**
`euro-chartpack/.env` 의 `ECOS_API_KEY` 를 씁니다. 표 코드는 다음과 같습니다.
- 2020년 기준: 271Y110·112~116 (대분류), 271Y120·122~126 (중분류)
- 2015년 기준: 271Y008·010~014, 271Y068·070~074

```bash
.venv\Scripts\python.exe build\fetch_ecos.py 271Y110 271Y112 271Y113 271Y114 271Y115 271Y116   # data/raw/ecos/*.json
.venv\Scripts\python.exe build\build_io_kr.py
```

새 연장표(예: 2025년)가 ECOS에 올라오면 위 두 줄만 다시 돌리면 연도 목록에 자동으로 추가됩니다.

**OECD ICIO**
- 2023년판(2011–2020년)은 OECD 통계 서버 공개 링크에서 받았습니다(`data/raw/icio/ICIO2023_*_SML.zip`).
- **2025년판(1995–2022년, 81개 경제권 × 50개 산업)**: OECD 새 서버(webfs-sti.oecd.org)가 스크립트 다운로드를 막아 자동으로 받지 못했습니다. 브라우저로 직접 받아야 합니다.
  1. <https://www.oecd.org/en/data/datasets/inter-country-input-output-tables.html> 에서 "2025 release (regular ICIO)"의 `2016-2022` 등 zip을 받습니다.
  2. `data/raw/icio/ICIO2025_2016-2022_SML.zip` 처럼 이름 앞에 `ICIO2025_` 를 붙여 저장합니다.
  3. `build\build_icio.py` 를 실행하면 화면의 'ICIO 판' 선택에 2025가 생깁니다. 국가·산업 한글명(50개 산업 포함)은 이미 들어 있습니다.

**잠재성장률 자료 (ECOS Open API)**
실질 GDP(200Y104), 경제활동인구(901Y027, 계절조정), 소비자물가(901Y009·901Y010), 제조업 평균가동률(901Y025), 실질 생산자본스톡(291Y504), 자본서비스 물량지수(291Y522 전산업 통크비스트), 기준금리(722Y001), 기대인플레이션(511Y003)을 ECOS 에서, 평균 근로시간을 OECD(2008년~)·Penn World Table(FRED, 그 이전 증가율)·사업체노동력조사(ECOS 901Y148·901Y052, 당해 연도 증가율)에서 받아 `data/processed/potential/kr_quarterly.csv` 로 만듭니다. 연간 자본서비스·근로시간은 Denton 방식으로 분기화합니다. 분기 GDP 가 새로 나오면 이 한 줄을 돌리고 push 합니다.

```bash
.venv\Scripts\python.exe build\build_potential.py
```

## 배포 (무료 공개: GitHub + Streamlit Community Cloud)

코드는 GitHub 저장소 `tskang89/macro-lab` 에 두고, Streamlit Community Cloud가 그 저장소를 실행합니다. 저장소에 push 하면 사이트도 자동으로 갱신됩니다.

1. share.streamlit.io 에 GitHub 계정으로 로그인합니다.
2. Create app → 저장소 `tskang89/macro-lab`, 브랜치 `main`, 파일 `app.py` 를 고릅니다.
3. Advanced settings 에서 Python 3.12 를 고르고 Deploy 합니다.

메모 사항
- `.npz` 자료(약 216MB, 파일당 22MB 이하)는 Git LFS 없이 일반 파일로 넣었습니다. GitHub 무료 LFS 대역폭(월 1GB)은 배포 몇 번이면 소진되기 때문입니다.
- 메모리: 국제표 한 해를 불러오면 약 320MB, 두 해를 캐시하면 약 1GB를 씁니다. Streamlit Cloud 한도(약 2.7GB) 안입니다.
- HANK처럼 메모리를 많이 쓰는 모듈을 붙여 한도를 넘으면 `Dockerfile` 로 컨테이너 호스팅(Hugging Face Spaces PRO 등)으로 옮깁니다. 2026-11 현재 Hugging Face 의 Docker Space 는 무료 계정에서 만들 수 없습니다.

## 방법론 요약

**잠재성장률** (`engines/potential/methods.py`, 상태공간 `statespace.py`): 모든 방법이 y=100·log(실질 GDP)에서 잠재 GDP 를 구하고 GDP 갭 = y − y*. 분류와 생산함수·준구조 설정은 한국은행 권지호 외(2019, 조사통계월보 2019.8)·이은경 외(2024, BOK 이슈노트 2024-33)를 따름.

**끝점 보정** (`engines/potential/forecast.py`, 전망 숫자 `data/processed/potential/bok_forecast.json`): 한국은행 경제전망보고서의 GDP 분기 경로·반기·연간 성장률로 실질 GDP 를 다음 해 4분기까지 연장(기본은 전년동기비 재현, 선택으로 전기비 경로). 취업자·실업률·물가는 전망치, 인구·근로시간·자본서비스는 최근 3년 추세, 가동률은 최근 평균, 금리·기대인플레이션은 마지막 값 유지. 상태공간모형(LW·생산함수 다변량 필터)은 모수를 실적 구간에서만 추정하고 전망 구간은 평활에만 씀. 새 경제전망(2·5·8·11월)이 나오면 JSON 을 고쳐 반영. 전망 시계 뒤 3년은 AR 예측 대신 중기 성장률 가정(기본 2.0%)으로 이어 붙여 끝점을 처리(AR 은 2008년 이후 평균 2.7%로 되돌아가 끝점 추세를 끌어올림). 기본 방법 구성은 한국은행 4개 모형(HP·LW형·생산함수 다변량 필터·생산함수 접근) 단순 평균. 생산함수모형은 'TFP 추세 고정 시점'(기본 2025Q4) 이후 성장을 순환(수요 압력)으로 보아 TFP 추세를 그 시점 기울기로 연장.
- 통계모형: 선형(구조변화 꺾은선), HP(λ=1600, AR(4) 12분기 연장으로 끝점 보정), 단방향 HP(실시간), CF 대역통과(주기 32분기 초과를 잠재로), Hamilton(2018) h=8·p=4, BN 필터(Kamber·Morley·Wong 2018, δ 고정 제약 최소제곱), 비관측 요인 모형(평활 추세+AR(2)).
- 준구조모형: LW형(IS곡선·필립스곡선·오쿤의 관계식, 칼만 필터 최우추정; λ_g=0.06·λ_z=0.03 고정, 잠재 GDP 수준 충격 표준편차 상한 0.25%; 실질금리 = 기준금리 − 소비자 1년 기대인플레이션; 중립금리·자연실업률 동시 추정), 다변량 HP(필립스곡선·오쿤, Laxton-Tetlow), 구조 VAR(Blanchard-Quah).
- 생산함수모형: 생산함수 접근(노동소득분배율 0.61; 잠재노동 = 15세이상인구×참가율 추세×(1−자연실업률)×평균 근로시간 추세, 또는 2024년 방식 고용률 추세; 자본 = 자본서비스 물량지수(추세로 간주); TFP 추세는 가동률 효과 제거 후 HP), 생산함수 다변량 필터(Fleischman-Roberts/FRB/US형: 생산함수 + 참가율·고용률·근로시간 오쿤식 + 필립스곡선을 하나의 상태공간모형으로; 갭 AR 계수 합 ≤ 0.9).
- 노동의 질 지수는 미시자료가 필요해 넣지 않음(그 효과는 TFP 에 포함).
- 방법 평균: 기본은 모형군(통계·준구조·생산함수) 평균의 평균, 선택하면 전 방법 단순 평균.

- **국내 수량모형**: Δx = (I−Ad)⁻¹ Δfd. Ad는 한국은행 생산유발계수표에서 역산했으므로 유발계수가 공표치와 일치합니다(테스트로 확인). 최종수요의 국산 비율은 최종재 수입을 수출 외 최종수요에 비례 배분해 구합니다.
- **국내 가격모형**: Δp' = (Δπ'Am + Δw')L. 환율·유가·수입품 가격·단위당 임금·이윤·세금 충격을 넣을 수 있고, 소비자물가는 가계소비 가중입니다.
- **공급 차질**: Ghosh 전방 파급과 레온티에프 후방 파급을 나란히 보입니다(합산하지 않음).
- **ICIO**: A = Z/x, L = (I−A)⁻¹ 는 역행렬 대신 LU 분해로 계산합니다(3,465차, 약 1초).
  - 관세: 탄력성 × 관세율로 수입 감소를 정하고, 국내·제3국 대체를 거친 수요 이동으로 파급합니다. 가격 쪽은 수입 중간재 비용 → 국제 가격모형으로 계산합니다.
  - GVC: VAX = diag(v)·L·E 로 DVA·FVA·DVX 를, VAFD = diag(v)·L·Y 로 부가가치 기준 교역을 구합니다.
- 모든 계산은 고정계수·1차 파급입니다. 소득→소비 2차 효과와 가격·환율·금리의 반응은 들어 있지 않습니다. 화면과 엑셀의 '방법론' 상자에 가정을 적어 두었습니다.

## 자료 출처

- 한국은행, 산업연관표(투입산출표, 생산자가격), ECOS
- OECD (2023), Inter-Country Input-Output Tables, <https://oe.cd/icio>

개인 연구용 분석 도구이며 어떤 기관의 공식 견해도 아닙니다.
