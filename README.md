# 거시 분석 플랫폼 (macro-lab)

산업연관표와 거시 모형으로 **시나리오를 넣어 파급효과를 계산하고 엑셀로 내려받는** 웹 분석 도구입니다.
왼쪽 메뉴에서 분석을 고르고, 옵션과 충격을 넣으면 요약 지표·차트·상세 표가 바로 나옵니다.
엑셀 파일에는 요약, 입력조건, 결과표, 엑셀 차트, 방법론 시트가 들어갑니다.

| 메뉴 | 상태 | 내용 |
|---|---|---|
| 산업연관표 분석 | 운영 | 한국은행 산업연관표 2015–2024년, 대분류 33·중분류 83부문. 수요 파급 · 가격 파급 · 공급 차질 · 연관 구조 |
| 국제 산업연관표 분석 | 운영 | OECD ICIO 2023년판 2011–2020년, 76개국+기타 지역 × 45개 산업. 국가 간 수요 파급 · 관세 · 공급망 차질 · 가격 파급 · GVC |
| 잠재성장률 추정 | 준비 중 | HP·칼만필터 UC·생산함수 접근 비교 |
| DSGE 모형 | 준비 중 | Dynare 7.1/Octave로 추정, 웹은 Python 선형 해법 |
| HANK 모형 | 준비 중 | `D:\Claude\Projects\HANK\hank2_soe` 연결(SSJ) |

## 로컬 실행

```bash
cd D:\Claude\Projects\macro-lab
.venv\Scripts\python.exe -m streamlit run app.py
```

브라우저에서 <http://localhost:8501> 을 엽니다. 가상환경을 새로 만들 때는 `python -m venv .venv` 다음 `pip install -r requirements-dev.txt` 를 실행합니다.

테스트(34건: 한국은행 공표 유발계수 대조, 항등식, 엑셀 출력):

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
