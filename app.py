"""거시 분석 플랫폼 진입점(모든 메뉴).

    streamlit run app.py

메뉴는 modules/ 폴더를 자동으로 읽어 만든다(core/registry.py).
새 분석을 추가하려면 modules/<이름>/module.py 에 META 와 render() 를 만들면 된다.
주제별 전용 앱은 app_potential.py · app_neutral.py · app_io.py (core/shell.py 참고).
"""
from core.shell import run

run(caption="산업연관 · 국제산업연관 · 잠재성장률 · (예정) DSGE · HANK")
