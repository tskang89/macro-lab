"""산업연관표 분석 전용 앱(국내 · 기본부문 · 국제).  streamlit run app_io.py"""
from core.shell import run

run(["io-domestic", "io-basic", "io-international"], page_title="산업연관표 분석",
    page_icon=":material/account_tree:", caption="한국은행 산업연관표 · OECD 국제산업연관표")
