"""중립금리 추정 전용 앱.  streamlit run app_neutral.py"""
from core.shell import run

run(["neutral-rate"], page_title="중립금리 추정", page_icon=":material/balance:")
