"""ECOS 통계표 원자료(JSON) 내려받기: python build/fetch_ecos.py 271Y110 271Y112 ...
키: 환경변수 ECOS_API_KEY, 없으면 euro-chartpack/.env"""
import json, urllib.request, sys, os
key=os.environ.get("ECOS_API_KEY") or [l.split("=",1)[1].strip() for l in open(r"D:\Claude\Projects\euro-chartpack\.env",encoding="utf-8") if l.startswith("ECOS_API_KEY=")][0]
for code in sys.argv[1:]:
    out=[]; start=1
    while True:
        url=f"https://ecos.bok.or.kr/api/StatisticSearch/{key}/json/kr/{start}/{start+9999}/{code}/A/2015/2025"
        d=json.load(urllib.request.urlopen(url,timeout=120))
        if "StatisticSearch" not in d: print(code,d); break
        rows=d["StatisticSearch"]["row"]; out+=rows
        tot=d["StatisticSearch"]["list_total_count"]
        start+=10000
        if start>tot: break
    os.makedirs("data/raw/ecos",exist_ok=True)
    json.dump(out,open(f"data/raw/ecos/{code}.json","w",encoding="utf-8"),ensure_ascii=False)
    print(code,len(out), out[0] if out else None)
