"""Build the HazeCast MY demo video: slides + app shots, piper narration, ffmpeg."""
import json
import os
import subprocess
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

V = Path("/home/claude/video")
SH, FR, AU, SC = V / "shots", V / "frames", V / "audio", V / "scenes"
for d in (FR, AU, SC):
    d.mkdir(exist_ok=True)
W, H = 1920, 1080

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
* { box-sizing: border-box; margin: 0; }
body { width: 1920px; height: 1080px; background: #E8ECEE; color: #1D2B33; font-family: 'IBM Plex Sans', sans-serif; overflow: hidden; }
.slide { padding: 90px 150px; height: 100%; display: flex; flex-direction: column; justify-content: center; }
h1 { font-family: 'Bricolage Grotesque'; font-size: 150px; font-weight: 700; letter-spacing: -0.02em; line-height: 1; }
h2 { font-family: 'Bricolage Grotesque'; font-size: 76px; font-weight: 700; line-height: 1.05; margin-bottom: 48px; max-width: 1500px; }
.sub { font-size: 50px; color: #3E5059; margin-top: 30px; max-width: 1400px; line-height: 1.25; }
.meta { font-size: 30px; color: #5B6B73; margin-top: 80px; }
.rule { width: 160px; height: 10px; background: #B7791F; margin-bottom: 50px; }
.stat { display: flex; align-items: baseline; gap: 40px; margin: 14px 0; }
.num { font-family: 'Bricolage Grotesque'; font-weight: 700; font-size: 74px; min-width: 520px; color: #1D2B33; }
.lbl { font-size: 36px; color: #3E5059; line-height: 1.3; max-width: 1050px; }
ul { font-size: 40px; line-height: 1.45; padding-left: 40px; max-width: 1550px; }
li { margin: 12px 0; }
.shot { position: absolute; left: 50%; top: 40px; transform: translateX(-50%); width: 1600px; height: 900px;
        border-radius: 12px; box-shadow: 0 10px 40px rgba(29,43,51,0.25); }
.cap { position: absolute; left: 0; right: 0; bottom: 0; height: 140px; display: flex; align-items: center; justify-content: center;
       font-size: 34px; font-weight: 500; padding: 0 120px; text-align: center; }
.tag { position: absolute; top: 60px; right: 80px; font-size: 26px; color: #5B6B73; }
"""

SCENES = [
    {"id": "01_title", "kind": "slide", "html": """<div class='slide'><div class='rule'></div><h1>HazeCast MY</h1>
      <div class='sub'>A one-day warning for Malaysia's haze, for every DOE air-quality station.</div>
      <div class='meta'>Neural Storm 2026 &nbsp;|&nbsp; Green Planet, GP-01 &nbsp;|&nbsp; Shermaine Yap Shi Min</div></div>""",
     "say": "Every haze season, Malaysians find out the air is unhealthy only when the index has already turned red. "
            "The official A P I M S site shows the last hour, never tomorrow. "
            "HazeCast M Y gives every monitoring station a day of warning."},
    {"id": "02_problem", "kind": "slide", "html": """<div class='slide'><h2>The air turns red before anyone is warned</h2>
      <div class='stat'><div class='num'>48 stations</div><div class='lbl'>Unhealthy at the same time in September 2019. Schools closed across several states.</div></div>
      <div class='stat'><div class='num'>20 areas</div><div class='lbl'>Unhealthy to hazardous on 3 September 2026 (The Star).</div></div>
      <div class='stat'><div class='num'>1 day</div><div class='lbl'>The notice a school, a clinic or an employer of outdoor workers actually needs.</div></div></div>""",
     "say": "In September twenty nineteen, forty eight stations were unhealthy at the same time, and schools closed across several states. "
            "It is still happening. On the third of September twenty twenty six, twenty areas recorded unhealthy to hazardous air. "
            "Closing a school or protecting outdoor workers needs a day's notice, not an hour."},
    {"id": "03_map", "kind": "shot", "img": "s3_map_peak.jpg",
     "cap": "Replay of 17 Sep 2019, 08:00: 35 stations flagged for unhealthy air tomorrow",
     "say": "This is HazeCast replaying the morning of the seventeenth of September, twenty nineteen. "
            "Every forecast you see comes from a model that never saw that year. "
            "Thirty five stations are flagged for unhealthy air tomorrow, across the Klang Valley, Melaka and Johor."},
    {"id": "04_agent", "kind": "shot", "img": "s4_agent.jpg",
     "cap": "HazeAgent decides ALERT, WATCH, CLEAR or INSUFFICIENT DATA, and writes a bulletin by state",
     "say": "Haze Agent turns forecasts into decisions: alert, watch, clear, or insufficient data, where it declines to guess. "
            "It writes a bulletin by state, and labels each episode as a widespread, transboundary haze signature, or a local build up."},
    {"id": "05_station", "kind": "shot", "img": "s5_station.jpg",
     "cap": "Forecast, 80% range, chance of unhealthy air, and the reason why",
     "say": "For each station you get the twenty four and forty eight hour forecast, an eighty percent range, and the chance of unhealthy air. "
            "The dotted line is what the model predicted a day earlier, against what really happened. "
            "The driver chart explains why, using exact contributions from the model."},
    {"id": "06_nearmiss", "kind": "shot", "img": "s6_nearmiss.jpg",
     "cap": "We show the misses too: an 87% alert for Klang, but readings stayed below 100",
     "say": "We show its mistakes too. On the eleventh of August twenty nineteen, it gave Klang an eighty seven percent chance of unhealthy air. "
            "The readings peaked just below one hundred, and then fell."},
    {"id": "07_how", "kind": "slide", "html": """<div class='slide'><h2>How it works</h2><ul>
      <li>3.8 million station-hours of official DOE hourly data, 2013 to 2022</li>
      <li>36 leakage-free features: the station's own trend, its 3 nearest stations, regional and nationwide haze</li>
      <li>LightGBM + Ridge ensemble that predicts the change from now</li>
      <li>Conformal prediction for honest 80% ranges</li>
      <li>Walk-forward testing on 2019, 2020 and 2021–22, years the model never saw</li></ul></div>""",
     "say": "Under the hood: three point eight million station hours of official hourly data. "
            "Thirty six features per station, including its own trend, its three nearest neighbours, and regional and nationwide haze signals. "
            "An ensemble of Light G B M and ridge regression predicts the change from now, conformal prediction gives the ranges, "
            "and everything is tested walk forward, on years the model never saw."},
    {"id": "08_results", "kind": "slide", "html": """<div class='slide'><h2>Results on unseen years</h2>
      <div class='stat'><div class='num'>6.70 vs 7.35</div><div class='lbl'>Error of the 24-hour forecast in API points, against "tomorrow = today" (8.8% better)</div></div>
      <div class='stat'><div class='num'>0.84 vs 0.67</div><div class='lbl'>Alert ranking (PR-AUC) in 2019, against persistence; better in every test period</div></div>
      <div class='stat'><div class='num'>159 of 169</div><div class='lbl'>New haze episodes in 2019 warned in advance, median 19 hours ahead</div></div>
      <div class='stat'><div class='num'>79%</div><div class='lbl'>Coverage of the 80% ranges</div></div>
      <div class='stat'><div class='num'>0.79 s</div><div class='lbl'>To forecast all 67 stations on a 2-core CPU</div></div></div>""",
     "say": "Results. The twenty four hour error is six point seven A P I points, against seven point three five for simply assuming tomorrow equals today. "
            "Alert ranking beats both simple rules in every test period. "
            "In twenty nineteen, it warned of one hundred and fifty nine of one hundred and sixty nine new haze episodes, a median nineteen hours ahead. "
            "The ranges cover seventy nine percent, against an eighty percent target. "
            "And all sixty seven stations are forecast in under a second."},
    {"id": "09_evidence", "kind": "shot", "img": "s9b_evidence.jpg",
     "cap": "Every number is in the app's evidence tab and in the repository",
     "say": "Every number, including the ones where a simple rule does better, is in the evidence tab and in the repository."},
    {"id": "10_end", "kind": "slide", "html": """<div class='slide'><h2>Limits, and what's next</h2><ul>
      <li>Sees only the composite index, not individual pollutants</li>
      <li>Ranges are too narrow during haze; the public archive ends in May 2022</li>
      <li>Next: wind, rain and fire-hotspot data, to see haze before it arrives</li></ul>
      <div class='meta' style='font-size:40px;color:#1D2B33'>Live: hazecast-my.streamlit.app<br>Code: github.com/ShermaineYap/neural-storm</div></div>""",
     "say": "Its limits: it sees only the composite index, its ranges are too narrow during haze, and the public archive ends in twenty twenty two. "
            "Next, we add wind, rain and fire hotspot data, so it can see haze coming before it arrives. "
            "HazeCast M Y. Build A I that matters, and give Malaysia a day's warning."},
]


def page_html(s):
    if s["kind"] == "slide":
        body = s["html"]
    else:
        body = (f"<img class='shot' src='file://{SH / s['img']}'>"
                f"<div class='cap'>{s['cap']}</div>")
    return f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}</body></html>"


def render_frames():
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": W, "height": H})
        for s in SCENES:
            f = FR / f"{s['id']}.html"
            f.write_text(page_html(s))
            pg.goto(f"file://{f}")
            pg.wait_for_load_state("networkidle")
            time.sleep(0.8)
            pg.screenshot(path=str(FR / f"{s['id']}.png"))
        b.close()


def tts():
    for s in SCENES:
        out = AU / f"{s['id']}.wav"
        subprocess.run(["python3", "-m", "piper", "-m", str(V / "voice/en_US-lessac-high.onnx"),
                        "-f", str(out), "--sentence-silence", "0.35"],
                       input=s["say"].encode(), check=True, capture_output=True)


def dur(path):
    return float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                          "-of", "csv=p=0", str(path)]).decode().strip())


def scenes():
    for s in SCENES:
        a = AU / f"{s['id']}.wav"
        d = dur(a) + 1.1
        tmp, out = SC / f"{s['id']}.tmp.mp4", SC / f"{s['id']}.mp4"
        fade = f"fade=t=in:st=0:d=0.4,fade=t=out:st={d - 0.4:.2f}:d=0.4"
        cmd = ["ffmpeg", "-y", "-v", "error", "-framerate", "30", "-i", str(FR / f"{s['id']}.png"),
               "-i", str(a),
               "-filter_complex",
               f"[0:v]format=yuv420p,loop=-1:1,trim=duration={d:.2f},{fade},setsar=1[v];"
               f"[1:a]adelay=500|500,apad,atrim=duration={d:.2f},aresample=48000[a]",
               "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
               "-pix_fmt", "yuv420p", "-r", "30", "-c:a", "aac", "-b:a", "160k", "-ac", "2", str(tmp)]
        subprocess.run(cmd, check=True)
        os.replace(tmp, out)
        print(s["id"], round(d, 1), flush=True)


def join():
    lst = SC / "list.txt"
    lst.write_text("".join(f"file '{SC / (s['id'] + '.mp4')}'\n" for s in SCENES))
    out = V / "HazeCast_MY_demo.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-c", "copy", "-movflags", "+faststart", str(V / "tmp_demo.mp4")], check=True)
    os.replace(V / "tmp_demo.mp4", out)
    print("total", round(dur(out), 1))


if __name__ == "__main__":
    import sys
    steps = sys.argv[1:] or ["frames", "tts", "scenes", "join"]
    if "frames" in steps: render_frames()
    if "tts" in steps: tts()
    if "scenes" in steps: scenes()
    if "join" in steps: join()
