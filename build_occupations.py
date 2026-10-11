"""
build_occupations.py  (v2: matched to O*NET 31.0 CSV files)

Joins O*NET interests, job zones, and skills with BLS OEWS wages into one
occupations.json that the web app loads directly.

RUN FROM the project folder that contains db_31_0_csv/ :
    pip install pandas openpyxl
    python build_occupations.py

BLS is optional for the first run: if bls/national_oews.xlsx is missing,
wages are left empty and you get a warning. Download it later from
bls.gov/oes/tables.htm (national file, xlsx) and re-run.

Keep the O*NET attribution text required by their license on your site.
"""
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ONET_DIR = Path("db_31_0_csv")
BLS_FILE = Path("bls/national_oews.xlsx")
ONET_VERSION = "31.0"
BLS_LABEL = "FILL_IN (e.g. May 2024)"   # the release year shown on the BLS file
OUT_FILE = "occupations.json"
SKILL_NAMES_FILE = "skill_names.txt"

# Riasec element IDs in career_interest_types.csv (Scale ID "OI")
RIASEC = {
    "1.B.1.a": "realistic",
    "1.B.1.b": "investigative",
    "1.B.1.c": "artistic",
    "1.B.1.d": "social",
    "1.B.1.e": "enterprising",
    "1.B.1.f": "conventional",
}

# The survey's skills checklist. The script checks these against the real data.
CHECKLIST = [
    "Writing", "Speaking", "Active Listening", "Mathematics", "Critical Thinking",
    "Complex Problem Solving", "Programming", "Systems Analysis", "Troubleshooting",
    "Persuasion", "Negotiation", "Instructing", "Service Orientation",
    "Time Management", "Judgment and Decision Making", "Quality Control Analysis",
]


def read_onet(name: str) -> pd.DataFrame:
    path = ONET_DIR / f"{name}.csv"
    if not path.exists():
        sys.exit(f"Missing {path}. Run `ls {ONET_DIR}` and check the file name.")
    return pd.read_csv(path, dtype=str)


def require_columns(df: pd.DataFrame, cols: list, name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        sys.exit(f"{name}.csv is missing columns {missing}. Found: {list(df.columns)}")


# ---------- 0. Scale ranges (read from the data, not hardcoded) ----------
scales = read_onet("scales_reference")
RANGE = {r["Scale ID"]: (float(r["Minimum"]), float(r["Maximum"])) for _, r in scales.iterrows()}
for needed in ("OI", "IM", "LV"):
    if needed not in RANGE:
        sys.exit(f"Scale {needed} not found in scales_reference.csv")


def norm(values, scale_id):
    lo, hi = RANGE[scale_id]
    return (values.astype(float) - lo) / (hi - lo)


# ---------- 1. Master list ----------
occ = read_onet("occupation_data")
require_columns(occ, ["O*NET-SOC Code", "Title", "Description"], "occupation_data")
occ = occ.rename(columns={"O*NET-SOC Code": "onet_code", "Title": "title", "Description": "description"})

# ---------- 2. Interests (RIASEC, Scale ID "OI") ----------
ci = read_onet("career_interest_types")
require_columns(ci, ["O*NET-SOC Code", "Element ID", "Scale ID", "Data Value", "Domain Source"], "career_interest_types")
ci = ci[ci["Scale ID"] == "OI"].copy()
ci["type"] = ci["Element ID"].map(RIASEC)
ci = ci.dropna(subset=["type"])
ci["value"] = norm(ci["Data Value"], "OI")
riasec = ci.pivot_table(index="O*NET-SOC Code", columns="type", values="value").reset_index()
riasec = riasec.rename(columns={"O*NET-SOC Code": "onet_code"})
# Some interest scores are machine-learning estimates rather than analyst/expert ratings.
# Keep the source so the UI can be honest about it.
interest_src = (
    ci.groupby("O*NET-SOC Code")["Domain Source"].first().reset_index()
    .rename(columns={"O*NET-SOC Code": "onet_code", "Domain Source": "interest_source"})
)
riasec = riasec.merge(interest_src, on="onet_code")

# ---------- 3. Job Zones (preparation level 1-5) ----------
jz = read_onet("job_zones")
require_columns(jz, ["O*NET-SOC Code", "Job Zone"], "job_zones")
jz = jz[["O*NET-SOC Code", "Job Zone"]].rename(columns={"O*NET-SOC Code": "onet_code", "Job Zone": "job_zone"})
jz["job_zone"] = jz["job_zone"].astype(int)

# ---------- 4. Skills: essential + transferable, importance (IM) and level (LV) ----------
frames = []
for fname, kind in [("essential_skills", "essential"), ("transferable_skills", "transferable")]:
    d = read_onet(fname)
    require_columns(d, ["O*NET-SOC Code", "Element Name", "Scale ID", "Data Value", "Recommend Suppress"], fname)
    d = d[d["Scale ID"].isin(["IM", "LV"])]
    d = d[d["Recommend Suppress"] != "Y"].copy()   # drop values O*NET recommends suppressing
    d["skill_type"] = kind
    frames.append(d)
sk = pd.concat(frames, ignore_index=True)
sk["Data Value"] = sk["Data Value"].astype(float)
wide = sk.pivot_table(
    index=["O*NET-SOC Code", "Element Name", "skill_type"], columns="Scale ID", values="Data Value"
).reset_index().dropna(subset=["IM", "LV"])
wide["importance"] = norm(wide["IM"], "IM")
wide["level"] = norm(wide["LV"], "LV")

# Keep ALL skills per occupation (about 30), sorted by importance, so the gap
# analysis can look up any checklist skill. The app can show the top few.
skills_by_occ = {}
for code, g in wide.sort_values("importance", ascending=False).groupby("O*NET-SOC Code"):
    skills_by_occ[code] = [
        {"name": r["Element Name"], "type": r["skill_type"],
         "importance": round(r["importance"], 3), "level": round(r["level"], 3)}
        for _, r in g.iterrows()
    ]

# Compare the survey checklist against the real skill names
all_names = sorted(wide["Element Name"].unique())
Path(SKILL_NAMES_FILE).write_text("\n".join(all_names))
missing_checklist = [c for c in CHECKLIST if c not in all_names]
print(f"{len(all_names)} distinct skill names written to {SKILL_NAMES_FILE}")
if missing_checklist:
    print("WARNING: these checklist skills do not match any O*NET skill name exactly:")
    for m in missing_checklist:
        print("   -", m)
    print("Edit the survey checklist to use names from skill_names.txt.")
else:
    print("All checklist skills match O*NET skill names.")

# ---------- 5. BLS wages (optional on first run) ----------
if BLS_FILE.exists():
    bls = pd.read_excel(BLS_FILE, dtype=str)
    require_columns(bls, ["OCC_CODE", "O_GROUP", "A_MEDIAN"], "BLS file")
    bls = bls[bls["O_GROUP"] == "detailed"].copy()
    bls["median_wage"] = pd.to_numeric(bls["A_MEDIAN"], errors="coerce")  # '*' and '#' become NaN
    bls["wage_note"] = bls["A_MEDIAN"].apply(
        lambda v: "top-coded or suppressed in BLS data" if v in ("#", "*") else ""
    )
    bls = bls[["OCC_CODE", "median_wage", "wage_note"]].rename(columns={"OCC_CODE": "soc7"})
    have_wages = True
else:
    print(f"WARNING: {BLS_FILE} not found. Wages will be empty. Download it and re-run.")
    bls = pd.DataFrame(columns=["soc7", "median_wage", "wage_note"])
    have_wages = False

# ---------- 6. Join ----------
df = occ.merge(riasec, on="onet_code").merge(jz, on="onet_code")
df["soc7"] = df["onet_code"].str[:7]          # 15-2051.01 -> 15-2051 (BLS uses the 7-character code)
df = df.merge(bls, on="soc7", how="left")
df["skills"] = df["onet_code"].map(skills_by_occ)

core = list(RIASEC.values()) + ["job_zone"]
before = len(df)
df = df.dropna(subset=core)
df = df[df["skills"].notna()]
print(f"Kept {len(df)} of {before} occupations that have interests, job zone and skills")
if have_wages:
    print(f"{df['median_wage'].isna().sum()} kept occupations have no BLS wage (show 'wage not available')")
print(f"Interest score sources: {df['interest_source'].value_counts().to_dict()}")

# ---------- 7. Export ----------
records = df.where(df.notna(), None).to_dict(orient="records")
payload = {
    "meta": {
        "built_on": date.today().isoformat(),
        "onet_version": ONET_VERSION,
        "bls_source": f"BLS OEWS national, {BLS_LABEL}" if have_wages else None,
        "wage_label": "Occupational median annual wage, not a career changer's starting pay",
    },
    "occupations": records,
}
with open(OUT_FILE, "w") as f:
    json.dump(payload, f, indent=1)
print(f"Wrote {OUT_FILE}")

# ---------- 8. Sanity checks: compare these by hand against onetonline.org ----------
for code in ["11-1011.00", "15-2051.00", "17-2141.00", "13-1161.00"]:
    row = df[df["onet_code"] == code]
    if row.empty:
        print(f"WARNING: {code} not in output. Check the code or a dropped row.")
    else:
        r = row.iloc[0]
        print(f"{code} {r['title']} | zone {r['job_zone']} | wage {r['median_wage']} | "
              f"R={r['realistic']:.2f} I={r['investigative']:.2f} | {len(r['skills'])} skills")