"""Aggregate the LLM_Deployment census (merged.xlsx) into the two small CSVs the
motivation_geo figure reads. Run once; the outputs are committed so the figure
does not depend on the 48 MB xlsx.

    python3 build_supply_csv.py \
        ../../datasets/llm_deployment/merged.xlsx .

Needs: pandas, openpyxl.
"""
import sys
import pandas as pd

XLSX = sys.argv[1] if len(sys.argv) > 1 else "../../datasets/llm_deployment/merged.xlsx"
OUTDIR = sys.argv[2] if len(sys.argv) > 2 else "."

df = pd.read_excel(XLSX).rename(columns={"国家代码": "cc", "asn组织": "org"})
tot = len(df)


def prov(o):
    """Normalize raw ASN org strings into cloud-provider buckets."""
    if not isinstance(o, str):
        return "Other"
    s = o.lower()
    if "amazon" in s or "aws" in s:
        return "Amazon"
    if "cloudflare" in s:
        return "Cloudflare"
    if "akamai" in s or "linode" in s:
        return "Akamai"
    if "alibaba" in s or "aliyun" in s:
        return "Alibaba"
    if "tencent" in s:
        return "Tencent"
    if any(k in s for k in ("chinanet", "china telecom", "china unicom",
                            "china169", "chinamobile")):
        return "China Telco"
    if "microsoft" in s or "azure" in s:
        return "Microsoft"
    if "google" in s:
        return "Google"
    if "hetzner" in s:
        return "Hetzner"
    return "Other"


df["prov"] = df["org"].map(prov)

# deployment share per country (all 144 codes)
(df["cc"].value_counts() / tot * 100).round(4).to_csv(
    f"{OUTDIR}/supply_country_share.csv", header=["supply_pct"])

# provider mix (%) within each of the top-10 serving countries
PROVS = ["Amazon", "Cloudflare", "Akamai", "Alibaba", "Tencent", "China Telco",
         "Microsoft", "Google", "Hetzner", "Other"]
rows = []
for c in df["cc"].value_counts().head(10).index:
    sub = df[df["cc"] == c]
    n = len(sub)
    rows.append({"cc": c, "n": n,
                 **{p: round(100 * (sub["prov"] == p).sum() / n, 2) for p in PROVS}})
pd.DataFrame(rows).to_csv(f"{OUTDIR}/supply_country_provider.csv", index=False)

print(f"wrote supply_country_share.csv & supply_country_provider.csv "
      f"(total endpoints: {tot})")
