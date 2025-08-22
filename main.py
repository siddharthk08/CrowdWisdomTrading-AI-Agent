import os
import json
import argparse
import time
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any
import requests
import feedparser
import pandas as pd
from dateutil import parser as dtparser
import pytz
from dotenv import load_dotenv
load_dotenv()

# --- CrewAI imports ---
from crewai import Agent, Task, Crew, Process

# --- LiteLLM (Groq) ---
from litellm import completion

SEC_ATOM_RECENT = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&count=200&output=atom"
DEFAULT_USER_AGENT = os.getenv("SEC_USER_AGENT", "siddharthkoli843@gmail.com CrewAI-SEC-Task")

MODEL = os.getenv("GROQ_MODEL", "groq/llama-3.1-8b-instant")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

assert GROQ_API_KEY, "Please set GROQ_API_KEY env var (free at https://console.groq.com)"

HEADERS = {"User-Agent": DEFAULT_USER_AGENT}

def within_last_hrs(dt: datetime, hours: int = 48) -> bool:
    return (datetime.now(timezone.utc) - dt).total_seconds() <= hours * 3600

def parse_atom_datetime(dt_str: str) -> datetime:
    # feedparser returns struct_time; we may also have ISO strings from 'updated'/'published'
    try:
        return dtparser.parse(dt_str).astimezone(timezone.utc)
    except Exception:
        return datetime.now(timezone.utc)

def fetch_sec_recent_filings() -> List[Dict[str, Any]]:
    """
    Fetch latest SEC filings via Atom feed and filter last 48 hours.
    Returns a list of dicts with keys: title, link, updated, summary, company, form_type
    """
    resp = requests.get(SEC_ATOM_RECENT, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    feed = feedparser.parse(resp.text)

    items = []
    for e in feed.entries:
        updated = parse_atom_datetime(e.get("updated", e.get("published", "")))
        if not within_last_hrs(updated, 48):
            continue
        title = e.get("title", "")
        link = e.get("link", "")
        summary = e.get("summary", "")
        # crude parse: "Company Name (CIK: ... ) (Filing) Form 8-K"
        form_type = ""
        company = ""
        t = title or ""
        if "form" in t.lower():
            # try to grab the form at the end
            parts = t.split("Form ")
            if len(parts) > 1:
                form_type = "Form " + parts[-1].strip()
        # company approx: before " - " or " (Filer)"
        if " - " in t:
            company = t.split(" - ")[0].strip()
        elif " (" in t:
            company = t.split(" (")[0].strip()
        items.append({
            "title": title,
            "link": link,
            "updated": updated.isoformat(),
            "summary": summary,
            "company": company,
            "form_type": form_type
        })
    return items

def filter_form4_insiders(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for x in entries:
        f = x.get("form_type", "").lower()
        t = x.get("title", "").lower()
        if "form 4" in f or "form 4" in t:
            out.append(x)
    return out

def group_insider_activity_today(form4s: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # aggregate by company for items dated today (UTC -> local Asia/Kolkata by default for "today")
    india = pytz.timezone("Asia/Kolkata")
    today_local = datetime.now(india).date()
    agg = {}
    for f in form4s:
        try:
            dt_utc = dtparser.parse(f["updated"]).astimezone(india)
            if dt_utc.date() == today_local:
                company = f.get("company") or "Unknown"
                agg[company] = agg.get(company, 0) + 1
        except Exception:
            pass
    rows = [{"company": k, "form4_count_today": v} for k, v in agg.items()]
    rows.sort(key=lambda r: r["form4_count_today"], reverse=True)
    return rows

def run_sentiment_groq(text: str) -> str:
  def run_sentiment_groq(text: str) -> str:
    """Call Groq via LiteLLM for quick sentiment: Positive / Neutral / Negative + short reason"""
    prompt = f"""
Classify the sentiment (Positive / Neutral / Negative) and give a short one-sentence rationale.

Text:
{text}

Answer as: <label> - <reason>
"""
    r = completion(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=120
    )
    return r["choices"][0]["message"]["content"]


# ----------------- Define Agents -----------------
class DataFetcherAgent(Agent):
    def __init__(self):
        super().__init__(
            role="SEC Data Fetcher",
            goal="Fetch SEC recent filings for the last 48 hours.",
            backstory="Specialist in SEC EDGAR feeds and compliant, polite scraping.",
            allow_delegation=False
        )

    def fetch(self) -> List[Dict[str, Any]]:
        return fetch_sec_recent_filings()

class InsiderTradeAgent(Agent):
    def __init__(self):
        super().__init__(
            role="Insider Trade Analyzer",
            goal="Extract insider (Form 4) activity from fetched filings.",
            backstory="Understands insider reporting and aggregates activity.",
            allow_delegation=False
        )
    def extract(self, entries: List[Dict[str, Any]]) -> Dict[str, Any]:
        form4s = filter_form4_insiders(entries)
        most_active_today = group_insider_activity_today(form4s)
        return {"form4": form4s, "most_active_today": most_active_today}

class SentimentAgent(Agent):
    def __init__(self):
        super().__init__(
            role="Sentiment Analyst",
            goal="Run LLM-based sentiment on up to 10 creators/companies based on recent filings context.",
            backstory="Fast and frugal analyst using Groq models via LiteLLM.",
            allow_delegation=False
        )

    def analyze(self, creators: List[str], context_items: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        # build a tiny context per creator (recent filings mentioning the name)
        out = []
        for c in creators[:10]:
            related = [i for i in context_items if c.lower() in (i.get("company","").lower() + " " + i.get("title","").lower())]
            snippet = "\\n".join([f"- {i.get('title','')} ({i.get('updated','')})" for i in related[:5]])
            text = f"Creator/Company: {c}\\nRecent context:\\n{snippet or 'No direct filings; general market context.'}"
            try:
                sentiment = run_sentiment_groq(text)
            except Exception as e:
                sentiment = f"Error: {e}"
            out.append({"creator": c, "sentiment": sentiment})
        return out

class ReportAgent(Agent):
    def __init__(self):
        super().__init__(
            role="Reporting Specialist",
            goal="Compile JSON, CSV, Excel, and PDF reports and print a console summary.",
            backstory="Turns raw signals into clean, decision-ready reports.",
            allow_delegation=False
        )

    def save(self, data: Dict[str, Any], out_dir: str) -> Dict[str, str]:
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = os.path.join(out_dir, f"sec_report_{ts}.json")
        csv_path = os.path.join(out_dir, f"sec_report_{ts}.csv")
        excel_path = os.path.join(out_dir, f"sec_report_{ts}.xlsx")
        pdf_path = os.path.join(out_dir, f"sec_report_{ts}.pdf")

        # Save JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # Prepare insider activity table
        today_rows = data.get("insiders", {}).get("most_active_today", [])
        if today_rows:
            df = pd.DataFrame(today_rows)
        else:
            df = pd.DataFrame([{"company": "No insider activity today", "form4_count_today": 0}])

        # Save CSV + Excel
        df.to_csv(csv_path, index=False)
        df.to_excel(excel_path, index=False)

        # Save PDF
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet

        doc = SimpleDocTemplate(pdf_path, pagesize=A4)
        styles = getSampleStyleSheet()
        elements = []

        elements.append(Paragraph("SEC Insider Activity Report", styles['Title']))
        elements.append(Spacer(1, 12))

        table_data = [["Company", "Form 4 Count Today"]] + df.values.tolist()
        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.grey),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0,0), (-1,0), 12),
            ('BACKGROUND', (0,1), (-1,-1), colors.beige),
        ]))
        elements.append(table)

        doc.build(elements)

        return {"json": json_path, "csv": csv_path, "excel": excel_path, "pdf": pdf_path}


class SupervisorAgent(Agent):
    def __init__(self):
        super().__init__(
            role="Supervisor",
            goal="Apply guardrails and sanity checks at each stage.",
            backstory="Keeps the pipeline reliable and output well-formed.",
            allow_delegation=False
        )

    # --- Guardrails ---
    def validate_filings(self, filings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not isinstance(filings, list):
            raise ValueError("Filings must be a list.")
        for i in filings:
            if "updated" not in i or "title" not in i:
                raise ValueError("Each filing must include 'title' and 'updated'.")
        return filings

    def validate_insiders(self, insiders: Dict[str, Any]) -> Dict[str, Any]:
        if "form4" not in insiders or "most_active_today" not in insiders:
            raise ValueError("Insider dict missing required keys.")
        return insiders

    def validate_sentiment(self, sentiments: List[Dict[str, str]]) -> List[Dict[str, str]]:
        if len(sentiments) == 0:
            raise ValueError("No sentiments produced.")
        return sentiments

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--creators", type=str, default="", help="Comma-separated list of up to 10 creators/companies")
    parser.add_argument("--out", type=str, default="out", help="Output directory")
    args = parser.parse_args()

    creators = [c.strip() for c in args.creators.split(",") if c.strip()]
    # --- Instantiate agents ---
    data_agent = DataFetcherAgent()
    insider_agent = InsiderTradeAgent()
    sentiment_agent = SentimentAgent()
    report_agent = ReportAgent()
    supervisor = SupervisorAgent()

    # --- Tasks (simple sequential flow) ---
    # 1) Fetch filings
    filings = data_agent.fetch()
    filings = supervisor.validate_filings(filings)

    # 2) Insider activity
    insiders = insider_agent.extract(filings)
    insiders = supervisor.validate_insiders(insiders)

    # 3) Decide creators
    if not creators:
        # pick top companies by frequency in filings
        freq = {}
        for f in filings:
            co = f.get("company") or "Unknown"
            freq[co] = freq.get(co, 0) + 1
        creators = sorted(freq.keys(), key=lambda k: freq[k], reverse=True)[:10]

    # 4) Sentiment
    sentiments = sentiment_agent.analyze(creators, filings)
    sentiments = supervisor.validate_sentiment(sentiments)

    # 5) Report
    final = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "filings_48h": filings,
        "insiders": insiders,
        "sentiment": sentiments,
    }
    paths = report_agent.save(final, args.out)

    # Console summary
    top = insiders.get("most_active_today", [])[:5]
    print("\\n=== CrewAI SEC Task (Groq + LiteLLM) ===")
    print(f"Saved: {paths['json']} and {paths['csv']}")
    print("Top insider-active companies today:")
    for row in top:
        print(f" - {row['company']}: {row['form4_count_today']} form 4(s)")
    print("\\nCreators Sentiment:")
    for s in sentiments:
        print(f" - {s['creator']}: {s['sentiment']}")

if __name__ == "__main__":
    main()
