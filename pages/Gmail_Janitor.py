import streamlit as st
import streamlit.components.v1 as components

# ── Configuration ──────────────────────────────────────────────────────────────
# Set this to the deployed Gmail Janitor app URL once running on your server.
# e.g. "https://denissoulimaportfolio.com/gmail-janitor"
# Leave empty to show deployment instructions instead.
GMAIL_JANITOR_APP_URL = ""

# ── Page header ────────────────────────────────────────────────────────────────
st.title("🧹 Denis Soulima's Gmail Janitor")
st.caption("AI-powered email cleanup · Gemini 2.5 Flash · Active Learning · Risk-aware deletion")

st.sidebar.markdown("### Gmail Janitor")
section = st.sidebar.radio(
    "Navigation",
    [
        "📘 Overview",
        "⚙️ Architecture",
        "🤖 AI Pipeline",
        "🛡️ Safety Design",
        "🛠️ Tech Stack",
        "▶️ Play with it",
        "🔒 Privacy & Terms",
    ],
)

# ══════════════════════════════════════════════════════════════════════════════
if section == "📘 Overview":
    st.header("📘 Project Overview")

    # Visible legal links — required for Google OAuth verification
    st.markdown(
        "**Legal:** "
        "[Privacy Policy](https://denissoulimaportfolio.com/Privacy_Policy) · "
        "[Terms of Service](https://denissoulimaportfolio.com/Terms_of_Service)"
    )
    st.divider()

    st.info(
        "**Denis Soulima's Gmail Janitor** is a portfolio project built by Denis Soulima "
        "that demonstrates AI-powered email triage using Google's Gemini 2.5 Flash LLM. "
        "Its purpose is to intelligently classify and clean Gmail inboxes using large language "
        "models with risk-aware deletion policies. Built for educational and portfolio "
        "demonstration purposes only."
    )

    st.markdown("""
**Denis Soulima's Gmail Janitor** is an AI-powered email cleanup assistant that uses
**Gemini 2.5 Flash** (Google Vertex AI) to intelligently classify, triage, and clean
Gmail inboxes — without ever permanently deleting an email you might need.

**Purpose:** Help users reclaim their inbox by using an LLM to understand email *intent*
and apply configurable risk-aware deletion policies, rather than brittle keyword rules.

The core problem: most people have thousands of unread, low-value emails (newsletters,
job alerts, marketing) drowning out important messages. Traditional filters are
rule-based and inflexible. This app uses AI to reason about each email contextually.

**⚠️ Warning:** This tool interacts with your real Gmail account and can move emails
to Trash. While the default policy never hard-deletes (emails go to a review label first),
use it at your own risk. The developer assumes no responsibility for any emails moved or
deleted. See the [Terms of Service](https://denissoulimaportfolio.com/Terms_of_Service).
""")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Lines of Code",       "~2,800")
    col2.metric("AI Model",            "Gemini 2.5 Flash")
    col3.metric("Email Categories",    "8")
    col4.metric("Safety: Hard Delete", "Never (default)")

    st.subheader("Key Features")
    features = {
        "🤖 LLM Classification":      "Gemini scores each email on importance, junk probability, and risk-of-wrong-deletion. Outputs structured JSON via Pydantic.",
        "⚡ Pre-filtering":            "Rule-based fast-path (no Gemini API call) for whitelisted senders, known spam domains, and explicit user rules.",
        "🧠 Active Learning":          "For uncertain emails, Gemini generates targeted follow-up questions. User answers update sender-level preferences and thresholds.",
        "🔄 Natural Language Planner": "Users can type commands like 'Trash all Red Cross marketing emails older than 30 days' — parsed by Gemini into structured action plans.",
        "↩️ Full Undo Support":        "Every action is logged to `actions_log.json`. Users can undo the last cleanup run with one click.",
        "🔒 OAuth + Privacy":          "Google OAuth 2.0 — never stores email content, only metadata and classification scores.",
    }
    for title, desc in features.items():
        with st.expander(title):
            st.markdown(desc)

    st.subheader("App UI (4 Tabs)")
    st.markdown("""
| Tab | Purpose |
|-----|---------|
| **Run Cleanup** | Set thresholds, choose search criteria, preview and execute classification |
| **Quarantine** | Browse `GmailJanitor/Review` label, approve/reject bulk actions |
| **Rules & Preferences** | Edit whitelist/blacklist domains, per-category rules |
| **Audit / Undo** | View full action history, undo last run |
""")


# ══════════════════════════════════════════════════════════════════════════════
elif section == "⚙️ Architecture":
    st.header("⚙️ System Architecture")

    st.code("""
┌─────────────────────────────────────────────────────────┐
│                    Streamlit Frontend (app.py)           │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐ │
│  │ Run Cleanup  │  │  Quarantine  │  │ Rules & Audit  │ │
│  └──────┬───────┘  └──────────────┘  └────────────────┘ │
│         │                                                 │
└─────────┼───────────────────────────────────────────────┘
          │
┌─────────▼───────────────────────────────────────────────┐
│                   Backend (main.py)                      │
│                                                          │
│  ┌─────────────────┐   ┌──────────────────────────────┐ │
│  │  Pre-filter      │   │   Gemini Classification      │ │
│  │  (rule-based)    │──▶│   (importance + junk score)  │ │
│  └─────────────────┘   └──────────────┬───────────────┘ │
│                                        │                 │
│  ┌─────────────────────────────────────▼───────────────┐ │
│  │             Decision Policy                          │ │
│  │  KEEP if importance ≥ 0.75                          │ │
│  │  TRASH if junk ≥ 0.85 AND risk ≤ 0.20              │ │
│  │  REVIEW (safe default) otherwise                    │ │
│  └─────────────────────────────────────────────────────┘ │
│                                                          │
│  ┌──────────────┐   ┌─────────────────────────────────┐  │
│  │  Active      │   │  Persistence Layer               │  │
│  │  Learning    │   │  preferences.json                │  │
│  │  (feedback)  │   │  sender_stats.json               │  │
│  └──────────────┘   │  actions_log.json                │  │
│                     │  cache_classifications.json      │  │
│                     └─────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
          │
┌─────────▼──────────────────┐    ┌──────────────────────┐
│   Gmail API (OAuth 2.0)    │    │  Planner Service      │
│   google-api-python-client │    │  (NL → action plan)   │
└────────────────────────────┘    └──────────────────────┘
""", language="text")

    st.subheader("Data Flow")
    st.markdown("""
1. **Auth** → OAuth 2.0 token stored in `tokens/` directory (per Google account)
2. **Fetch** → Gmail API pulls emails matching search criteria (keywords, date, labels)
3. **Pre-filter** → Whitelisted senders pass through as KEEP; known spam → TRASH immediately
4. **Classify** → Remaining emails sent to Gemini with structured prompt from `prompts.yml`
5. **Decide** → Policy engine applies thresholds to scores → KEEP / TRASH / REVIEW
6. **Execute** → Actions applied via Gmail API (labels added, emails moved, not hard-deleted)
7. **Log** → Every action written to `actions_log.json` for audit/undo
8. **Learn** → User feedback updates `sender_stats.json` + preference thresholds
""")

# ══════════════════════════════════════════════════════════════════════════════
elif section == "🤖 AI Pipeline":
    st.header("🤖 AI Classification Pipeline")

    st.subheader("Gemini Classification Schema")
    st.code("""
class EmailClassification(BaseModel):
    category: Literal[
        "job_alert", "marketing", "receipt",
        "financial", "social", "personal",
        "system", "unknown"
    ]
    importance_score:          float  # 0-1 (higher = more important)
    junk_score:                float  # 0-1 (higher = more likely junk)
    risk_of_wrong_deletion:    float  # 0-1 (higher = riskier to trash)
    confidence:                float  # 0-1 (model certainty)
    reasoning:                 str    # brief explanation
    suggested_action:          str    # KEEP / TRASH / REVIEW
    follow_up_question:        str | None  # for uncertain cases
""", language="python")

    st.subheader("Decision Policy")
    st.code("""
def apply_policy(classification, prefs):
    imp   = classification.importance_score
    junk  = classification.junk_score
    risk  = classification.risk_of_wrong_deletion

    # Configurable thresholds (user can tune these in UI)
    if imp  >= prefs.importance_threshold:   # default 0.75
        return "KEEP"
    if (junk >= prefs.junk_threshold         # default 0.85
        and risk <= prefs.max_risk):         # default 0.20
        return "TRASH"
    return "REVIEW"    # safe default
""", language="python")

    st.subheader("Natural Language Planner")
    st.markdown("Users can type commands in plain English — Gemini parses them into structured action plans:")
    examples = [
        ("Trash all Red Cross emails older than 30 days",
         "ActionPlan(action='trash', from_domain='redcross.org', older_than_days=30)"),
        ("Keep all LinkedIn job alerts",
         "ActionPlan(action='keep', category='job_alert', from_domain='linkedin.com')"),
        ("Show me emails I might have missed last week",
         "ActionPlan(action='review', recent_days=7, importance_min=0.6)"),
    ]
    for cmd, result in examples:
        with st.expander(f'"{cmd}"'):
            st.code(result, language="python")

    st.subheader("Active Learning Loop")
    st.markdown("""
When `confidence < 0.7` or `risk > 0.5`, Gemini generates a targeted question:

> *"This email from careers@company.com looks like a job alert, but it references your application from 3 months ago. Is this a recruiter follow-up you'd want to keep?"*

User's yes/no answer:
- Updates `sender_stats.json` (override rate for that sender)
- Adjusts category-level importance threshold
- Improves future classifications for similar emails
""")

# ══════════════════════════════════════════════════════════════════════════════
elif section == "🛡️ Safety Design":
    st.header("🛡️ Safety-First Design")

    st.success("**Core principle:** Gmail Janitor never permanently deletes emails by default.")

    st.markdown("""
### Layered Safety Model

| Layer | Mechanism |
|-------|-----------|
| **No hard deletes** | Emails moved to `GmailJanitor/Review` label, not permanently deleted |
| **Risk scoring** | Gemini rates each email's deletion risk; high-risk emails go to REVIEW regardless of junk score |
| **Whitelist** | Configured senders/domains are always KEEP — never analyzed |
| **Confidence gate** | Low-confidence classifications trigger human review |
| **Undo log** | Every action logged with message-id, timestamp, and action type |
| **Audit tab** | Full history visible in UI — undo last run with one click |
| **Per-category rules** | Users can override defaults per category (e.g., always keep `financial`) |
""")

    st.subheader("Undo Architecture")
    st.code("""
# actions_log.json structure
{
  "run_id": "2026-02-22T14:30:00",
  "actions": [
    {
      "message_id": "18e2f4...",
      "subject": "Your Amazon order shipped",
      "action": "TRASH",
      "classification": {"junk_score": 0.91, "risk": 0.08, ...},
      "applied_at": "2026-02-22T14:30:05"
    },
    ...
  ]
}
""", language="json")

# ══════════════════════════════════════════════════════════════════════════════
elif section == "🛠️ Tech Stack":
    st.header("🛠️ Tech Stack")

    stack = {
        "AI / LLM":          "Google Gemini 2.5 Flash via Vertex AI (`google-generativeai`)",
        "Email API":         "Gmail API via `google-api-python-client` with OAuth 2.0",
        "Schema Validation": "Pydantic — structured LLM output parsing",
        "Frontend":          "Streamlit — 4-tab UI with session state management",
        "Backend":           "Python 3.11 — 1,547-line orchestration module (`main.py`)",
        "Config":            "YAML (`prompts.yml`) for Gemini system prompts",
        "Persistence":       "JSON files per account — preferences, sender stats, audit log, cache",
        "Natural Language":  "Planner service parses user commands → structured `ActionPlan` objects",
    }
    for tech, desc in stack.items():
        col1, col2 = st.columns([1, 3])
        col1.markdown(f"**{tech}**")
        col2.markdown(desc)
        st.divider()

    st.subheader("Project Stats")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("main.py",            "1,547 lines")
    c2.metric("app.py (UI)",        "1,264 lines")
    c3.metric("planner_service.py", "151 lines")
    c4.metric("Total",              "~2,962 lines")

    st.subheader("Key Design Decisions")
    st.markdown("""
- **Gemini over GPT-4**: Free tier availability + deep Google Workspace integration
- **Pydantic schemas**: Forces structured JSON output from LLM, prevents hallucinated field names
- **Label-based actions**: More reversible than hard delete, familiar to Gmail users
- **Per-account data isolation**: Preferences/stats stored in separate directories per Google account
- **Prompt-in-YAML**: Decouples prompt engineering from application code — easier to iterate
""")

# ══════════════════════════════════════════════════════════════════════════════
elif section == "▶️ Play with it":
    st.header("▶️ Play with Denis Soulima's Gmail Janitor")

    st.warning(
        "⚠️ **This tool connects to your real Gmail account.** "
        "Emails may be moved to Trash or a review label. "
        "Use the **Audit / Undo** tab inside the app to reverse any actions. "
        "By proceeding you acknowledge the [Privacy Policy & Terms](#privacy--terms) below."
    )

    if GMAIL_JANITOR_APP_URL:
        st.success(f"App is live at: {GMAIL_JANITOR_APP_URL}")
        st.link_button("Open Gmail Janitor in full screen ↗", url=GMAIL_JANITOR_APP_URL)
        st.divider()
        components.iframe(GMAIL_JANITOR_APP_URL, height=800, scrolling=True)
    else:
        st.info(
            "The live app is not yet configured. "
            "To enable it, deploy the Gmail Janitor backend on your server and set "
            "`GMAIL_JANITOR_APP_URL` at the top of this file."
        )
        st.subheader("What the app looks like")
        st.markdown("""
The live Gmail Janitor UI has four tabs:

| Tab | What you can do |
|-----|----------------|
| **Run Cleanup** | Choose how many emails to scan, set junk/importance thresholds, preview results before executing |
| **Quarantine** | Review emails the AI wasn't sure about — approve or reject each one |
| **Rules & Preferences** | Add sender whitelists/blacklists, per-category overrides |
| **Audit / Undo** | Full history of every action taken — one-click undo for the last run |

Once authenticated with Google OAuth, the app fetches your emails, runs them through
Gemini 2.5 Flash, and presents you with a prioritised action plan before touching anything.
""")

# ══════════════════════════════════════════════════════════════════════════════
elif section == "🔒 Privacy & Terms":
    st.header("🔒 Privacy Policy & Terms of Service")
    st.caption("Last updated: February 2026 · Denis Soulima's Gmail Janitor")

    st.subheader("Privacy Policy")
    st.markdown("""
**TL;DR — We do not want your data. We do not store, sell, or share it.**

1. **No data collection.** Denis Soulima's Gmail Janitor does not collect, store, or transmit
   any email content, personal information, or usage data to any server controlled by the
   developer. All processing occurs locally within your browser session and on your own
   Google account.

2. **Google OAuth scope.** The app requests OAuth access to your Gmail account solely to
   read, label, and move emails within that account. The access token is stored locally
   in the `tokens/` directory on the machine running the app and is never sent to the
   developer or any third party.

3. **Email content.** Email subjects, snippets, and metadata are sent to **Google Gemini
   2.5 Flash** (via the Google Vertex AI API) for classification. This is subject to
   [Google's Privacy Policy](https://policies.google.com/privacy). No email content is
   stored by this application.

4. **No analytics.** No cookies, tracking pixels, or analytics are used.

5. **Portfolio purpose only.** This project exists to demonstrate software engineering
   skills. It is not a commercial product. Use it at your own discretion.

6. **Data deletion.** To revoke access, go to [Google Account Permissions](https://myaccount.google.com/permissions)
   and remove "Denis Soulima's Gmail Janitor". Delete the local `tokens/` directory to
   remove any cached credentials from the machine running the app.
""")

    st.divider()
    st.subheader("Terms of Service")
    st.markdown("""
By using Denis Soulima's Gmail Janitor you agree to the following:

1. **Portfolio use only.** This application is provided as a portfolio demonstration project
   by Denis Soulima. It is not intended for production or commercial use.

2. **Use at your own risk.** The application interacts with your real Gmail inbox and may
   move emails to Trash or apply labels. While the default configuration does not
   permanently delete emails, the developer makes **no warranty** and accepts **no
   liability** for any emails moved, deleted, or otherwise affected by use of this tool.

3. **No warranty.** The software is provided "as is", without warranty of any kind,
   express or implied, including but not limited to the warranties of merchantability,
   fitness for a particular purpose, and non-infringement.

4. **Indemnification.** You agree to indemnify and hold harmless Denis Soulima from any
   claim, damage, or expense arising from your use of the application.

5. **Google's Terms.** Your use of Gmail through this application is also subject to
   [Google's Terms of Service](https://policies.google.com/terms).

6. **Changes.** These terms may be updated at any time. Continued use constitutes acceptance.

---
*Questions? Contact Denis Soulima via [denissoulimaportfolio.com](https://denissoulimaportfolio.com)*
""")
