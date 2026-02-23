import streamlit as st

st.title("🧹 Gmail Janitor")
st.caption("AI-powered email cleanup using Gemini 2.5 Flash with active learning and risk-aware deletion policies.")

section = st.sidebar.radio(
    "Navigation",
    ["📘 Overview", "⚙️ Architecture", "🤖 AI Pipeline", "🛡️ Safety Design", "🛠️ Tech Stack"],
)

# ══════════════════════════════════════════════════════════════════════════════
if section == "📘 Overview":
    st.header("📘 Project Overview")

    st.markdown("""
Gmail Janitor is an AI-powered email cleanup assistant that uses **Gemini 2.5 Flash**
(Google Vertex AI) to intelligently classify, triage, and clean Gmail inboxes — without
ever permanently deleting an email you might need.

The core problem: most people have thousands of unread, low-value emails (newsletters,
job alerts, marketing) drowning out important messages. Traditional filters are brittle
and rule-based. Gmail Janitor uses an LLM to understand email *intent* and applies
risk-aware deletion policies with a safety-first approach.
""")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Lines of Code",     "~2,800")
    col2.metric("AI Model",          "Gemini 2.5 Flash")
    col3.metric("Email Categories",  "8")
    col4.metric("Safety: Hard Delete", "Never (default)")

    st.subheader("Key Features")
    features = {
        "🤖 LLM Classification":   "Gemini scores each email on importance, junk probability, and risk-of-wrong-deletion. Outputs structured JSON via Pydantic.",
        "⚡ Pre-filtering":        "Rule-based fast-path (no Gemini API call) for whitelisted senders, known spam domains, and explicit user rules.",
        "🧠 Active Learning":      "For uncertain emails, Gemini generates targeted follow-up questions. User answers update sender-level preferences and thresholds.",
        "🔄 Natural Language Planner": "Users can type commands like 'Trash all Red Cross marketing emails older than 30 days' — parsed by Gemini into structured action plans.",
        "↩️ Full Undo Support":    "Every action is logged to `actions_log.json`. Users can undo the last cleanup run with one click.",
        "🔒 OAuth + Privacy":      "Google OAuth 2.0 — never stores email content, only metadata and classification scores.",
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
    c1.metric("main.py",           "1,547 lines")
    c2.metric("app.py (UI)",       "1,264 lines")
    c3.metric("planner_service.py","151 lines")
    c4.metric("Total",             "~2,962 lines")

    st.subheader("Key Design Decisions")
    st.markdown("""
- **Gemini over GPT-4**: Free tier availability + deep Google Workspace integration
- **Pydantic schemas**: Forces structured JSON output from LLM, prevents hallucinated field names
- **Label-based actions**: More reversible than hard delete, familiar to Gmail users
- **Per-account data isolation**: Preferences/stats stored in separate directories per Google account
- **Prompt-in-YAML**: Decouples prompt engineering from application code — easier to iterate
""")
