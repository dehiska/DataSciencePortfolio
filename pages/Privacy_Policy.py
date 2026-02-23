import streamlit as st

st.title("🔒 Privacy Policy & Terms of Service")
st.caption("Denis Soulima · denissoulimaportfolio.com · Last updated: February 2026")

st.markdown("""
This page covers the privacy policy and terms of service for all tools and applications
published at **denissoulimaportfolio.com** by **Denis Soulima**, including
**Denis Soulima's Gmail Janitor**.
""")

st.divider()

st.header("Privacy Policy")
st.markdown("""
**TL;DR — We do not want your data. We do not store, sell, or share it.**

### 1. No Data Collection
Denis Soulima's applications do not collect, store, or transmit any personal information,
email content, or usage data to any server controlled by the developer. All processing
occurs within your own session and on your own connected accounts (e.g., Gmail).

### 2. Google OAuth (Gmail Janitor)
The Gmail Janitor requests OAuth access to your Gmail account solely to read, label, and
move emails within that account on your behalf. The OAuth access token is stored locally
on the machine running the app and is **never** sent to the developer or any third party.

### 3. Email Content & Third-Party AI
Email subjects, snippets, and metadata may be sent to **Google Gemini 2.5 Flash**
(Google Vertex AI) for AI-based classification. This processing is governed by
[Google's Privacy Policy](https://policies.google.com/privacy). This application does
not store email content.

### 4. No Analytics or Tracking
No cookies, tracking pixels, analytics scripts, or third-party advertising are used on
this portfolio site.

### 5. Purpose
This portfolio and all its tools exist solely to demonstrate software engineering and
data science skills. They are not commercial products.

### 6. Revoking Access
To revoke Gmail access granted to Denis Soulima's Gmail Janitor:
- Visit [Google Account Permissions](https://myaccount.google.com/permissions)
- Remove "Denis Soulima's Gmail Janitor"
- Delete the local `tokens/` directory to clear cached credentials

### 7. Contact
For privacy questions, contact Denis Soulima via [denissoulimaportfolio.com](https://denissoulimaportfolio.com).
""")

st.divider()

st.header("Terms of Service")
st.markdown("""
By using any application at denissoulimaportfolio.com you agree to the following:

### 1. Portfolio Use Only
All applications on this site are provided as portfolio demonstration projects by
Denis Soulima. They are not intended for production or commercial use.

### 2. Use at Your Own Risk
Applications that interact with external accounts (e.g., Gmail) may modify data in
those accounts. The developer makes **no warranty** and accepts **no liability** for
any data modified, moved, or deleted through use of these tools.

### 3. No Warranty
The software is provided "as is", without warranty of any kind, express or implied.

### 4. Google's Terms
Use of Gmail-connected features is also subject to
[Google's Terms of Service](https://policies.google.com/terms).

### 5. Changes
These terms may be updated at any time. Continued use constitutes acceptance.
""")
