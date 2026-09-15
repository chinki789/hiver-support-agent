"""
AmazonHelp — front / login page
Run with:  streamlit run app.py
Requires Streamlit >= 1.39 (container/widget `key` produces the st-key-* CSS class).
"""

import streamlit as st

st.set_page_config(
    page_title="AmazonHelp | AI Customer Support",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ------------------------------------------------------------------
# Session state
# ------------------------------------------------------------------
st.session_state.setdefault("login_mode", "existing")   # existing | new
st.session_state.setdefault("login_method", "phone")    # phone | email
st.session_state.setdefault("otp_sent", False)
st.session_state.setdefault("logged_in", False)


# ------------------------------------------------------------------
# Styles
# ------------------------------------------------------------------
st.markdown(
    """
<style>
#MainMenu, header, footer {visibility: hidden;}

/* ---------- page background ---------- */
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at 8% 6%,
            rgba(255,157,36,0.90) 0px,
            rgba(255,157,36,0.45) 90px,
            rgba(255,157,36,0.10) 200px,
            transparent 340px),
        radial-gradient(circle at 93% 55%,
            rgba(0,190,220,0.50) 0px,
            rgba(0,160,205,0.22) 160px,
            transparent 400px),
        radial-gradient(circle at 87% 88%,
            rgba(42,80,190,0.60) 0px,
            rgba(42,80,190,0.26) 180px,
            transparent 420px),
        radial-gradient(circle at 14% 74%,
            rgba(20,55,140,0.45) 0px,
            transparent 240px),
        linear-gradient(135deg, #061326 0%, #071a35 46%, #08233b 100%);
}

/* decorative arcs */
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed;
    width: 540px; height: 540px;
    left: -245px; top: -280px;
    border: 3px solid rgba(255,173,65,0.45);
    border-radius: 50%;
    pointer-events: none;
}
[data-testid="stAppViewContainer"]::after {
    content: "";
    position: fixed;
    width: 720px; height: 720px;
    right: -370px; bottom: -360px;
    border: 3px solid rgba(40,170,230,0.28);
    border-radius: 50%;
    pointer-events: none;
}

/* ---------- outer layout: transparent, just centres the card ---------- */
[data-testid="stMain"] {
    padding-bottom: 60px;
}

[data-testid="stMainBlockContainer"] {
    max-width: 660px !important;
    padding: 5vh 20px 0 20px !important;
    background: transparent !important;
}

/* ---------- the white card ---------- */
.st-key-card {
    background: #ffffff;
    border-radius: 24px;
    box-shadow: 0 28px 80px rgba(0,0,0,0.48),
                0 5px 25px rgba(0,0,0,0.18);
}

/* header must sit flush against the card edge, so kill the stack gap */
.st-key-card > div > [data-testid="stVerticalBlock"] {
    gap: 0 !important;
}

/* ---------- brand header ---------- */
.brand-area {
    text-align: center;
    padding: 26px 20px 22px;
    /* the card no longer clips its children, so the header rounds itself */
    border-radius: 24px 24px 0 0;
    background:
        radial-gradient(circle at 6% 10%, rgba(255,178,60,0.18), transparent 26%),
        radial-gradient(circle at 95% 82%, rgba(55,130,220,0.12), transparent 28%),
        linear-gradient(135deg, #f8fafc 0%, #eef2f6 100%);
    border-bottom: 1px solid #edf0f3;
}
.amazon-logo {
    font-family: Arial, Helvetica, sans-serif;
    font-size: 44px; line-height: 40px;
    font-weight: 800; letter-spacing: -2px;
    color: #18283a;
}
.swoosh {
    height: 10px; margin: 0 auto 12px;
    border-bottom: 4px solid #ff9900;
    border-radius: 50%;
}
.swoosh-lg { width: 70px; }
.swoosh-sm { width: 58px; height: 7px; border-bottom-width: 3px; margin: 5px auto 10px; }

.agent-title {
    font-family: Arial, Helvetica, sans-serif;
    color: #17283b; font-size: 25px; font-weight: 800; line-height: 30px;
}
.agent-title span { font-size: 22px; margin-right: 6px; }
.agent-subtitle {
    color: #72839b; font-family: Arial, Helvetica, sans-serif;
    font-size: 13px; font-weight: 700; letter-spacing: 0.7px;
}

/* ---------- card body ---------- */
.st-key-card_body { padding: 20px 50px 28px; }
.st-key-card_body [data-testid="stVerticalBlock"] { gap: 0.7rem; }

.welcome-title {
    text-align: center; color: #153052;
    font-family: Arial, Helvetica, sans-serif;
    font-size: 32px; font-weight: 800; margin: 0 0 8px;
}
.welcome-subtitle {
    text-align: center; color: #74869d;
    font-family: Arial, Helvetica, sans-serif;
    font-size: 15px; margin: 0 0 10px;
}

/* ---------- text inputs ---------- */
[data-testid="stTextInput"] input {
    height: 48px !important;
    box-sizing: border-box !important;
    border: 1px solid #dce4eb !important;
    border-radius: 11px !important;
    background: #fbfcfe !important;
    color: #24374b !important;
    font-size: 14px !important;
    padding: 0 14px !important;
}
[data-testid="stTextInput"] input::placeholder {
    color: #98a6b5 !important; opacity: 1 !important; font-style: italic;
}
[data-testid="stTextInput"] input:focus {
    border-color: #ff9900 !important;
    box-shadow: 0 0 0 3px rgba(255,153,0,0.13) !important;
}

/* ---------- joined country-code + phone field ---------- */
.st-key-phone_row [data-testid="stColumn"]:last-child,
.st-key-phone_row [data-testid="stHorizontalBlock"] > div:last-child {
    margin-left: 3px;
}
.st-key-phone_row {
    margin-top: 12px;
}
.st-key-phone_row [data-testid="stTextInput"] input {
    border-radius: 0 11px 11px 0 !important;
    border-left: 0 !important;
}
.country-code {
    height: 48px; box-sizing: border-box;
    display: flex; align-items: center; justify-content: center; gap: 6px;
    border: 1px solid #dce4eb;
    border-right: 0;
    border-radius: 11px 0 0 11px;
    background: #fbfcfe;
    color: #293b50;
    font-family: Arial, Helvetica, sans-serif;
    font-size: 14px; font-weight: 700;
}
.st-key-phone_row {
    margin-bottom: 9px;
}

/* ---------- buttons (shared) ---------- */
.stButton > button {
    height: 48px !important; min-height: 48px !important;
    border-radius: 11px !important;
    font-family: Arial, Helvetica, sans-serif !important;
    font-size: 15px !important; font-weight: 700 !important;
    transition: transform .15s ease, box-shadow .15s ease, background .15s ease !important;
}

/* primary action: Request OTP / Verify / Continue */
.st-key-request_otp button,
.st-key-verify_otp button,
.st-key-continue_email button {
    color: #ffffff !important; border: none !important;
    background: linear-gradient(135deg, #ff9900 0%, #ffab18 100%) !important;
    box-shadow: 0 8px 20px rgba(255,153,0,0.25) !important;
}
.st-key-request_otp button:hover,
.st-key-verify_otp button:hover,
.st-key-continue_email button:hover {
    transform: translateY(-1px);
    box-shadow: 0 10px 25px rgba(255,153,0,0.35) !important;
}

/* outlined alternate-method button */
.st-key-switch_method button {
    color: #f09500 !important;
    background: #ffffff !important;
    border: 1.5px solid #ffae22 !important;
}
.st-key-switch_method button:hover { background: #fff9ef !important; }

/* ---------- OR divider ---------- */
.or-divider {
    display: flex; align-items: center; gap: 13px;
    color: #78889b; font-family: Arial, Helvetica, sans-serif;
    font-size: 14px; font-weight: 700;
    margin: 12px 0 12px;
}
.or-divider::before, .or-divider::after {
    content: ""; flex: 1; height: 1px; background: #e2e7ec;
}

/* ---------- user-type pill toggle ---------- */
.st-key-user_toggle {
    margin: 14px auto 4px;
    max-width: 420px;
    border: 1px solid #e5e9ee;
    border-radius: 24px;
    background: #f7f9fb;
    overflow: hidden;
}
.st-key-user_toggle [data-testid="stHorizontalBlock"] { gap: 0 !important; }
.st-key-user_toggle .stButton > button {
    width: 100% !important;
    height: 45px !important; min-height: 45px !important;
    border: none !important;
    border-radius: 24px !important;
    font-size: 14px !important;
}
.st-key-user_toggle .stButton > button[kind="primary"] {
    color: #ffffff !important;
    background: linear-gradient(135deg, #ff9900, #ffab18) !important;
    box-shadow: 0 3px 10px rgba(255,153,0,0.22) !important;
}
.st-key-user_toggle .stButton > button[kind="secondary"] {
    color: #41536a !important;
    background: #f7f9fb !important;
}
.st-key-user_toggle .stButton > button[kind="secondary"]:hover {
    background: #eef2f6 !important;
}

/* ---------- footer ---------- */
.login-footer {
    text-align: center; color: #a0adbc;
    font-family: Arial, Helvetica, sans-serif;
    font-size: 11px; padding: 10px 0 6px;
}

[data-testid="stAlert"] { border-radius: 10px !important; font-size: 13px !important; }

/* ---------- mobile ---------- */
@media (max-width: 700px) {
    [data-testid="stMain"] { padding-bottom: 32px; }
    [data-testid="stMainBlockContainer"] { padding: 2vh 12px 0 12px !important; }
    .st-key-card_body { padding: 26px 28px 22px; }
    .amazon-logo { font-size: 38px; }
    .agent-title { font-size: 22px; }
    .welcome-title { font-size: 27px; }
}
</style>
""",
    unsafe_allow_html=True,
)


# ==================================================================
# The white card — everything below lives inside this container
# ==================================================================
with st.container(key="card"):

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    st.markdown(
        """
        <div class="brand-area">
            <div class="amazon-logo">amazon</div>
            <div class="swoosh swoosh-lg"></div>
            <div class="agent-title"><span>🤖</span>AmazonHelp</div>
            <div class="swoosh swoosh-sm"></div>
            <div class="agent-subtitle">AI CUSTOMER SUPPORT AGENT</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


    # ------------------------------------------------------------------
    # Body
    # ------------------------------------------------------------------
    with st.container(key="card_body"):

        new_user = st.session_state.login_mode == "new"

        st.markdown(
            f'<div class="welcome-title">{"Create your account" if new_user else "Welcome Back!"}</div>',
            unsafe_allow_html=True,
        )
        subtitle = (
            "Verify your number to set up your AI support account."
            if new_user
            else "Sign in to continue to your AI support dashboard."
        )
        st.markdown(
            f'<div class="welcome-subtitle">{subtitle}</div>',
            unsafe_allow_html=True,
        )

        # ---------- phone flow ----------
        if st.session_state.login_method == "phone":

            with st.container(key="phone_row"):
                col_code, col_phone = st.columns([0.27, 0.73], gap="small")
                with col_code:
                    st.markdown(
                        '<div class="country-code">🇮🇳 +91 ⌄</div>',
                        unsafe_allow_html=True,
                    )
                with col_phone:
                    phone = st.text_input(
                        "Phone number",
                        placeholder="Enter a valid 10 Digit Number",
                        max_chars=10,
                        label_visibility="collapsed",
                        key="phone_number",
                    )

            if st.button("Request OTP  →", key="request_otp", use_container_width=True):
                if phone.isdigit() and len(phone) == 10:
                    st.session_state.otp_sent = True
                    st.success("OTP sent. Demo code: 1234")
                else:
                    st.session_state.otp_sent = False
                    st.error("Enter a 10 digit mobile number.")

            if st.session_state.otp_sent:
                otp = st.text_input(
                    "OTP",
                    placeholder="Enter the 4 digit OTP",
                    max_chars=4,
                    label_visibility="collapsed",
                    key="otp",
                )
                if st.button("Verify OTP  →", key="verify_otp", use_container_width=True):
                    if otp == "1234":
                        st.session_state.logged_in = True
                        st.rerun()
                    else:
                        st.error("That code doesn't match. Use 1234 in this demo.")

        # ---------- email flow ----------
        else:
            email = st.text_input(
                "Email",
                placeholder="Enter your email address",
                label_visibility="collapsed",
                key="email",
            )
            if st.button("Continue with Email  →", key="continue_email", use_container_width=True):
                if "@" in email and "." in email.split("@")[-1]:
                    st.session_state.logged_in = True
                    st.rerun()
                else:
                    st.error("Enter a valid email address, like you@example.com")

        # ---------- divider ----------
        st.markdown('<div class="or-divider"><span>OR</span></div>', unsafe_allow_html=True)

        # ---------- switch login method ----------
        phone_mode = st.session_state.login_method == "phone"
        with st.container(key="switch_method"):
            if st.button(
                "✉  Login using Email" if phone_mode else "📱  Login using Phone",
                key="switch_method_btn",
                use_container_width=True,
            ):
                st.session_state.login_method = "email" if phone_mode else "phone"
                st.session_state.otp_sent = False
                st.rerun()

        # ---------- existing / new user ----------
        with st.container(key="user_toggle"):
            col_a, col_b = st.columns(2, gap="small")

            with col_a:
                if st.button(
                    "Existing User",
                    key="tab_existing",
                    type="secondary" if new_user else "primary",
                    use_container_width=True,
                ):
                    st.session_state.login_mode = "existing"
                    st.rerun()

            with col_b:
                if st.button(
                    "New User",
                    key="tab_new",
                    type="primary" if new_user else "secondary",
                    use_container_width=True,
                ):
                    st.session_state.login_mode = "new"
                    st.rerun()

        st.markdown(
            '<div class="login-footer">🔒 Secure • Local Demo • Not a Real Account</div>',
            unsafe_allow_html=True,
        )


# ------------------------------------------------------------------
# After login — hook your dashboard / agent up here
# ------------------------------------------------------------------
if st.session_state.logged_in:
    st.success("You're signed in. Connect your dashboard or agent code here.")