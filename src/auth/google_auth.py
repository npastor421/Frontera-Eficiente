"""
Google OAuth 2.0 Authentication and Multi-Tenant User Management Module.

Provides seamless Google Sign-In, profile management, and session state
synchronization for Streamlit applications.
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional

import streamlit as st


def get_oauth_config() -> Dict[str, str]:
    """
    Retrieve Google OAuth configuration from environment variables, Streamlit secrets, or secrets.toml.
    """
    config: Dict[str, str] = {}
    
    # 1. Environment variables (highest priority for deployment overrides and testing)
    config["client_id"] = os.environ.get("GOOGLE_CLIENT_ID", "").strip()
    config["client_secret"] = os.environ.get("GOOGLE_CLIENT_SECRET", "").strip()
    config["redirect_uri"] = os.environ.get("GOOGLE_REDIRECT_URI", "").strip()

    # 2. Try Streamlit secrets if not in env
    if not config.get("client_id") or not config.get("client_secret"):
        try:
            if hasattr(st, "secrets") and "google_oauth" in st.secrets:
                sec = st.secrets["google_oauth"]
                if not config.get("client_id"):
                    config["client_id"] = sec.get("client_id", "").strip()
                if not config.get("client_secret"):
                    config["client_secret"] = sec.get("client_secret", "").strip()
                if not config.get("redirect_uri"):
                    config["redirect_uri"] = sec.get("redirect_uri", "http://localhost:8501").strip()
        except Exception:
            pass

    # 3. Fallback to direct reading of .streamlit/secrets.toml
    if not config.get("client_id") or not config.get("client_secret"):
        try:
            import tomllib
            from pathlib import Path
            secrets_path = Path(__file__).resolve().parent.parent.parent / ".streamlit" / "secrets.toml"
            if secrets_path.exists():
                data = tomllib.loads(secrets_path.read_text(encoding="utf-8"))
                sec = data.get("google_oauth", {})
                if not config.get("client_id"):
                    config["client_id"] = sec.get("client_id", "").strip()
                if not config.get("client_secret"):
                    config["client_secret"] = sec.get("client_secret", "").strip()
                if not config.get("redirect_uri"):
                    config["redirect_uri"] = sec.get("redirect_uri", "http://localhost:8501").strip()
        except Exception:
            pass

    if not config.get("redirect_uri"):
        config["redirect_uri"] = "http://localhost:8501"
    else:
        config["redirect_uri"] = config["redirect_uri"].rstrip("/")

    return config


def is_oauth_configured() -> bool:
    """Check if Google OAuth client ID and client secret are configured."""
    cfg = get_oauth_config()
    return bool(cfg.get("client_id") and cfg.get("client_secret"))


def get_google_auth_url() -> str:
    """Generate the Google OAuth 2.0 authorization URL."""
    cfg = get_oauth_config()
    base_url = "https://accounts.google.com/o/oauth2/v2/auth"
    params = {
        "client_id": cfg.get("client_id", ""),
        "redirect_uri": cfg.get("redirect_uri", "http://localhost:8501"),
        "response_type": "code",
        "scope": "openid email profile",
        "access_type": "offline",
        "prompt": "select_account",
    }
    return f"{base_url}?{urllib.parse.urlencode(params)}"


def exchange_code_for_user_info(auth_code: str) -> Optional[Dict[str, Any]]:
    """
    Exchange authorization code for tokens and fetch user profile.
    """
    cfg = get_oauth_config()
    token_url = "https://oauth2.googleapis.com/token"
    
    data = urllib.parse.urlencode({
        "code": auth_code,
        "client_id": cfg.get("client_id", ""),
        "client_secret": cfg.get("client_secret", ""),
        "redirect_uri": cfg.get("redirect_uri", "http://localhost:8501"),
        "grant_type": "authorization_code",
    }).encode("utf-8")

    req = urllib.request.Request(token_url, data=data, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            token_data = json.loads(resp.read().decode("utf-8"))
            access_token = token_data.get("access_token")
            if not access_token:
                return None

        # Fetch User Profile
        userinfo_url = "https://www.googleapis.com/oauth2/v3/userinfo"
        user_req = urllib.request.Request(userinfo_url, method="GET")
        user_req.add_header("Authorization", f"Bearer {access_token}")

        with urllib.request.urlopen(user_req, timeout=10) as user_resp:
            user_data = json.loads(user_resp.read().decode("utf-8"))
            return {
                "id": user_data.get("sub", ""),
                "email": user_data.get("email", ""),
                "name": user_data.get("name", user_data.get("email", "")),
                "picture": user_data.get("picture", ""),
                "verified_email": user_data.get("email_verified", False),
            }
    except Exception as e:
        st.error(f"Error autenticando con Google: {e}")
        return None


def init_auth_session() -> None:
    """
    Check URL query params for OAuth authorization code and initialize user session.
    """
    if "user" not in st.session_state:
        st.session_state["user"] = None

    # Handle incoming OAuth code from URL redirect
    query_params = st.query_params
    if "code" in query_params:
        auth_code = query_params["code"]
        if isinstance(auth_code, list):
            auth_code = auth_code[0] if auth_code else ""

        if auth_code and is_oauth_configured():
            with st.spinner("Iniciando sesión con tu cuenta de Google..."):
                user_info = exchange_code_for_user_info(auth_code)
                if user_info:
                    st.session_state["user"] = user_info
                    # Clear query parameter from URL cleanly
                    st.query_params.clear()
                    st.rerun()


def get_active_user() -> Optional[Dict[str, Any]]:
    """Return the currently authenticated user dictionary, or None."""
    return st.session_state.get("user")


def get_active_user_id() -> str:
    """
    Return a unique string identifier for the active user session.
    Returns the user's email if authenticated, or 'guest' for unauthenticated sessions.
    """
    user = get_active_user()
    if user and user.get("email"):
        return str(user["email"]).strip().lower()
    return "guest"


def logout_user() -> None:
    """Clear the authenticated user session."""
    st.session_state["user"] = None
    st.query_params.clear()
    st.rerun()


def render_user_auth_sidebar() -> None:
    """
    Render Google Sign-In button, profile badge, and logout control in the sidebar.
    """
    user = get_active_user()
    oauth_ready = is_oauth_configured()

    st.sidebar.markdown("### 👤 Cuenta de Usuario")

    if user:
        # Authenticated Profile Card
        name = user.get("name", "Usuario")
        email = user.get("email", "")
        picture = user.get("picture", "")

        avatar_html = (
            f"<img src='{picture}' style='width: 44px; height: 44px; border-radius: 50%; margin-right: 12px; border: 2px solid #00FF66;' />"
            if picture
            else "<div style='width: 44px; height: 44px; border-radius: 50%; background: #1f4e79; color: white; display: flex; align-items: center; justify-content: center; font-weight: bold; margin-right: 12px;'>G</div>"
        )

        st.sidebar.markdown(
            f"""
            <div style='background: rgba(22, 27, 38, 0.9); border: 1px solid rgba(0, 255, 102, 0.3); border-radius: 10px; padding: 12px; display: flex; align-items: center; margin-bottom: 10px;'>
                {avatar_html}
                <div style='overflow: hidden;'>
                    <div style='font-weight: 600; color: #FFFFFF; font-size: 14px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden;'>{name}</div>
                    <div style='font-size: 11px; color: #A0AEC0; white-space: nowrap; text-overflow: ellipsis; overflow: hidden;'>{email}</div>
                    <div style='font-size: 10px; color: #00FF66;'>● Conectado (Portafolios Privados)</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.sidebar.button("🚪 Cerrar Sesión", use_container_width=True, key="btn_logout_google"):
            logout_user()
    else:
        # Guest Mode / Login CTA
        if oauth_ready:
            auth_url = get_google_auth_url()
            st.sidebar.markdown(
                f"""
                <a href="{auth_url}" target="_self" style="text-decoration: none;">
                    <div style="background: #FFFFFF; color: #3c4043; border-radius: 8px; padding: 10px 14px; display: flex; align-items: center; justify-content: center; font-weight: 600; font-size: 13px; box-shadow: 0 2px 4px rgba(0,0,0,0.25); cursor: pointer; border: 1px solid #dadce0; margin-bottom: 8px;">
                        <svg version="1.1" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48" style="width: 20px; height: 20px; margin-right: 10px;">
                            <path fill="#EA4335" d="M24 9.5c3.54 0 6.71 1.22 9.21 3.6l6.85-6.85C35.9 2.38 30.47 0 24 0 14.62 0 6.51 5.38 2.56 13.22l7.98 6.19C12.43 13.72 17.74 9.5 24 9.5z"/>
                            <path fill="#4285F4" d="M46.98 24.55c0-1.57-.15-3.09-.38-4.55H24v9.02h12.94c-.58 2.96-2.26 5.48-4.78 7.18l7.73 6c4.51-4.18 7.09-10.36 7.09-17.65z"/>
                            <path fill="#FBBC05" d="M10.53 28.59c-.48-1.45-.76-2.99-.76-4.59s.27-3.14.76-4.59l-7.98-6.19C.92 16.46 0 20.12 0 24c0 3.88.92 7.54 2.56 10.78l7.97-6.19z"/>
                            <path fill="#34A853" d="M24 48c6.48 0 11.93-2.13 15.89-5.81l-7.73-6c-2.15 1.45-4.92 2.3-8.16 2.3-6.26 0-11.57-4.22-13.47-9.91l-7.98 6.19C6.51 42.62 14.62 48 24 48z"/>
                        </svg>
                        Iniciar Sesión con Google
                    </div>
                </a>
                """,
                unsafe_allow_html=True,
            )
            st.sidebar.caption("🔒 Inicia sesión para que tus portafolios guardados sean privados y solo tú puedas verlos.")
        else:
            st.sidebar.info("👤 **Modo Invitado Activo**\n\nTus portafolios se guardan localmente. Puedes configurar Google OAuth en `.streamlit/secrets.toml` para inicio de sesión en la nube.")
            with st.sidebar.expander("🛠️ Simular Usuario (Demo)", expanded=False):
                demo_email = st.text_input("Correo Demo", value="amigo@gmail.com", key="demo_email_input")
                demo_name = st.text_input("Nombre Demo", value="Amigo Inversionista", key="demo_name_input")
                if st.button("Simular Login", use_container_width=True, key="btn_demo_login"):
                    st.session_state["user"] = {
                        "id": "demo_123",
                        "email": demo_email.strip(),
                        "name": demo_name.strip(),
                        "picture": "",
                        "verified_email": True,
                    }
                    st.rerun()

    st.sidebar.markdown("---")
