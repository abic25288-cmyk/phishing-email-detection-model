import re

SUSPICIOUS_WORDS = {
    "verify", "urgent", "suspended", "password", "confirm",
    "account", "login", "immediately", "payment", "invoice"
}

def extract_features(text: str) -> dict:
    text = str(text or "")
    lower = text.lower()
    urls = re.findall(r"https?://\S+|www\.\S+", lower)
    return {
        "url_count": len(urls),
        "has_ip_url": int(bool(re.search(r"https?://(?:\d{1,3}\.){3}\d{1,3}", lower))),
        "suspicious_word_count": sum(word in lower for word in SUSPICIOUS_WORDS),
        "exclamation_count": text.count("!"),
        "text_length": len(text),
    }
