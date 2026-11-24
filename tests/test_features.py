from src.features import extract_features

def test_extract_features_counts_url_and_keywords():
    result = extract_features("Urgent! Verify account at https://example.com")
    assert result["url_count"] == 1
    assert result["suspicious_word_count"] >= 2
    assert result["exclamation_count"] == 1
