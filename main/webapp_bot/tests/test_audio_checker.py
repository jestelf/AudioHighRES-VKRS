from audio_checker import predict
def test_predict_returns_label(silence_wav):
    out = predict(str(silence_wav))
    assert "BINARY:" in out and "CLASS:" in out
