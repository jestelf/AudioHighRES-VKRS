from server_bot import get_classifier
import pytest, asyncio

@pytest.mark.asyncio
async def test_safe_message():
    clf = get_classifier()
    res = await clf.analyse("Здравствуйте, как дела?")
    assert res["Безопасные сообщения"] > 0.5

@pytest.mark.asyncio
async def test_scam_message():
    clf = get_classifier()
    res = await clf.analyse("Вы выиграли миллион!")
    risky = {k:v for k,v in res.items() if k!="Безопасные сообщения"}
    assert max(risky.values()) > 0.4
