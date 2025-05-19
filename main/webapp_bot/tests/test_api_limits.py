from server_bot import tariff_info, set_tariff

def test_free_limit():
    info = tariff_info("dummy")   # несуществующий → free
    assert info == {"slots": 1, "daily_gen": 5}

def test_vip_limit():
    set_tariff("dummy", "vip")
    assert tariff_info("dummy")["slots"] == 6
