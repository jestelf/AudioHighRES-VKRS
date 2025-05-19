# Lessons Learned

| № | Lesson | Category | Actionable next-time |
|---|--------|----------|----------------------|
| 1 | Телеграм не отдаёт `web_app_data` при deep-link, если нет `/start`. | Integration | Всегда делать «hand-shake» `/start` перед WebApp. |
| 2 | XTTS v2 при `speed>2.5` ломается (NaN). | Tech Debt | Ограничили `_clamp()` до `<3.0`. Нужно патчить модель. |
| 3 | Pydub читает `ogg` медленно — 100 ms на 5 сэмплов. | Perf | Переключиться на ffmpeg-py / torchaudio stream. |
| 4 | Bandit найден Hard-coded token. | Security | Все секреты в `.env`, GH Secret Scanning enabled. |
