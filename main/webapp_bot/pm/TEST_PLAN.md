# Test Plan — Audio High-Res

| Версия документа | 1.0 |
| Автор           | QA Team |
| Дата            | 19.05.2025 |

## 1 Scope
Покрыть функциональные, нефункциональные (нагрузка ≥ 100 RPS) и security-аспекты Telegram-бота + Flask-API.

## 2 References
* SRS «Требования к ПО …», v 1.3  
* Release Notes  
* Risk Log

## 3 Risks
| ID | Риск | Вероятность | Урон | План |
| R1 | GPU недоступен на хосте | medium | medium | Fallback → CPU |
| R2 | Flood-limit Telegram | low | high | Rate-limit в bot |

## 4 Entry / Exit Criteria
* Entry — ветка `main`, CI «зелёная».  
* Exit — все P0 / P1 тест-кейсы PASSED, open-bugs ≤ 2 (P3).

## 5 Schedule
| Дата | Активность |
|------|-----------|
| 19-20 Май | smoke, unit |
| 21-22 Май | load (Low) |
| 23-24 Май | security-scan |
| 25 Май    | sign-off |

## 6 Test Items
| Модуль | Как проверяем |
|--------|---------------|
| `/voice/tts` | Positive / limits / bad payload |
| XTTSv2 wrapper | unit-fake CUDA |
| Web-App | Cypress e2e (out-of-scope CI) |

## 7 Strategy
* **Unit** — `pytest`, coverage ≥ 80 %.  
* **Load** — `locust` 100 users, 3 min, SLA p95 < 1 s.  
* **Security** — `bandit`, OWASP ZAP passive.  
* **Regression** — `tests/test_smoke.py` в CI.

## 8 Environment
* Ubuntu 22.04, Python 3.11, CUDA 11.8.  
* Staging: `http://stage.audiohighres.ai`.

## 9 Deliverables
* `htmlcov/` — coverage  
* `locust_stats.csv`  
* ZAP-report HTML

## 10 Staffing
| Имя | Роль |
|-----|------|
| Анна | QA Lead |
| Иван | Automation |
| Пётр | Security QA |

## 11 Approvals
| Имя | Дата | Статус |
|-----|------|--------|
| Team-Lead | 25 Май | ✅ |
