# Архитектура Audio High-Res

```mermaid
graph TD
  subgraph "External"
    TG["Telegram<br>Client"]:::ext
    Browser["Web-App<br>SPA"]:::ext
  end
  TG -->|Webhook / poll| Bot
  Browser -->|XHR / fetch| API
  Bot --> API

  subgraph "Flask API"
    API["Blueprint / Flask"]:::code
    VOICE["VoiceModule (XTTS-v2)"]:::code
    CLF["ScamClassifier<br>(HF transformers)"]:::code
    PAT["PatentTTS<br>Checker"]:::code
  end

  API --> VOICE
  API --> CLF
  API --> PAT

  subgraph "Storage"
    JSON1["tariffs_db.json"]:::data
    JSON2["user_settings.json"]:::data
    EMB["users_emb/*"]:::data
  end
  API -. R/W .-> JSON1 & JSON2 & EMB

  classDef code  fill:#24283b,color:#fff;
  classDef data  fill:#394260,color:#fff;
  classDef ext   fill:#2d5b71,color:#fff;
