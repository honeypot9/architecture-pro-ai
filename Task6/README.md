# Автоматическое ежедневное обновление базы знаний

1. Создан скрипт для обновления [link](./update_index.py).
2. Для настройки автоматики необходимо выполнить:
    1. crontab -e
    2. 0 6 * * * /usr/bin/python3 /path/to/update_index.py >> /tmp/cron.log 2>&1

## Архитектурная диаграмма

[link](./updateSchema.puml)

## Пример лога выполнения

[link](./update.log)
