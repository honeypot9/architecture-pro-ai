### Эмбеддинг

- Модель: `BAAI/bge-m3` (репозиторий: [Hugging Face](https://huggingface.co/BAAI/bge-m3))
- Размер эмбеддингов: 1024

1. `pip install -r requirements.txt`
2. Построение индексов `python build_index.py`
3. Тестирование `python test_index.py`

Статистика - [link](./indexing_metrics.json)
Индекс - [folder](./faiss_index)
