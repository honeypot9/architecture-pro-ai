#!/usr/bin/env python3

import json
import pickle
import faiss
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings

class FAISSSearchEngine:
    def __init__(self, index_path: str = "./faiss_index"):
        self.index_path = index_path

        # Загрузка модели эмбеддингов
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # Загрузка FAISS индекса
        self.index = faiss.read_index(f"{index_path}/faiss.index")

        # Загрузка метаданных
        with open(f"{index_path}/metadata.pkl", 'rb') as f:
            self.metadata = pickle.load(f)

        # Загрузка информации о модели
        with open(f"{index_path}/model_info.json", 'r', encoding='utf-8') as f:
            self.model_info = json.load(f)

        print(f"✅ FAISS индекс загружен")
        print(f"   Модель: {self.model_info['name']}")
        print(f"   Размерность: {self.model_info['embedding_size']}")
        print(f"   Всего чанков: {len(self.metadata['chunks'])}")

    def search(self, query: str, k: int = 3):
        """Поиск по запросу"""
        try:
            # Генерация эмбеддинга для запроса
            query_embedding = self.embedding_model.embed_query(query)
            query_vector = np.array([query_embedding]).astype('float32')

            # Поиск в FAISS
            distances, indices = self.index.search(query_vector, k=k)

            results = []
            for distance, idx in zip(distances[0], indices[0]):
                if idx < len(self.metadata['chunks']):
                    chunk_data = self.metadata['chunks'][idx]
                    results.append({
                        'content': chunk_data['content'],
                        'metadata': chunk_data['metadata'],
                        'similarity_score': float(distance)
                    })

            return results
        except Exception as e:
            print(f"Ошибка при поиске: {e}")
            return []

    def detailed_search_test(self, queries: list):
        """Детальное тестирование поиска"""
        print("\n" + "🔍 ТЕСТИРОВАНИЕ ПОИСКА")
        print("=" * 70)

        for i, query in enumerate(queries, 1):
            print(f"\n{i}. ЗАПРОС: '{query}'")
            print("-" * 70)

            results = self.search(query, k=3)

            if not results:
                print("   ❌ Не найдено результатов")
                continue

            for j, result in enumerate(results, 1):
                print(f"   {j}. {result['metadata']['source']}")
                print(f"       Сходство: {result['similarity_score']:.3f}")
                print(f"      ️ Категория: {result['metadata']['category']}")
                print(f"       ID чанка: {result['metadata']['chunk_id']}")
                print(f"       Размер: {result['metadata']['content_length']} символов")
                print(f"       Позиция: {result['metadata']['chunk_index'] + 1}/{result['metadata']['total_chunks']}")
                print(f"       Содержание:")
                print(f"      {result['content'][:150]}...")
                print()

    def get_index_stats(self):
        """Получение статистики индекса"""
        return {
            "total_chunks": len(self.metadata['chunks']),
            "embedding_dimension": self.model_info['embedding_size'],
            "embedding_model": self.model_info['name'],
            "index_type": "FAISS IndexFlatIP"
        }

def main():
    try:
        # Инициализация поискового движка
        print("🔄 Инициализация поискового движка...")
        search_engine = FAISSSearchEngine()

        # Тестовые запросы
        test_queries = [
            "Who was born in Eldamar?",
            "What is most important era?",
            "Who create the ring?"
        ]

        # Запуск тестов
        search_engine.detailed_search_test(test_queries)

        # Дополнительная информация
        stats = search_engine.get_index_stats()
        print("\n" + "📊 СВОДКА ИНДЕКСА")
        print("=" * 50)
        print(f"Всего чанков в индексе: {stats['total_chunks']}")
        print(f"Размерность векторов: {stats['embedding_dimension']}")
        print(f"Модель эмбеддингов: {stats['embedding_model']}")
        print(f"Тип индекса: {stats['index_type']}")

    except FileNotFoundError as e:
        print(f"❌ Файл не найден: {e}")
        print("Убедитесь, что вы сначала запустили build_index.py для создания индекса")
    except Exception as e:
        print(f"❌ Ошибка при тестировании: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
