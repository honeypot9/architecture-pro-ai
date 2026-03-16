#!/usr/bin/env python3

import json
import os
import time
import pickle
from pathlib import Path
from typing import List
import faiss
import numpy as np

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

class KnowledgeIndexBuilder:
    def __init__(self, data_source_path: str = "../Task2/knowledge_base"):
        self.data_source_path = Path(data_source_path)

        # Используем выбранную в Задании 1 модель BAAI/bge-m3
        self.embedding_model_name = "BAAI/bge-m3"
        self.embedding_engine = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={
                'normalize_embeddings': True,
                'batch_size': 32
            }
        )

        # Настройка сплиттера с оптимальными параметрами
        self.chunk_processor = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
        )

        self.source_documents = []
        self.processed_chunks = []
        self.embeddings = []

    def retrieve_documents(self) -> List[Document]:
        """Загрузка документов из базы знаний"""
        print("🔍 Загрузка документов...")

        if not self.data_source_path.exists():
            print(f"❌ Путь не существует: {self.data_source_path}")
            return []

        supported_extensions = ['*.txt', '*.md']

        for extension in supported_extensions:
            for file_path in self.data_source_path.rglob(extension):
                if file_path.is_file():
                    try:
                        content = ""
                        if file_path.suffix.lower() == '.txt':
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read().strip()
                        elif file_path.suffix.lower() == '.md':
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read().strip()

                        if content and len(content) > 50:
                            relative_path = file_path.relative_to(self.data_source_path)
                            category = relative_path.parts[0] if len(relative_path.parts) > 1 else "general"

                            metadata = {
                                "source": str(relative_path),
                                "category": category,
                                "filename": file_path.name,
                                "file_path": str(file_path),
                                "file_size": len(content),
                                "file_type": file_path.suffix
                            }

                            doc = Document(
                                page_content=content,
                                metadata=metadata
                            )
                            self.source_documents.append(doc)
                            print(f"   ✓ Загружен: {relative_path}")

                    except Exception as e:
                        print(f"   ✗ Ошибка загрузки {file_path}: {e}")

        print(f"📊 Загружено документов: {len(self.source_documents)}")
        return self.source_documents

    def segment_documents(self) -> List[Document]:
        """Разбивка документов на логические чанки"""
        print("✂️  Разбивка документов на чанки...")

        total_chunks_before_filter = 0

        for doc in self.source_documents:
            segments = self.chunk_processor.split_documents([doc])
            total_chunks_before_filter += len(segments)

            valid_segments = []
            for i, segment in enumerate(segments):
                clean_content = segment.page_content.strip()
                if len(clean_content) >= 100:
                    # Обновляем метаданные, сохраняя оригинальные
                    new_metadata = segment.metadata.copy()
                    new_metadata.update({
                        "chunk_id": f"{Path(doc.metadata['filename']).stem}_{i}",
                        "chunk_index": i,
                        "total_chunks": len(segments),
                        "content_length": len(clean_content)
                    })

                    valid_segment = Document(
                        page_content=clean_content,
                        metadata=new_metadata
                    )
                    valid_segments.append(valid_segment)

            self.processed_chunks.extend(valid_segments)
            print(f"   📄 {doc.metadata['filename']}: {len(segments)} → {len(valid_segments)} валидных чанков")

        print(f"📊 Всего чанков создано: {len(self.processed_chunks)} (из {total_chunks_before_filter} до фильтрации)")
        return self.processed_chunks

    def generate_embeddings(self) -> List[List[float]]:
        """Генерация эмбеддингов для всех чанков"""
        print("🧮 Генерация эмбеддингов...")

        if not self.processed_chunks:
            raise ValueError("Нет чанков для обработки. Сначала выполните segment_documents()")

        contents = [chunk.page_content for chunk in self.processed_chunks]

        # Пакетная обработка для оптимизации скорости
        batch_size = 32
        for i in range(0, len(contents), batch_size):
            batch_contents = contents[i:i + batch_size]
            try:
                batch_embeddings = self.embedding_engine.embed_documents(batch_contents)
                self.embeddings.extend(batch_embeddings)

                if (i // batch_size) % 10 == 0:
                    print(f"   Обработано: {min(i + batch_size, len(contents))}/{len(contents)} чанков")
            except Exception as e:
                print(f"   Ошибка при обработке батча {i}: {e}")
                # Пробуем обработать по одному документу
                for j, content in enumerate(batch_contents):
                    try:
                        embedding = self.embedding_engine.embed_documents([content])
                        self.embeddings.extend(embedding)
                    except Exception as single_error:
                        print(f"   Ошибка при обработке отдельного документа: {single_error}")
                        # Добавляем нулевой эмбеддинг в случае ошибки
                        self.embeddings.append([0.0] * 1024)  # BAAI/bge-m3 имеет размерность 1024

        print(f"✅ Сгенерировано эмбеддингов: {len(self.embeddings)}")
        return self.embeddings

    def build_faiss_index(self, storage_path: str = "./faiss_index"):
        """Создание и сохранение FAISS индекса"""
        print("🏗️  Создание FAISS индекса...")

        os.makedirs(storage_path, exist_ok=True)

        if not self.embeddings:
            raise ValueError("Эмбеддинги не сгенерированы. Сначала вызовите generate_embeddings()")

        # Преобразование в numpy array
        embeddings_array = np.array(self.embeddings).astype('float32')
        dimension = embeddings_array.shape[1]

        print(f"   Размерность эмбеддингов: {dimension}")
        print(f"   Форма массива: {embeddings_array.shape}")

        # Создание FAISS индекса
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings_array)

        # Сохранение индекса
        faiss.write_index(index, os.path.join(storage_path, "faiss.index"))

        # Сохранение метаданных
        metadata = {
            "chunks": [
                {
                    "content": chunk.page_content,
                    "metadata": chunk.metadata
                }
                for chunk in self.processed_chunks
            ],
            "embedding_model": self.embedding_model_name,
            "embedding_dimension": dimension,
            "total_chunks": len(self.processed_chunks)
        }

        with open(os.path.join(storage_path, "metadata.pkl"), 'wb') as f:
            pickle.dump(metadata, f)

        # Сохранение информации о модели
        model_info = {
            "name": self.embedding_model_name,
            "repository": "https://huggingface.co/BAAI/bge-m3",
            "embedding_size": dimension,
            "normalized": True
        }

        with open(os.path.join(storage_path, "model_info.json"), 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)

        print(f"💾 Индекс сохранен в: {storage_path}")
        return index

    def test_search(self, index, sample_queries: List[str] = None):
        """Тестирование поиска по индексу"""
        if sample_queries is None:
            sample_queries = [
                 "Who was born in Eldamar?",
                 "What is most important era?",
                 "Who create the ring?"
            ]

        print("\n🧪 Тестирование поиска:")
        print("=" * 60)

        for query in sample_queries:
            print(f"\n🔍 Запрос: '{query}'")

            try:
                # Эмбеддинг для запроса
                query_embedding = self.embedding_engine.embed_query(query)
                query_vector = np.array([query_embedding]).astype('float32')

                # Поиск 3 наиболее похожих чанков
                distances, indices = index.search(query_vector, k=3)

                for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                    if idx < len(self.processed_chunks):
                        chunk = self.processed_chunks[idx]
                        print(f"   {i+1}. 📁 {chunk.metadata['source']} (сходство: {distance:.3f})")
                        print(f"      {chunk.page_content[:120]}...")
                        print()
            except Exception as e:
                print(f"   Ошибка при поиске для запроса '{query}': {e}")

    def record_metrics(self, execution_duration: float):
        """Запись метрик выполнения"""
        embedding_size = len(self.embeddings[0]) if self.embeddings else 0

        metrics = {
            "model": {
                "name": self.embedding_model_name,
                "repository": "https://huggingface.co/BAAI/bge-m3",
                "embedding_size": embedding_size
            },
            "knowledge_base": {
                "path": str(self.data_source_path),
                "total_documents": len(self.source_documents),
                "total_chunks": len(self.processed_chunks)
            },
            "processing": {
                "time_seconds": execution_duration,
                "time_minutes": execution_duration / 60,
                "chunks_per_second": len(self.processed_chunks) / execution_duration if execution_duration > 0 else 0
            },
            "vectorstore": {
                "type": "FAISS",
                "persist_directory": "./faiss_index",
                "index_type": "IndexFlatIP"
            },
            "environment": {
                "langchain_version": "0.1.0+",
                "embedding_backend": "sentence-transformers"
            }
        }

        with open("indexing_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        return metrics

def main():
    print("🚀 Запуск создания векторного индекса")
    print("=" * 50)

    start_time = time.time()

    try:
        # Инициализация билдера
        builder = KnowledgeIndexBuilder()

        # Основной пайплайн
        print("\n📁 Этап 1: Загрузка документов")
        builder.retrieve_documents()

        print("\n✂️  Этап 2: Разбивка на чанки")
        builder.segment_documents()

        print("\n🧮 Этап 3: Генерация эмбеддингов")
        builder.generate_embeddings()

        print("\n🏗️  Этап 4: Построение FAISS индекса")
        index = builder.build_faiss_index()

        print("\n🧪 Этап 5: Тестирование поиска")
        builder.test_search(index)

        # Запись метрик
        execution_time = time.time() - start_time
        metrics = builder.record_metrics(execution_time)

        # Вывод итоговой информации
        print("\n" + "=" * 50)
        print("ВЕКТОРНЫЙ ИНДЕКС УСПЕШНО СОЗДАН")
        print("=" * 50)
        print(f"Модель эмбеддингов: {metrics['model']['name']}")
        print(f"Размерность векторов: {metrics['model']['embedding_size']}")
        print(f"Обработано документов: {metrics['knowledge_base']['total_documents']}")
        print(f"Создано чанков: {metrics['knowledge_base']['total_chunks']}")
        print(f"Общее время: {metrics['processing']['time_minutes']:.2f} минут")
        print(f"Скорость обработки: {metrics['processing']['chunks_per_second']:.1f} чанков/сек")
        print("=" * 50)

    except Exception as e:
        print(f"❌ Ошибка при создании индекса: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
