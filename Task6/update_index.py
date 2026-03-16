#!/usr/bin/env python3

import json
import os
import time
import pickle
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Set, Dict, Any
import faiss
import numpy as np

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("update.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class KnowledgeIndexUpdater:
    def __init__(self, data_source_path: str = "./knowledge_base", storage_path: str = "./faiss_index"):
        self.data_source_path = Path(data_source_path)
        self.storage_path = storage_path

        # Модель из исходного скрипта
        self.embedding_model_name = "BAAI/bge-m3"
        self.embedding_engine = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={
                'normalize_embeddings': True,
                'batch_size': 32
            }
        )

        # Параметры чанков из исходного скрипта
        self.chunk_processor = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
        )

        self.source_documents = []
        self.processed_chunks = []
        self.metadata_file = Path(self.storage_path) / "document_metadata.json"
        self.index_file = Path(self.storage_path) / "faiss.index"
        self.metadata_pkl = Path(self.storage_path) / "metadata.pkl"

    def calculate_file_hash(self, file_path: Path) -> str:
        """Вычисление хеша файла для отслеживания изменений"""
        hasher = hashlib.md5()
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating hash for {file_path}: {e}")
            return ""

    def retrieve_documents(self) -> List[Document]:
        """Загрузка документов из источника данных"""
        logger.info("🔍 Scanning for documents...")
        self.source_documents = []

        if not self.data_source_path.exists():
            logger.error(f"❌ Data source path does not exist: {self.data_source_path}")
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

                            file_hash = self.calculate_file_hash(file_path)
                            file_stats = file_path.stat()

                            metadata = {
                                "source": str(relative_path),
                                "category": category,
                                "filename": file_path.name,
                                "file_path": str(file_path),
                                "file_size": len(content),
                                "file_type": file_path.suffix,
                                "last_modified": file_stats.st_mtime,
                                "file_hash": file_hash,
                                "created_time": file_stats.st_ctime
                            }

                            doc = Document(
                                page_content=content,
                                metadata=metadata
                            )
                            self.source_documents.append(doc)
                            logger.info(f"   ✓ Loaded: {relative_path}")

                    except Exception as e:
                        logger.error(f"   ✗ Error loading {file_path}: {e}")

        logger.info(f"📊 Retrieved: {len(self.source_documents)} documents")
        return self.source_documents

    def load_existing_metadata(self) -> Dict[str, Any]:
        """Загрузка существующих метаданных документов"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading metadata: {e}")
        return {}

    def save_metadata(self, metadata: Dict[str, Any]):
        """Сохранение метаданных документов"""
        os.makedirs(self.storage_path, exist_ok=True)
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")

    def identify_changed_documents(self) -> tuple[List[Document], Set[str]]:
        """Идентификация измененных и новых документов"""
        existing_metadata = self.load_existing_metadata()
        changed_docs = []
        changed_sources = set()

        for doc in self.source_documents:
            source = doc.metadata["source"]
            current_hash = doc.metadata["file_hash"]
            last_modified = doc.metadata["last_modified"]

            # Проверка на новый или измененный документ
            if (source not in existing_metadata or
                existing_metadata[source].get("file_hash") != current_hash or
                existing_metadata[source].get("last_modified") != last_modified):

                changed_docs.append(doc)
                changed_sources.add(source)
                logger.info(f"📄 Changed/New document: {source}")

        return changed_docs, changed_sources

    def segment_documents(self, documents: List[Document]) -> List[Document]:
        """Разбивка документов на чанки"""
        chunks = []
        for doc in documents:
            segments = self.chunk_processor.split_documents([doc])
            valid_segments = []

            for i, segment in enumerate(segments):
                clean_content = segment.page_content.strip()
                if len(clean_content) >= 50:  # Минимальная длина чанка
                    # Обновляем метаданные
                    new_metadata = segment.metadata.copy()
                    new_metadata.update({
                        "chunk_id": f"{Path(doc.metadata['filename']).stem}_{i}",
                        "chunk_index": i,
                        "total_chunks": len(segments),
                        "content_length": len(clean_content),
                        "parent_source": doc.metadata["source"]
                    })

                    valid_segment = Document(
                        page_content=clean_content,
                        metadata=new_metadata
                    )
                    valid_segments.append(valid_segment)

            chunks.extend(valid_segments)
            logger.info(f"   📄 {doc.metadata['filename']}: {len(segments)} → {len(valid_segments)} valid chunks")

        return chunks

    def load_existing_faiss_data(self) -> tuple[Any, List[Dict]]:
        """Загрузка существующего FAISS индекса и метаданных"""
        try:
            if self.index_file.exists() and self.metadata_pkl.exists():
                index = faiss.read_index(str(self.index_file))
                with open(self.metadata_pkl, 'rb') as f:
                    metadata = pickle.load(f)
                logger.info("✅ Loaded existing FAISS index and metadata")
                return index, metadata
        except Exception as e:
            logger.warning(f"Could not load existing FAISS data: {e}")

        # Создание нового индекса если не существует
        dimension = 1024  # BAAI/bge-m3 имеет размерность 1024
        index = faiss.IndexFlatIP(dimension)
        return index, []

    def update_faiss_index(self) -> tuple[Any, int]:
        """Обновление FAISS индекса"""
        os.makedirs(self.storage_path, exist_ok=True)

        # Загрузка существующих данных
        index, existing_metadata = self.load_existing_faiss_data()

        # Получение и анализ документов
        self.retrieve_documents()
        changed_docs, changed_sources = self.identify_changed_documents()

        if not changed_docs:
            logger.info("✅ No changes detected in documents")
            return index, 0

        logger.info(f"🔄 Processing {len(changed_docs)} changed documents")

        # Обработка измененных документов
        new_chunks = self.segment_documents(changed_docs)

        if not new_chunks:
            logger.info("✅ No new chunks to add")
            return index, 0

        # Генерация эмбеддингов для новых чанков
        logger.info("🧮 Generating embeddings for new chunks...")
        contents = [chunk.page_content for chunk in new_chunks]

        try:
            new_embeddings = self.embedding_engine.embed_documents(contents)
            logger.info(f"✅ Generated {len(new_embeddings)} embeddings")
        except Exception as e:
            logger.error(f"❌ Error generating embeddings: {e}")
            return index, 0

        # Удаление старых версий измененных документов из существующих метаданных
        updated_metadata = [item for item in existing_metadata
                          if item.get("metadata", {}).get("parent_source") not in changed_sources]

        logger.info(f"🗑️  Removed {len(existing_metadata) - len(updated_metadata)} old chunks")

        # Добавление новых чанков в метаданные
        chunks_added = 0
        for chunk, embedding in zip(new_chunks, new_embeddings):
            chunk_data = {
                "content": chunk.page_content,
                "metadata": chunk.metadata,
                "embedding": embedding
            }
            updated_metadata.append(chunk_data)
            chunks_added += 1

        # Перестроение индекса с обновленными данными
        logger.info("🏗️  Rebuilding FAISS index...")

        if updated_metadata:
            all_embeddings = [item["embedding"] for item in updated_metadata]
            embeddings_array = np.array(all_embeddings).astype('float32')

            # Создание нового индекса
            dimension = len(all_embeddings[0])
            index = faiss.IndexFlatIP(dimension)
            index.add(embeddings_array)

            # Сохранение обновленного индекса и метаданных
            faiss.write_index(index, str(self.index_file))
            with open(self.metadata_pkl, 'wb') as f:
                pickle.dump(updated_metadata, f)

            logger.info(f"💾 Saved FAISS index with {len(updated_metadata)} total chunks")
        else:
            logger.warning("No chunks to index")

        # Обновление метаданных документов
        new_metadata = {
            doc.metadata["source"]: {
                "file_hash": doc.metadata["file_hash"],
                "last_modified": doc.metadata["last_modified"],
                "file_size": doc.metadata["file_size"],
                "category": doc.metadata["category"]
            }
            for doc in self.source_documents
        }
        self.save_metadata(new_metadata)

        return index, chunks_added

    def execute_sample_queries(self, index, metadata: List[Dict]):
        """Выполнение тестовых запросов для проверки обновления"""
        sample_queries = [
             "Who was born in Eldamar?",
             "What is most important era?",
             "Who create the ring?"
        ]

        logger.info("\n🧪 Testing search functionality:")

        for query in sample_queries:
            try:
                logger.info(f"\n🔍 Query: {query}")

                # Генерация эмбеддинга для запроса
                query_embedding = self.embedding_engine.embed_query(query)
                query_vector = np.array([query_embedding]).astype('float32')

                # Поиск в FAISS индексе
                distances, indices = index.search(query_vector, k=2)

                for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                    if idx < len(metadata):
                        chunk_data = metadata[idx]
                        source = chunk_data["metadata"].get("source", "Unknown")
                        content_preview = chunk_data["content"][:100] + "..." if len(chunk_data["content"]) > 100 else chunk_data["content"]
                        logger.info(f"   {i+1}. {source} (similarity: {distance:.3f})")
                        logger.info(f"      {content_preview}")

            except Exception as e:
                logger.error(f"Error executing query '{query}': {e}")

    def load_current_metadata(self) -> List[Dict]:
        """Загрузка текущих метаданных чанков"""
        try:
            with open(self.metadata_pkl, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading current metadata: {e}")
            return []

    def record_update_metrics(self, execution_duration: float, chunks_added: int):
        """Запись метрик обновления"""
        try:
            current_metadata = self.load_current_metadata()
            total_chunks = len(current_metadata)
        except:
            total_chunks = 0

        metrics = {
            "update_timestamp": datetime.now().isoformat(),
            "model": {
                "name": self.embedding_model_name,
                "embedding_size": 1024  # BAAI/bge-m3 имеет размерность 1024
            },
            "knowledge_base": {
                "source_path": str(self.data_source_path),
                "total_documents": len(self.source_documents),
                "changed_documents": len(self.identify_changed_documents()[0]),
                "chunks_added": chunks_added,
                "total_chunks_in_index": total_chunks
            },
            "processing": {
                "execution_time_seconds": round(execution_duration, 2),
                "execution_time_minutes": round(execution_duration / 60, 2),
                "chunks_per_second": round(chunks_added / execution_duration, 2) if execution_duration > 0 else 0
            },
            "vectorstore": {
                "type": "FAISS",
                "index_type": "IndexFlatIP",
                "persist_directory": self.storage_path
            },
            "status": "success" if chunks_added >= 0 else "error"
        }

        # Сохранение метрик в файл
        with open("update_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        return metrics

def main():
    """Основная функция обновления"""
    start_time = time.time()
    logger.info("🚀 Starting Knowledge Base Update Process")

    try:
        # Инициализация и обновление
        updater = KnowledgeIndexUpdater()
        index, chunks_added = updater.update_faiss_index()

        # Загрузка метаданных для тестовых запросов
        current_metadata = updater.load_current_metadata()

        # Тестовые запросы
        if current_metadata:
            updater.execute_sample_queries(index, current_metadata)

        # Запись метрик
        execution_time = time.time() - start_time
        metrics = updater.record_update_metrics(execution_time, chunks_added)

        # Итоговый отчет
        logger.info("\n" + "=" * 60)
        logger.info("📊 UPDATE COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"Timestamp: {metrics['update_timestamp']}")
        logger.info(f"Model: {metrics['model']['name']}")
        logger.info(f"Total Documents: {metrics['knowledge_base']['total_documents']}")
        logger.info(f"Changed Documents: {metrics['knowledge_base']['changed_documents']}")
        logger.info(f"Chunks Added: {metrics['knowledge_base']['chunks_added']}")
        logger.info(f"Total Chunks in Index: {metrics['knowledge_base']['total_chunks_in_index']}")
        logger.info(f"  Execution Time: {metrics['processing']['execution_time_seconds']}s")
        logger.info(f"  Vector Store: {metrics['vectorstore']['type']}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"❌ Update failed: {e}")
        import traceback
        traceback.print_exc()

        # Запись ошибки в метрики
        error_metrics = {
            "update_timestamp": datetime.now().isoformat(),
            "status": "error",
            "error_message": str(e)
        }
        with open("update_metrics.json", "w", encoding="utf-8") as f:
            json.dump(error_metrics, f, indent=2, ensure_ascii=False)
        raise

if __name__ == "__main__":
    main()
