#!/usr/bin/env python3

import os
import logging
import asyncio
from typing import Dict, List, Any
from dataclasses import dataclass

import faiss
import pickle
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models.yandex import ChatYandexGPT

from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ConversationHandler,
    ContextTypes,
    filters
)

# Logging setup
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Conversation states
SELECTING_ACTION, ASKING_QUESTION, SELECTING_METHOD = range(3)

# Keyboards
main_keyboard = [["Ask Question", "Change Method"], ["Show Sources", "Bot Info"]]
main_markup = ReplyKeyboardMarkup(main_keyboard, one_time_keyboard=True)

method_keyboard = [["standard", "few_shot"], ["cot", "Back"]]
method_markup = ReplyKeyboardMarkup(method_keyboard, one_time_keyboard=True)

@dataclass
class RAGConfig:
    """RAG system configuration"""
    faiss_index_path: str = "../../Task3/faiss_index"
    embedding_model: str = "BAAI/bge-m3"
    yandex_model: str = "yandexgpt-lite"
    temperature: float = 0.7
    max_tokens: int = 800
    top_k: int = 3
    similarity_threshold:  float = 0.4

class SimpleRetriever:
    """Simple FAISS retriever"""

    def __init__(self, config: RAGConfig):
        self.config = config

        # Load embedding model
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=config.embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # Load FAISS index and metadata
        try:
            self.index = faiss.read_index(f"{config.faiss_index_path}/faiss.index")
            with open(f"{config.faiss_index_path}/metadata.pkl", 'rb') as f:
                self.metadata = pickle.load(f)
            logger.info(f"Retriever loaded: {len(self.metadata['chunks'])} chunks")
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            raise

    def search(self, query: str) -> List[Dict]:
        """Search for relevant documents"""
        try:
            query_embedding = self.embedding_model.embed_query(query)
            query_vector = np.array([query_embedding]).astype('float32')

            distances, indices = self.index.search(query_vector, k=self.config.top_k)

            documents = []
            for distance, idx in zip(distances[0], indices[0]):
                if idx < len(self.metadata['chunks']):
                    chunk_data = self.metadata['chunks'][idx]
                    documents.append({
                        'content': chunk_data['content'],
                        'source': chunk_data['metadata'].get('source', 'Unknown'),
                        'similarity': float(distance)
                    })
            logger.info(f"Similarity : {documents}")
            return [doc for doc in documents if doc['similarity'] >= self.config.similarity_threshold]

        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

class SimpleRAGEngine:
    """Simplified RAG engine"""

    def __init__(self):
        self.config = RAGConfig()
        self.retriever = SimpleRetriever(self.config)
        self.llm = self._setup_llm()

    def _setup_llm(self):
        """Setup YandexGPT"""
        try:
            api_key = os.getenv("YANDEX_API_KEY")
            folder_id = os.getenv("YANDEX_FOLDER_ID")

            if not api_key or not folder_id:
                raise ValueError("Set YANDEX_API_KEY and YANDEX_FOLDER_ID environment variables")

            return ChatYandexGPT(
                api_key=api_key,
                folder_id=folder_id,
                model_name=self.config.yandex_model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
        except Exception as e:
            logger.error(f"LLM setup error: {e}")
            return None

    def _build_prompt(self, context: str, question: str, method: str) -> str:
        """Build English prompt based on method"""

        if method == "few_shot":
            examples = """
Example 1:
Question: Who is Borin Stoneheart?
Answer: Borin Stoneheart is a Dwarf of the Stone Lineage, son of Glóin, and a member of The Company of the Ring. He became the Lord of the Crystal Caverns at Stone Gorge after the defeat of Morghul.

Example 2:
Question: How did Borin Stoneheart get the name 'Lockbearer'?
Answer: He received the name 'Lockbearer' after Lyra Silverwood gave him three strands of her hair, which he intended to preserve in crystal as an heirloom.


Now answer this question:
"""
            prompt = examples + f"\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"

        elif method == "cot":
            prompt = f"""Analyze step by step and then provide the final answer.

Context:
{context}

Question: {question}

Step-by-step reasoning:
1. First, identify what the user is asking about
2. Find relevant information in the context
3. Analyze the found information
4. Formulate a clear and complete answer

Final answer:"""

        else:  # standard
            prompt = f"""Use the provided context to answer the question accurately and helpfully.

Context:
{context}

Question: {question}

Answer:"""

        return prompt

    def _prepare_context(self, documents: List[Dict]) -> str:
        """Prepare context from documents"""
        if not documents:
            return "No relevant information found in the knowledge base."

        context_parts = []
        for i, doc in enumerate(documents, 1):
            context_parts.append(f"[Document {i}] Source: {doc['source']}, Similarity: {doc['similarity']:.3f}\n{doc['content']}")

        return "\n\n".join(context_parts)

    def _error_response(self, message: str) -> Dict[str, Any]:
        """Create error response"""
        return {
            "answer": f"❌ {message}",
            "sources": [],
            "has_answer": False
        }

    async def create_response(self, query: str, method: str = "standard") -> Dict[str, Any]:
        """Create response for user query"""
        logger.info(f"Processing: {query} with {method}")

        if not self.llm:
            return self._error_response("YandexGPT is not configured")

        # Search for documents
        documents = self.retriever.search(query)

        if not documents:
            return self._error_response("No relevant information found in knowledge base")

        # Prepare context and prompt
        context = self._prepare_context(documents)
        prompt = self._build_prompt(context, query, method)

        try:
            # Generate response
            response = await self.llm.ainvoke(prompt)
            answer = response.content

            # Prepare sources
            sources = [
                {
                    "source": doc['source'],
                    "similarity": doc['similarity'],
                    "preview": doc['content'][:100] + "..."
                }
                for doc in documents
            ]

            return {
                "answer": answer,
                "sources": sources,
                "method": method,
                "has_answer": True
            }

        except Exception as e:
            logger.error(f"Generation error: {e}")
            return self._error_response("Error generating response")

# User session management
user_sessions: Dict[int, Dict[str, Any]] = {}

def get_user_session(user_id: int) -> Dict[str, Any]:
    """Get or create user session"""
    if user_id not in user_sessions:
        user_sessions[user_id] = {
            "current_method": "standard",
            "last_response": None
        }
    return user_sessions[user_id]

# Telegram bot handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Start command handler"""
    user = update.effective_user
    session = get_user_session(user.id)

    welcome_text = f"""
🤖 Welcome, {user.first_name}!

I'm a RAG-powered assistant with YandexGPT. I can help you find information using different prompting techniques.

Current method: {session['current_method']}

Use the keyboard below to get started!
"""

    await update.message.reply_text(welcome_text, reply_markup=main_markup)
    return SELECTING_ACTION

async def select_action(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle main menu actions"""
    text = update.message.text
    user_id = update.effective_user.id
    session = get_user_session(user_id)

    if text == "Ask Question":
        await update.message.reply_text("💬 Please enter your question:", reply_markup=ReplyKeyboardRemove())
        return ASKING_QUESTION

    elif text == "Change Method":
        await update.message.reply_text(f"🛠️ Current method: {session['current_method']}\nSelect new method:", reply_markup=method_markup)
        return SELECTING_METHOD

    elif text == "Show Sources":
        if session.get('last_response') and session['last_response'].get('sources'):
            sources = session['last_response']['sources']
            sources_text = "📚 Sources from last response:\n\n"
            for i, source in enumerate(sources, 1):
                sources_text += f"{i}. {source['source']}\n   Similarity: {source['similarity']:.3f}\n\n"
            await update.message.reply_text(sources_text, reply_markup=main_markup)
        else:
            await update.message.reply_text("No sources available. Ask a question first.", reply_markup=main_markup)
        return SELECTING_ACTION

    elif text == "Bot Info":
        info_text = """
🤖 RAG Telegram Bot

Features:
• YandexGPT integration
• FAISS vector search
• Multiple prompting techniques
• Source citation

Use /start to begin!
"""
        await update.message.reply_text(info_text, reply_markup=main_markup)
        return SELECTING_ACTION

    else:
        await update.message.reply_text("Please select from menu:", reply_markup=main_markup)
        return SELECTING_ACTION

async def select_method(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle method selection"""
    text = update.message.text
    user_id = update.effective_user.id
    session = get_user_session(user_id)

    if text in ["standard", "few_shot", "cot"]:
        session['current_method'] = text
        await update.message.reply_text(f"✅ Method set to: {text}", reply_markup=main_markup)
        return SELECTING_ACTION
    elif text == "Back":
        await update.message.reply_text("Back to main menu", reply_markup=main_markup)
        return SELECTING_ACTION
    else:
        await update.message.reply_text("Please select a method:", reply_markup=method_markup)
        return SELECTING_METHOD

async def ask_question(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle user questions"""
    user_question = update.message.text
    user_id = update.effective_user.id
    session = get_user_session(user_id)

    logger.info(f"User {user_id} asked: {user_question}")

    # Send typing action
    await update.message.chat.send_action(action="typing")

    try:
        # Initialize RAG engine and get response
        rag_engine = SimpleRAGEngine()
        response_data = await rag_engine.create_response(user_question, session['current_method'])

        # Store response
        session['last_response'] = response_data

        # Prepare response text
        if response_data['has_answer']:
            response_text = f"💡 **Answer** ({response_data['method']}):\n\n{response_data['answer']}"
            if response_data['sources']:
                response_text += f"\n\n📚 Found {len(response_data['sources'])} relevant sources"
        else:
            response_text = response_data['answer']

        await update.message.reply_text(response_text, reply_markup=main_markup, parse_mode='Markdown')

    except Exception as e:
        logger.error(f"Error processing request: {e}")
        await update.message.reply_text("❌ Error processing request. Try again.", reply_markup=main_markup)

    return SELECTING_ACTION

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Cancel conversation"""
    await update.message.reply_text("Operation cancelled. Use /start to begin.", reply_markup=ReplyKeyboardRemove())
    return ConversationHandler.END

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle errors"""
    logger.error(f"Error: {context.error}")
    if update and update.effective_message:
        await update.effective_message.reply_text("❌ An error occurred.", reply_markup=main_markup)

def main():
    """Start the bot"""
    # Check environment variables
    telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
    yandex_api_key = os.getenv("YANDEX_API_KEY")
    yandex_folder_id = os.getenv("YANDEX_FOLDER_ID")

    if not telegram_token:
        logger.error("TELEGRAM_BOT_TOKEN not set")
        return

    if not yandex_api_key or not yandex_folder_id:
        logger.error("YANDEX_API_KEY and YANDEX_FOLDER_ID must be set")
        return


    # Create application
    application = ApplicationBuilder().token(telegram_token).build()

    # Add conversation handler
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            SELECTING_ACTION: [MessageHandler(filters.TEXT & ~filters.COMMAND, select_action)],
            SELECTING_METHOD: [MessageHandler(filters.TEXT & ~filters.COMMAND, select_method)],
            ASKING_QUESTION: [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_question)],
        },
        fallbacks=[CommandHandler("cancel", cancel)]
    )

    application.add_handler(conv_handler)
    application.add_error_handler(error_handler)

    # Start bot
    logger.info("Starting Telegram bot...")
    application.run_polling()

if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()

    main()
