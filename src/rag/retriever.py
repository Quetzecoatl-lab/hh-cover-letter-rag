from pathlib import Path
from typing import List
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


class ResumeRAG:
    """Универсальный RAG-поисковик: принимает любое PDF-резюме и ищет опыт под вакансию."""

    def __init__(self, embedding_model: str = "intfloat/multilingual-e5-small"):
        # Загружаем легкую мультиязычную модель эмбеддингов (работает на CPU ноутбука)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},  # Нормализация для точного косинусного сходства
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=550,
            chunk_overlap=80,
            separators=["\n\n", "\n- ", "\n", ". "],
        )
        self.vector_store: FAISS | None = None

    def load_pdf_resume(self, pdf_path: Path) -> int:
        """Читает любой PDF-файл резюме, нарезает на чанки и строит векторный индекс в памяти."""
        reader = PdfReader(str(pdf_path))
        full_text = "\n".join(page.extract_text() or "" for page in reader.pages)

        chunks = self.splitter.split_text(full_text)
        # Для моделей семейства E5 добавляется префикс 'passage: ' перед индексацией текста
        docs = [
            Document(page_content=f"passage: {chunk}", metadata={"source": pdf_path.name, "chunk_id": i})
            for i, chunk in enumerate(chunks)
            if len(chunk.strip()) > 50
        ]

        self.vector_store = FAISS.from_documents(docs, self.embeddings)
        return len(docs)

    def retrieve_relevant_experience(self, vacancy_text: str, top_k: int = 3) -> List[str]:
        """Ищет наиболее релевантные абзацы из резюме под текст конкретной вакансии."""
        if not self.vector_store:
            raise ValueError("Сначала загрузите резюме через load_pdf_resume()!")

        # Для моделей E5 поисковый запрос предваряется префиксом 'query: '
        results = self.vector_store.similarity_search(f"query: {vacancy_text}", k=top_k)

        # Убираем технический префикс 'passage: ' перед подачей в LLM
        clean_chunks = [doc.page_content.replace("passage: ", "", 1).strip() for doc in results]
        return clean_chunks


if __name__ == "__main__":
    # Тестируем работу RAG на пробном PDF-резюме
    sample_pdf = Path("data/resume/test_resume.pdf")
    if not sample_pdf.exists():
        print(f"Положите тестовый PDF-файл резюме по пути: {sample_pdf.resolve()}")
    else:
        rag = ResumeRAG()
        num_chunks = rag.load_pdf_resume(sample_pdf)
        print(f"Резюме успешно проиндексировано! Разбито на {num_chunks} смысловых фрагментов.\n")

        # Проверим на двух разных вакансиях: 1) чисто SQL/Backend и 2) LLM/RAG
        test_vacancies = [
            "Ищем Python Backend разработчика для оптимизации тяжелых SQL-запросов в PostgreSQL и рефакторинга корпоративных систем.",
            "Требуется AI/ML инженер для построения RAG-пайплайнов на LangChain, векторных баз Qdrant и дообучения Qwen методом QLoRA.",
        ]

        for vac in test_vacancies:
            print(f"=== Вакансия: {vac} ===")
            matches = rag.retrieve_relevant_experience(vac, top_k=2)
            for idx, m in enumerate(matches, 1):
                print(f"[{idx}] {m}\n")