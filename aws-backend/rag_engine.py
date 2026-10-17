import uuid
import fitz
import chromadb
from chromadb.utils import embedding_functions

class ChromaRAGEngine:
    def __init__(self):
        self.chroma_client = chromadb.PersistentClient(path="/app/chroma_data")
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu"
        )

    def build_context_from_pdf_and_jd(self, pdf_bytes: bytes, jd_text: str) -> tuple[str, str]:
        session_id = f"sess_{uuid.uuid4().hex[:10]}"
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        resume_text = "\n".join([page.get_text("text") for page in doc])

        def chunk_words(text: str, size: int = 220) -> list[str]:
            words = text.split()
            return [" ".join(words[i:i + size]) for i in range(0, len(words), size // 2)] or [text[:500]]

        r_chunks = chunk_words(resume_text)
        j_chunks = chunk_words(jd_text)
        all_docs = r_chunks + j_chunks

        collection = self.chroma_client.create_collection(
            name=session_id,
            embedding_function=self.embedding_fn
        )
        collection.add(
            documents=all_docs,
            ids=[f"id_{i}" for i in range(len(all_docs))],
            metadatas=[{"type": "resume"} for _ in r_chunks] + [{"type": "jd"} for _ in j_chunks]
        )

        results = collection.query(
            query_texts=["Candidate technical stack, system architecture projects, and core job requirements"],
            n_results=min(8, len(all_docs))
        )
        retrieved_context = "\n---\n".join(results["documents"][0])
        return session_id, retrieved_context
