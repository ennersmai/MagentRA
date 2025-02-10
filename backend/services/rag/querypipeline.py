from typing import List, Tuple
from langchain.schema import Document
from langchain_community.vectorstores import Neo4jVector, SupabaseVectorStore
from langchain_openai import OpenAIEmbeddings
import supabase
import os

class QueryPipeline:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        self.supabase = supabase.create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_ANON_KEY")  # Use anonymous key for queries
        )

    def hybrid_search(self, query: str, k: int = 5, database: str = "neo4j") -> List[Tuple[Document, float]]:
        """Search across configured vector databases"""
        results = []
        
        if database in ["neo4j", "both"]:
            results.extend(self._neo4j_search(query, k))
            
        if database in ["supabase", "both"]:
            results.extend(self._supabase_search(query, k))
            
        return self._deduplicate_and_sort(results)[:k]

    def _neo4j_search(self, query: str, k: int) -> List[Tuple[Document, float]]:
        store = Neo4jVector(
            embedding=self.embeddings,
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD"),
            index_name="magentra_knowledge_v1",
            node_label="Chunk",
            text_node_property="text"
        )
        return store.similarity_search_with_relevance_scores(query, k=k)

    def _supabase_search(self, query: str, k: int) -> List[Tuple[Document, float]]:
        store = SupabaseVectorStore(
            client=self.supabase,
            embedding=self.embeddings,
            table_name="documents",
            query_name="match_documents",
            embedding_column="embedding",
            content_column="content",
            metadata_column="metadata"
        )
        return store.similarity_search_with_relevance_scores(query, k=k)

    def _deduplicate_and_sort(self, results: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
        seen = set()
        deduped = []
        for doc, score in sorted(results, key=lambda x: x[1], reverse=True):
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                deduped.append((doc, score))
        return deduped 

    def query_documents(self, question: str, database: str = "supabase"):
        # Perform the hybrid search query
        results = self.hybrid_search(question)
        if not isinstance(results, list):
            results = [results]
        # Map each result tuple to a dictionary for JSON serialization
        mapped_results = [{
            "content": doc.page_content,
            "score": score,
            "database": doc.metadata.get("database", "unknown"),
            "metadata": doc.metadata
        } for doc, score in results]
        return mapped_results 