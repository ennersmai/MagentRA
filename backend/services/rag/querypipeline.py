from typing import List, Tuple
from langchain.schema import Document
from langchain_community.vectorstores import Neo4jVector, SupabaseVectorStore
from langchain_openai import OpenAIEmbeddings
import supabase
import os
from langchain.retrievers import EnsembleRetriever
import asyncio

class QueryPipeline:
    def __init__(self, embeddings: OpenAIEmbeddings):
        self.supabase = None  # Initialize attribute
        self.embeddings = embeddings
        self.retriever = self._configure_retriever()
        
    def _configure_retriever(self):
        # Initialize vector stores
        neo4j_store = Neo4jVector(
            embedding=self.embeddings,
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD"),
            index_name="magentra_knowledge_v1",
            node_label="Chunk",
            text_node_property="text"
        )
        
        supabase_store = SupabaseVectorStore(
            client=supabase.create_client(
                str(os.getenv("SUPABASE_URL", "")),
                str(os.getenv("SUPABASE_ANON_KEY", ""))
            ),
            embedding=self.embeddings,
            table_name="documents",
            query_name="match_documents_embeddings"
        )
        
        # Create retrievers with proper search parameters
        neo4j_retriever = neo4j_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        supabase_retriever = supabase_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        # Create ensemble retriever
        return EnsembleRetriever(
            retrievers=[neo4j_retriever, supabase_retriever],
            weights=[0.4, 0.6]
        )

    async def hybrid_search(self, query: str):
        """Official LangChain hybrid search implementation"""
        return await self.retriever.ainvoke(query)

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
            table_name="documents"
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
        results = asyncio.run(self.hybrid_search(question))
        # Map each result tuple to a dictionary for JSON serialization.
        mapped_results = []
        for doc, score in results:
            if isinstance(doc, Document):
                content = doc.page_content
                metadata = doc.metadata
            else:
                content = str(doc)
                metadata = {}
            mapped_results.append({
                "content": content,
                "score": score,
                "database": metadata.get("database", "unknown"),
                "metadata": metadata
            })
        return mapped_results 