from dotenv import load_dotenv
import os
import uuid
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Neo4jVector  # For Neo4j ingestion
import supabase
from neo4j import GraphDatabase
load_dotenv()
class IngestionPipeline:
    def __init__(self):
        # Initialize the embeddings service
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Load Supabase credentials from env variables
        supabase_url = os.getenv("SUPABASE_URL")
        service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        anon_key = os.getenv("SUPABASE_ANON_KEY")
        if not all([supabase_url, service_key, anon_key]):
            missing = []
            if not supabase_url: missing.append("SUPABASE_URL")
            if not service_key: missing.append("SUPABASE_SERVICE_ROLE_KEY")
            if not anon_key: missing.append("SUPABASE_ANON_KEY")
            raise ValueError(f"Missing Supabase credentials: {', '.join(missing)}")
        
        # Create Supabase clients for backend and frontend as needed
        self.supabase = supabase.create_client(supabase_url, service_key)
        self.public_supabase = supabase.create_client(supabase_url, anon_key)
        
        # Set table name (the expected table in Supabase)
        self.table_name = os.getenv("SUPABASE_DOCS_TABLE", "magentra_docs")
        print(f"Connected to Supabase at: {supabase_url}")
        print(f"Using table: {self.table_name}")
        
        # Setup Neo4j configuration from env variables
        neo4j_uri = os.getenv("NEO4J_URI")
        if not neo4j_uri:
            neo4j_uri = "bolt://localhost:7687"
        neo4j_user = os.getenv("NEO4J_USERNAME") or "neo4j"
        neo4j_pass = os.getenv("NEO4J_PASSWORD") or "password"
        self.neo4j_config = {
            "uri": neo4j_uri,
            "user": neo4j_user,
            "password": neo4j_pass
        }
        print(f"Neo4j Configuration: URI: {neo4j_uri}, User: {neo4j_user}")
        
        # Setup Neo4j schema/indexes (ensures unique uuids on Chunk nodes)
        self._setup_neo4j_schema()
    
    def _setup_neo4j_schema(self):
        try:
            driver = GraphDatabase.driver(self.neo4j_config["uri"],
                                            auth=(self.neo4j_config["user"], self.neo4j_config["password"]))
            with driver.session() as session:
                # Ensure that each Chunk node has a unique uuid
                session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE c.uuid IS UNIQUE")
                # Optionally, create an index on the source property for efficiency. Uncomment if needed.
                session.run("CREATE INDEX IF NOT EXISTS FOR (c:Chunk) ON (c.source)")
                print("Neo4j schema setup completed")
            driver.close()
        except Exception as e:
            print("Error setting up Neo4j schema:", e)

    def ingest_document(self, file_path: str, database: str = "both") -> dict:
        """High-level ingestion method for Supabase and/or Neo4j."""
        print(f"Ingesting: {os.path.abspath(file_path)}")
        docs = self._load_and_split(file_path)
        results = {}
        if database in ["neo4j", "both"]:
            results["neo4j"] = self._ingest_neo4j(docs)
        if database in ["supabase", "both"]:
            results["supabase"] = self._ingest_supabase(docs)
        return results

    def _load_and_split(self, file_path: str):
        """Load the file and split it into chunks."""
        loader = TextLoader(file_path)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=200,
            separators=["\n\n## ", "\n\n", "\n", ". "],
            add_start_index=True
        )
        docs = splitter.split_documents(documents)
        # Add source metadata to each document
        for doc in docs:
            doc.metadata["source"] = file_path
            if not doc.page_content.strip():
                raise ValueError("Empty text content in document chunk")
        return docs

    def _ingest_supabase(self, docs) -> int:
        """Insert documents into Supabase.
           Each document's embedding is computed and then stored with its
           content and metadata.
        """
        insert_data = []
        expected_dim = 1536  # Expected vector dimension
        for doc in docs:
            doc_id = str(uuid.uuid4())
            # Compute the embedding; embed_documents returns a list of embeddings—
            # so we take the first (and only) result.
            embedding = self.embeddings.embed_documents([doc.page_content])[0]
            # Verify the embedding has the proper dimension
            if len(embedding) != expected_dim:
                raise ValueError(
                    f"Embedding dimension is {len(embedding)}; expected {expected_dim}"
                )
            insert_data.append({
                "id": doc_id,
                "content": doc.page_content,
                "metadata": doc.metadata,
                "embedding": embedding   # Passed as a JSON array
            })
        response = self.supabase.table(self.table_name).insert(insert_data).execute()
        print("Supabase response:", response)
        print(f"Inserted {len(insert_data)} documents into Supabase")
        return {"supabase_inserted": len(insert_data), "supabase_response": response}

    def _ingest_neo4j(self, docs) -> int:
        """Insert documents into Neo4j.
           Adds a UUID into each document's metadata before ingestion.
        """
        # Annotate each document with a UUID
        for doc in docs:
            doc.metadata["uuid"] = str(uuid.uuid4())
        Neo4jVector.from_documents(
            documents=docs,
            embedding=self.embeddings,
            url=self.neo4j_config["uri"],
            username=self.neo4j_config["user"],
            password=self.neo4j_config["password"],
            database="neo4j",      # Use your Neo4j database
            index_name="magentra_knowledge_v1",
            node_label="Chunk",
            text_node_property="text",
            embedding_node_property="embedding",
            pre_delete_collection=False
        )
        inserted = len(docs)
        print(f"Inserted {inserted} documents into Neo4j")
        return {"neo4j_inserted": inserted}

    