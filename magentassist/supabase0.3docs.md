SupabaseVectorStore
class langchain_community.vectorstores.supabase.SupabaseVectorStore(client: supabase.client.Client, embedding: Embeddings, table_name: str, chunk_size: int = 500, query_name: str | None = None)[source]
Supabase Postgres vector store.

It assumes you have the pgvector extension installed and a match_documents (or similar) function. For more details: https://integrations.langchain.com/vectorstores?integration_name=SupabaseVectorStore

You can implement your own match_documents function in order to limit the search space to a subset of documents based on your own authorization or business logic.

Note that the Supabase Python client does not yet support async operations.

If you’d like to use max_marginal_relevance_search, please review the instructions below on modifying the match_documents function to return matched embeddings.

Examples:

from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import SupabaseVectorStore
from supabase.client import create_client

docs = [
    Document(page_content="foo", metadata={"id": 1}),
]
embeddings = OpenAIEmbeddings()
supabase_client = create_client("my_supabase_url", "my_supabase_key")
vector_store = SupabaseVectorStore.from_documents(
    docs,
    embeddings,
    client=supabase_client,
    table_name="documents",
    query_name="match_documents",
    chunk_size=500,
)
To load from an existing table:

from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from supabase.client import create_client


embeddings = OpenAIEmbeddings()
supabase_client = create_client("my_supabase_url", "my_supabase_key")
vector_store = SupabaseVectorStore(
    client=supabase_client,
    embedding=embeddings,
    table_name="documents",
    query_name="match_documents",
)
Initialize with supabase client.

Attributes

embeddings

Access the query embedding object if available.

Methods

__init__(client, embedding, table_name[, ...])

Initialize with supabase client.

aadd_documents(documents, **kwargs)

Async run more documents through the embeddings and add to the vectorstore.

aadd_texts(texts[, metadatas, ids])

Async run more texts through the embeddings and add to the vectorstore.

add_documents(documents, **kwargs)

Add or update documents in the vectorstore.

add_texts(texts[, metadatas, ids])

Run more texts through the embeddings and add to the vectorstore.

add_vectors(vectors, documents, ids)

adelete([ids])

Async delete by vector ID or other criteria.

afrom_documents(documents, embedding, **kwargs)

Async return VectorStore initialized from documents and embeddings.

afrom_texts(texts, embedding[, metadatas, ids])

Async return VectorStore initialized from texts and embeddings.

aget_by_ids(ids, /)

Async get documents by their IDs.

amax_marginal_relevance_search(query[, k, ...])

Async return docs selected using the maximal marginal relevance.

amax_marginal_relevance_search_by_vector(...)

Async return docs selected using the maximal marginal relevance.

as_retriever(**kwargs)

Return VectorStoreRetriever initialized from this VectorStore.

asearch(query, search_type, **kwargs)

Async return docs most similar to query using a specified search type.

asimilarity_search(query[, k])

Async return docs most similar to query.

asimilarity_search_by_vector(embedding[, k])

Async return docs most similar to embedding vector.

asimilarity_search_with_relevance_scores(query)

Async return docs and relevance scores in the range [0, 1].

asimilarity_search_with_score(*args, **kwargs)

Async run similarity search with distance.

delete([ids])

Delete by vector IDs.

from_documents(documents, embedding, **kwargs)

Return VectorStore initialized from documents and embeddings.

from_texts(texts, embedding[, metadatas, ...])

Return VectorStore initialized from texts and embeddings.

get_by_ids(ids, /)

Get documents by their IDs.

match_args(query, filter)

max_marginal_relevance_search(query[, k, ...])

Return docs selected using the maximal marginal relevance.

max_marginal_relevance_search_by_vector(...)

Return docs selected using the maximal marginal relevance.

search(query, search_type, **kwargs)

Return docs most similar to query using a specified search type.

similarity_search(query[, k, filter])

Return docs most similar to query.

similarity_search_by_vector(embedding[, k, ...])

Return docs most similar to embedding vector.

similarity_search_by_vector_returning_embeddings(...)

similarity_search_by_vector_with_relevance_scores(...)

similarity_search_with_relevance_scores(query)

Return docs and relevance scores in the range [0, 1].

similarity_search_with_score(*args, **kwargs)

Run similarity search with distance.

Parameters
:
client (supabase.client.Client)

embedding (Embeddings)

table_name (str)

chunk_size (int)

query_name (Union[str, None])

__init__(client: supabase.client.Client, embedding: Embeddings, table_name: str, chunk_size: int = 500, query_name: str | None = None) → None[source]
Initialize with supabase client.

Parameters
:
client (supabase.client.Client)

embedding (Embeddings)

table_name (str)

chunk_size (int)

query_name (Union[str, None])

Return type
:
None

async aadd_documents(documents: list[Document], **kwargs: Any) → list[str]
Async run more documents through the embeddings and add to the vectorstore.

Parameters
:
documents (list[Document]) – Documents to add to the vectorstore.

kwargs (Any) – Additional keyword arguments.

Returns
:
List of IDs of the added texts.

Raises
:
ValueError – If the number of IDs does not match the number of documents.

Return type
:
list[str]

async aadd_texts(texts: Iterable[str], metadatas: list[dict] | None = None, *, ids: list[str] | None = None, **kwargs: Any) → list[str]
Async run more texts through the embeddings and add to the vectorstore.

Parameters
:
texts (Iterable[str]) – Iterable of strings to add to the vectorstore.

metadatas (list[dict] | None) – Optional list of metadatas associated with the texts. Default is None.

ids (list[str] | None) – Optional list

**kwargs (Any) – vectorstore specific parameters.

Returns
:
List of ids from adding the texts into the vectorstore.

Raises
:
ValueError – If the number of metadatas does not match the number of texts.

ValueError – If the number of ids does not match the number of texts.

Return type
:
list[str]

add_documents(documents: list[Document], **kwargs: Any) → list[str]
Add or update documents in the vectorstore.

Parameters
:
documents (list[Document]) – Documents to add to the vectorstore.

kwargs (Any) – Additional keyword arguments. if kwargs contains ids and documents contain ids, the ids in the kwargs will receive precedence.

Returns
:
List of IDs of the added texts.

Raises
:
ValueError – If the number of ids does not match the number of documents.

Return type
:
list[str]

add_texts(texts: Iterable[str], metadatas: List[Dict[Any, Any]] | None = None, ids: List[str] | None = None, **kwargs: Any) → List[str][source]
Run more texts through the embeddings and add to the vectorstore.

Parameters
:
texts (Iterable[str]) – Iterable of strings to add to the vectorstore.

metadatas (List[Dict[Any, Any]] | None) – Optional list of metadatas associated with the texts.

ids (List[str] | None) – Optional list of IDs associated with the texts.

**kwargs (Any) – vectorstore specific parameters. One of the kwargs should be ids which is a list of ids associated with the texts.

Returns
:
List of ids from adding the texts into the vectorstore.

Raises
:
ValueError – If the number of metadatas does not match the number of texts.

ValueError – If the number of ids does not match the number of texts.

Return type
:
List[str]

add_vectors(vectors: List[List[float]], documents: List[Document], ids: List[str]) → List[str][source]
Parameters
:
vectors (List[List[float]])

documents (List[Document])

ids (List[str])

Return type
:
List[str]

async adelete(ids: list[str] | None = None, **kwargs: Any) → bool | None
Async delete by vector ID or other criteria.

Parameters
:
ids (list[str] | None) – List of ids to delete. If None, delete all. Default is None.

**kwargs (Any) – Other keyword arguments that subclasses might use.

Returns
:
True if deletion is successful, False otherwise, None if not implemented.

Return type
:
Optional[bool]

async classmethod afrom_documents(documents: list[Document], embedding: Embeddings, **kwargs: Any) → VST
Async return VectorStore initialized from documents and embeddings.

Parameters
:
documents (list[Document]) – List of Documents to add to the vectorstore.

embedding (Embeddings) – Embedding function to use.

kwargs (Any) – Additional keyword arguments.

Returns
:
VectorStore initialized from documents and embeddings.

Return type
:
VectorStore

async classmethod afrom_texts(texts: list[str], embedding: Embeddings, metadatas: list[dict] | None = None, *, ids: list[str] | None = None, **kwargs: Any) → VST
Async return VectorStore initialized from texts and embeddings.

Parameters
:
texts (list[str]) – Texts to add to the vectorstore.

embedding (Embeddings) – Embedding function to use.

metadatas (list[dict] | None) – Optional list of metadatas associated with the texts. Default is None.

ids (list[str] | None) – Optional list of IDs associated with the texts.

kwargs (Any) – Additional keyword arguments.

Returns
:
VectorStore initialized from texts and embeddings.

Return type
:
VectorStore

async aget_by_ids(ids: Sequence[str], /) → list[Document]
Async get documents by their IDs.

The returned documents are expected to have the ID field set to the ID of the document in the vector store.

Fewer documents may be returned than requested if some IDs are not found or if there are duplicated IDs.

Users should not assume that the order of the returned documents matches the order of the input IDs. Instead, users should rely on the ID field of the returned documents.

This method should NOT raise exceptions if no documents are found for some IDs.

Parameters
:
ids (Sequence[str]) – List of ids to retrieve.

Returns
:
List of Documents.

Return type
:
list[Document]

Added in version 0.2.11.

async amax_marginal_relevance_search(query: str, k: int = 4, fetch_k: int = 20, lambda_mult: float = 0.5, **kwargs: Any) → list[Document]
Async return docs selected using the maximal marginal relevance.

Maximal marginal relevance optimizes for similarity to query AND diversity among selected documents.

Parameters
:
query (str) – Text to look up documents similar to.

k (int) – Number of Documents to return. Defaults to 4.

fetch_k (int) – Number of Documents to fetch to pass to MMR algorithm. Default is 20.

lambda_mult (float) – Number between 0 and 1 that determines the degree of diversity among the results with 0 corresponding to maximum diversity and 1 to minimum diversity. Defaults to 0.5.

kwargs (Any)

Returns
:
List of Documents selected by maximal marginal relevance.

Return type
:
list[Document]

async amax_marginal_relevance_search_by_vector(embedding: list[float], k: int = 4, fetch_k: int = 20, lambda_mult: float = 0.5, **kwargs: Any) → list[Document]
Async return docs selected using the maximal marginal relevance.

Maximal marginal relevance optimizes for similarity to query AND diversity among selected documents.

Parameters
:
embedding (list[float]) – Embedding to look up documents similar to.

k (int) – Number of Documents to return. Defaults to 4.

fetch_k (int) – Number of Documents to fetch to pass to MMR algorithm. Default is 20.

lambda_mult (float) – Number between 0 and 1 that determines the degree of diversity among the results with 0 corresponding to maximum diversity and 1 to minimum diversity. Defaults to 0.5.

**kwargs (Any) – Arguments to pass to the search method.

Returns
:
List of Documents selected by maximal marginal relevance.

Return type
:
list[Document]

as_retriever(**kwargs: Any) → VectorStoreRetriever
Return VectorStoreRetriever initialized from this VectorStore.

Parameters
:
**kwargs (Any) –

Keyword arguments to pass to the search function. Can include: search_type (Optional[str]): Defines the type of search that

the Retriever should perform. Can be “similarity” (default), “mmr”, or “similarity_score_threshold”.

search_kwargs (Optional[Dict]): Keyword arguments to pass to the
search function. Can include things like:
k: Amount of documents to return (Default: 4) score_threshold: Minimum relevance threshold

for similarity_score_threshold

fetch_k: Amount of documents to pass to MMR algorithm
(Default: 20)

lambda_mult: Diversity of results returned by MMR;
1 for minimum diversity and 0 for maximum. (Default: 0.5)

filter: Filter by document metadata

Returns
:
Retriever class for VectorStore.

Return type
:
VectorStoreRetriever

Examples:

# Retrieve more documents with higher diversity
# Useful if your dataset has many similar documents
docsearch.as_retriever(
    search_type="mmr",
    search_kwargs={'k': 6, 'lambda_mult': 0.25}
)

# Fetch more documents for the MMR algorithm to consider
# But only return the top 5
docsearch.as_retriever(
    search_type="mmr",
    search_kwargs={'k': 5, 'fetch_k': 50}
)

# Only retrieve documents that have a relevance score
# Above a certain threshold
docsearch.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={'score_threshold': 0.8}
)

# Only get the single most similar document from the dataset
docsearch.as_retriever(search_kwargs={'k': 1})

# Use a filter to only retrieve documents from a specific paper
docsearch.as_retriever(
    search_kwargs={'filter': {'paper_title':'GPT-4 Technical Report'}}
)
async asearch(query: str, search_type: str, **kwargs: Any) → list[Document]
Async return docs most similar to query using a specified search type.

Parameters
:
query (str) – Input text.

search_type (str) – Type of search to perform. Can be “similarity”, “mmr”, or “similarity_score_threshold”.

**kwargs (Any) – Arguments to pass to the search method.

Returns
:
List of Documents most similar to the query.

Raises
:
ValueError – If search_type is not one of “similarity”, “mmr”, or “similarity_score_threshold”.

Return type
:
list[Document]

async asimilarity_search(query: str, k: int = 4, **kwargs: Any) → list[Document]
Async return docs most similar to query.

Parameters
:
query (str) – Input text.

k (int) – Number of Documents to return. Defaults to 4.

**kwargs (Any) – Arguments to pass to the search method.

Returns
:
List of Documents most similar to the query.

Return type
:
list[Document]

async asimilarity_search_by_vector(embedding: list[float], k: int = 4, **kwargs: Any) → list[Document]
Async return docs most similar to embedding vector.

Parameters
:
embedding (list[float]) – Embedding to look up documents similar to.

k (int) – Number of Documents to return. Defaults to 4.

**kwargs (Any) – Arguments to pass to the search method.

Returns
:
List of Documents most similar to the query vector.

Return type
:
list[Document]

async asimilarity_search_with_relevance_scores(query: str, k: int = 4, **kwargs: Any) → list[tuple[Document, float]]
Async return docs and relevance scores in the range [0, 1].

0 is dissimilar, 1 is most similar.

Parameters
:
query (str) – Input text.

k (int) – Number of Documents to return. Defaults to 4.

**kwargs (Any) –

kwargs to be passed to similarity search. Should include: score_threshold: Optional, a floating point value between 0 to 1 to

filter the resulting set of retrieved docs

Returns
:
List of Tuples of (doc, similarity_score)

Return type
:
list[tuple[Document, float]]

async asimilarity_search_with_score(*args: Any, **kwargs: Any) → list[tuple[Document, float]]
Async run similarity search with distance.

Parameters
:
*args (Any) – Arguments to pass to the search method.

**kwargs (Any) – Arguments to pass to the search method.

Returns
:
List of Tuples of (doc, similarity_score).

Return type
:
list[tuple[Document, float]]

delete(ids: List[str] | None = None, **kwargs: Any) → None[source]
Delete by vector IDs.

Parameters
:
ids (List[str] | None) – List of ids to delete.

kwargs (Any)

Return type
:
None

classmethod from_documents(documents: list[Document], embedding: Embeddings, **kwargs: Any) → VST
Return VectorStore initialized from documents and embeddings.

Parameters
:
documents (list[Document]) – List of Documents to add to the vectorstore.

embedding (Embeddings) – Embedding function to use.

kwargs (Any) – Additional keyword arguments.

Returns
:
VectorStore initialized from documents and embeddings.

Return type
:
VectorStore

classmethod from_texts(texts: List[str], embedding: Embeddings, metadatas: List[dict] | None = None, client: supabase.client.Client | None = None, table_name: str | None = 'documents', query_name: str | None = 'match_documents', chunk_size: int = 500, ids: List[str] | None = None, **kwargs: Any) → SupabaseVectorStore[source]
Return VectorStore initialized from texts and embeddings.

Parameters
:
texts (List[str])

embedding (Embeddings)

metadatas (Optional[List[dict]])

client (Optional[supabase.client.Client])

table_name (Optional[str])

query_name (Union[str, None])

chunk_size (int)

ids (Optional[List[str]])

kwargs (Any)

Return type
:
SupabaseVectorStore

get_by_ids(ids: Sequence[str], /) → list[Document]
Get documents by their IDs.

The returned documents are expected to have the ID field set to the ID of the document in the vector store.

Fewer documents may be returned than requested if some IDs are not found or if there are duplicated IDs.

Users should not assume that the order of the returned documents matches the order of the input IDs. Instead, users should rely on the ID field of the returned documents.

This method should NOT raise exceptions if no documents are found for some IDs.

Parameters
:
ids (Sequence[str]) – List of ids to retrieve.

Returns
:
List of Documents.

Return type
:
list[Document]

Added in version 0.2.11.

match_args(query: List[float], filter: Dict[str, Any] | None) → Dict[str, Any][source]
Parameters
:
query (List[float])

filter (Dict[str, Any] | None)

Return type
:
Dict[str, Any]

max_marginal_relevance_search(query: str, k: int = 4, fetch_k: int = 20, lambda_mult: float = 0.5, **kwargs: Any) → List[Document][source]
Return docs selected using the maximal marginal relevance.

Maximal marginal relevance optimizes for similarity to query AND diversity among selected documents.

Parameters
:
query (str) – Text to look up documents similar to.

k (int) – Number of Documents to return. Defaults to 4.

fetch_k (int) – Number of Documents to fetch to pass to MMR algorithm.

lambda_mult (float) – Number between 0 and 1 that determines the degree of diversity among the results with 0 corresponding to maximum diversity and 1 to minimum diversity. Defaults to 0.5.

kwargs (Any)

Returns
:
List of Documents selected by maximal marginal relevance.

Return type
:
List[Document]

max_marginal_relevance_search requires that query_name returns matched embeddings alongside the match documents. The following function demonstrates how to do this:

```sql CREATE FUNCTION match_documents_embeddings(query_embedding vector(1536),

match_count int)

RETURNS TABLE(
id uuid, content text, metadata jsonb, embedding vector(1536), similarity float)

LANGUAGE plpgsql AS $$ # variable_conflict use_column

BEGIN
RETURN query SELECT

id, content, metadata, embedding, 1 -(docstore.embedding <=> query_embedding) AS similarity

FROM
docstore

ORDER BY
docstore.embedding <=> query_embedding

LIMIT match_count;

END; $$; ```

max_marginal_relevance_search_by_vector(embedding: List[float], k: int = 4, fetch_k: int = 20, lambda_mult: float = 0.5, **kwargs: Any) → List[Document][source]
Return docs selected using the maximal marginal relevance.

Maximal marginal relevance optimizes for similarity to query AND diversity among selected documents.

Parameters
:
embedding (List[float]) – Embedding to look up documents similar to.

k (int) – Number of Documents to return. Defaults to 4.

fetch_k (int) – Number of Documents to fetch to pass to MMR algorithm.

lambda_mult (float) – Number between 0 and 1 that determines the degree of diversity among the results with 0 corresponding to maximum diversity and 1 to minimum diversity. Defaults to 0.5.

kwargs (Any)

Returns
:
List of Documents selected by maximal marginal relevance.

Return type
:
List[Document]

search(query: str, search_type: str, **kwargs: Any) → list[Document]
Return docs most similar to query using a specified search type.

Parameters
:
query (str) – Input text

search_type (str) – Type of search to perform. Can be “similarity”, “mmr”, or “similarity_score_threshold”.

**kwargs (Any) – Arguments to pass to the search method.

Returns
:
List of Documents most similar to the query.

Raises
:
ValueError – If search_type is not one of “similarity”, “mmr”, or “similarity_score_threshold”.

Return type
:
list[Document]

similarity_search(query: str, k: int = 4, filter: Dict[str, Any] | None = None, **kwargs: Any) → List[Document][source]
Return docs most similar to query.

Parameters
:
query (str) – Input text.

k (int) – Number of Documents to return. Defaults to 4.

**kwargs (Any) – Arguments to pass to the search method.

filter (Dict[str, Any] | None)

**kwargs

Returns
:
List of Documents most similar to the query.

Return type
:
List[Document]

similarity_search_by_vector(embedding: List[float], k: int = 4, filter: Dict[str, Any] | None = None, **kwargs: Any) → List[Document][source]
Return docs most similar to embedding vector.

Parameters
:
embedding (List[float]) – Embedding to look up documents similar to.

k (int) – Number of Documents to return. Defaults to 4.

**kwargs (Any) – Arguments to pass to the search method.

filter (Dict[str, Any] | None)

**kwargs

Returns
:
List of Documents most similar to the query vector.

Return type
:
List[Document]

similarity_search_by_vector_returning_embeddings(query: List[float], k: int, filter: Dict[str, Any] | None = None, postgrest_filter: str | None = None) → List[Tuple[Document, float, ndarray]][source]
Parameters
:
query (List[float])

k (int)

filter (Dict[str, Any] | None)

postgrest_filter (str | None)

Return type
:
List[Tuple[Document, float, ndarray]]

similarity_search_by_vector_with_relevance_scores(query: List[float], k: int, filter: Dict[str, Any] | None = None, postgrest_filter: str | None = None, score_threshold: float | None = None) → List[Tuple[Document, float]][source]
Parameters
:
query (List[float])

k (int)

filter (Dict[str, Any] | None)

postgrest_filter (str | None)

score_threshold (float | None)

Return type
:
List[Tuple[Document, float]]

similarity_search_with_relevance_scores(query: str, k: int = 4, filter: Dict[str, Any] | None = None, **kwargs: Any) → List[Tuple[Document, float]][source]
Return docs and relevance scores in the range [0, 1].

0 is dissimilar, 1 is most similar.

Parameters
:
query (str) – Input text.

k (int) – Number of Documents to return. Defaults to 4.

**kwargs (Any) –

kwargs to be passed to similarity search. Should include: score_threshold: Optional, a floating point value between 0 to 1 to

filter the resulting set of retrieved docs.

filter (Dict[str, Any] | None)

**kwargs

Returns
:
List of Tuples of (doc, similarity_score).

Return type
:
List[Tuple[Document, float]]

similarity_search_with_score(*args: Any, **kwargs: Any) → list[tuple[Document, float]]
Run similarity search with distance.

Parameters
:
*args (Any) – Arguments to pass to the search method.

**kwargs (Any) – Arguments to pass to the search method.

Returns
:
List of Tuples of (doc, similarity_score).

Return type
:
list[tuple[Document, float]]