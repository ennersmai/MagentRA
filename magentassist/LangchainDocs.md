Retrievers
[Prerequisites]
Vector stores
Embeddings
Text splitters
Overview
Many different types of retrieval systems exist, including vectorstores, graph databases, and relational databases. With the rise on popularity of large language models, retrieval systems have become an important component in AI application (e.g., RAG). Because of their importance and variability, LangChain provides a uniform interface for interacting with different types of retrieval systems. The LangChain retriever interface is straightforward:

Input: A query (string)
Output: A list of documents (standardized LangChain Document objects)
Key concept
Retriever

All retrievers implement a simple interface for retrieving documents using natural language queries.

Interface
The only requirement for a retriever is the ability to accepts a query and return documents. In particular, LangChain's retriever class only requires that the _getRelevantDocuments method is implemented, which takes a query: string and returns a list of Document objects that are most relevant to the query. The underlying logic used to get relevant documents is specified by the retriever and can be whatever is most useful for the application.

A LangChain retriever is a runnable, which is a standard interface is for LangChain components. This means that it has a few common methods, including invoke, that are used to interact with it. A retriever can be invoked with a query:

const docs = await retriever.invoke(query);

Retrievers return a list of Document objects, which have two attributes:

pageContent: The content of this document. Currently is a string.
metadata: Arbitrary metadata associated with this document (e.g., document id, file name, source, etc).
[Further reading]
See our how-to guide on building your own custom retriever.
Common types
Despite the flexibility of the retriever interface, a few common types of retrieval systems are frequently used.

Search apis
It's important to note that retrievers don't need to actually store documents. For example, we can be built retrievers on top of search APIs that simply return search results!

Relational or graph database
Retrievers can be built on top of relational or graph databases. In these cases, query analysis techniques to construct a structured query from natural language is critical. For example, you can build a retriever for a SQL database using text-to-SQL conversion. This allows a natural language query (string) retriever to be transformed into a SQL query behind the scenes.

[Further reading]
See our tutorial for context on how to build a retreiver using a SQL database and text-to-SQL.
See our tutorial for context on how to build a retreiver using a graph database and text-to-Cypher.
Lexical search
As discussed in our conceptual review of retrieval, many search engines are based upon matching words in a query to the words in each document. BM25 and TF-IDF are two popular lexical search algorithms. LangChain has retrievers for many popular lexical search algorithms / engines.

[Further reading]
See the BM25 retriever integration.
Vector store
Vector stores are a powerful and efficient way to index and retrieve unstructured data. An vectorstore can be used as a retriever by calling the asRetriever() method.

const vectorstore = new MyVectorStore();
const retriever = vectorstore.asRetriever();

Advanced retrieval patterns
Ensemble
Because the retriever interface is so simple, returning a list of Document objects given a search query, it is possible to combine multiple retrievers using ensembling. This is particularly useful when you have multiple retrievers that are good at finding different types of relevant documents. It is easy to create an ensemble retriever that combines multiple retrievers with linear weighted scores:

// Initialize the ensemble retriever
const ensembleRetriever = new EnsembleRetriever({
  retrievers: [bm25Retriever, vectorStoreRetriever],
  weights: [0.5, 0.5],
});

When ensembling, how do we combine search results from many retrievers? This motivates the concept of re-ranking, which takes the output of multiple retrievers and combines them using a more sophisticated algorithm such as Reciprocal Rank Fusion (RRF).

Source document retention
Many retrievers utilize some kind of index to make documents easily searchable. The process of indexing can include a transformation step (e.g., vectorstores often use document splitting). Whatever transformation is used, can be very useful to retain a link between the transformed document and the original, giving the retriever the ability to return the original document.

Retrieval with full docs

This is particularly useful in AI applications, because it ensures no loss in document context for the model. For example, you may use small chunk size for indexing documents in a vectorstore. If you return only the chunks as the retrieval result, then the model will have lost the original document context for the chunks.

LangChain has two different retrievers that can be used to address this challenge. The Multi-Vector retriever allows the user to use any document transformation (e.g., use an LLM to write a summary of the document) for indexing while retaining linkage to the source document. The ParentDocument retriever links document chunks from a text-splitter transformation for indexing while retaining linkage to the source document.

Name	Index Type	Uses an LLM	When to Use	Description
ParentDocument	Vector store + Document Store	No	If your pages have lots of smaller pieces of distinct information that are best indexed by themselves, but best retrieved all together.	This involves indexing multiple chunks for each document. Then you find the chunks that are most similar in embedding space, but you retrieve the whole parent document and return that (rather than individual chunks).
Multi Vector	Vector store + Document Store	Sometimes during indexing	If you are able to extract information from documents that you think is more relevant to index than the text itself.	This involves creating multiple vectors for each document. Each vector could be created in a myriad of ways - examples include summaries of the text and hypothetical questions.

Vector stores
[Prerequisites]
Embeddings
Text splitters
[Note]
This conceptual overview focuses on text-based indexing and retrieval for simplicity. However, embedding models can be multi-modal and vector stores can be used to store and retrieve a variety of data types beyond text.

Overview
Vector stores are specialized data stores that enable indexing and retrieving information based on vector representations.

These vectors, called embeddings, capture the semantic meaning of data that has been embedded.

Vector stores are frequently used to search over unstructured data, such as text, images, and audio, to retrieve relevant information based on semantic similarity rather than exact keyword matches.

Vector stores

Integrations
LangChain has a large number of vectorstore integrations, allowing users to easily switch between different vectorstore implementations.

Please see the full list of LangChain vectorstore integrations.

Interface
LangChain provides a standard interface for working with vector stores, allowing users to easily switch between different vectorstore implementations.

The interface consists of basic methods for writing, deleting and searching for documents in the vector store.

The key methods are:

addDocuments: Add a list of texts to the vector store.
deleteDocuments / delete: Delete a list of documents from the vector store.
similaritySearch: Search for similar documents to a given query.
Initialization
Most vectors in LangChain accept an embedding model as an argument when initializing the vector store.

We will use LangChain's MemoryVectorStore implementation to illustrate the API.

import { MemoryVectorStore } from "langchain/vectorstores/memory";
// Initialize with an embedding model
const vectorStore = new MemoryVectorStore(new SomeEmbeddingModel());

Adding documents
To add documents, use the addDocuments method.

This API works with a list of Document objects. Document objects all have pageContent and metadata attributes, making them a universal way to store unstructured text and associated metadata.

import { Document } from "@langchain/core/documents";

const document1 = new Document(
    pageContent: "I had chocalate chip pancakes and scrambled eggs for breakfast this morning.",
    metadata: { source: "tweet" },
)

const document2 = new Document(
    pageContent: "The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees.",
    metadata: { source: "news" },
)

const documents = [document1, document2]

await vectorStore.addDocuments(documents)

You should usually provide IDs for the documents you add to the vector store, so that instead of adding the same document multiple times, you can update the existing document.

await vectorStore.addDocuments(documents, { ids: ["doc1", "doc2"] });

Delete
To delete documents, use the deleteDocuments method which takes a list of document IDs to delete.

await vectorStore.deleteDocuments(["doc1"]);

or the delete method:

await vectorStore.deleteDocuments({ ids: ["doc1"] });

Search
Vector stores embed and store the documents that added. If we pass in a query, the vectorstore will embed the query, perform a similarity search over the embedded documents, and return the most similar ones. This captures two important concepts: first, there needs to be a way to measure the similarity between the query and any embedded document. Second, there needs to be an algorithm to efficiently perform this similarity search across all embedded documents.

Similarity metrics
A critical advantage of embeddings vectors is they can be compared using many simple mathematical operations:

Cosine Similarity: Measures the cosine of the angle between two vectors.
Euclidean Distance: Measures the straight-line distance between two points.
Dot Product: Measures the projection of one vector onto another.
The choice of similarity metric can sometimes be selected when initializing the vectorstore. Please refer to the documentation of the specific vectorstore you are using to see what similarity metrics are supported.

[Further reading]
See this documentation from Google on similarity metrics to consider with embeddings.
See Pinecone's blog post on similarity metrics.
See OpenAI's FAQ on what similarity metric to use with OpenAI embeddings.
Similarity search
Given a similarity metric to measure the distance between the embedded query and any embedded document, we need an algorithm to efficiently search over all the embedded documents to find the most similar ones. There are various ways to do this. As an example, many vectorstores implement HNSW (Hierarchical Navigable Small World), a graph-based index structure that allows for efficient similarity search. Regardless of the search algorithm used under the hood, the LangChain vectorstore interface has a similaritySearch method for all integrations. This will take the search query, create an embedding, find similar documents, and return them as a list of Documents.

const query = "my query";
const docs = await vectorstore.similaritySearch(query);

Many vectorstores support search parameters to be passed with the similaritySearch method. See the documentation for the specific vectorstore you are using to see what parameters are supported. As an example Pinecone several parameters that are important general concepts: Many vectorstores support the k, which controls the number of Documents to return, and filter, which allows for filtering documents by metadata.

query (string) – Text to look up documents similar to.
k (number) – Number of Documents to return. Defaults to 4.
filter (Record<string, any> | undefined) – Object of argument(s) to filter on metadata
[Further reading]
See the how-to guide for more details on how to use the similaritySearch method.
See the integrations page for more details on arguments that can be passed in to the similaritySearch method for specific vectorstores.
Metadata filtering
While vectorstore implement a search algorithm to efficiently search over all the embedded documents to find the most similar ones, many also support filtering on metadata. This allows structured filters to reduce the size of the similarity search space. These two concepts work well together:

Semantic search: Query the unstructured data directly, often using via embedding or keyword similarity.
Metadata search: Apply structured query to the metadata, filtering specific documents.
Vector store support for metadata filtering is typically dependent on the underlying vector store implementation.

Here is example usage with Pinecone, showing that we filter for all documents that have the metadata key source with value tweet.

await vectorstore.similaritySearch(
  "LangChain provides abstractions to make working with LLMs easy",
  2,
  {
    // The arguments of this field are provider specific.
    filter: { source: "tweet" },
  }
);

[Further reading]
See Pinecone's documentation on filtering with metadata.
See the list of LangChain vectorstore integrations that support metadata filtering.
Advanced search and retrieval techniques
While algorithms like HNSW provide the foundation for efficient similarity search in many cases, additional techniques can be employed to improve search quality and diversity. For example, maximal marginal relevance is a re-ranking algorithm used to diversify search results, which is applied after the initial similarity search to ensure a more diverse set of results.

Name	When to use	Description
Maximal Marginal Relevance (MMR)	When needing to diversify search results.	MMR attempts to diversify the results of a search to avoid returning similar and redundant documents.

Embedding models
[Prerequisites]
Documents
[Note]
This conceptual overview focuses on text-based embedding models.

Embedding models can also be multimodal though such models are not currently supported by LangChain.

Imagine being able to capture the essence of any text - a tweet, document, or book - in a single, compact representation. This is the power of embedding models, which lie at the heart of many retrieval systems. Embedding models transform human language into a format that machines can understand and compare with speed and accuracy. These models take text as input and produce a fixed-length array of numbers, a numerical fingerprint of the text's semantic meaning. Embeddings allow search system to find relevant documents not just based on keyword matches, but on semantic understanding.

Key concepts
Conceptual Overview

(1) Embed text as a vector: Embeddings transform text into a numerical vector representation.

(2) Measure similarity: Embedding vectors can be comparing using simple mathematical operations.

Embedding
Historical context
The landscape of embedding models has evolved significantly over the years. A pivotal moment came in 2018 when Google introduced BERT (Bidirectional Encoder Representations from Transformers). BERT applied transformer models to embed text as a simple vector representation, which lead to unprecedented performance across various NLP tasks. However, BERT wasn't optimized for generating sentence embeddings efficiently. This limitation spurred the creation of SBERT (Sentence-BERT), which adapted the BERT architecture to generate semantically rich sentence embeddings, easily comparable via similarity metrics like cosine similarity, dramatically reduced the computational overhead for tasks like finding similar sentences. Today, the embedding model ecosystem is diverse, with numerous providers offering their own implementations. To navigate this variety, researchers and practitioners often turn to benchmarks like the Massive Text Embedding Benchmark (MTEB) here for objective comparisons.

[Further reading]
See the seminal BERT paper.
See Cameron Wolfe's excellent review of embedding models.
See the Massive Text Embedding Benchmark (MTEB) leaderboard for a comprehensive overview of embedding models.
Interface
LangChain provides a universal interface for working with them, providing standard methods for common operations. This common interface simplifies interaction with various embedding providers through two central methods:

embedDocuments: For embedding multiple texts (documents)
embedQuery: For embedding a single text (query)
This distinction is important, as some providers employ different embedding strategies for documents (which are to be searched) versus queries (the search input itself). To illustrate, here's a practical example using LangChain's .embedDocuments method to embed a list of strings:

import { OpenAIEmbeddings } from "@langchain/openai";
const embeddingsModel = new OpenAIEmbeddings();
const embeddings = await embeddingsModel.embedDocuments([
  "Hi there!",
  "Oh, hello!",
  "What's your name?",
  "My friends call me World",
  "Hello World!",
]);

console.log(`(${embeddings.length}, ${embeddings[0].length})`);
// (5, 1536)

For convenience, you can also use the embedQuery method to embed a single text:

const queryEmbedding = await embeddingsModel.embedQuery(
  "What is the meaning of life?"
);

[Further reading]
See the full list of LangChain embedding model integrations.
See these how-to guides for working with embedding models.
Integrations
LangChain offers many embedding model integrations which you can find on the embedding models integrations page.

Measure similarity
Each embedding is essentially a set of coordinates, often in a high-dimensional space. In this space, the position of each point (embedding) reflects the meaning of its corresponding text. Just as similar words might be close to each other in a thesaurus, similar concepts end up close to each other in this embedding space. This allows for intuitive comparisons between different pieces of text. By reducing text to these numerical representations, we can use simple mathematical operations to quickly measure how alike two pieces of text are, regardless of their original length or structure. Some common similarity metrics include:

Cosine Similarity: Measures the cosine of the angle between two vectors.
Euclidean Distance: Measures the straight-line distance between two points.
Dot Product: Measures the projection of one vector onto another.
The choice of similarity metric should be chosen based on the model. As an example, OpenAI suggests cosine similarity for their embeddings, which can be easily implemented:

function cosineSimilarity(vec1: number[], vec2: number[]): number {
  const dotProduct = vec1.reduce((sum, val, i) => sum + val * vec2[i], 0);
  const norm1 = Math.sqrt(vec1.reduce((sum, val) => sum + val * val, 0));
  const norm2 = Math.sqrt(vec2.reduce((sum, val) => sum + val * val, 0));
  return dotProduct / (norm1 * norm2);
}

const similarity = cosineSimilarity(queryResult, documentResult);
console.log("Cosine Similarity:", similarity);

[Further reading]
See Simon Willison's nice blog post and video on embeddings and similarity metrics.
See this documentation from Google on similarity metrics to consider with embeddings.
See Pinecone's blog post on similarity metrics.
See OpenAI's FAQ on what similarity metric to use with OpenAI embeddings.

Text splitters
[Prerequisites]
Documents
Tokenization
Overview
Document splitting is often a crucial preprocessing step for many applications. It involves breaking down large texts into smaller, manageable chunks. This process offers several benefits, such as ensuring consistent processing of varying document lengths, overcoming input size limitations of models, and improving the quality of text representations used in retrieval systems. There are several strategies for splitting documents, each with its own advantages.

Key concepts
Conceptual Overview

Text splitters split documents into smaller chunks for use in downstream applications.

Why split documents?
There are several reasons to split documents:

Handling non-uniform document lengths: Real-world document collections often contain texts of varying sizes. Splitting ensures consistent processing across all documents.
Overcoming model limitations: Many embedding models and language models have maximum input size constraints. Splitting allows us to process documents that would otherwise exceed these limits.
Improving representation quality: For longer documents, the quality of embeddings or other representations may degrade as they try to capture too much information. Splitting can lead to more focused and accurate representations of each section.
Enhancing retrieval precision: In information retrieval systems, splitting can improve the granularity of search results, allowing for more precise matching of queries to relevant document sections.
Optimizing computational resources: Working with smaller chunks of text can be more memory-efficient and allow for better parallelization of processing tasks.
Now, the next question is how to split the documents into chunks! There are several strategies, each with its own advantages.

[Further reading]
See Greg Kamradt's chunkviz to visualize different splitting strategies discussed below.
Approaches
Length-based
The most intuitive strategy is to split documents based on their length. This simple yet effective approach ensures that each chunk doesn't exceed a specified size limit. Key benefits of length-based splitting:

Straightforward implementation
Consistent chunk sizes
Easily adaptable to different model requirements
Types of length-based splitting:

Token-based: Splits text based on the number of tokens, which is useful when working with language models.
Character-based: Splits text based on the number of characters, which can be more consistent across different types of text.
Example implementation using LangChain's CharacterTextSplitter with character based splitting:

import { CharacterTextSplitter } from "@langchain/textsplitters";
const textSplitter = new CharacterTextSplitter({
  chunkSize: 100,
  chunkOverlap: 0,
});
const texts = await textSplitter.splitText(document);

[Further reading]
See the how-to guide for token-based splitting.
See the how-to guide for character-based splitting.
Text-structured based
Text is naturally organized into hierarchical units such as paragraphs, sentences, and words. We can leverage this inherent structure to inform our splitting strategy, creating split that maintain natural language flow, maintain semantic coherence within split, and adapts to varying levels of text granularity. LangChain's RecursiveCharacterTextSplitter implements this concept:

The RecursiveCharacterTextSplitter attempts to keep larger units (e.g., paragraphs) intact.
If a unit exceeds the chunk size, it moves to the next level (e.g., sentences).
This process continues down to the word level if necessary.
Here is example usage:

import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";

const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 100,
  chunkOverlap: 0,
});
const texts = await textSplitter.splitText(document);

[Further reading]
See the how-to guide for recursive text splitting.
Document-structured based
Some documents have an inherent structure, such as HTML, Markdown, or JSON files. In these cases, it's beneficial to split the document based on its structure, as it often naturally groups semantically related text. Key benefits of structure-based splitting:

Preserves the logical organization of the document
Maintains context within each chunk
Can be more effective for downstream tasks like retrieval or summarization
Examples of structure-based splitting:

Markdown: Split based on headers (e.g., #, ##, ###)
HTML: Split using tags
JSON: Split by object or array elements
Code: Split by functions, classes, or logical blocks
[Further reading]
See the how-to guide for Code splitting.
Semantic meaning based
Unlike the previous methods, semantic-based splitting actually considers the content of the text. While other approaches use document or text structure as proxies for semantic meaning, this method directly analyzes the text's semantics. There are several ways to implement this, but conceptually the approach is split text when there are significant changes in text meaning. As an example, we can use a sliding window approach to generate embeddings, and compare the embeddings to find significant differences:

Start with the first few sentences and generate an embedding.
Move to the next group of sentences and generate another embedding (e.g., using a sliding window approach).
Compare the embeddings to find significant differences, which indicate potential "break points" between semantic sections.
This technique helps create chunks that are more semantically coherent, potentially improving the quality of downstream tasks like retrieval or summarization.