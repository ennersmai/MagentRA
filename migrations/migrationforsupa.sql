CREATE OR REPLACE FUNCTION public.match_documents_embeddings(
  query_embedding vector(1536),
  match_count int DEFAULT 5
)
RETURNS TABLE(
  id uuid,
  content text,
  metadata jsonb,
  embedding vector(1536),
  similarity float8
) AS $$
BEGIN
  RETURN QUERY
  SELECT
    d.id,
    d.content,
    d.metadata,
    d.embedding,
    1 - (d.embedding <=> query_embedding) AS similarity
  FROM documents d
  ORDER BY d.embedding <=> query_embedding
  LIMIT match_count;
END;
$$ LANGUAGE plpgsql STABLE;