from langchain_community.vectorstores import SupabaseVectorStore

class CustomSupabaseVectorStore(SupabaseVectorStore):
    def __init__(self, *args, relevance_score_fn=None, **kwargs):
        # Store the custom relevance scoring function.
        self.relevance_score_fn = relevance_score_fn
        super().__init__(*args, **kwargs)

    def _select_relevance_score_fn(self):
        # If a custom relevance score function was provided, return it.
        if self.relevance_score_fn is not None:
            return self.relevance_score_fn
        # Otherwise, use the parent's version.
        return super()._select_relevance_score_fn() 