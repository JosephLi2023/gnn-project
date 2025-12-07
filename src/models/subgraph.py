class Subgraph:
    def __init__(self, graph_id, movie_ids, embeddings):
        self.graph_id = graph_id
        self.movie_ids = movie_ids
        self.embeddings = embeddings  # List[torch.Tensor] or torch.Tensor
