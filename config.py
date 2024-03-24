class Config:
    def __init__(self, model_name='llama2-13b-chat', temperature=0, top_p=0.95,
                 repetition_penalty=1.15, split_chunk_size=800, split_overlap=100,
                 embeddings_model_name="all-MiniLM-L6-v2",
                 K=6, PDFs_path="dataset/",
                 embeddings_path="embeddings/", output_folder="outputs/"):
        self.model_name = model_name  # 'llama2-13b-chat'  # wizardlm, llama2-7b-chat, llama2-13b-chat, mistral-7B
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.split_chunk_size = split_chunk_size
        self.split_overlap = split_overlap
        self.embeddings_model_name = embeddings_model_name
        self.K = K
        self.PDFs_path = PDFs_path
        self.embeddings_path = embeddings_path
        self.output_folder = output_folder
