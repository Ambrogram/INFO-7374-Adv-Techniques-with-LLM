from transformers import AutoTokenizer, AutoModel, pipeline
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np


# Retrieve Relevant Documents based on cosine similarity.
def retrieve_documents(query, top_k=3):
    # Generate embeddings for query
    query_embedding = embedding_model.encode([query])[0]

    # Compare embeddings of Query with embeddings of documents in Knowledge base based on Cosine Similarity
    similarities = cosine_similarity([query_embedding], document_embeddings)[0]

    #select top-k most similar embeddings (sort embeddings by similarity then select the biggest ones)
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    #Return a list of top similar documents
    return [documents[i] for i in top_indices], similarities[top_indices]


# Generate a response using the query and retrieved documents.
def generate_response(query, relevant_docs):
    # Add relevant docs(information) to the prompt as the context
    context = " ".join(relevant_docs)
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"

    # Return a response from generator based on query and relevant information
    response = generator(prompt, max_length=100, num_return_sequences=1)
    return response[0]["generated_text"]


# Model for embeddings
embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Model for generation
generator = pipeline("text2text-generation", model="google/flan-t5-small")

# Knowledge Base
documents = [
    "The Eiffel Tower is in the capital of France.",
    "The Eiffel Tower is a symbol of Paris.",
    "The Great Wall of China is in China.",
    "Mount Everest is the tallest mountain on Earth.",
    "The Amazon River is the longest river in South America."
]


# Generate embeddings for the documents
document_embeddings = embedding_model.encode(documents)


# Query
query = "Where is the Eiffel Tower?"

# Retrieve two relevant documents based on query
relevant_docs, scores = retrieve_documents(query, top_k=2)

# Generate a response
response = generate_response(query, relevant_docs)

# Output
print("\nQuery:", query)
print("\nRelevant Documents:")
for doc, score in zip(relevant_docs, scores):
    print(f"- {doc} (Score: {score:.4f})")
print("\nGenerated Response:")
print(response)
