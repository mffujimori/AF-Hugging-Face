# Instalar:
# pip install transformers sentence-transformers faiss-cpu

from transformers import AutoTokenizer, AutoModel
import torch
import faiss
import numpy as np

# 1. Carregar modelo HuggingFace
model_name = "sentence-transformers/all-MiniLM-L6-v2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Função para gerar embeddings
def encode(texts):
    tokens = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        output = model(**tokens)
    embeddings = output.last_hidden_state.mean(dim=1)
    return embeddings.numpy()

# 2. Base de documentos
docs = [
    "trabalho_vector_database.py",
    "projeto_busca_semantica_vector_db.py",
    "trabalho_huggingface_faiss.py",
    "atividade_pratica_vector_db.py",
    "implementacao_vector_database.py",
    "projeto_aplicacao_semantic_search.py",
    "trabalho_aplicacao_faiss_embeddings.py",
    "atividade_ia_busca_semantica.py",
    "sistema_busca_semantica_faiss.py",
    "trabalho_integrador_semantic_search.py"
]

# 3. Criar embeddings
embeds = encode(docs)

# 4. Criar índice FAISS
index = faiss.IndexFlatL2(embeds.shape[1])
index.add(np.array(embeds))


# ==========================
#        LOOP PRINCIPAL
# ==========================
while True:
    query = input("\nDigite sua pergunta: ")

    # Gerar embedding da consulta
    q_emb = encode([query])

    # Buscar documento mais similar
    dist, idx = index.search(q_emb, 1)

    print("\n🔎 Pergunta:", query)
    print("📄 Documento mais parecido:", docs[idx[0][0]])
    print("📉 Distância:", dist[0][0])

    # Perguntar se deseja continuar
    again = input("\nDeseja fazer outra pergunta? (s/n): ").strip().lower()

    if again not in ["s", "sim", "y", "yes"]:
        print("\n👋 Encerrando. Até mais!")
        break
