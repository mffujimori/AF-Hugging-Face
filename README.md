# AtividadeIA - AF 11 6AN CC
Script para localizar arquivos através de palavras-chave. 

# Integrantes
Gabriel Quirino <br>
Higor Rocha <br>
Julio Cesar <br>
Matheus Fermino <br>
Mateo Rodriguez <br>
Nicolas Medina <br>

# Problema
  - Dificuldade na localização de arquivos, dentro de uma grande base de dados
    
# Solução
HuggingFace (para gerar embeddings)
  - Transformamos nomes de arquivos em vetores numéricos que representam o significado do texto.,

FAISS (Vector Database)
  - Criamos um índice vetorial para fazer busca rápida por similaridade.,

Python + Loop interativo

O usuário pode:
digitar o que procura <br>
o modelo gera embeddings <br>
o FAISS encontra o arquivo mais semântico <br>
o sistema pergunta se ele quer continuar <br>
