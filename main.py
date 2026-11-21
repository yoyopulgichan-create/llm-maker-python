import re
import numpy as np
import matplotlib.pyplot as plt

RNG_SEED = 42


class Tokenizer:
    """1️⃣ Tokenização: separa o texto em palavras e pontuação (tokens)."""

    def __init__(self):
        self.token_to_id = {}
        self.id_to_token = {}

    def _get_id(self, token):
        if token not in self.token_to_id:
            novo_id = len(self.token_to_id)
            self.token_to_id[token] = novo_id
            self.id_to_token[novo_id] = token
        return self.token_to_id[token]

    def encode(self, text):
        tokens = re.findall(r"\w+|[^\w\s]", text.lower(), flags=re.UNICODE)
        ids = [self._get_id(t) for t in tokens]
        return tokens, ids


class Embedder:
    """2️⃣ Codificação numérica: cada token vira um vetor de "significado".
    Palavras parecidas (do mesmo grupo) ficam pertinho no espaço."""

    DIM = 6

    GRUPOS = {
        "saudacao": ["oi", "ola", "olá", "opa", "e", "aí", "eai"],
        "pergunta": ["?", "como", "que", "quem", "quando", "onde", "qual", "por"],
        "positivo": ["ótimo", "otimo", "legal", "incrível", "incrivel", "demais", "bem", "adorei", "uau", "bom"],
        "negativo": ["não", "nao", "ruim", "triste", "chato"],
        "acao": ["ajudar", "conversar", "aprender", "tentar", "brincar", "posso", "vou"],
    }

    def __init__(self):
        rng = np.random.default_rng(RNG_SEED)
        self.ancoras = {grupo: rng.normal(size=self.DIM) for grupo in self.GRUPOS}
        self._cache = {}
        self._token_para_grupo = {}
        for grupo, palavras in self.GRUPOS.items():
            for p in palavras:
                self._token_para_grupo[p] = grupo

    def vetor(self, token):
        if token not in self._cache:
            rng = np.random.default_rng(abs(hash(token)) % (2**32))
            grupo = self._token_para_grupo.get(token)
            if grupo:
                v = self.ancoras[grupo] + rng.normal(scale=0.3, size=self.DIM)
            else:
                v = rng.normal(size=self.DIM)
            self._cache[token] = v
        return self._cache[token]

    def vetores(self, tokens):
        return np.array([self.vetor(t) for t in tokens])


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


class LLMBrain:
    """3️⃣ Cálculo e previsão: uma rede neural com atenção escolhe,
    passo a passo, o próximo token mais provável."""

    VOCABULARIO_RESPOSTAS = [
        "oi", "tudo", "ótimo", "bem", "legal", "demais", "como", "posso",
        "ajudar", "você", "estou", "aprendendo", "com", "uau", "que",
        "incrível", "não", "sei", "ainda", "vamos", "conversar", "mais",
        "obrigado", "por", "perguntar", ".", "!", "?",
    ]

    def __init__(self, embedder):
        self.embedder = embedder
        dim = embedder.DIM
        rng = np.random.default_rng(RNG_SEED + 1)
        self.Wq = rng.normal(scale=0.5, size=(dim, dim))
        self.Wk = rng.normal(scale=0.5, size=(dim, dim))
        self.Wv = rng.normal(scale=0.5, size=(dim, dim))
        self.resp_vetores = embedder.vetores(self.VOCABULARIO_RESPOSTAS)

    def atencao(self, vetores):
        """Self-attention bem pequenininha: cada token "olha" pros outros. 🔍"""
        Q = vetores @ self.Wq
        K = vetores @ self.Wk
        V = vetores @ self.Wv
        scores = Q @ K.T / np.sqrt(vetores.shape[1])
        pesos = np.array([softmax(linha) for linha in scores])
        atendido = pesos @ V
        contexto = atendido.mean(axis=0)
        return contexto

    def prever_proximo(self, contexto, temperatura=0.9):
        logits = self.resp_vetores @ contexto / np.sqrt(len(contexto))
        probs = softmax(logits / temperatura)
        escolhido = np.random.choice(len(self.VOCABULARIO_RESPOSTAS), p=probs)
        return self.VOCABULARIO_RESPOSTAS[escolhido], probs

    def gerar_resposta(self, tokens_entrada, max_tokens=8):
        vetores_contexto = list(self.embedder.vetores(tokens_entrada))
        gerados = []
        primeiro_probs = None
        for passo in range(max_tokens):
            contexto = self.atencao(np.array(vetores_contexto))
            palavra, probs = self.prever_proximo(contexto)
            if passo == 0:
                primeiro_probs = probs
            if palavra == "." and gerados:
                gerados.append(palavra)
                break
            gerados.append(palavra)
            vetores_contexto.append(self.embedder.vetor(palavra))
        return gerados, primeiro_probs


def decodificar(tokens):
    """4️⃣ Decodificação em frase: os tokens escolhidos viram uma frase."""
    frase = ""
    for t in tokens:
        if t in ".,!?" and frase:
            frase = frase.rstrip() + t + " "
        else:
            frase += t + " "
    frase = frase.strip()
    return frase[0].upper() + frase[1:] if frase else frase


# ================== 🎬 começa a conversa ==================

llm_name = input("Kid, what's the name of your chatbot? ").strip()
print(f"🎉 Yay! Your chatbot is called **{llm_name}**! 🤖💕")

tokenizer = Tokenizer()
embedder = Embedder()
brain = LLMBrain(embedder)

new_chat = input("New chat! ").strip()
print(f"you: {new_chat}")

# 1️⃣ Tokenização
tokens, ids = tokenizer.encode(new_chat)
tokens_para_mostrar = tokens if tokens else ["..."]
print(f"\n1️⃣ Tokenização — {llm_name} separou sua frase em tokens: {tokens_para_mostrar}")

# 2️⃣ Codificação numérica
print(f"2️⃣ Codificação numérica — cada token virou um ID: {ids if ids else '[]'}")
vetor_exemplo = embedder.vetor(tokens_para_mostrar[0])
print(f"   e um vetor de significado, tipo: '{tokens_para_mostrar[0]}' ➡️ {np.round(vetor_exemplo, 2)}")

# 3️⃣ Cálculo e previsão
gerados, primeiro_probs = brain.gerar_resposta(tokens_para_mostrar)
print(f"3️⃣ Cálculo e previsão — a atenção olhou pro contexto e foi escolhendo: {gerados}")

# 4️⃣ Decodificação em frase
resposta = decodificar(gerados)
print(f"\n4️⃣ Decodificação em frase — {llm_name}: {resposta}")

# 📊 desenha o espaço de significado + a previsão do primeiro token
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

V = embedder.vetores(tokens_para_mostrar)
V_centrado = V - V.mean(axis=0)
if V.shape[0] > 1:
    _, _, VT = np.linalg.svd(V_centrado, full_matrices=False)
    pontos_2d = V_centrado @ VT[:2].T
else:
    pontos_2d = np.zeros((1, 2))

ax1.scatter(pontos_2d[:, 0], pontos_2d[:, 1], color="hotpink", s=80)
for (x, y), palavra in zip(pontos_2d, tokens_para_mostrar):
    ax1.annotate(palavra, (x, y), xytext=(4, 4), textcoords="offset points")
ax1.set_title("Espaco de significado")
ax1.set_xlabel("dimensao 1")
ax1.set_ylabel("dimensao 2")

top = np.argsort(primeiro_probs)[-6:]
ax2.barh([brain.VOCABULARIO_RESPOSTAS[i] for i in top], primeiro_probs[top], color="hotpink")
ax2.set_title("Previsao do proximo token")
ax2.set_xlabel("probabilidade")

plt.tight_layout()
plt.savefig("brainwave.png")
print("\n📊 Saved a cute little brainwave picture as brainwave.png! 🌈
    

