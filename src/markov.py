"""Funções gerais de cadeias de Markov usadas nas Etapas 1 a 8.

Convenções
----------
* Os níveis de alerta são 1, 2, 3, 4 (verde, amarelo, laranja, vermelho).
  Dentro das matrizes, o nível ``k`` ocupa o índice ``k - 1``.
* Distribuições são vetores-linha: pi_(t+1) = pi_(t) @ P.
* Semanas epidemiológicas (SE) vêm no formato AAAASS (ex.: 202453).

Somente numpy é usado aqui, para que os testes rodem com o mínimo de
dependências. Os laços do método das potências e dos quadrados sucessivos
são escritos explicitamente, como pede a Seção 7 do enunciado.
"""

# [IA] Claude (Claude Opus 5.5) - 2026-11-21 - funções de cadeias de Markov (estimação, pi, espectro, potências, primeira passagem, cadeia absorvente)
from datetime import date, timedelta

import numpy as np

N_NIVEIS = 4
TOL_ESTOCASTICA = 1e-12



# Semanas epidemiológicas


def inicio_ano_epidemiologico(ano):
    """Domingo em que começa a SE 1 do ano.

    A SE 1 é a semana (domingo a sábado) que contém pelo menos quatro dias
    de janeiro, isto é, a semana que contém o dia 4 de janeiro.
    """
    d = date(ano, 1, 4)
    return d - timedelta(days=(d.weekday() + 1) % 7)  # recua até o domingo


def semanas_no_ano(ano):
    """Número de semanas epidemiológicas do ano (52 ou 53)."""
    return (inicio_ano_epidemiologico(ano + 1) - inicio_ano_epidemiologico(ano)).days // 7


def indice_semana(se):
    """Converte AAAASS em um inteiro sequencial: semanas consecutivas no
    calendário epidemiológico têm índices consecutivos (inclusive na virada
    de ano e nos anos com SE 53)."""
    se = int(se)
    ano, semana = divmod(se, 100)
    if not 1 <= semana <= semanas_no_ano(ano):
        raise ValueError(f"SE inválida: {se} (o ano {ano} tem {semanas_no_ano(ano)} semanas)")
    domingo = inicio_ano_epidemiologico(ano) + timedelta(weeks=semana - 1)
    return domingo.toordinal() // 7



# Contagem e estimação


def contar_transicoes(semanas, niveis, n=N_NIVEIS):
    """Matriz de contagens C = (n_ij) a partir de uma série de um município.

    Uma transição (X_t, X_t+1) só é contada se as duas semanas forem
    consecutivas no calendário epidemiológico e ambas tiverem nível em
    {1, ..., n}. Lacunas (semana ausente ou nível ausente) interrompem a
    sequência. Semanas duplicadas geram erro: o critério de escolha entre
    registros duplicados deve ser aplicado antes (Etapa 1).
    """
    semanas = np.asarray(semanas)
    niveis = np.asarray(niveis, dtype=float)
    if semanas.shape != niveis.shape:
        raise ValueError("semanas e niveis devem ter o mesmo tamanho")

    idx = np.array([indice_semana(s) for s in semanas], dtype=int)
    ordem = np.argsort(idx, kind="stable")
    idx, niveis = idx[ordem], niveis[ordem]
    if np.any(np.diff(idx) == 0):
        raise ValueError("há semanas duplicadas na série")

    C = np.zeros((n, n), dtype=int)
    for t in range(len(idx) - 1):
        a, b = niveis[t], niveis[t + 1]
        consecutivas = idx[t + 1] - idx[t] == 1
        validos = (not np.isnan(a)) and (not np.isnan(b)) and 1 <= a <= n and 1 <= b <= n
        if consecutivas and validos:
            C[int(a) - 1, int(b) - 1] += 1
    return C


def estimar_P(C, alpha=0.0):
    """Estimador de máxima verossimilhança p_ij = n_ij / n_i.

    Com ``alpha > 0`` aplica a suavização de Laplace
    p_ij = (n_ij + alpha) / (n_i + n * alpha).
    """
    C = np.asarray(C, dtype=float)
    n = C.shape[0]
    linhas = C.sum(axis=1, keepdims=True)
    if alpha == 0 and np.any(linhas == 0):
        vazias = (np.where(linhas.ravel() == 0)[0] + 1).tolist()
        raise ValueError(f"níveis sem nenhuma transição de saída: {vazias}")
    return (C + alpha) / (linhas + n * alpha)


def erros_padrao(C):
    """Erro-padrão aproximado sqrt(p_ij (1 - p_ij) / n_i) de cada entrada."""
    C = np.asarray(C, dtype=float)
    P = estimar_P(C)
    return np.sqrt(P * (1 - P) / C.sum(axis=1, keepdims=True))


def eh_estocastica(P, tol=TOL_ESTOCASTICA):
    """Verifica P >= 0 e P 1 = 1 (com tolerância)."""
    P = np.asarray(P, dtype=float)
    return bool(np.all(P >= -tol) and np.allclose(P.sum(axis=1), 1.0, rtol=0, atol=tol))


def vetor_base(nivel, n=N_NIVEIS):
    """Vetor-linha e_i correspondente ao nível ``nivel`` (1 a n)."""
    e = np.zeros(n)
    e[nivel - 1] = 1.0
    return e



# Distribuição estacionária (três caminhos independentes)


def pi_sistema(P):
    """Resolve (P^T - I) pi^T = 0 trocando a última equação por sum(pi) = 1.

    numpy.linalg.solve usa fatoração LU com pivotamento parcial (LAPACK gesv).
    """
    P = np.asarray(P, dtype=float)
    n = P.shape[0]
    A = P.T - np.eye(n)
    A[-1, :] = 1.0
    b = np.zeros(n)
    b[-1] = 1.0
    return np.linalg.solve(A, b)


def pi_potencias(P, pi0=None, tol=1e-12, max_iter=1_000_000):
    """Método das potências: pi_(k+1) = pi_(k) P até ||pi_(k+1) - pi_(k)||_1 < tol.

    Retorna (pi, número de iterações).
    """
    P = np.asarray(P, dtype=float)
    n = P.shape[0]
    pi = np.full(n, 1.0 / n) if pi0 is None else np.asarray(pi0, dtype=float).copy()
    for k in range(1, max_iter + 1):
        novo = pi @ P
        if np.abs(novo - pi).sum() < tol:
            return novo, k
        pi = novo
    raise RuntimeError(f"método das potências não convergiu em {max_iter} iterações")


def pi_autovetor(P):
    """Autovetor à esquerda de P para o autovalor 1 (autovetor à direita de P^T).

    numpy.linalg.eig usa o algoritmo QR com deslocamentos sobre a forma de
    Hessenberg (LAPACK geev).
    """
    P = np.asarray(P, dtype=float)
    valores, vetores = np.linalg.eig(P.T)
    k = int(np.argmin(np.abs(valores - 1.0)))
    v = np.real(vetores[:, k])
    return v / v.sum()



# Espectro, primitividade e potências


def espectro(P):
    """Autovalores de P ordenados por módulo decrescente."""
    valores = np.linalg.eigvals(np.asarray(P, dtype=float))
    return valores[np.argsort(-np.abs(valores), kind="stable")]


def modulo_lambda2(P):
    """|lambda_2|: segundo maior módulo do espectro."""
    return float(np.abs(espectro(P)[1]))


def meia_vida(P):
    """n_1/2 = ln(1/2) / ln|lambda_2|, em semanas."""
    return float(np.log(0.5) / np.log(modulo_lambda2(P)))


def menor_k_positiva(P, k_max=None):
    """Menor k com P^k > 0 entrada a entrada.

    Por padrão procura até o limite de Wielandt (n-1)^2 + 1; se nenhuma
    potência até lá for positiva, a matriz não é primitiva e retorna None.
    """
    P = np.asarray(P, dtype=float)
    n = P.shape[0]
    if k_max is None:
        k_max = (n - 1) ** 2 + 1
    Pk = np.eye(n)
    for k in range(1, k_max + 1):
        Pk = Pk @ P
        if np.all(Pk > 0):
            return k
    return None


def potencia_quadrados(P, expoente):
    """P^expoente por quadrados sucessivos (exponenciação binária explícita)."""
    P = np.asarray(P, dtype=float)
    resultado = np.eye(P.shape[0])
    base = P.copy()
    e = int(expoente)
    while e > 0:
        if e % 2 == 1:
            resultado = resultado @ base
        base = base @ base
        e //= 2
    return resultado


def potencia_diagonalizacao(P, expoente):
    """P^expoente = V Lambda^expoente V^-1.

    Retorna (P^expoente, kappa_1(V)), com kappa_1(V) = ||V||_1 ||V^-1||_1.
    """
    P = np.asarray(P, dtype=float)
    valores, V = np.linalg.eig(P)
    V_inv = np.linalg.inv(V)
    Pn = V @ np.diag(valores ** expoente) @ V_inv
    kappa = np.linalg.norm(V, 1) * np.linalg.norm(V_inv, 1)
    return np.real_if_close(Pn, tol=1000).real, float(kappa)


def distancia_estacionaria(P, pi0, pi, n_max):
    """||pi0 P^n - pi||_1 para n = 0, ..., n_max."""
    P = np.asarray(P, dtype=float)
    atual = np.asarray(pi0, dtype=float).copy()
    distancias = []
    for _ in range(n_max + 1):
        distancias.append(np.abs(atual - pi).sum())
        atual = atual @ P
    return np.array(distancias)



# Permanência e primeira passagem


def permanencia_media(P):
    """Permanência média 1 / (1 - p_ii) em cada nível (distribuição geométrica)."""
    return 1.0 / (1.0 - np.diag(np.asarray(P, dtype=float)))


def primeira_passagem(P):
    """Matriz M = (m_ij) de tempos médios de primeira passagem.

    Para cada destino j resolve o sistema (n-1)x(n-1)
        m_ij = 1 + sum_{k != j} p_ik m_kj,   i != j,
    e depois m_jj = 1 + sum_{k != j} p_jk m_kj.
    """
    P = np.asarray(P, dtype=float)
    n = P.shape[0]
    M = np.zeros((n, n))
    for j in range(n):
        outros = [k for k in range(n) if k != j]
        A = np.eye(n - 1) - P[np.ix_(outros, outros)]
        m = np.linalg.solve(A, np.ones(n - 1))
        M[outros, j] = m
        M[j, j] = 1.0 + P[j, outros] @ m
    return M


def primeira_passagem_Z(P):
    """M pela matriz fundamental Z = (I - P + 1 pi)^-1 (caminho independente).

    m_ij = (z_jj - z_ij) / pi_j para i != j e m_jj = 1 / pi_j.
    """
    P = np.asarray(P, dtype=float)
    n = P.shape[0]
    pi = pi_sistema(P)
    Z = np.linalg.inv(np.eye(n) - P + np.outer(np.ones(n), pi))
    M = (np.diag(Z)[None, :] - Z) / pi[None, :]
    M[np.diag_indices(n)] = 1.0 / pi
    return M



# Cadeia absorvente


def cadeia_absorvente(P, transientes, absorventes):
    """Matriz fundamental e grandezas derivadas.

    ``transientes`` e ``absorventes`` são listas de níveis (1 a n). Retorna
    um dicionário com Q, R, N = (I - Q)^-1, t = N 1, var = (2N - I) t - t∘t
    e B = N R (linhas na ordem de ``transientes``, colunas na ordem de
    ``absorventes``).
    """
    P = np.asarray(P, dtype=float)
    T = [k - 1 for k in transientes]
    A = [k - 1 for k in absorventes]
    Q = P[np.ix_(T, T)]
    R = P[np.ix_(T, A)]
    I = np.eye(len(T))
    N = np.linalg.inv(I - Q)
    t = N @ np.ones(len(T))
    var = (2 * N - I) @ t - t * t
    B = N @ R
    return {"Q": Q, "R": R, "N": N, "t": t, "var": var, "B": B}


def neumann_parcial(Q, K):
    """Soma parcial sum_{k=0}^{K} Q^k da série de Neumann."""
    Q = np.asarray(Q, dtype=float)
    soma = np.eye(Q.shape[0])
    termo = np.eye(Q.shape[0])
    for _ in range(K):
        termo = termo @ Q
        soma = soma + termo
    return soma
# [/IA]
