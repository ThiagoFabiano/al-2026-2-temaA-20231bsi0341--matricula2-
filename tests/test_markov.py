# [IA] Claude (Claude Opus 5.5) - 2026-11-21 - testes da Tabela 1 e das propriedades de src/markov.py
import numpy as np
import pytest

from src import markov as mk

TOL = 1e-6  # tolerância exigida para a Tabela 1

P_REF = np.array([
    [0.90, 0.08, 0.02, 0.00],
    [0.20, 0.65, 0.13, 0.02],
    [0.05, 0.15, 0.65, 0.15],
    [0.00, 0.05, 0.20, 0.75],
])

PI_REF = np.array([0.498970, 0.205954, 0.174125, 0.120951])


# ---------------------------------------------------------------------------
# Tabela 1
# ---------------------------------------------------------------------------

def test_pi_tabela1():
    assert np.allclose(mk.pi_sistema(P_REF), PI_REF, atol=TOL)


def test_espectro_tabela1():
    esperado = [1, 0.869971, 0.618816, 0.461213]
    assert np.allclose(np.abs(mk.espectro(P_REF)), esperado, atol=TOL)


def test_meia_vida_tabela1():
    assert mk.meia_vida(P_REF) == pytest.approx(4.976086, abs=TOL)


def test_menor_k_tabela1():
    assert mk.menor_k_positiva(P_REF) == 2


def test_e1_P4_tabela1():
    esperado = [0.728667, 0.172331, 0.077617, 0.021385]
    e1 = mk.vetor_base(1)
    assert np.allclose(e1 @ mk.potencia_quadrados(P_REF, 4), esperado, atol=TOL)


def test_distancia_e1_P20_tabela1():
    d = mk.distancia_estacionaria(P_REF, mk.vetor_base(1), mk.pi_sistema(P_REF), 20)
    assert d[20] == pytest.approx(0.045556, abs=TOL)


def test_primeira_passagem_tabela1():
    M = mk.primeira_passagem(P_REF)
    assert M[0, 3] == pytest.approx(46.873065, abs=TOL)  # m14
    assert M[3, 0] == pytest.approx(17.035647, abs=TOL)  # m41


def test_retorno_tabela1():
    esperado = [2.004128, 4.855455, 5.743011, 8.267802]
    assert np.allclose(np.diag(mk.primeira_passagem(P_REF)), esperado, atol=TOL)


def test_absorcao_nivel4_tabela1():
    r = mk.cadeia_absorvente(P_REF, transientes=[1, 2, 3], absorventes=[4])
    assert np.allclose(r["t"], [46.873065, 39.473684, 26.470588], atol=TOL)
    assert np.allclose(r["var"], [1788.363542, 1732.320352, 1427.271899], atol=TOL)


def test_B_tabela1():
    r = mk.cadeia_absorvente(P_REF, transientes=[2, 3], absorventes=[1, 4])
    esperado = [[0.742718, 0.257282], [0.461165, 0.538835]]
    assert np.allclose(r["B"], esperado, atol=TOL)


# ---------------------------------------------------------------------------
# Concordância entre caminhos independentes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pi0", [mk.vetor_base(1), mk.vetor_base(4), None])
def test_pi_tres_metodos_concordam(pi0):
    a = mk.pi_sistema(P_REF)
    b, iteracoes = mk.pi_potencias(P_REF, pi0)
    c = mk.pi_autovetor(P_REF)
    assert np.abs(a - b).sum() < 1e-10
    assert np.abs(a - c).sum() < 1e-10
    assert iteracoes > 0


def test_residuo_pi():
    pi = mk.pi_sistema(P_REF)
    assert np.abs(pi @ P_REF - pi).sum() < 1e-12


def test_potencias_concordam():
    por_quadrados = mk.potencia_quadrados(P_REF, 52)
    por_diag, kappa = mk.potencia_diagonalizacao(P_REF, 52)
    por_produto = np.linalg.matrix_power(P_REF, 52)
    assert np.abs(por_quadrados - por_diag).sum() < 1e-10
    assert np.abs(por_quadrados - por_produto).sum() < 1e-10
    assert kappa >= 1


def test_M_dois_metodos_concordam():
    assert np.allclose(mk.primeira_passagem(P_REF), mk.primeira_passagem_Z(P_REF), atol=1e-10)


def test_N_inversa_e_neumann():
    r = mk.cadeia_absorvente(P_REF, transientes=[1, 2, 3], absorventes=[4])
    erros = [np.abs(mk.neumann_parcial(r["Q"], K) - r["N"]).max() for K in (10, 100, 2000)]
    assert erros[0] > erros[1] > erros[2]
    assert erros[2] < 1e-10


# ---------------------------------------------------------------------------
# Propriedades
# ---------------------------------------------------------------------------

def test_kac():
    pi = mk.pi_sistema(P_REF)
    assert np.allclose(np.diag(mk.primeira_passagem(P_REF)), 1 / pi, atol=1e-10)


def test_B_linhas_somam_1():
    r = mk.cadeia_absorvente(P_REF, transientes=[2, 3], absorventes=[1, 4])
    assert np.allclose(r["B"].sum(axis=1), 1.0, atol=1e-12)


def test_produto_de_estocasticas_eh_estocastico():
    rng = np.random.default_rng(2026)
    A = rng.random((4, 4))
    A /= A.sum(axis=1, keepdims=True)
    assert mk.eh_estocastica(A @ P_REF)
    assert mk.eh_estocastica(mk.potencia_quadrados(P_REF, 37))


def test_traco_e_determinante():
    lam = mk.espectro(P_REF)
    assert abs(lam.sum() - np.trace(P_REF)) < 1e-10
    assert abs(np.prod(lam) - np.linalg.det(P_REF)) < 1e-10


def test_P_estimada_eh_estocastica():
    C = np.array([[50, 4, 1, 0], [6, 30, 5, 1], [1, 4, 20, 3], [0, 1, 4, 15]])
    for alpha in (0.0, 0.5):
        assert mk.eh_estocastica(mk.estimar_P(C, alpha))


def test_laplace_formula():
    C = np.array([[3, 1, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 2, 2]])
    P = mk.estimar_P(C, alpha=0.5)
    assert P[0, 0] == pytest.approx((3 + 0.5) / (4 + 2))
    assert np.allclose(P[1], 0.25)  # linha sem dados vira uniforme


def test_linha_vazia_sem_suavizacao_gera_erro():
    C = np.array([[3, 1, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 2, 2]])
    with pytest.raises(ValueError):
        mk.estimar_P(C)


def test_matriz_nao_primitiva():
    ciclo = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]], dtype=float)
    assert mk.menor_k_positiva(ciclo) is None


# ---------------------------------------------------------------------------
# Contagem de transições (série pequena construída à mão)
# ---------------------------------------------------------------------------

def test_contagem_serie_com_lacuna():
    # 202401..202404 consecutivas; 202405 ausente; 202406..202408 consecutivas;
    # 202409 sem nível; 202410 isolada.
    semanas = [202401, 202402, 202403, 202404, 202406, 202407, 202408, 202409, 202410]
    niveis = [1, 1, 2, 3, 3, 4, 4, np.nan, 2]
    C = mk.contar_transicoes(semanas, niveis)
    esperado = np.zeros((4, 4), dtype=int)
    esperado[0, 0] = 1  # 01 -> 02
    esperado[0, 1] = 1  # 02 -> 03
    esperado[1, 2] = 1  # 03 -> 04
    esperado[2, 3] = 1  # 06 -> 07
    esperado[3, 3] = 1  # 07 -> 08
    assert np.array_equal(C, esperado)
    assert C.sum() == 5


def test_contagem_independe_da_ordem():
    semanas = [202403, 202401, 202402]
    niveis = [3, 1, 2]
    C = mk.contar_transicoes(semanas, niveis)
    assert C[0, 1] == 1 and C[1, 2] == 1 and C.sum() == 2


def test_virada_de_ano_com_SE53():
    assert mk.semanas_no_ano(2020) == 53
    assert mk.semanas_no_ano(2025) == 53
    assert mk.semanas_no_ano(2024) == 52
    # 2020 tem SE 53: 202052 -> 202053 -> 202101 são consecutivas
    C = mk.contar_transicoes([202052, 202053, 202101], [1, 2, 3])
    assert C.sum() == 2
    # 2024 não tem SE 53: 202452 -> 202501 é consecutiva
    C = mk.contar_transicoes([202452, 202501], [1, 1])
    assert C[0, 0] == 1
    # 202052 -> 202101 pula a SE 53: não é transição
    C = mk.contar_transicoes([202052, 202101], [1, 1])
    assert C.sum() == 0


def test_SE_inexistente_gera_erro():
    with pytest.raises(ValueError):
        mk.indice_semana(202453)


def test_duplicata_gera_erro():
    with pytest.raises(ValueError):
        mk.contar_transicoes([202401, 202401, 202402], [1, 2, 2])
# [/IA]
