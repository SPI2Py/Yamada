import pickle
from cypari import pari


def get_coefficients_and_exponents(poly):

    """
    A helper function to extract the coefficients and exponents from a Yamada polynomial.

    The Yamada polynomial calculator was originally written with SageMath and the Laurent polynomial objects
    had explicit attributes for coefficients and exponents that you could directly query. However, switching
    to the cypari library to improve OS compatibility added a few complications, including that there is no native
    method to access the coefficients and exponents of Yamada polynomials. This function obtains them.
    """

    # Assumes all denominators are only A**n with no coefficient
    coefficients = poly.numerator().Vec()
    coeff_len = len(coefficients)

    exponents = []
    degree = poly.poldegree()

    for _ in range(coeff_len):
        exponents.append(degree)
        degree -= 1

    return coefficients, exponents


def pickle_yamada(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)


def load_yamada(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)


def reverse_poly(poly):
    """
    """

    a = pari('A')

    coeffs, exps = get_coefficients_and_exponents(poly)

    ans = pari(0)

    for coeff, exp in zip(coeffs, exps):
        ans += coeff * a ** (-exp)

    return ans


def normalize_poly(yamada_polynomial):
    """normalized_yamada_polynomial
    """

    a = pari('A')

    _, exps = get_coefficients_and_exponents(yamada_polynomial)

    ans1 = (-a) ** (-min(exps)) * yamada_polynomial
    ans2 = (-a) ** max(exps) * reverse_poly(yamada_polynomial)

    normalized_poly = min([ans1, ans2], key=list)

    return normalized_poly
