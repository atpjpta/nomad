"""Exact endpoint checks for the consecutive-order certificate. Run with unittest."""
import unittest
from fractions import Fraction as Q

import torch

if __package__:
    from .nomad_error_bounds import certify_step
    from .nomad_solver import NomadSolver
else:
    from nomad_error_bounds import certify_step
    from nomad_solver import NomadSolver


def doubles(values):
    return torch.tensor(values, dtype=torch.float64)


class CertifiedSteps(unittest.TestCase):
    def assert_contains(self, interval, exact):
        self.assertLessEqual(interval[0], exact)
        self.assertLessEqual(exact, interval[1])

    def assert_endpoint(self, cert, exact):
        self.assert_contains(cert.signed_low_error, exact - cert.low_output)
        self.assert_contains(cert.signed_high_error, exact - cert.high_output)
        self.assert_contains(cert.absolute_low_error, abs(exact - cert.low_output))

    def test_rational_flow_with_duplicate_columns(self):
        # x' = 2 x² - x². With x(0) = 1 the endpoint is exactly 1/(1-h).
        for p in (1, 2, 4, 7):
            for h in (0.125, 0.0625):
                with self.subTest(p=p, h=h):
                    cert, = certify_step(doubles([[2, -1]]), torch.tensor([[2, 2]]),
                                         doubles([1]), h, p, doubles([0.5]))
                    H = Q(h)
                    self.assertEqual(cert.low_output, sum(H**k for k in range(p+1)))
                    self.assertEqual(cert.observed_increment, H**(p+1))
                    self.assertEqual(cert.remainder_radius, H**(p+2)*Q(3, 2)**(p+3))
                    self.assert_endpoint(cert, 1/(1-H))

    def test_vanishing_increment(self):
        # u'=1, v'=u^(p+1) has endpoint (h, h^(p+2)/(p+2)).
        for p in (1, 2, 4):
            with self.subTest(p=p):
                u, v = certify_step(doubles([[1, 0], [0, 1]]),
                                    torch.tensor([[0, p+1], [0, 0]]),
                                    doubles([0, 0]), 0.25, p, doubles([0.5, 0.5]))
                exact = Q(1, 4)**(p+2)/(p+2)
                self.assertEqual(u.observed_increment, 0)
                self.assertEqual(v.observed_increment, 0)
                self.assertEqual(v.remainder_radius, exact)
                self.assert_endpoint(u, Q(1, 4))
                self.assert_endpoint(v, exact)

    def test_block_of_higher_orders(self):
        a = doubles([[1, 0], [0, 1]])
        k = torch.tensor([[0, 3], [0, 0]])
        z, radii = doubles([0, 0]), doubles([0.5, 0.5])
        _, v = certify_step(a, k, z, 0.25, 2, radii, high_order=4)
        exact = Q(1, 4)**4 / 4
        self.assertEqual(v.observed_increment, exact)
        self.assertEqual(v.remainder_radius, 0)
        self.assertEqual(v.signed_low_error, (exact, exact))
        with self.assertRaisesRegex(ValueError, "high_order"):
            certify_step(a, k, z, 0.25, 2, radii, high_order=2)

    def test_shrinking_support_and_coordinates(self):
        # x'=x² and y'=1 exhibit shrinking support and distinguish physical rows.
        a, k = doubles([[1, 0], [0, 1]]), torch.tensor([[2, 0], [0, 0]])
        x, y = certify_step(a, k, doubles([1, 3]), 0.125, 3, doubles([0.5, 0.5]))
        self.assertEqual(x.low_output, Q(585, 512))
        self.assertEqual(y.low_output, Q(25, 8))
        self.assert_endpoint(x, Q(8, 7))
        self.assert_endpoint(y, Q(25, 8))

    def test_notebook_coupled_field(self):
        # At (1,2) the fourth raw derivative of the notebook field is (-91,-49).
        a, k = doubles([[-1, 1, 0, 0], [0, 0, 1, -1]]), torch.tensor([[1, 0, 2, 0], [0, 2, 0, 1]])
        h = Q(1, 32)
        certs = certify_step(a, k, doubles([1, 2]), float(h), 3, doubles([0.5, 0.5]))
        for cert, z, first, second, third, fourth in zip(certs, (1, 2), (3, -1),
                                                        (-7, 7), (37, -3), (-91, -49)):
            exact_low = z + first*h + second*h**2/2 + third*h**3/6
            self.assertEqual(cert.exact_high_taylor, exact_low + fourth*h**4/24)
            self.assertLessEqual(abs(cert.low_output - exact_low), Q(1, 2**51))
            self.assertLessEqual(abs(cert.high_output - cert.exact_high_taylor), Q(1, 2**51))
            self.assertGreater(cert.remainder_radius, 0)

    def test_negative_signed_error(self):
        cert, = certify_step(doubles([[-1]]), torch.tensor([[2]]), doubles([1]),
                             0.125, 2, doubles([0.5]))
        self.assertLess(cert.signed_low_error[1], 0)
        self.assert_endpoint(cert, Q(8, 9))

    def test_roundoff_with_zero_truncation(self):
        # At z=1, binary64 rounds 1 + 2^-54 to 1 despite an exactly linear flow.
        h = 2.0**-54
        cert, = certify_step(doubles([[1]]), torch.tensor([[0]]),
                             doubles([1]), h, 2, doubles([1]))
        self.assertEqual(cert.observed_increment, 0)
        self.assertEqual(cert.remainder_radius, 0)
        self.assertEqual(cert.signed_low_error, (Q(h), Q(h)))
        self.assert_endpoint(cert, 1 + Q(h))

    def test_zero_field_and_rejected_inputs(self):
        a, k = doubles([[0]]), torch.tensor([[2]])
        cert, = certify_step(a, k, doubles([1]), 0.25, 2, doubles([1]))
        self.assertEqual(cert.signed_low_error, (0, 0))
        with self.assertRaisesRegex(ValueError, "box test"):
            certify_step(doubles([[1]]), k, doubles([1]), 0.25, 2, doubles([1]))
        with self.assertRaisesRegex(ValueError, "fractional"):
            certify_step(a, doubles([[0.5]]), doubles([1]), 0.25, 2, doubles([1]))

    def test_original_second_order_regressions(self):
        # Both the batched public step and the asymmetric field expose the port regressions.
        for a, k, z, expected in (
            ([[1]], [[1]], [[[1, 2]]], [[[1.28125, 2.5625]]]),
            ([[1], [0]], [[2], [0]], [[[1, 1]], [[2, 3]]],
             [[[1.3125, 1.3125]], [[2, 3]]]),
            ([[1 + 2**-30]], [[1]], [[[1]]], None),
        ):
            with self.subTest(a=a):
                solver = NomadSolver(doubles(a), torch.tensor(k), n=2, device='cpu')
                if expected is None:
                    factor = a[0][0]
                    self.assertEqual(solver.a_hats['1'].item(), factor * factor / 2)
                else:
                    self.assertTrue(torch.equal(solver.step(doubles(z), 0.25), doubles(expected)))


if __name__ == '__main__':
    unittest.main()
