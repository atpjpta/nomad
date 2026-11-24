import Mathlib.Data.Complex.Trigonometric
import Mathlib.Tactic.Ring
import Mathlib.Tactic.FieldSimp

/-! The twelve exponential-quotient selections in the notebook. -/
noncomputable section
open Complex
namespace Nomad

def evenExp (z : ℂ) := exp z + exp (-z)
def oddExp (z : ℂ) := exp z - exp (-z)

private theorem even_h (z : ℂ) : evenExp z = 2 * cosh z := by
  unfold evenExp Complex.cosh
  ring
private theorem odd_h (z : ℂ) : oddExp z = 2 * sinh z := by
  unfold oddExp Complex.sinh
  ring
private theorem even_i (z : ℂ) : evenExp (z*I) = 2 * cos z := by
  rw [even_h, cosh_mul_I]
private theorem odd_i (z : ℂ) : oddExp (z*I) = 2 * (sin z * I) := by
  rw [odd_h, sinh_mul_I]
private theorem even_zero : evenExp 0 = 2 := by simp [evenExp]; ring

/-- Conditional rows give the ordinary domains of the reciprocal or quotient functions.
The argument can in particular be the value of a polynomial. -/
theorem selectors (z w : ℂ) :
    (-I*w * oddExp (z*I) / evenExp 0 = w*sin z) ∧
    (w * oddExp z / evenExp 0 = w*sinh z) ∧
    (w * evenExp (z*I) / evenExp 0 = w*cos z) ∧
    (w * evenExp z / evenExp 0 = w*cosh z) ∧
    (cos z ≠ 0 → w * evenExp 0 / evenExp (z*I) = w / cos z) ∧
    (cosh z ≠ 0 → w * evenExp 0 / evenExp z = w / cosh z) ∧
    (sin z ≠ 0 → I*w * evenExp 0 / oddExp (z*I) = w / sin z) ∧
    (sinh z ≠ 0 → w * evenExp 0 / oddExp z = w / sinh z) ∧
    (cos z ≠ 0 → -I*w * oddExp (z*I) / evenExp (z*I) = w*tan z) ∧
    (cosh z ≠ 0 → w * oddExp z / evenExp z = w*tanh z) ∧
    (sin z ≠ 0 → I*w * evenExp (z*I) / oddExp (z*I) = w*(cos z / sin z)) ∧
    (sinh z ≠ 0 → w * evenExp z / oddExp z = w*(cosh z / sinh z)) := by
  simp only [even_zero, even_i, odd_i]
  simp only [even_h, odd_h, Complex.tan_eq_sin_div_cos,
    Complex.tanh_eq_sinh_div_cosh]
  repeat' apply And.intro
  all_goals (try intro h)
  all_goals field_simp [Complex.I_ne_zero, *]
  all_goals ring_nf
  all_goals simp [Complex.I_sq]

end Nomad
