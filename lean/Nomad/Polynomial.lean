import Mathlib.Algebra.MvPolynomial.PDeriv
import Mathlib.Analysis.Calculus.Taylor
import Mathlib.Tactic.Ring

/-! The coefficient table, its expansion into paths, and its meaning on a solution. -/
noncomputable section
open MvPolynomial Set
open scoped BigOperators

namespace Nomad

abbrev Poly (m : ℕ) := MvPolynomial (Fin m) ℝ
abbrev Exponent (m : ℕ) := Fin m →₀ ℕ

variable {m : ℕ}

/-- Each input index labels a column. Equal exponents add in the polynomial ring. -/
def table {w : ℕ} (k : Fin w → Exponent m) (c : Fin w → ℝ) : Poly m :=
  ∑ i, monomial (k i) (c i)

/-- The gradient multiplier and the second (possibly mixed) partial of one
monomial. The second multiplier reads the exponent left by the first derivative. -/
theorem monomial_hessian (u : Exponent m) (c : ℝ) (j l : Fin m) :
    pderiv l (pderiv j (monomial u c)) =
      monomial (u - Finsupp.single j 1 - Finsupp.single l 1)
        ((c * (u j : ℝ)) * ((u - Finsupp.single j 1 : Exponent m) l : ℝ)) := by
  rw [pderiv_monomial, pderiv_monomial]

def lie (F : Fin m → Poly m) (P : Poly m) : Poly m :=
  ∑ j, pderiv j P * F j

def jet (F : Fin m → Poly m) : ℕ → Fin m → Poly m
  | 0, l => X l
  | n + 1, l => lie F (jet F n l)

theorem lie_X (F : Fin m → Poly m) (l : Fin m) : lie F (X l) = F l := by
  classical
  unfold lie
  rw [Finset.sum_eq_single l]
  · simp
  · intro j _ h
    rw [pderiv_X_of_ne (Ne.symm h), zero_mul]
  · simp

theorem lie_add (F : Fin m → Poly m) (P Q : Poly m) :
    lie F (P + Q) = lie F P + lie F Q := by
  simp [lie, add_mul, Finset.sum_add_distrib]

theorem lie_mul (F : Fin m → Poly m) (P Q : Poly m) :
    lie F (P * Q) = lie F P * Q + P * lie F Q := by
  simp only [lie, pderiv_mul, add_mul, Finset.sum_add_distrib, Finset.sum_mul, Finset.mul_sum]
  congr 1 <;> apply Finset.sum_congr rfl <;> intros <;> ring

/-- One differentiation of every source column against every field column. The
truncated subtraction is harmless on an inactive branch because its coefficient is zero. -/
theorem lie_table {v w : ℕ} (k : Fin w → Exponent m) (A : Fin m → Fin w → ℝ)
    (q : Fin v → Exponent m) (c : Fin v → ℝ) :
    lie (fun j => table k (A j)) (table q c) =
      ∑ j, ∑ a, ∑ i, monomial (q a - Finsupp.single j 1 + k i)
        ((c a * (q a j : ℝ)) * A j i) := by
  simp only [lie, table, map_sum, Finset.sum_mul, Finset.mul_sum, pderiv_monomial, monomial_mul]
  apply Finset.sum_congr rfl
  intro j _
  exact Finset.sum_comm

theorem collect_table {w : ℕ} (k : Fin w → Exponent m) (c : Fin w → ℝ) (u : Exponent m) :
    coeff u (table k c) = ∑ i, if k i = u then c i else 0 := by
  simp only [table, coeff_sum, coeff_monomial]

/-- Expand every choice of differentiated coordinate and field column. The
recursive coefficient is the notebook's product of the encountered exponents and entries of A. -/
def paths {w : ℕ} (k : Fin w → Exponent m) (A : Fin m → Fin w → ℝ) :
    ℕ → Exponent m → ℝ → Poly m
  | 0, u, c => monomial u c
  | n + 1, u, c => ∑ j, ∑ i, paths k A n (u - Finsupp.single j 1 + k i)
      ((c * (u j : ℝ)) * A j i)

theorem paths_eq_iterate {w : ℕ} (k : Fin w → Exponent m) (A : Fin m → Fin w → ℝ)
    (n : ℕ) (u : Exponent m) (c : ℝ) :
    paths k A n u c = (lie (fun j => table k (A j)))^[n] (monomial u c) := by
  classical
  let F := fun j => table k (A j)
  let L : AddMonoid.End (Poly m) :=
    { toFun := lie F, map_zero' := by simp [lie], map_add' := lie_add F }
  change paths k A n u c = L^[n] (monomial u c)
  induction n generalizing u c with
  | zero => rfl
  | succ n ih =>
    rw [Function.iterate_succ_apply, paths]
    have h : L (monomial u c) = ∑ j, ∑ i, monomial (u - Finsupp.single j 1 + k i)
        ((c * (u j : ℝ)) * A j i) := by
      change lie (fun j => table k (A j)) (monomial u c) = _
      simp only [lie, table,
        Finset.mul_sum, pderiv_monomial, monomial_mul]
    rw [h]
    have distribute {ι : Type} [Fintype ι] (g : ι → Poly m) :
        L^[n] (∑ i, g i) = ∑ i, L^[n] (g i) :=
      map_sum (L ^ n) g Finset.univ
    simp only [ih, distribute]

theorem jet_eq_paths {w : ℕ} (k : Fin w → Exponent m) (A : Fin m → Fin w → ℝ)
    (n : ℕ) (l : Fin m) :
    jet (fun j => table k (A j)) n l = paths k A n (Finsupp.single l 1) 1 := by
  rw [paths_eq_iterate]
  induction n with
  | zero => rfl
  | succ n ih => rw [jet, Function.iterate_succ_apply', ih]

theorem paths_zero {w : ℕ} (k : Fin w → Exponent m) (A : Fin m → Fin w → ℝ)
    (n : ℕ) (u : Exponent m) : paths k A n u 0 = 0 := by
  induction n generalizing u with
  | zero => simp [paths]
  | succ n ih => simp [paths, ih]

/-- Start from the first derivative, exactly as in the notebook's path formula. -/
theorem field_paths {w : ℕ} (k : Fin w → Exponent m) (A : Fin m → Fin w → ℝ)
    (n : ℕ) (l : Fin m) :
    jet (fun j => table k (A j)) (n+1) l = ∑ i, paths k A n (k i) (A l i) := by
  rw [jet_eq_paths, paths, Finset.sum_eq_single l]
  · simp
  · intro j _ h
    simp [Finsupp.single_apply, h, Ne.symm h, paths_zero]
  · simp

/-- For a fixed field-column index i and offset d the slide has exactly one possible
source index i+d. These overlap limits work for either ordering of the widths. -/
theorem slide_bounds (P Q i d : ℤ) :
    (max 0 (-d) ≤ i ∧ i < min Q (P-d)) ↔
      (0 ≤ i ∧ i < Q ∧ 0 ≤ i+d ∧ i+d < P) := by
  rw [max_le_iff, lt_min_iff]
  omega

/-- The coefficients actually stored for a Taylor step. -/
def normalized (F : Fin m → Poly m) (n : ℕ) (l : Fin m) : Poly m :=
  C (n.factorial : ℝ)⁻¹ * jet F n l

theorem normalized_succ (F : Fin m → Poly m) (n : ℕ) (l : Fin m) :
    normalized F (n+1) l = C ((n+1 : ℕ) : ℝ)⁻¹ * lie F (normalized F n l) := by
  unfold normalized
  rw [jet, lie_mul, show lie F (C (n.factorial : ℝ)⁻¹) = 0 by simp [lie, pderiv_C]]
  simp only [zero_mul, zero_add, Nat.factorial_succ, Nat.cast_mul, mul_inv_rev, map_mul]
  ring

def SolvesOn (F : Fin m → Poly m) (x : ℝ → Fin m → ℝ) (s : Set ℝ) : Prop :=
  ∀ t ∈ s, ∀ j, HasDerivWithinAt (fun u => x u j) (eval (x t) (F j)) s t

theorem along_lie {F : Fin m → Poly m} {x : ℝ → Fin m → ℝ} {s : Set ℝ}
    (hx : SolvesOn F x s) (P : Poly m) {t : ℝ} (ht : t ∈ s) :
    HasDerivWithinAt (fun u => eval (x u) P) (eval (x t) (lie F P)) s t := by
  induction P using MvPolynomial.induction_on with
  | C a => simpa [lie, pderiv_C] using (hasDerivWithinAt_const t s a)
  | add P Q ih jh => simpa [lie_add] using ih.add jh
  | mul_X P j ih => simpa [lie_mul, lie_X] using ih.mul (hx t ht j)

theorem jet_derivative {F : Fin m → Poly m} {x : ℝ → Fin m → ℝ} {s : Set ℝ}
    (hx : SolvesOn F x s) (hs : UniqueDiffOn ℝ s) (n : ℕ) (l : Fin m) :
    EqOn (iteratedDerivWithin n (fun u => x u l) s) (fun u => eval (x u) (jet F n l)) s := by
  induction n with
  | zero => intro t _; simp [jet]
  | succ n ih =>
    intro t ht
    rw [iteratedDerivWithin_succ, derivWithin_congr ih (ih ht)]
    exact (along_lie hx (jet F n l) ht).derivWithin (hs t ht)

theorem jet_differentiable {F : Fin m → Poly m} {x : ℝ → Fin m → ℝ} {s : Set ℝ}
    (hx : SolvesOn F x s) (hs : UniqueDiffOn ℝ s) (n : ℕ) (l : Fin m) :
    DifferentiableOn ℝ (iteratedDerivWithin n (fun u => x u l) s) s :=
  by
    have h : DifferentiableOn ℝ (fun u => eval (x u) (jet F n l)) s :=
      fun t ht => (along_lie hx (jet F n l) ht).differentiableWithinAt
    exact DifferentiableOn.congr h (jet_derivative hx hs n l)

end Nomad
