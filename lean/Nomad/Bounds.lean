import Nomad.Polynomial
import Mathlib.Tactic.Linarith

/-! Local consecutive-order error enclosures for scalar components of a polynomial ODE. -/
noncomputable section
open MvPolynomial Set
open scoped BigOperators
namespace Nomad

variable {m : ℕ}

def term (F : Fin m → Poly m) (z : Fin m → ℝ) (h : ℝ) (l : Fin m) (n : ℕ) : ℝ :=
  eval z (normalized F n l) * h ^ n

def step (F : Fin m → Poly m) (z : Fin m → ℝ) (h : ℝ) (l : Fin m) (p : ℕ) : ℝ :=
  ∑ k ∈ Finset.range (p+1), term F z h l k

theorem step_succ (F : Fin m → Poly m) (z : Fin m → ℝ) (h : ℝ) (l : Fin m) (p : ℕ) :
    step F z h l (p+1) = step F z h l p + term F z h l (p+1) := by
  exact Finset.sum_range_succ _ _

theorem step_eq_taylor {F : Fin m → Poly m} {x : ℝ → Fin m → ℝ} {s : Set ℝ}
    (hx : SolvesOn F x s) (hs : UniqueDiffOn ℝ s) {a : ℝ} (ha : a ∈ s)
    (b : ℝ) (l : Fin m) (p : ℕ) :
    step F (x a) (b-a) l p = taylorWithinEval (fun t => x t l) p s a b := by
  rw [taylor_within_apply]
  apply Finset.sum_congr rfl
  intro k _
  rw [jet_derivative hx hs k l ha]
  simp [term, normalized, mul_assoc, mul_left_comm, mul_comm]

/-- The highest evaluated polynomial is normalized, so B already includes the factorial. -/
theorem local_remainder {F : Fin m → Poly m} {x : ℝ → Fin m → ℝ} {a b B : ℝ}
    (hab : a < b) (hx : SolvesOn F x (Icc a b)) (l : Fin m) (p : ℕ)
    (hB : ∀ t ∈ Icc a b, |eval (x t) (normalized F (p+1) l)| ≤ B) :
    |x b l - step F (x a) (b-a) l p| ≤ B * (b-a)^(p+1) := by
  have hs := uniqueDiffOn_Icc hab
  have hf : ContDiffOn ℝ p (fun t => x t l) (Icc a b) := by
    apply contDiffOn_of_differentiableOn_deriv
    exact fun n _ => jet_differentiable hx hs n l
  obtain ⟨t, ht, he⟩ := taylor_mean_remainder_lagrange hab hf
    ((jet_differentiable hx hs p l).mono Ioo_subset_Icc_self)
  have ht' : t ∈ Icc a b := Ioo_subset_Icc_self ht
  rw [step_eq_taylor hx hs (left_mem_Icc.mpr hab.le), he, jet_derivative hx hs _ l ht']
  have hr : eval (x t) (jet F (p+1) l) * (b-a)^(p+1) / ((p+1).factorial : ℝ) =
      eval (x t) (normalized F (p+1) l) * (b-a)^(p+1) := by
    simp [normalized, div_eq_mul_inv, mul_assoc, mul_left_comm, mul_comm]
  rw [hr, abs_mul, abs_of_nonneg (pow_nonneg (sub_nonneg.mpr hab.le) _)]
  exact mul_le_mul_of_nonneg_right (hB t ht') (pow_nonneg (sub_nonneg.mpr hab.le) _)

/-- A computable componentwise bound for any table on an axis-aligned box. -/
def majorant {w : ℕ} (k : Fin w → Exponent m) (c : Fin w → ℝ) (ρ : Fin m → ℝ) : ℝ :=
  ∑ i, |c i| * (k i).prod (fun j e => ρ j ^ e)

theorem table_bound {w : ℕ} (k : Fin w → Exponent m) (c : Fin w → ℝ)
    {z ρ : Fin m → ℝ} (hz : ∀ j, |z j| ≤ ρ j) :
    |eval z (table k c)| ≤ majorant k c ρ := by
  simp only [table, majorant, map_sum, eval_monomial, Finsupp.prod]
  apply (Finset.abs_sum_le_sum_abs _ _).trans
  apply Finset.sum_le_sum
  intro i _
  rw [abs_mul, Finset.abs_prod]
  apply mul_le_mul_of_nonneg_left _ (abs_nonneg _)
  apply Finset.prod_le_prod (fun j _ => abs_nonneg _)
  intro j _
  rw [abs_pow]
  exact pow_le_pow_left₀ (abs_nonneg _) (hz j) _

/-- Sharp signed enclosure from orders p, p+1 and a bound for p+2. -/
theorem consecutive {F : Fin m → Poly m} {x : ℝ → Fin m → ℝ} {a b B : ℝ}
    (hab : a < b) (hx : SolvesOn F x (Icc a b)) (l : Fin m) (p : ℕ)
    (hB : ∀ t ∈ Icc a b, |eval (x t) (normalized F (p+2) l)| ≤ B) :
    |(x b l - step F (x a) (b-a) l p) - term F (x a) (b-a) l (p+1)| ≤
      B * (b-a)^(p+2) := by
  have h := local_remainder hab hx l (p+1) hB
  rwa [step_succ, ← sub_sub] at h

/-- Add any finite block of consecutive higher Taylor terms. -/
theorem step_add (F : Fin m → Poly m) (z : Fin m → ℝ) (h : ℝ) (l : Fin m) (p k : ℕ) :
    step F z h l (p+k) = step F z h l p +
      ∑ i ∈ Finset.range k, term F z h l (p+1+i) := by
  induction k with
  | zero => simp
  | succ k ih =>
    rw [Nat.add_succ, step_succ, ih, Finset.sum_range_succ, add_assoc]
    rw [show p+k+1 = p+1+k by omega]

theorem block_enclosure {F : Fin m → Poly m} {x : ℝ → Fin m → ℝ} {a b B : ℝ}
    (hab : a < b) (hx : SolvesOn F x (Icc a b)) (l : Fin m) (p k : ℕ)
    (hB : ∀ t ∈ Icc a b, |eval (x t) (normalized F (p+k+1) l)| ≤ B) :
    |(x b l - step F (x a) (b-a) l p) -
      ∑ i ∈ Finset.range k, term F (x a) (b-a) l (p+1+i)| ≤ B * (b-a)^(p+k+1) := by
  have h := local_remainder hab hx l (p+k) hB
  rwa [step_add, ← sub_sub] at h

/-- A first-exit estimate for a solution on the step. Speed bounds are only
required at points inside the box. -/
theorem stays_in_box {F : Fin m → Poly m} {x : ℝ → Fin m → ℝ} {a b : ℝ}
    (hx : SolvesOn F x (Icc a b)) (r M : Fin m → ℝ) (hM : ∀ j, 0 ≤ M j)
    (hstep : ∀ j, M j * (b-a) < r j)
    (hspeed : ∀ t ∈ Icc a b, (∀ j, |x t j - x a j| < r j) →
      ∀ j, |eval (x t) (F j)| ≤ M j) :
    ∀ t ∈ Icc a b, ∀ j, |x t j - x a j| < r j := by
  classical
  let bad (j : Fin m) : Set ℝ := {t ∈ Icc a b | r j ≤ |x t j - x a j|}
  have compactBad : IsCompact (⋃ j, bad j) := by
    apply isCompact_iUnion
    intro j
    have hc : ContinuousOn (fun t => x t j) (Icc a b) :=
      fun t ht => (hx t ht j).continuousWithinAt
    exact isCompact_Icc.of_isClosed_subset
      (isClosed_Icc.isClosed_le continuousOn_const (hc.sub continuousOn_const).abs)
      (fun _ ht => ht.1)
  intro t ht j
  by_contra! hj
  obtain ⟨u, hu, hfirst⟩ := compactBad.exists_isLeast
    ⟨t, mem_iUnion.mpr ⟨j, ht, hj⟩⟩
  obtain ⟨l, hul, hboundary⟩ := mem_iUnion.mp hu
  have hsub : Icc a u ⊆ Icc a b := Icc_subset_Icc_right hul.2
  have hbefore : ∀ v ∈ Ico a u, ∀ i, |x v i - x a i| < r i := by
    intro v hv i
    by_contra! hi
    exact (not_le.mpr hv.2) (hfirst (mem_iUnion.mpr ⟨i, hsub (Ico_subset_Icc_self hv), hi⟩))
  have hdisp := norm_image_sub_le_of_norm_deriv_le_segment'
    (fun v hv => (hx v (hsub hv) l).mono hsub)
    (fun v hv => (Real.norm_eq_abs _).trans_le
      (hspeed v (hsub (Ico_subset_Icc_self hv)) (hbefore v hv) l))
    u (right_mem_Icc.mpr hul.1)
  rw [Real.norm_eq_abs] at hdisp
  exact (not_lt.mpr hboundary) (hdisp.trans_lt
    ((mul_le_mul_of_nonneg_left (sub_le_sub_right hul.2 a) (hM l)).trans_lt (hstep l)))

/-- End-to-end bound from the input field table, the next normalized table and
the box radii. No derivative bound along the unknown trajectory is an input. -/
theorem boxed_consecutive {w v : ℕ} (k : Fin w → Exponent m) (A : Fin m → Fin w → ℝ)
    {x : ℝ → Fin m → ℝ} {a b : ℝ} (hab : a < b)
    (hx : SolvesOn (fun j => table k (A j)) x (Icc a b))
    (r : Fin m → ℝ) (hr : ∀ j, 0 < r j) (l : Fin m) (p : ℕ)
    (q : Fin v → Exponent m) (c : Fin v → ℝ)
    (hnext : normalized (fun j => table k (A j)) (p+2) l = table q c)
    (hstep : ∀ j, majorant k (A j) (fun i => |x a i| + r i) * (b-a) < r j) :
    |(x b l - step (fun j => table k (A j)) (x a) (b-a) l p) -
      term (fun j => table k (A j)) (x a) (b-a) l (p+1)| ≤
        majorant q c (fun i => |x a i| + r i) * (b-a)^(p+2) := by
  have hin {t : ℝ} (h : ∀ i, |x t i - x a i| < r i) (i : Fin m) :
      |x t i| ≤ |x a i| + r i := by
    linarith [abs_sub_abs_le_abs_sub (x t i) (x a i), h i]
  have hcenter (i : Fin m) : |x a i| ≤ |x a i| + r i := le_add_of_nonneg_right (hr i).le
  have hbox := stays_in_box hx r _ (fun i =>
    (abs_nonneg _).trans (table_bound k (A i) hcenter)) hstep
    (fun _ _ ht i => table_bound k (A i) (hin ht))
  apply consecutive hab hx l p
  intro t ht
  rw [hnext]
  exact table_bound q c (hin (hbox t ht))

theorem absolute_enclosure {e d R : ℝ} (h : |e-d| ≤ R) :
    max 0 (|d|-R) ≤ |e| ∧ |e| ≤ |d|+R := by
  have h₁ := abs_sub_abs_le_abs_sub d e
  have h₂ := abs_sub_abs_le_abs_sub e d
  rw [abs_sub_comm d e] at h₁
  exact ⟨max_le (abs_nonneg _) (by linarith), by linarith⟩

/-- Transfer the exact higher-order bound to the actual returned low-order value.
D is the observed increment and (T - Yhi) is the signed rounding correction. -/
theorem rounded_enclosure {x T Ylo Yhi R : ℝ} (h : |x-T| ≤ R) :
    |(x-Ylo) - ((Yhi-Ylo) + (T-Yhi))| ≤ R := by
  convert h using 2
  ring

end Nomad
