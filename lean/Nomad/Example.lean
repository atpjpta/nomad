import Nomad.Bounds

/-! A concrete replay of the quadratic-flow example in the companion paper. -/
noncomputable section
open MvPolynomial Set
namespace Nomad.Example

private def field (_ : Fin 1) : Poly 1 := X 0 ^ 2

private theorem square_normalized (n : ℕ) : normalized field n 0 = X 0 ^ (n+1) := by
  induction n with
  | zero => simp [normalized, jet]
  | succ n ih =>
    rw [normalized_succ, ih]
    simp only [lie, Fin.sum_univ_one, field, pderiv_pow, pderiv_X_self, mul_one,
      Nat.add_sub_cancel]
    rw [← map_natCast (C : ℝ →+* Poly 1) (n+1)]
    rw [← mul_assoc, ← mul_assoc, ← map_mul, inv_mul_cancel₀ (by positivity), map_one, one_mul]
    rw [← pow_add]

/-- The values in the worked PyTorch example, with x(a)=1, h=1/8 and p=2.
The box radius is 1/2. All solution coordinates reduce to the one scalar. -/
theorem quadratic_certificate {x : ℝ → Fin 1 → ℝ}
    (hx : SolvesOn field x (Icc 0 (1/8))) (hz : x 0 0 = 1) :
    13/131072 ≤ x (1/8) 0 - 73/64 ∧ x (1/8) 0 - 73/64 ≤ 499/131072 := by
  have hbox := stays_in_box hx (fun _ => 1/2) (fun _ => 9/4) (by intros; norm_num)
    (by intros; norm_num) (by
      intro t _ ht j
      fin_cases j
      have hi := ht 0
      rw [hz] at hi
      have hb : |x t 0| ≤ 3/2 := by linarith [abs_sub_abs_le_abs_sub (x t 0) 1, abs_one (α := ℝ)]
      simpa [field, abs_pow] using pow_le_pow_left₀ (abs_nonneg _) hb 2 |>.trans (by norm_num : (3/2:ℝ)^2 ≤ 9/4))
  have h := consecutive (by norm_num : (0:ℝ)<1/8) hx 0 2 (B := (3/2)^5) (by
    intro t ht
    have hi := hbox t ht 0
    rw [hz] at hi
    have hb : |x t 0| ≤ 3/2 := by linarith [abs_sub_abs_le_abs_sub (x t 0) 1, abs_one (α := ℝ)]
    rw [square_normalized]
    simpa [abs_pow] using pow_le_pow_left₀ (abs_nonneg _) hb 5)
  norm_num [step, term, Finset.sum_range_succ, square_normalized, hz] at h
  obtain ⟨hl, hu⟩ := abs_le.mp h
  constructor <;> linarith

end Nomad.Example
