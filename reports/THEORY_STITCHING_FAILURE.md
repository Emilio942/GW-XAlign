# Theoretical Appendix: Why Zero-Shot GW Stitching Fails Functionally

*Mathematical account of Exp N (functional cliff) and its reconciliation with Exp M (structural transfer).*

---

## 1. Setup and notation

Let A, B be autoregressive transformers with L_A, L_B decoder blocks. For input
tokens t = (t_1..t_T), model M produces hidden states

    h_M^{(i)}(t) ∈ R^{d_M},   i = 0..L_M,   h^{(0)} = embedding,

and a task head. Define the *tail map* of B at depth i,

    B_{>i} := Head ∘ F_{L_B} ∘ ... ∘ F_{i+1},

where F_i is the i-th decoder block (attention + MLP + residual update).
Write the two state distributions (token clouds, pooled or token-level) as

    μ_A^i = (h_A^{(i)})_# ν,   μ_B^i = (h_B^{(i)})_# ν,

for a data distribution ν over inputs, with empirical versions μ̂_A^i, μ̂_B^i.

**Stitching operator.** The stitched model S_{A→B}^{(i)} runs A up to layer i,
applies a transport map T: R^{d_A} → R^{d_B}, and continues with B's tail:

    S(x) = B_{>i}( T( h_A^{(i)}(x) ) ).

**Functional target.** Stitching is successful iff

    (F)   S(x) ≈ B(x)  for all x ~ ν,
    i.e.  B_{>i}( T(h_A^{(i)}(x)) ) ≈ B_{>i}( h_B^{(i)}(x) ).

## 2. What the GW problem actually determines

The entropic GW problem is

    J(T) = ∑_{i,j,k,l} | d_A(x_i,x_k) − d_B(y_j,y_l) |² T_ij T_kl − ε H(T),
    γ* = argmin_{T ∈ Π(a,b)} J(T),

and the barycentric map is

    T̂(x) = argmin_{y ∈ R^{d_B}} ∑_j w'_j(x) ‖y − y_j‖²,
    w'_j(x) = ∑_i (γ*_ij / a_i) κ(x, x_i).

Note what enters J: **only the intra-cloud distance matrices** D_A ∈ R^{N×N},
D_B ∈ R^{N×N}. Coordinates, bases, orientations, scales of individual states
do not appear anywhere in the problem's data.

## 3. The gauge obstruction (core theorem)

**Gauge group.** Let G be the group of isometries of (R^{d_B}, ‖·‖):

    G = { y ↦ U y + c : U ∈ O(d_B), c ∈ R^{d_B} }.

**Proposition 1 (G-invariance of GW).**
For every g ∈ G acting on the B-side marginals (y_j ↦ g(y_j)):

    J(g·T) = J(T).

*Proof.* d_B(g y_j, g y_l) = ‖U(y_j − y_l)‖ = ‖y_j − y_l‖ = d_B(y_j, y_l).
Every summand of J is unchanged; the transport polytope Π(a,b) is unchanged
(marginals are measures on points, and the points move with g). ∎

Consequently the minimiser is determined, at best, as a **G-orbit**:

    γ* is unique (entropy ⇒ strict convexity in T),
    but T̂ is determined only modulo the orbit { g ∘ T̂ : g ∈ G }.

**Proposition 2 (functional risk is not G-invariant).**
For a generic target model B (generic weight matrices W), the functional risk

    R(T) = E_{x∼ν} ℓ( B_{>i}( T(h_A^{(i)}(x)) ), f*(x) )

is *not* invariant under G: there exist g₁, g₂ ∈ G with R(g₁·T̂) ≠ R(g₂·T̂).

*Proof sketch.* Take d_A = d_B and suppose the idealised case that A and B are
exactly geometrically isometric at depth i, i.e. h_B^{(i)} = U₀ h_A^{(i)} for a
fixed U₀ ∈ O(d_B) (any harder case inherits the argument). Then the map
T = U₀ has R(T) = 0 (perfect stitching). But for a generic V ∈ O(d_B), V ≠ U₀:

    B_{>i}( V h_B^{(i)} ) ≠ B_{>i}( h_B^{(i)} )

because B's tail contains generic matrices W; the set
{U ∈ O(d_B) : W_{L}···W_{i+1} U = W_{L}···W_{i+1}} is the stabiliser of the
tail, which for generic W is finite (dimension 0), while dim O(d_B) =
d_B(d_B−1)/2 → ∞. Hence the G-orbit of T̂ has positive dimension and
R is non-constant on it, varying by O(1) in task units. ∎

**Corollary 1 (information deficit).**
The GW problem's input data (D_A, D_B) determine the solution only in the
quotient ℳ/G, while (F) requires a specific representative of the coset.
The information identifying *which* representative is B's actual encoding is
**not present in the GW problem**. Zero-shot GW minimisation therefore cannot,
even with infinite samples and exact optimisation, select a functional map.
This is an identifiability failure, not an optimisation failure.

## 4. Why stitched ≈ shuffled (Exp N), quantitatively

Both maps push the source cloud into (approximately) the same *pushforward
measure* on B-space: the barycentric image of the cloud under γ*. They differ
only in the *assignment* (which target point each source token receives).

For B's forward pass, an injected token state enters (a) its own residual
update and (b) other tokens' attention. Under both maps:

    { T̂(x_t) }_t  and  { (g∘T̂)(x_t) }_t = { T̂(x_t) } up to per-token re-labelling
    within the same cloud, with identical pairwise geometry,

so the *distribution* of injected states is the same and attention-weighted
aggregations agree in distribution to first order. Hence

    R(shuffled) − R(stitched) → 0 (second order in the assignment),

which is exactly what was measured: mean stitched/ceiling = 0.62,
shuffled/ceiling = 0.63, stitched better in 9/35 ≈ coin flip.

**Why ~0.62 and not 0?** The injected states are convex combinations of
B-typical states (barycentric map ⇒ image ⊂ conv(supp μ̂_B)), so they remain
in-distribution in the coarse sense: B's downstream layers still receive
"language-model-like" states and produce language-model-like continuations —
just with the token-conditional content scrambled. A shuffled map preserves
the scrambling; both deliver the same (partial) marginal information. Performance
above chance and below ceiling is the signature of *marginal information
without correspondence information*.

## 5. Why the structural results (Exp M) are consistent

Neighbourhood-based metrics are **G-invariant**. For any g ∈ G:

    y' ∈ N_k(g·T̂(x))  ⟺  T̂(x') with x' ∈ N_k(x)   (isometry preserves balls),

so kNN class transfer, correlation of distance profiles, and cluster structure
are unaffected by which orbit representative we picked. Exp M's high structural
transfer (0.82–0.88) and Exp N's functional failure are therefore *the same
fact seen twice*: the map is a good isometric embedding modulo gauge, and
function is gauge-dependent.

## 6. What breaks the gauge: two principled repairs

**(a) Anchors (closed form).** With k ≥ d_B paired states
(H_A ∈ R^{k×d_A}, H_B ∈ R^{k×d_B} of *corresponding* tokens), select the orbit
representative by orthogonal Procrustes:

    U* = argmin_{U ∈ O(d_B)} ‖U H_A − H_B‖_F²
       = argmax_{U ∈ O(d_B)} tr(U H_A H_Bᵀ)
       = V Wᵀ   where  H_A H_Bᵀ = W Σ Vᵀ   (SVD).

U* exists in closed form, costs O(k d²), and with generic anchors the gauge is
fixed exactly. The stitching map becomes T = U* ∘ T̂. This is why the stitching
literature succeeds *with* a (small) trained connector: training on k anchors
solves the Procrustes problem (plus the linear-tail readout) by gradient descent.

**(b) Trained connector as our repair.** Minimising R directly over a
connector W_c:

    min_{W_c} E ℓ( B_{>i}( W_c h_A^{(i)}(x) ), f*(x) )

is a convex problem if the tail were linear, and practically a small
well-conditioned fit otherwise. The GW map serves as *initialisation in the
correct pushforward region* — it fixes the marginal (scale, anisotropy,
support), which is the part of the information GW *does* determine, and leaves
the gauge to be learned from k ≪ N anchors. Prediction: GW-init + light
connector training ≥ random-init + same training, with the gap shrinking as k
grows. (Untested here — the indicated next experiment.)

## 7. Summary table: measurement ↔ theory

| Measurement | Theory entity | Result |
|---|---|---|
| EGW ≈ 0 on self-pair (Exp M) | J(T) minimal on identical spaces | ✔ |
| kNN transfer 0.82–0.88 (Exp M) | G-invariant neighbourhood structure | ✔ passes |
| stitched = shuffled, 9/35 (Exp N) | same G-orbit ⇒ same risk | ✔ passes |
| stitched/ceiling ≈ 0.62 ≫ 0 | marginal information preserved (barycentric image) | ✔ |
| stitched/ceiling ≪ 1 | gauge not resolved ⇒ functional content lost | ✔ |
| ρ(EGW, acc) weak across families (Exp L/M) | J sees only distance geometry, not head/task readout | ✔ |

## 8. One-sentence conclusion

The GW problem is a *relaxation of the stitching problem that quotients out
exactly the information the stitching problem needs* (the O(d_B)-gauge); its
solution is provably determined only up to that gauge, the functional objective
varies on the gauge orbit, and therefore zero-shot GW stitching fails
identifiably — with the shuffled control as the empirical image of the orbit.
