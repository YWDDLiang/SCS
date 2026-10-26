# Scheduling Definition

SCS operates on a raw training-event microbatch $\mathcal{B}_t$ at stage $t$.
For each event $e_i\in\mathcal{B}_t$, the scheduler computes a
representation-spectrum novelty score $\nu_t(e_i)$ from the current model
representation and the online second-moment state, then standardizes scores
within $\mathcal{B}_t$ to obtain $\widetilde{\nu}_t(e_i)$. Let $c(e_i)$ be the
number of distinct users in the original
training split with the same adjacent transition, represented by complete
item semantic identifiers. The coverage-qualified pool is

$$
\mathcal{C}_t = \{e_i\in\mathcal{B}_t : c(e_i) \geq c_{\min}\}.
$$

The spectrum-priority distribution combines the novelty score, the pool, and
the experimental multiplier for events that admit an alternative history:

$$
q_t(i) = \frac{\exp(\widetilde{\nu}_t(e_i)/\tau)\,\mathbf{1}[e_i\in\mathcal{C}_t]\,w_t(i)}
{\sum_j \exp(\widetilde{\nu}_t(e_j)/\tau)\,\mathbf{1}[e_j\in\mathcal{C}_t]\,w_t(j)},
\qquad
w_t(i)=\begin{cases}4,& e_i\text{ admits a same-length alternative history},\\1,&\text{otherwise.}\end{cases}
$$

Both exact repeats and history views draw event identities from $q_t$. Within
the fixed history-view quota, the output-head gradient-conflict signal adjusts
the relative probability of choosing a view. History views preserve the
prediction target, visible history length, latest history item, and temporal
order. Every repeat or view consumes one unit from the additional-presentation
budget; original base events remain in training.

## Reported Configuration

- Minimum transition coverage $c_{\min}=2$ distinct original-training users.
- Warm-up: 200 optimizer updates using the base stream only.
- Additional-presentation budget: 9,757,428 (Toys), 13,434,229 (Beauty), and
  6,742,922 (Sports).
- History-view quota: 75% of additional presentations.
- Spectrum sketch dimension: 32; second-moment decay: 0.99; ridge: 0.01.
- Conflict-reference decay: 0.90.
- Seeds: model 42, spectrum 20261020, history views 45519.

Coverage is computed from the original training sequences at dataloader
initialization. Repeated instances of a transition within one user count once;
the event's own user is included. Validation and test sequences do not enter
this calculation. View readiness is incorporated directly into $q_t$ and does
not define a separate event-selection stage.
