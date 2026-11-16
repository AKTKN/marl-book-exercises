## Diversity methods solving bottleneck of the model-free
In sarsa, making $\pi$ fully greedy and deterministic would violate trying all combination infinitely often.
When the transition dynamics $T$ and reward function $R$ are unknown, unvisited state-action pairs retain inaccurate initial value estimates,
which results in selecting the already-observed sub-optimal action repeatedly for greedy policy improvement.

Besides $\epsilon$-greedy policies, initializing an ensemble of $K$ diverse deterministic policies $\Pi_0 = \{\pi_0^{(1)}, \dots, \pi_0^{(K)}\}$ broadens state-action coverage compared to a single policy. 
However, random multi-start initialization does not scale efficiently: if discovering a reward requires a specific sequence of $H$ actions out of $m$ choices per state, a single random policy succeeds with probability $p = m^{-H}$, requiring $K = O(m^H)$ policies to achieve a constant probability of success. 
Efficiency thus depends heavily on how policy diversity is maintained.

**Related Literature**
* **Monte Carlo Control with Exploring Starts (Sutton & Barto, 2018, Sec. 5.3):** The most direct classical solution to this problem. It retains a deterministic greedy policy while sampling initial state-action pairs $(s_0, a_0) \sim \rho$ ($\rho(s,a) > 0$) to guarantee sufficient coverage without $\epsilon$-greedy randomization.
* **Bootstrapped DQN (Osband et al., NeurIPS 2016):** Directly aligns with ensembling by maintaining $K$ value function estimators ($\widehat{Q}_1, \dots, \widehat{Q}_K$). Sampling one estimator per episode drives temporally extended ("deep") exploration.
* **Generalized Policy Improvement (Barreto et al., NeurIPS 2017):** Combines evaluations across $K$ policies via $\pi_{\mathrm{new}}(s) \in \arg\max_a \max_{1 \le j \le K} Q^{\pi_j}(s,a)$, guaranteeing $V^{\pi_{\mathrm{new}}}(s) \ge \max_j V^{\pi_j}(s)$ for all $s \in S$.

---
## Stopping criteria and Policy Evaluation in Modern RL

In modern reinforcement learning (RL) research, the standard practice is to run training for a fixed number of environment steps while periodically evaluating the policy along the way. 
Relying solely on TD errors or the magnitude of changes in $Q$ to determine convergence is generally not used as a stopping criterion, 
especially in large-scale environments. 
This is because value function convergence and policy performance do not necessarily improve at the same rate.

#### 1. Convergence Detection Using $Q$ and TD Errors
Defining the TD error in Sarsa as
$$\delta_t = r_t + \gamma Q_t(s_{t+1}, a_{t+1}) - Q_t(s_t, a_t),$$
the update magnitude is given by
$$\Delta Q_t(s_t, a_t) = \alpha_t \delta_t.$$
Two candidate stopping criteria are:

$$\vert{}\Delta Q_t(s_t, a_t)\vert{} < \eta \quad \text{or} \quad \vert{}\delta_t\vert{} < \eta.$$

However, neither criterion is sufficient on its own to judge convergence:

1. **Decaying Learning Rates:** If the learning rate $\alpha_t$ decays over time, $\vert{}\Delta Q_t(s_t, a_t)\vert{} \to 0$ can occur as $\alpha_t \to 0$ even while the TD error remains large.
2. **Stochastic Environments:** In a stochastic MDP, individual TD errors generally do not vanish even after reaching the true action-value function. For a fixed policy $\pi$, even when $Q = Q^\pi$ holds, we only have

$$\mathbb{E}[\delta_t \mid s_t = s, a_t = a] = 0,$$

while the variance may remain strictly positive:

$$\operatorname{Var}(\delta_t \mid s_t = s, a_t = a) > 0.$$

In other words, what should approach zero is not the individual TD error, but the conditional expected TD error.
If the environment model is known, one can use the Bellman optimality residual with respect to the Bellman optimality operator $\mathcal{T}^*$:

$$B(Q) = \Vert{}\mathcal{T}^* Q - Q\Vert{}_\infty.$$

In a finite discounted MDP, the contraction property guarantees

$$\Vert{}Q - Q^*\Vert{}_\infty \leq \frac{B(Q)}{1 - \gamma},$$

meaning that $B(Q) \leq (1 - \gamma)\eta$ bounds the value function error within $\eta$. 
However, in model-free Sarsa, $\mathcal{T}^*$ cannot be computed directly, and sample-based residual estimates introduce statistical error. 
Consequently, practitioners evaluate the learned policy itself rather than tracking update magnitudes in $Q$.

#### 2. Stopping and Evaluation Criteria Used in Practice
Commonly adopted criteria can be summarized as follows:

| Method | Measured Quantity | Primary Use Case |
| --- | --- | --- |
| **Fixed training budget** | Cumulative environment steps | Research experiments & benchmarks |
| **Performance threshold** | Mean evaluation return | Reaching a target performance level |
| **Performance plateau** | Lack of improvement in evaluation return over a set period | Early stopping |
| **Changes in $Q$ / Bellman residual** | Value function convergence metrics | Small-scale tabular RL & theoretical verification |

During policy evaluation, a greedy policy is extracted from the current estimate at time $t$:
$$\hat{\pi}_t(s) \in \arg\max_a Q_t(s, a),$$
and $N$ evaluation episodes are executed independently of training. Letting $G_t^{(j)}$ denote the discounted return of the $j$-th evaluation episode, performance is measured by

$$\widehat{J}_t = \frac{1}{N} \sum_{j=1}^{N} G_t^{(j)}.$$

Even if $\epsilon$-greedy exploration is used during training, exploration is turned off during evaluation when assessing the performance of the final greedy policy. 
This approach avoids sweeping over the entire state-action space; if the average episode length is $H$, the interaction complexity is $O(NH)$. For example, Stable-Baselines3 implements `EvalCallback` alongside `StopTrainingOnRewardThreshold` (which halts training once evaluation return exceeds a threshold) and `StopTrainingOnNoModelImprovement` (which halts training if no improvement is seen across a specified number of evaluations).

#### 3. Scaling to Large State-Action Spaces

In tabular Sarsa, maintaining $Q \in \mathbb{R}^{\vert{}S\vert{} \times \vert{}A\vert{}}$ requires $O(\vert{}S\vert{}\vert{}A\vert{})$ memory, and checking updates across all state-action pairs at every step becomes intractable in large environments.

In modern deep RL, the action-value function is approximated using a parameterized function:


$$Q_\theta(s, a) \approx Q^\pi(s, a).$$


This eliminates the explicit Q-table, making memory usage primarily dependent on the number of network parameters. Here again, continuously tracking the evaluation return of the learned policy is standard practice rather than inspecting changes in individual $Q_\theta(s, a)$ values. Note, however, that evaluation return measures


$$J(\pi) = \mathbb{E}_{s_0 \sim \mu}[V^\pi(s_0)]$$


with respect to the initial state distribution $\mu$ used during evaluation, and does not guarantee that $Q_\theta(s, a)$ is accurate across all states.

#### 4. Typical Training Step Budgets

The required number of training steps depends on the scale of the environment and the algorithm, with no universal standard. Representative budgets from published implementations include:

| Environment / Algorithm | Example Training Budget |
| --- | --- |
| Mars Rover (Textbook Section 2.7) | $500$ steps |
| CleanRL PPO (MuJoCo) | $10^6$ steps |
| CleanRL DQN (Atari) | $10^7$ steps |
| Original DQN (Atari; Mnih et al., 2015) | $5 \times 10^7$ agent steps |

These figures represent standardized evaluation budgets rather than guarantees of convergence to an optimal policy, and cannot be directly compared across different environments.

To ensure fair comparisons across algorithms on the same environment, researchers allocate an identical budget of environment steps and compare the resulting learning curves (similarly, Section 2.7 of the textbook plots cumulative environment steps rather than episode counts on the horizontal axis). Furthermore, to account for stochasticity in training, experiments are run across multiple random seeds and reported with confidence intervals. Agarwal et al. (NeurIPS 2021) analyzed the statistical instability of evaluating RL with few seeds and recommended comparison protocols using metrics such as stratified bootstrap confidence intervals.

#### 5. Configuration for the 5-State Sarsa Simulator

For our 5-state robot environment, the following configuration is sufficient as a starting point. Note that these parameters are intended for observing learning curves rather than providing formal convergence guarantees:

| Parameter | Recommended Setting |
| --- | --- |
| **Max training steps** | $5,000$ |
| **Evaluation interval** | $250$ steps |
| **Evaluation policy** | Greedy ($\epsilon_{\mathrm{eval}} = 0$) |
| **Early stopping** | No improvement in evaluation return for $5$ consecutive evaluations |
| **Independent training seeds** | $5\text{--}10$ |

Because both transitions and rewards in this environment are deterministic, repeated rollouts of the same greedy policy from a fixed initial state are unnecessary. Instead, evaluating the policy starting from every non-terminal state provides a much clearer picture of policy improvement across the state space.

Moreover, because we have access to the exact environment model in our implementation, we can withhold the model from the learning agent while using value iteration offline to compute the exact optimal $Q^*$ purely for evaluation. This allows us to directly track both


$$E_t = \Vert{}Q_t - Q^*\Vert{}_\infty \quad \text{and} \quad J(\hat{\pi}_t) - J(\pi^*),$$


separating value function convergence from policy optimization.

Finally, note that our previous simulator held $\alpha$ and $\epsilon$ constant. Sarsa with a fixed $\epsilon > 0$ does not generally converge to $Q^*$, and a constant learning rate leaves persistent variance in the value estimates. To observe theoretical convergence to $Q^*$, one must use a GLIE (Greedy in the Limit with Infinite Exploration) exploration schedule alongside a learning rate satisfying the Robbins-Monro conditions (Equation 2.54). At this experimental scale, logging the estimation error of $Q_t$, the evaluation return of the greedy policy, and the action-matching rate against the optimal policy offers the most direct way to verify Sarsa's learning dynamics.

---

### Mechanism Analysis and Standard Evaluation Protocols in Reinforcement Learning

While the general direction of your hypothesis is sound, the observed results are more accurately explained by **state-action visitation frequencies driven by exploration** and the **rate of information acquisition required for value estimation**, rather than "escaping local optima."

Furthermore, while your current experiment effectively measures the time to reach an optimal policy, that metric alone cannot evaluate the overall performance of a learning algorithm. In RL research, algorithms are typically evaluated across four core dimensions: **online performance during training**, **sample efficiency**, **final performance**, and **statistical reproducibility**.

#### 1. Mechanism Behind the Observed Results

The implemented environment is a one-dimensional MDP:


$$s_1 \;-\; s_2 \;-\; s_3 \;-\; s_4 \;-\; s_5$$


where $s_1$ and $s_5$ are terminal states with rewards $+1$ and $+5$, respectively, and $s_3$ is the initial state.

When $\epsilon$ is small, if early greedy actions select the path leading to the low-reward terminal state ($s_1$), only the action values along that trajectory are repeatedly updated. For instance, if the left action initially acquires a positive value estimate such that


$$Q_t(s_3, L) > Q_t(s_3, R),$$


the agent selects $L$ at $s_3$ with high probability, drastically reducing the frequency of experiences reaching $s_4$ and the high-reward terminal state ($s_5$). This phenomenon is explained by the **interdependence between exploration and estimation**: sub-optimal value estimates bias action selection, which in turn biases the data distribution used for subsequent estimation.

Conversely, increasing $\epsilon$ promotes exploration independent of greedy action choices. Because the distance to the high-reward state is short and the state-action space is small in this environment, the information-gathering benefits of increased exploration are strongly pronounced.

##### Examining the $\epsilon = 1$ Case

Analyzing $\epsilon = 1$ reveals an important structural property of this MDP. At $\epsilon = 1$, Sarsa's behavior policy is uniformly random, meaning the target of evaluation is the fixed-policy action-value function $Q^{\pi_{\mathrm{random}}}$. Solving the Bellman equations for this MDP with $\gamma = 0.9$ yields:

| State | $Q^{\pi_{\mathrm{random}}}(s, L)$ | $Q^{\pi_{\mathrm{random}}}(s, R)$ |
| --- | --- | --- |
| **$s_2$** | $1.000$ | $2.042$ |
| **$s_3$** | $1.369$ | $3.169$ |
| **$s_4$** | $2.042$ | $5.000$ |

Extracting the greedy policy from these values gives


$$\arg\max_a Q^{\pi_{\mathrm{random}}}(s, a) = R$$


for every non-terminal state, which coincides identically with the optimal policy $\pi^*$. In other words, in this MDP, merely learning the action values of a uniformly random policy suffices to establish the exact action ranking required to construct the optimal policy. Thus, the rapid discovery of the optimal greedy policy at $\epsilon = 1$ is directly explained by the reward structure and action-value ordering of this MDP (note that because a constant learning rate is used, $Q_t$ fluctuates around rather than strictly converging to the exact values in the table above).

On the other hand, the policy actually executed by Sarsa at $\epsilon = 1$ remains uniformly random. Its expected return from the initial state $s_3$ is approximately $2.269$, which falls well below the optimal policy's expected return of $4.5$.

##### Recommended Summary Statement for Your Results

> *"In this experiment, increasing the exploration rate $\epsilon$ reduced the number of steps required to first reach the optimal greedy policy. In a small-scale MDP, a higher exploration rate increases the visitation frequency of state-action pairs, enabling earlier acquisition of the experiences necessary for accurate value estimation. Specifically, because the greedy policy extracted from the action-value function of a uniformly random policy is already optimal in this environment, increasing $\epsilon$ suppressed early action-selection bias and accelerated the discovery of the optimal greedy policy."*

---

#### 2. Standard Evaluation Metrics in RL Research

Evaluation metrics in RL are chosen based on the specific property an algorithm aims to improve. In single-agent RL, the following metrics are widely used:

| Evaluation Metric | Mathematical Definition / Measurement | Property Evaluated |
| --- | --- | --- |
| **Evaluation return** | $\displaystyle J(\pi_t) = \mathbb{E}_{\pi_t}\left[\sum_{k=0}^\infty \gamma^k r_k\right]$ | Performance of the learned policy |
| **Sample efficiency** | Relationship between return and environment steps | Ability to learn strong policies from limited experience |
| **Time to threshold** | $\displaystyle \tau_c = \inf\{t : J(\pi_t) \geq c\}$ | Speed of reaching a target performance level |
| **Cumulative reward / regret** | Total reward accrued during training; cumulative loss relative to $\pi^*$ | Online performance inclusive of exploration |
| **Success rate** | Probability of achieving the episode objective | Reliability of task completion |
| **Generalization** | Return on unseen initial conditions or environments | Adaptability to novel scenarios |
| **Computational efficiency** | Wall-clock time, memory footprint, etc. | Computational cost of training and inference |

##### Learning Curves and Sample Efficiency

The standard visualization is a **learning curve** plotting cumulative environment steps on the horizontal axis against evaluation return on the vertical axis (as used in Figure 2.4 of the textbook, Section 2.7, which evaluates the greedy policy at regular intervals and plots the mean and standard deviation across multiple seeds).

To quantify overall sample efficiency across a fixed training budget $B$—distinguishing, for example, an algorithm with rapid initial learning from one with slower initial progress but higher asymptotic performance—researchers compute the Area Under the Learning Curve (AUC):


$$\operatorname{AUC}(B) = \frac{1}{B}\int_0^B J(\pi_t)\,dt.$$

Furthermore, when evaluating online performance during training, one measures the return of the exploratory behavior policy. Recording both the **greedy policy evaluation return** and the **$\epsilon$-greedy behavior policy return** separately enables a direct analysis of the trade-off between the benefits and costs of exploration.

##### Statistical Evaluation

Because RL learning curves depend heavily on random seeds and environmental stochasticity, results must be reported across multiple seeds alongside uncertainty bounds. Agarwal et al. (*Deep Reinforcement Learning at the Edge of the Statistical Precipice*, NeurIPS 2021) established standard statistical reporting practices for multi-task RL benchmarks, proposing:

* **Interquartile Mean (IQM):** A robust aggregate metric taking the mean across the middle $50\%$ of the score distribution.
* **Stratified bootstrap confidence intervals:** Confidence intervals estimated by resampling training runs per task.
* **Performance profiles:** Visualizations comparing score distributions across multiple tasks and seeds.

---

#### 3. Representative Benchmark Environments

RL algorithms are typically evaluated across multiple tasks with distinct structural properties rather than a single environment:

| Benchmark | Representative Tasks | Primary Evaluation Focus |
| --- | --- | --- |
| **Gymnasium Toy Text** | `CliffWalking`, `FrozenLake`, `Taxi` | Tabular RL, exploration, stochastic transitions |
| **Arcade Learning Environment (ALE)** | Atari 2600 game suite | Visual observations, sample efficiency, multi-task performance |
| **MuJoCo** | `Hopper`, `HalfCheetah`, `Walker2d`, `Ant` | Continuous state-action spaces, locomotion control |
| **Procgen** | $16$ procedurally generated games | Sample efficiency, generalization to unseen levels |

At your current stage of exploring Sarsa, **`CliffWalking`** and **`FrozenLake`** are particularly well-suited next steps—both are more complex than the 5-state MDP while remaining tractable for tabular Sarsa:

* **`CliffWalking`:** A $4 \times 12$ gridworld where stepping off a cliff incurs a large negative reward. A classic testbed for comparing Sarsa against Q-learning and analyzing how $\epsilon$-exploration affects online training return.
* **`FrozenLake`:** A $4 \times 4$ gridworld ($16$ states, $4$ actions). With slippery ice enabled, transitions become stochastic and goal rewards are sparse, allowing direct study of the relationship between exploration rates and learning efficiency.

---

#### 4. Upgrading Your Experiment to Standard RL Evaluation Protocols

You can elevate your current $\epsilon$-sweep experiment to match standard research protocols while keeping your existing setup intact. At each training step $t$, extract both the greedy policy and the behavior policy from $Q_t$:


$$\begin{aligned} \pi_t^{\mathrm{g}}(s) &\in \arg\max_a Q_t(s, a), \\ \pi_t^{\epsilon}(a \mid s) &= \epsilon\text{-greedy}(Q_t(s, \cdot)), \end{aligned}$$


and measure their respective expected returns:


$$J_t^{\mathrm{g}} = J(\pi_t^{\mathrm{g}}), \qquad J_t^{\epsilon} = J(\pi_t^{\epsilon}).$$

Because this MDP is small and its exact dynamics are known, you can solve the Bellman equations directly during evaluation to compute exact expected returns without Monte Carlo sampling noise.

##### Recommended Experimental Setup

| Parameter / Metric | Specification |
| --- | --- |
| **$\epsilon$ values** | Keep existing $9$ conditions |
| **Training budget** | $5,000$ environment steps |
| **Evaluation interval** | Every $50$ steps |
| **Independent seeds** | $100$ |
| **Metric 1** | Greedy policy evaluation return ($J_t^{\mathrm{g}}$) |
| **Metric 2** | Behavior policy evaluation return ($J_t^{\epsilon}$) |
| **Metric 3** | Area Under the Return Curve ($\operatorname{AUC}$) |
| **Metric 4** | Time to reach $95\%$ of optimal return ($\tau_{0.95 J^*}$) |
| **Metric 5** | First hitting time to the optimal greedy policy (existing metric) |

This setup elevates your initial finding (*"larger $\epsilon$ discovers the optimal greedy policy faster"*) into a broader scientific investigation: *"How does the exploration rate govern the trade-off between learning efficiency and online policy performance?"* Specifically, at $\epsilon = 1$, $J_t^{\mathrm{g}}$ rapidly approaches the optimal return while $J_t^{\epsilon}$ remains stuck at the uniformly random baseline—a critical degradation in execution performance that would be completely missed if only the greedy policy were evaluated. This distinction is fundamental not only to tabular Sarsa, but to evaluating exploration algorithms across all of RL.

