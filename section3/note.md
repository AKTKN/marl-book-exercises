## Before diving, let's think for myself
### Simple Janken
今、repeated normal-form gamesを読んだが、少しここで2章で読んだ内容を踏まえて、この後を読む前に少し自分で問題を色々考えてみる。
例えば、two agentのじゃんけんを何回か繰り返すとする。このとき、相手のagentの中身はわからないとする。そして、勝てば+3, 負ければ-2、アイコなら-1という形でrewardを定義する。
ここで、相手の出す手の傾向が実はグーしかださないであるとする。このとき、もしじゃんけんAIを実装するとなったら、2章における、model-free learningとして、状態が一つ、actionが3つのモデルでsarsaやq-leaningをすれば、いずれ、この相手に勝つようなpolicyが導けるという理解でよいか？

---
はい。その理解で正しいです。相手が常にグーを出す固定された方策を持つなら、**1状態・3行動のMDPとして定式化でき、SarsaやQ-learningによって最適方策を学習できます。**
自分をagent 1、相手をagent 2とし、
$$
 S=\\{s\\},\qquad A_1=\\{R,P,S\\} 
$$
とします。相手の方策は未知ですが、実際には
$$
 \pi_2(R)=1,\qquad \pi_2(P)=\pi_2(S)=0 
$$
であるとします。
このとき、自分が観測する報酬関数は
$$
 R(s,a)= \begin{cases} -1 & (a=R),\\\ +3 & (a=P),\\\ -2 & (a=S). \end{cases} 
$$
状態遷移は任意の行動について $T(s\mid s,a)=1$ です。
割引率 $0\leq\gamma<1$ の無限期間問題を考えると、Bellman最適方程式は
$$
 Q^\*(s,a)=R(s,a)+\gamma\max\_{a'}Q^\*(s,a') 
$$
となります。したがって、
$$
 \begin{aligned} Q^\*(s,R)&=-1+\gamma V^\*(s),\\\ Q^\*(s,P)&=3+\gamma V^\*(s),\\\ Q^\*(s,S)&=-2+\gamma V^\*(s). \end{aligned} 
$$
ここで $V^\*(s)=\max_a Q^\*(s,a)$ なので、
$$
 V^\*(s)=3+\gamma V^\*(s) =\frac{3}{1-\gamma}. 
$$
従って最適方策は
$$
 \pi_1^\*(P\mid s)=1 
$$

となり、常にパーを選択します。

Q-learningでは、各試行から得られる報酬を利用して

$$
 Q(s,a_t)\leftarrow Q(s,a_t)+\alpha_t \left[ r_t+\gamma\max\_{a'}Q(s,a')-Q(s,a_t) \right] 
$$

を繰り返します。相手の方策 $\pi_2$ を直接知る必要はありません。

教科書の第2.6節で説明されるように、すべての行動が無限回探索され、学習率が式(2.54)の条件を満たせば、Q-learningは最適行動価値関数に収束します。Sarsaも、十分な探索を維持しつつ $\epsilon\to0$ とする適切な方策改善の下で最適方策を学習できます。

今回の例では、相手の方策が固定されているため、自分から見た環境は定常的です。この条件により、通常のsingle-agent model-free RLをそのまま適用できます。

なお、$\epsilon$-greedy方策で $\epsilon>0$ を固定すると探索によってパー以外も選択されますが、学習された貪欲方策はパーを選択するようになります。

### Reward design in MARL
cooporativeではなく、基本的にじゃんけんや勝ち負けゲームのようなシンプルに自身のrewardを最大化子さえすれば良い場合は他のagentも環境として考えれば良いということですね。では、協力系とかrewardの最適化に複数のagentが関わる場合どうすればよいのか。一つは、reward関数をそのまま共通化してしまう。というかそれなら、その共通のinteractionを持つagentを一つにまとめ、stateやactionの次元を上げればよいのではとも考えられるが、なんだかそれだとscalableではない気がする。どうなのだろうか？それか、まとめるではなく、reward系を変えて、お互いを独立に動かした時に、理想的な状態がナッシュ均衡になるようにするとかrewardの設計を工夫するとか言う戦略もあるのか。どうでしょうか？
---
考えている方向性は正しいです。挙げられた方法は、実際にMARLで研究されてきた代表的なアプローチに対応します。特に、**共通報酬による協調、中央集権的な学習、独立学習、報酬設計による均衡の制御**は、それぞれ異なる問題を扱っています。

まず、他のagentを環境の一部として扱えるかどうかは、競争・協力の区別ではなく、他のagentの方策が固定されているかに依存します。

agent $i$ から見た状態遷移確率は、他のagentの方策を $\pi\_{-i}$ とすると、

$$
 T_i(s'\mid s,a_i) =\sum\_{a\_{-i}}\pi\_{-i}(a\_{-i}\mid s) T(s'\mid s,a_i,a\_{-i}) 
$$

となります。相手の方策が固定なら通常のMDPとして扱えます。一方、複数のagentが同時に学習すれば $\pi\_{-i}$ が変化するため、agent $i$ から見た環境は非定常になります。これは協力ゲームでも同様です。

次に、協力ゲームに対する挙げられた方法を考えます。

#### 1. 共通報酬を設定し、全agentを一つのMDPとして扱う方法

$n$ 個のagentが共通報酬を受け取るとします。

$$
 R_1(s,\mathbf a,s')=\cdots=R_n(s,\mathbf a,s') =R(s,\mathbf a,s') 
$$

ここで $\mathbf a=(a_1,\ldots,a_n)$ です。

このとき、全agentの行動をまとめれば、

$$
 \mathcal A=A_1\times A_2\times\cdots\times A_n 
$$

という行動空間を持つ単一のMDPに帰着できます。したがって、第2章のQ-learningをそのまま適用できます。

$$
 Q(s,\mathbf a)\leftarrow Q(s,\mathbf a)+\alpha \left[ r+\gamma\max\_{\mathbf a'}Q(s',\mathbf a') -Q(s,\mathbf a) \right] 
$$

これは教科書の第5.3.1節で紹介される *Central Learning* に相当します。

問題は、指摘された通りscalabilityです。各agentが $m$ 個の行動を持つ場合、
$$
 |\mathcal A|=m^n 
$$
となり、joint-action spaceがagent数に対して指数的に増加します。さらに、実行時にも中央のcontrollerが各agentの行動を決定する必要があります。

#### 2. 共通報酬を用いるが、各agentは独立に学習する方法
各agentが独立した行動価値関数
$$
 Q_i(s,a_i) 
$$
を持ち、全agentが同一の報酬 $r$ を受け取る構成です。
各agentは第2章のQ-learningを独立に実行します。これは *Independent Q-Learning (IQL)* と呼ばれ、教科書の第5.3.2節で扱われています。
行動価値関数の大きさは各agentについて $O(|S|m)$ で済みます。
しかし、二つの問題が生じます。

- Non-stationarity：他のagentの方策が学習中に変化する。
- Multi-agent credit assignment：共通報酬から、各agentの行動がどれだけ貢献したかを識別しにくい。

この問題に対して、後の章では *Centralized Training with Decentralized Execution (CTDE)* や *Value Decomposition* が導入されます。
例えば、VDNはjoint-action valueを
$$
 Q\_{\mathrm{tot}}(s,\mathbf a) \approx\sum\_{i=1}^{n}Q_i(s,a_i) 
$$
という形で近似します。
これにより、共通報酬に基づいて各agentの価値関数を共同学習しながら、行動の選択自体は各agentが独立に実行できます。

#### 3. 報酬関数を設計して、望ましい状態をNash equilibriumにする方法
これも実際に研究される戦略です。ゲーム理論では *Mechanism Design* や *Incentive Design*、関連する枠組みとして *Potential Games* が存在します。
例えば、2-agentのnormal-form gameで、あるjoint action $\mathbf a^\*$ を実現したいとします。
そのjoint actionがNash equilibriumであるための条件は、
$$
 \forall i,\ \forall a_i\in A_i: \quad R_i(\mathbf a^\*)\geq R_i(a_i,\mathbf a^\*\_{-i}) 
$$
です。
つまり、他のagentが $\mathbf a^\*\_{-i}$ を選んでいるとき、どのagentも自分だけ行動を変える動機を持たないように報酬を設計します。
興味深いことに、共通報酬を与えるだけでも、共通報酬の大域的最大点は必ずNash equilibriumになります。
しかし、次のような共通報酬ゲームを考えてください。

|            | Agent 2: A | Agent 2: B |
| ---------- | ---------- | ---------- |
| Agent 1: A | 10         | 0          |
| Agent 1: B | 0          | 8          |

この場合、$(A,A)$ が大域的最適解ですが、$(B,B)$ もNash equilibriumです。
つまり、共通報酬を設定するだけでは、agentたちが最適な均衡を選択するとは限りません。
そこで、例えば全体の報酬を $G(\mathbf a)$ とし、各agentに
$$
 \widetilde R_i(\mathbf a) = G(\mathbf a)-G(a_i^0,\mathbf a\_{-i}) 
$$
という報酬を与える方法があります。ここで $a_i^0$ は基準行動です。これは *Difference Rewards* と呼ばれる設計の一例です。
この場合、
$$
 \widetilde R_i(a_i,\mathbf a\_{-i}) -\widetilde R_i(a_i',\mathbf a\_{-i}) = G(a_i,\mathbf a\_{-i}) -G(a_i',\mathbf a\_{-i}) 
$$
が成立します。
したがって、各agentの報酬改善方向は全体報酬の改善方向と一致します。このような構造はexact potential gameに対応し、個々のagentの行動と全体目的を整合させることができます。
ただし、この設計でも局所的な均衡は残り得ます。

以上を整理すると、提案された三つの発想はいずれも妥当であり、互いに排他的でもありません。共通報酬は「何を最適化するか」、中央集権化・独立学習は「どのように最適化するか」、報酬設計は「各agentの利得をどう全体目的に整合させるか」という異なる側面を扱っています。

教科書でも、第5.3節でCentral LearningとIndependent Learningを比較し、第9.5節でValue Decompositionを導入するという形で、この問題が発展的に扱われています。
