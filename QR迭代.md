# QR迭代
特征值问题求解的QR迭代算法
证明过程及实现过程

## QR迭代基础
### 补充：幂法（Power Method）
用于求解矩阵的主特征值（绝对值最大的特征值）及其对应的特征向量。它通过不断乘以矩阵 \( A \) 和一个初始向量 \( x_0 \)，使得向量逐步收敛到主特征向量。
#### 迭代过程：
>选择一个初始向量 \( x_0 \)
 \( x_{k+1} = A x_k \)（迭代）
 \( x_{k+1} = \frac{x_{k+1}}{\|x_{k+1}\|} \)（归一化）
 重复上述过程，直到 \( x_k \) 收敛到主特征向量

主特征值可以通过Rayleigh商来估计：\( \lambda_k = \frac{x_k^T A x_k}{x_k^T x_k} \)

### 迭代原理
正交相似变换保证特征值与特征向量不变，矩阵经正交相似变换，变为上hessenberg矩阵，再经QR分解，Givens处理，求解特征值和特征向量。

对上hessenberg矩阵进行QR迭代，迭代结果仍为上hessenberg矩阵，即**不可约性**。

### 迭代过程
#### Algorithm Orthogonal Iteraton正交迭代算法
对初始化\(n * p\)正交矩阵\(Z_0，i=0\)
运用幂法思想求特征值

>   repeat
      \(Y_{i+1} = AZ_i\) (左乘矩阵A)
      \(Y_{i+1} = Z_{i+1}R_{i+1}\) (QR分解)
      \(i = i+1\)
   until convergence

特征值可以通过 \( Q_k^T A Q_k \) 的对角元素来估计。
#### Algorithm QR Iteration  QR迭代算法
Given \(A_0\), we iterate \(i = 0\)
>   repeat
   \(A_i = Q_i R_i\) (the \(QR\) decomposition)
   \(A_{i+1} = R_iQ_i\)
   \(i = i + 1\)
   until convergence

### QR迭代收敛到schur form
#### 什么是schur form？
**Schur form of A**，即一个上三角矩阵，其特征值位于对角线上或者分块上三角。上三角矩阵 T 的对角线元素是 A 的特征值。

实矩阵的QR运算中会出现复共轭特征值，通过QR迭代所得的Am逼近的可能不是一个上三角阵，但会逼近实Schur标准形

#### 迭代过程分析
\(
\operatorname{span}\left\{Z_{i+1}\right\}=\operatorname{span}\left\{Y_{i+1}\right\}=\operatorname{span}\left\{A Z_i\right\} 
\)
\(
\operatorname{span}\left\{Z_i\right\}=\operatorname{span}\left\{A^i Z_0\right\}=\operatorname{span}\left\{S \Lambda^i S^{-1} Z_0\right\}
\)
\(
\begin{aligned} S \Lambda^i S^{-1} Z_0 & =S \operatorname{diag}\left(\lambda_1^i, \cdots \lambda_n^i\right) S^{-1} Z_0 \\ & =\lambda_p^i S\left[\begin{array}{ccccc}\left(\lambda_1 / \lambda_p\right)^i & & & & \\ & \ddots & & & \\ & & 1 & & \\ & & & \ddots & \\ & & & & \left(\lambda_n / \lambda_p\right)^i\end{array}\right] S^{-1} Z_0\end{aligned}
\)


Since \(\left|\frac{\lambda_i}{\lambda_p}\right| \geq 1\) if \(i \leq p\), and \(\left|\frac{\lambda_i}{\lambda_p}\right|<1\) if \(i>p\), we get（以P为界分块
\(
\left[\begin{array}{ccc}
\left(\lambda_1 / \lambda_p\right)^i & & \\
& \ddots & \\
& & \left(\lambda_n / \lambda_p\right)^i
\end{array}\right] S^{-1} Z_0=\left[\begin{array}{c}
X_i^{p \times p} \\
Y_i^{(n-p) \times p}
\end{array}\right]
\)

其中\(Y_i \rightarrow 0\)

\(S=\left[\begin{array}{lll}s_1 & \cdots & s_n\end{array}\right] \equiv \left[S_p^{n \times p} \hat{S}_p^{n \times(n-p)}\right]\)

$S \Lambda^i S^{-1} Z_0=\lambda_p^i S\left[\begin{array}{c}X_i \\ Y_i\end{array}\right]=\lambda_p^i\left(S_p X_i+\hat{S}_p Y_i\right)$

$\operatorname{span}\left(Z_i\right)=\operatorname{span}\left(S \Lambda^i S^{-1} Z_0\right)=\operatorname{span}\left(S_p X_i+\hat{S}_p Y_i\right)
\rightarrow
\operatorname{span}\left(S_p X_i\right)=\operatorname{span}\left(S_p\right),当 X_i可逆则相等,
i \rightarrow \infin$

#### 引理：$A_i = Z_i^TAZ_i$ 收敛到\(A\)的schur form
在\(p = n，Z_0 = I\)的条件下对A进行正交迭代。
如果\(A\)的特征值有n个互异绝对值且所有顺序主子式都满秩，
则$A_i = Z_i^TAZ_i$ 收敛到\(A\)的schur form

注：有n个相异且对称的特征值时为对角；非对称n个互异特征向量无法正交，为上三角，第k个正交向量近似为前k个特征向量的组合
**证明**：
$Z_i = \left[\begin{array}{c}Z_{i1} & Z_{i2}\end{array}\right]，Z_{i1} \in R^{n \times p}$

$A_i = Z_i^T A Z_i 
=\left[\begin{array}{c}Z_{i1}^T \\ Z_{i2}^T\end{array}\right]A\left[\begin{array}{c}Z_{i1} & Z_{i2}\end{array}\right]
= \left[\begin{array}{c}Z_{i1}^TAZ_{i1} & Z_{i1}^TAZ_{i2} \\ Z_{i2}^TAZ_{i1} & Z_{i2}^TAZ_{i2} \end{array}\right]
$

其中 ,

$Z_{i2}^T = \left[\begin{array}{c}z_{p+1}^T \\ \vdots \\ z_{n}^T\end{array}\right]，AZ_{i1} \rightarrow Z_{i+1，1}，$

由于正交，$Z_{i2}^TAZ_{i1} \rightarrow 0 $，则$A_i \rightarrow R $

#### 下证：QR迭代形式等价于$A_i = Z_i^TAZ_i$
在QR迭代中，
$ A_i = Q_i R_i$ （对\(A\)进行QR分解）
$A_{i+1} = R_i Q_i$ （赋值给 \(A_{i+1}\) ）
**归纳证明：**
$ A_0 = IA_0I$，假设 $ A_i = Z_i^T A Z_i$，要推出$A_i = Z_{i+1}^T A Z_{i+1}$

$A Z_i (=QR) = Z_{i+1} R_i$，这是在正交迭代中的QR分解，$R_i = Z_{i+1}^T A Z_i$
$\begin{aligned}A_i  = Z_i^T A Z_i & = Z_i^T Z_{i+1} R_i \\ & = Q_i R_i \end{aligned}$
则$ A_{i+1} := R_i Q_i = Z_i^T A Z_i Z_i^T Z_{i+1} = Z_{i+1}^T A Z_{i+1}$，得证


## 通过上Hessenberg矩阵的实现方法降低QR迭代计算代价
事实上这个实现是隐式的，即不明确计算H的QR分解，而是通过Givens Rotation和其他正交矩阵的乘积来构造Q

### A变换成hessenberg矩阵
对A做正交相似变换，将矩阵变为上hessenberg矩阵
```matlab
function u = my_house(x)
u = x;
u(1) = u(1) + sign(x(1))*norm(x);
u = u / norm(u);
```

```matlab
function [Q,H] = uphessenberg(A)
n = size(A,1); Q = eye(n)
for i = 1 : n-2
   u = my_house(A(i+1:n, i));
   u2 = 2*u;
   A(i+1:n, i:n) = A(i+1:n, i:n) + u2*(u'*A(i+1:n, i:n))；
   A(:, i+1:n) = A(:, i+1:n) + (A(:, i+1:n)*u2)*u';
   Q(i+1:n, :) = Q(i+1:n, :) - u2*(u'*Q(i+1:n, :));
end
H = A;
```

### implicit Q theorem隐式 Q 定理
隐式 Q 定理是隐式 QR 迭代算法的理论基础。在隐式 QR 迭代中，我们不需要显式地对矩阵 \(A\) 进行 QR 分解，而是通过构造一个正交矩阵 \(Q\)，使其第一列满足特定条件，从而直接计算出迭代矩阵 \(A_{k+1}\)。这种方法大大减少了计算量。

通过隐式 Q 定理，我们可以高效地实现 QR 迭代，从而求解非对称特征值问题。
#### **定理**：
设 \(H = Q^T A Q \in \mathbb{R}^{n \times n}\) 是一个不可约上 Hessenberg 矩阵，其中 \(Q \in \mathbb{R}^{n \times n}\) 是正交矩阵，若有 \(\tilde{Q}^T A \tilde{Q} = \tilde{H}\), and \(Q\) 的第一列与 \(\tilde{Q}\) 的第一列相等，则\(Q = \tilde{Q}\)。

**即 \(Q\) 的第 2 列到第 \(n\) 列由 \(Q\) 的第一列唯一确定（可相差一个符号）。**

#### 证明
1. **不可约上 Hessenberg 矩阵的定义**  
   不可约上 Hessenberg 矩阵 \(H\) 的次对角线元素 \(h_{i+1,i} \neq 0\)，即矩阵形式为：
   \[
   H = \begin{bmatrix}
   h_{11} & h_{12} & h_{13} & \cdots & h_{1n} \\
   h_{21} & h_{22} & h_{23} & \cdots & h_{2n} \\
   0 & h_{32} & h_{33} & \cdots & h_{3n} \\
   \vdots & \vdots & \vdots & \ddots & \vdots \\
   0 & 0 & 0 & \cdots & h_{nn}
   \end{bmatrix}
   \]
   其中 \(h_{i+1,i} \neq 0\)。

2. **假设两个上 Hessenberg 分解**  
   假设矩阵 \(A\) 有两个上 Hessenberg 分解：
   \[Q^T A Q = H, \quad V^T A V = G\]
   其中 \(Q\) 和 \(V\) 是正交矩阵，且 \(Q\) 和 \(V\) 的第一列相等，即\((Q)_1 = (V)_1\)
   \(H\) 和 \(G\) 是不可约上 Hessenberg 矩阵。
   则有：
   \[A Q = Q H, \quad V^T A = G V^T\]
3. **矩阵元素分析**
    设\(W = V^T Q\)，则\((W)_1 = (1,0,...,0)^T = e_1\)
    \[GW = GV^TQ = V^TQH = WH\]
    \[(GW)_1 = G(W)_1 = [g_{11}， g_{21}，0，...，0]^T = h_{11} e_1 + h_{21} (W)_2\]

    则\[h_{21} (W)_2 = [g_{11} - h_{11}，g_{21}，0，...，0]^T\]
    由于\(W\)是正交矩阵，\(w_{21} = 0\)，所以\((W)_2 = e_2\)



    推得：
    $$(GW)_i = G(W)_i = (WH)_i= \sum_{j=1}^{i+1}h_{ji} (W)_i$$
    所以\[h_{i+1,i} (W)_{i+1} = G (W)_i - \sum_{j=1}^ih_{ji}(W)_j\]

    \(W\)是上三角矩阵且正交，\(W\)近似单位阵（元素相差正负号）

    则两分解相等，即$Q=V$，得证。


### 做Givens旋转
对H做QR分解
旋转矩阵第一列与H分解后的Q第一列相等/成比例，由隐式Q定理就能保证矩阵相等

通过正交变换将A变成上hessenberg矩阵，这个变换是特别的，这也是保证隐式Q定理成立的关键

旋转矩阵Q1、Q2是实的，则迭代出的A2也是实的


```matlab
function [c,s]=givens(x)
    r=sqrt((x')*x);
    c=x(1)/r;
    s=x(2)/r;
end

function A=qr_givens(H)%输入一个上hessenberg矩阵，用givens进行合同变换，输出仍为上hessenberg矩阵
    A=H;
    sigma=A(end,end);
    [n,~]=size(A);
    for i=1:n-1
        %[c,s]=givens(A([i,i+1],i));%不位移
        [c,s]=givens(A([i,i+1],i)-[sigma;0]);%位移A(n,n)
        G=blkdiag(eye(i-1),[c,s;-s,c],eye(n-i-1));
        A=G*A*(G');
    end
end
```

## 隐式QR迭代
隐式QR迭代是一种高效的数值方法，用于计算矩阵的特征值和特征向量。以下是实现隐式QR迭代的步骤：

### 1. 将矩阵化为上Hessenberg矩阵
首先，通过Householder变换将目标矩阵 \( A \) 化为上Hessenberg矩阵 \( H \)。上Hessenberg矩阵的特点是次对角线下方的元素均为零。具体步骤如下：
- 对矩阵 \( A \) 的每一列，构造Householder矩阵 \( H_k \)，使得 \( H_k A \) 的第 \( k \) 列下方的元素变为零。
- 重复上述过程，直到矩阵 \( A \) 被化为上Hessenberg矩阵。

### 2. 选择位移
位移的选取对隐式QR迭代的收敛速度和稳定性至关重要。常见的位移选择方法包括：
- **单步位移**：取矩阵右下角的对角元素 \( \sigma_k = H_k(n,n) \) 作为位移。
- **双步位移**：取矩阵右下角2×2子矩阵的特征值作为位移，称为**Francis位移**。
- **Wilkinson位移**：对于对称矩阵，取右下角2×2子矩阵的两个特征值中靠近 \( H_k(n,n) \) 的一个作为位移。

### 3. 进行隐式QR迭代
隐式QR迭代的核心是通过构造正交变换矩阵，避免显式的QR分解，从而减少计算量。具体步骤如下：
- 对化为上Hessenberg矩阵的 \( H \)，构造矩阵 \( H - \sigma I \)，其中 \( \sigma \) 是位移。
- 通过Givens变换或Householder变换，逐步将 \( H - \sigma I \) 的“bulge”（小矩阵）向右下角“赶”，最终消除所有“bulge”，得到新的上Hessenberg矩阵。
- 更新矩阵 \( H \) 为 \( H' = Q^T H Q + \sigma I \)，其中 \( Q \) 是所有变换矩阵的乘积。

### 代码实现（含显式qr迭代）
```matlab
function H=householder(x)
    e=zeros(length(x),1);
    e(1)=1;
    v=x-sqrt((x')*x)*e;
    H=eye(length(x))-2/((v')*v)*v*(v');
end

function H=hessenberg(A)
    H=A;
    [n,~]=size(A);
    for i=1:n-2
        Q1=householder(H(i+1:n,i));
        Q2=blkdiag(eye(i),Q1);
        H=Q2*H*(Q2');
    end
end

function eigv=im_qr_iter(A)%隐式
    B=hessenberg(A);
    last=diag(B,-1);
    err=0;
    tol=1e-3;%精度
    max_iter=1000;%最大迭代次数
    while(max_iter)
        B=qr_givens(B);
        err=norm(diag(B,-1)-last);%次对角线的收敛性
        last=diag(B,-1);
        if(err<tol)
            break;
        end
        max_iter=max_iter-1;
    end
    if(max_iter==0)
        disp('达到最大迭代次数');
    end
    eigv=diag(B);
end

function eigv=qr_iter(A)%显式
    A=hessenberg(A);
    last=diag(A,-1);
    n=size(A,1);
    err=0;
    tol=1e-6;%精度
    max_iter=1000;%最大迭代次数
    while(max_iter)
        sigma=A(end,end);
        [Q,R]=qr(A-sigma*eye(n));
        A=R*Q+sigma*eye(n);
        max_iter=max_iter-1;
        err=norm(diag(A,-1)-last);%次对角线的收敛性
        last=diag(A,-1);
        if(err<tol)
            break;
        end
    end
    if(max_iter==0)
        disp('达到最大迭代次数');
    end
    eigv=diag(R)+sigma;
end
```

