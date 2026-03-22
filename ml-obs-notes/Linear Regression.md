In regression problems, the inputs $x \in \mathbb{R}^D$ are mapped to the corr. function values $f(x) \in \mathbb{R}$. The inputs are set of training values $x_n$ and noisy observations $y_n = f(x_n) + \epsilon$ , where $\epsilon$ is an i.i.d. rand-var describing noise.

To solve a regression problem, the following problems must be solved:
	- **Choice of the model & parametrisation** - Which polynomial (and of what degree) passes good to model the data?
	- **Finding good parameters** - Optimise params $\theta$ based on loss functions
	- **Overfitting & model selection** 
	- **Relationship between loss-funcs & parameter priors** - Prior assumptions induce losses
	- **Uncertainty modelling**

### Problem formulation

The noise is modelled using a likelihood function. For the following examples, it will be
	$p(y|x) = \mathcal{N}(y|f(x),\sigma^2)$ 
	with $x \in \mathbb{R}^D$ - inputs and $y=f(x) + \epsilon \in \mathbb{R}$ - targets, and with $\epsilon \sim \mathcal{N}(0,\sigma^2)$ - i.i.d. noise distribution

Linear regression is given by:
	$p(y|x,\theta) = \mathcal{N}(y|x^T\theta, \sigma^2)$ 
		$p(y|x,\theta)$ is the prob-density func of $y$ evaluated at $x^T\theta$ 
	$\Leftrightarrow y=x^T\theta +\epsilon, ~ \epsilon \sim \mathcal{N}(0,\sigma^2)$ , where $\theta \in \mathbb{R}^D$ are the searched params
	

For the input $\mathcal{D} := {(x_1,y_1),...,(x_N,y_N)}$ the observations $y_i$ and $y_j$ are conditionally independent given respective inputs $x_i$ and $x_j$. Therefore the likelihood factorises:
	$p(\mathcal{Y}|\mathcal{X},\theta) = p(y_1,...,y_N|x_1,...,x_N,\theta) = \displaystyle\prod_{n=1}^N p(y_n|x_n,\theta)$
	$\Leftrightarrow \displaystyle\prod_{n=1}^N \mathcal{N}(y_n|x_n^T\theta,\sigma^2))$ 

The goal is to find the optimal parameters $\theta^*$. When these are found, function values are predicted from distribution
	$p(y_*|x_*,\theta^*)=\mathcal{N}(y_*|x_*^T\theta^*,\sigma^2)$ 

### Maximum Likelihood Estimation

The optimal params $\theta_{ML}$ are chosen by maximising the likelihood 
	$\theta_{ML} \in argmax_{\theta}~p(\mathcal{Y}|\mathcal{X},\theta)$ 

Instead of direct maximisation of the likelihood, the log transformation is applied and negative log-likelihood is minimised.
	Optimum of the $f$ is the same as the optimum of $logf$ 
	Moreover, log-transform doesn't suffer from numerical underflow and has simpler differentiation rules.

Log-transform makes the optimal function a sum of logs, instead of a product:
	$-log~p(\mathcal{Y}|\mathcal{X},\theta) = -log~\displaystyle\prod_{n=1}^N~p(y_n|x_n,\theta)$ 
	$\Leftrightarrow -\displaystyle\sum_{n=1}^N~log~p(y_n|x_n,\theta)$  

For linear regressions with Gaussian likelihood, we get
	$log~p(y_n|x_n,\theta)=-\cfrac{1}{2\sigma^2}(y_n-x_n^T\theta)^2 + const$ 
Thus, the negative log-likelihood becomes
	$\mathcal{L}(\theta)=\cfrac{1}{2\sigma^2}\displaystyle\sum_{n=1}^N~(y_n-x_n^T\theta)$
		$=\cfrac{1}{2\sigma^2} ||y-X\theta||^2$  

For this quadratic function, the minimum is searched, by computing the gradient of $\mathcal{L}$, setting it to 0 and solving for $\theta$
	$\cfrac{d\mathcal{L}}{d\theta}=...=\cfrac{1}{\sigma^2}(-y^TX+\theta^TX^TX) \in \mathbb{R}^{1\times D}$ , with $rk(X)=D$ 

Setting it to 0 and solving yields
	$\cfrac{d\mathcal{L}}{d\theta}=0^T\Leftrightarrow \theta_{ML}^TX^TX=y^TX$ 
	$\Leftrightarrow \theta_{ML}=(X^TX)^{-1}X^Ty$


For the more complex data, the data can be expressed via a non-linear transformation of the inputs $\phi(x)$, to fir within the linear regression framework (regression must be linear only in the parameters)
	$p(y|x,\theta)=\mathcal{N}(y|\phi^T(x)\theta,\sigma^2)$
	$\Leftrightarrow y=\phi^T(x)\theta + \epsilon = \displaystyle\sum_{k=0}^{K-1}~\theta_k~\phi_k~(x)+\epsilon$
	where $\phi:\mathbb{R}^D \rightarrow \mathbb{R}^K$ is a (nonlinear) transformation of the inputs $x$ and $\phi_k:\mathbb{R}^D \rightarrow \mathbb{R}$ is the kth component of the **feature vector ${\phi}$**. The model params $\theta$ still only appear linearly

For training inputs $x_n \in \mathbb{R}^D$ and targets $y_n \in \mathbb{R}, n=1,...,N$, the feature matrix is defined as:

$\begin{bmatrix} \phi^T(x_1) \\ \vdots \\ \phi^T(x_N) \end{bmatrix}$ $=$ $\begin{bmatrix} \phi_0(x_1) & \cdots & \phi_{K-1}(x_1) \\ \vdots & \ddots & \vdots \\ \phi_0(x_N) & \cdots &\phi_{K-1}(x_N) \end{bmatrix}$ $\in \mathbb{R}^{N\times K}$ 

Where $\Phi_{ij}=\phi_j(x_i)$ and $\phi_j:\mathbb{R}^D \rightarrow \mathbb{R}$ 

With the feature matrix $\Phi$, the negative log-likelihood for the linear regression has the form
	$-log~p(\mathcal{Y}|\mathcal{X},\theta)$ $=$ $\cfrac{1}{2\sigma^2}(y-\Phi \theta)^T(y-\Phi \theta) + const$ 
And the MLE
	$\theta_{ML}=(\Phi^T \Phi)^{-1}~\Phi^Ty$  <- MLE for linear regression with nonlinear features.

MLE can also be used to estimate the variance $\sigma^2_{ML}$ , for that the derivative of the log-likelihood with respect to $\sigma^2>0$ is computed, set to 0, and solved.
Log-likelihood 
$log~p(\mathcal{Y}|\mathcal{X},\theta,\sigma^2) = \displaystyle\sum_{n=1}^{N}~log~\mathcal{N}(y_n|\phi^T(x_n)\theta,\sigma^2)$ 
	$= \displaystyle\sum_{n=1}^{N}(-\cfrac{1}{2}log(2\pi)-\cfrac{1}{2}log~\sigma^2 - \cfrac{1}{2\sigma^2}(y_n-\phi^T(x_n)\theta)^2)$
	$=\cfrac{N}{2}~log~\sigma^2-\cfrac{1}{2\sigma^2}~\underbrace{\displaystyle\sum_{n=1}^{N}(y_n-\phi^T(x_n)\theta)^2}_{=:s}+const$

Then the partial derivative with respect to $\sigma^2$ is computed
	$\cfrac{\partial log~p(\mathcal{Y}|\mathcal{X},\theta,\sigma^2)}{\partial\sigma^2}$ $=$ $-\cfrac{N}{2\sigma^2}+\cfrac{1}{2\sigma^4}s$ $= 0$ 
	$\Leftrightarrow \cfrac{N}{2\sigma^2}=\cfrac{s}{2\sigma^4}$ 

And then, the most likely noise variance $\sigma_{ML}^2$
	$\sigma_{ML}^2 = \cfrac{s}{N}=\cfrac{1}{N}\displaystyle\sum_{n=1}^N~(y_n-\phi^T(x_n)\theta)^2$ 
	so, the MLE of the noise variance is the empirical mean of the distance between the noisy observations $y_n$ and corresponding function values $\phi^T(x_n)\theta$ 

MLE can be seen as an algorithm that finds the best subspace for a set of points by minimising their combined lengths of orthogonal projections onto this subspace.

### Maximum a Posteriori

MLE is prone to overfitting, and when overfitting the magnitude of parameter values becomes larger.

To mitigate that, a prior can be placed on parameter values $p(\theta)$, pre-encoding which parameter values are plausible before seeing the data. Then, instead of the likelihood, the posterior $p(\theta|\mathcal{X},\mathcal{Y})$ is maximised via choosing optimal parameters $\theta$.

To find the MAP estimate for a linear regression, first the similar (to MLE) log-transform is made, and log-posterior is computed
	$log p(θ|\mathcal{X},{Y}) = log p(\mathcal Y|\mathcal X,θ) + log p(θ) + const$ <- the final MAP estimate is a compromise between a prior assumption $p(\theta)$ and max-likelihood $p(\mathcal Y | \mathcal X, \theta)$

Then, negative log-posterior is minimised
	$\theta_{MAP} ∈ arg~min_θ \lbrace −log p(Y|X,θ)−log p(θ)\rbrace$ 

The gradient of the negative log-posterior is
	 $-\cfrac{d~log p(θ|\mathcal X,\mathcal Y)}{d~θ}$ $=$ $\underbrace{-\cfrac{d~log p(\mathcal Y|\mathcal X,θ)}{d~θ}}_{grad ~of~ \mathcal L_{\theta}}$ $-\cfrac{d~log p(θ)}{d~θ}$

With a conjugate Gaussian prior $p(θ) =\mathcal N (0, b^2I)$ (Continuing example from prev. part) on params $\theta$, negative log posterior looks like:
	$−log~p(θ|\mathcal X,\mathcal Y) = \cfrac {1}{2σ^2} (y−Φθ)^T(y−Φθ) + \cfrac{1}{2b^2}θ^Tθ+ const$ 

Then, its gradient is computed as:
	$-\cfrac{d~log p(θ|\mathcal X,\mathcal Y)}{d~θ}$ $=$ $\cfrac{1}{\sigma^2}~(θ^TΦ^TΦ−y^TΦ) + \cfrac{1}{b^2}θ^T$ 

It is then set to $0^T$ and solved for $\theta_{MAP}$
	$\cfrac{1}{σ^2} (θ^⊤Φ^⊤Φ−y^⊤Φ) + \cfrac{1}{b^2} θ^⊤= 0^⊤$ 
	$⇐⇒θ^⊤( \cfrac{1}{σ^2} Φ^⊤Φ + \cfrac{1}{b^2} I)−\cfrac{1}{\sigma^2}y^⊤Φ= 0^⊤$ 
	$⇐⇒θ^⊤= y^⊤Φ (Φ^⊤Φ + \cfrac{σ^2}{b^2} I)^{-1}$ 

So:
	$θ_{MAP} = (Φ^⊤Φ +\cfrac{σ^2}{b^2} I)^{-1}~Φ^⊤y$ 
	The only difference with the MLE estimation is the additional $\cfrac{σ^2}{b^2} I$, which ensures that the inverse matrix is symmetric and positive definite, and reflects the impact of the regulariser.

### MAP as Regularisation

Instead of placing a prior distribution on parameters θ, it is also possible to mitigate overfitting by penalising the amplitude of the parameter (via regularisation). In **regularised least squares**,  loss function is defined as
	$||y-Φθ||^2 + λ||θ||_2^2$, which is minimised with respect to $\theta$ 
	$||y-Φθ||^2$ <- data-fit term (misfit term) $\propto$ neg. log-likelihood
	$λ||θ||_2^2$ <- regulariser, with reg. parameter $\lambda⩾0$  
	$||\cdot||_2$ <- Euclidean, or any other $p$-norm $|| \cdot ||_p$ 

For the $λ=\cfrac{1}{2b^2}$ , the regulariser and negative log-Gaussian prior (from MAP) are identical (with the same Gaussian prior being $p(θ) = \mathcal N{0, b^2I}$ )

After minimising, the RLS estimate closely resembles the MAP estimate
	$θ_{RLS} = (Φ^⊤Φ + λI)^{−1}~Φ^⊤y$ 

### Bayesian Linear Regression

Same motivation - distribution of params instead of point estimates.

For a model with
	prior $p(θ) = \mathcal N(m_0, S_0)$,
	likelihood $p(y|x,θ) = \mathcal N (y|ϕ^⊤(x)θ, σ^2)$ ,

We turn the set of params $\theta$ into a prior distribution $p(θ) = \mathcal N(m_0, S_0)$

In this case, the full joint distribution looks like $p(y,θ|x) = p(y|x,θ)p(θ)$ 

At an input $x_*$, prediction is made by integrating out $\theta$:
	$p(y_*|x_*)$ $=$ $\int p(y∗|x∗,θ)p(θ)dθ$ $=$ $\mathbb E_θ[p(y∗|x∗,θ)]$ <- this is an average prediction of $y∗|x∗,θ$
	for all plausible params $\theta$ according to the distribution $p(\theta)$ 

Following the example, predictive distribution of defined lin-reg model would be
	$p(y_∗|x_∗) = \mathcal N(ϕ^⊤(x∗)m_0, ϕ^⊤(x∗)S_0ϕ(x∗) + σ^2$ 
	This prediction is Gaussian
	$ϕ^⊤(x∗)S_0ϕ(x∗)$ describes the uncertainty associated with the params $\theta$
	$\sigma^2$ describes uncertainty added due to noise.
	$y_*$ is a linear transform of $\theta$, so mean and covar of the prediction can be computed analytically.
	

The Gaussian noise is independent, so 
	$\mathbb V[y_∗] = \mathbb V_θ[ϕ^⊤(x_∗)θ] +\mathbb V_ϵ[ϵ]$ ($\mathbb V_{\theta}$ being a variance of $[\cdot]$ with respect to $\theta$ )


Posterior $p(θ|X,Y)$ is computed using Bayes'.
	$p(θ|\mathcal X,\mathcal Y) = \cfrac{p(\mathcal Y|\mathcal X,θ)p(θ)}{p(\mathcal Y|\mathcal X)}$

Here, $p(\mathcal Y|\mathcal X,θ)$ is the likelihood, $p(θ)$ the parameter prior, and
	$p(\mathcal Y|\mathcal X) = \mathbb E_{\theta}[p(\mathcal Y|\mathcal X,\theta)]$ 

For the example model, the parameter posterior is computed as
	$p(θ|\mathcal X,\mathcal Y) = \mathcal N (θ|m_N, S_N)$
	$S_N = (S_0^{-1} + σ^{-2}Φ^⊤Φ)^{-1}$
	$m_N = S_N(S_0^{-1} m_0 + σ^{−2}Φ^⊤y)$

And the posterior predictive distribution is defined as
	$p(y_∗|\mathcal X,\mathcal Y,x_∗) =\int p(y_∗|x_∗,θ)p(θ|\mathcal X,\mathcal Y)dθ$ 
	$= \int \mathcal N (y_∗|ϕ^⊤(x_∗)θ, σ^2)\mathcal N (θ|m_N, S_N) dθ$ 
	$= \mathcal N (y_∗|ϕ^⊤(x_∗)m_N, ϕ^⊤(x_∗)S_Nϕ(x_∗) + σ^2)$ 

$ϕ^⊤(x_∗)S_Nϕ(x_∗)$ here reflects the posterior uncertainty about the parameters $\theta$. $S_N$ depends on the training input through $\phi$ 

### Marginal Likelihood

For samples generated from distributions
	$θ∼\mathcal N (m_0, S_0)$
	$y_n|x_n,θ∼ \mathcal N (x^⊤_nθ, σ^2)$

The marginal likelihood is given by
	$p(\mathcal Y|\mathcal X) = \int p(\mathcal Y|\mathcal X,θ)p(θ)dθ$
	$=\int \mathcal N (y|Xθ, σ^2I) ~\mathcal N (θ|m_0, S_0) dθ$
	Parameters $\theta$ are integrated out. The marginal likelihood can be interpreted as the expected likelihood under the prior: $\mathbb E_{\theta}[p(\mathcal Y|\mathcal X,θ)]$ 

The mean of the marginal likelihood is computed as
	$\mathbb E[\mathcal Y|\mathcal X] =\mathbb E_{θ,ϵ}[Xθ+ ϵ] = X \mathbb E_θ[θ] = Xm_0$ 

Its covariance matrix is given as
	$Cov[\mathcal Y|\mathcal X] = Cov_{θ,ϵ}[Xθ+ ϵ] = Cov_θ[Xθ] + σ^2I$
	$= XCov_θ[θ]X^⊤+ σ^2I= XS_0X^⊤+ σ^2I$ 

Hence, the marginal likelihood is
	$p(\mathcal Y|\mathcal X) = \mathcal N( y|Xm_0, XS_0X^⊤+ σ^2I)$ 


## Broader usage outline

Linear regression can work as a standalone model, that takes inputs $x$, performs the linear transformation with params $\theta$ and produces output $y$ 

It can also function as the final layer of a neural network, i.e. the network learns the linear transformation $\phi(x)$, which is then used to get outputs.

But linear regression can not model complex patterns, because all linear transformations / stacked layers stay linear.

**In a deep learning scenario:**
Say, the model is supposed to display a non-linear distribution. Pure linear regression doesn't work, but it can be used on the last layer to decide how to weight the features produced by the deeper layers.

Linear regression (via MLE, or others) gets the most suitable $\theta$ for $y \approx \theta^T \phi(x)$ , and $\phi(x)$ is received by the deeper network learning. 
So, the network is searching for a representation $\phi(x)$ such that the data becomes linearly predictable.
	$\min_{\phi, \theta} ||y - \theta^T \phi(x)||^2$ 
	where $\phi(x)$ is parametrised by the network weights
	$\phi(x) = \phi(x; W_1, W_2, \dots, W_{L-1})$ 

So the overall optimisation problem (for all layers) can be displayed as
	$\min_{W_1,\dots,W_L} ||y - W_L \cdot \phi(x; W_1,\dots,W_{L-1})||^2$ 

While linear regression solves analytically, neural network approximates via backprop. Earlier layers learn feature detectors, later layers combine them linearly.