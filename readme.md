<h1>0.Introduction</h1>
Below there is a list of projects I've prepared for this repository.
<ul>
	<li><a href = "https://github.com/lukasy09/Machine-Learning-From-Scratch/tree/master/SimpleLinearRegression">Simple Linear Regression</a></li>
	<li><a href = "https://github.com/lukasy09/Machine-Learning-From-Scratch/tree/master/MultipleLinearRegression">Multiple Linear Regression</a></li>
</ul>

<h1>1.Supervised learning algorithms</h1>



<h2 id ="SLR">1.1 Simple Linear Regression</h2>

<p>As a first subproject I've implemented SimpleLinearRegression class. This model enables to fit the input data
 (that is 1-Dimensional vector/tensor) to a continous set of labels.</p>
<p>In this particular example each point represents a single human - worker, X-axis his position at work, Y- axis his salary. The model tries to predict how much people should earn.</p>
<p>The data comes from UDEMY ML Course [A-Z]</p>
<p align = "center">
<img src = "./assets/SLR/slr.gif" ></img>
</p>
<p align = "center">
<img src = "./assets/SLR/loss.png" ></img>
</p>

<p>On the first picture/gif there is a plot representing how the model changes in time(when the epochs are growing). The final epoch is equal to 100k.</p>
<p>On the second one we can see a graph displaying the dependency of loss on the epoch(in range 0-1000). After 10000th epoch the change is really very small and is comparable with sklearn model.</p>


<h2 id ="MLR">1.2 Multiple Linear Regression</h2>

<p>More general version of the previous class. This time regression object takes a matrix of data as input. We can use regression for more compound problems</p>
<p>Note: After about ~5000 epochs the model in most cases is almost as good as the Sklearn's LineaRegression</p>
<p align = "center">
<img src = "./assets/MLR/multi-loss.png" ></img>
</p>

<h2 id ="LR">1.3 Logistic Regression</h2>

<p>In this project I've implemented (binary) Logistic Regression learning model. </p>
<p>The data below have been generated and split into 2 classes. The model's task is to make a difference among the points and classify them.</p>

<p align = "center">
<img src = "./assets/LogisticRegression/model_log_10kiter.png" ></img>
</p>

<h1>2.Unsupervised Learning</h1>

<h2>2.1 PCA - Principal Component Analysis</h2>
<p>An unsupervised learning algorithm reducing the dimensionality of data.</p>

<h3>1.Blue - Input data sized (1000,3)</h3>
<h3>2.Orange - Output data sized (1000,2)</h3>

<p align = "center">
<img src = "./assets/PCA/PCA3d_2d.png" ></img>
</p>
