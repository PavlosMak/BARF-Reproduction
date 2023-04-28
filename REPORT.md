# Reproducing and testing BARF 🤮
by Rodrigo Alvarez Lucendo, Pavlos Makridis and Jorge Romeu Huidobro

In this blog post, we present our process for reproducing the results of **BARF: Bundle-Adjusting Neural Radiance Fields**. First, we give a review of Neural Radiance Fields (NeRF) and of BARF's contributions. Then we showcase the reproduced results on a subset of the scenes in table 2 of the original paper. Additionally, we evaluate the ability of the network to perform in visually uniform scenes and reason about the observed behaviour. 


## Introduction 

We start with reviewing the basics of both NeRF and BARF.

### NeRF: Neural Radiance Fields
Originally presented in [1], Neural Radiance Fields (NeRFs) showed remarkable ability for *novel view synthesis*. Simply put, view synthesis is concerned with taking a collection of different posed images of a scene and then rendering a new view of the same scene as if the camera was placed in a new location. So how is it done? The idea lies in viewing the scene as a sort of field (or function), the *Radiance Field*.

But what exactly are Radience Fields? Radiance fields represent the scene as a continuous function that takes a 5D vector as an input and outputs a colour and an intensity value. This 5D vector consists of two parts, a 3D location $\vec{x} = (x,y,z)$ and a 2D viewing direction $(\theta, \phi)$, where $\theta$ and $\phi$ can be seen as spherical coordinates. If you think about it, this is a natural way to represent how a scene looks like. For example, when you look at an object, the colour and the intensity that reach your eyes depend first on the point of the object you are looking at (the 3D location) and from what direction (the 2D vector) you are looking at it (e.g. by tilting your head you might observe that the same point has a slightly different colour). Below we summarise the above in a concise formula. For a deeper understanding of the mathematics behind radiance, we recommend [chapter 5 of *Physically Based Rendering*](https://pbr-book.org/3ed-2018/Color_and_Radiometry). 

$$F: (x,y,z,\theta,\phi) \rightarrow (\text{color},\text{intensity})$$

Although this formulation might seem a bit complicated, the important thing to understand is that it gives us a way to reason about view synthesis as a function approximation problem. Therefore now we can use a Multi-Layer Perceptron (MLP) to learn that function! This is exactly what NeRF networks do, and they can create amazing results like the one shown below (taken from [1]).
![](https://i.imgur.com/Kl6TbAM.png)


### BARF: Bundle-Adjusting Neural Radiance Fields
Although great, NeRFs have one significant limitation: they require accurate knowledge of the camera's position and rotation when training. This is problematic for many real-life scenarios since perfectly localising camera positions is not trivial. By using Bundle-Adjustment, BARF relaxes this requirement and can learn both the scene representations and the camera positions simultaneously.

To understand how BARF works, we must start looking at how the camera pose is represented in NeRF. The key idea is utilising a *positional encoding* scheme to map 3D coordinates to higher dimensions. You can think of this as an additional layer applied between the input and the MLP. Concretely, this positional encoding, $\gamma$, works by converting the position vector to an encoding with $L$ different sinusoidal bases: 
$$\gamma(\vec{x}) = [\vec{x}, \gamma_0(\vec{x}), \gamma_1(\vec{x}), ..., \gamma_{L-1}(\vec{x})]$$
where the $k$-th entry, $\gamma_k(\vec{x})$ in this vector is:
$$\gamma_k(\vec{x}) = [\cos(2^k\pi \vec{x}), \sin(2^k\pi\vec{x})]$$ 

With this scheme in place, if we want to learn the camera positions using gradient descent, we need to differentiate the above formulation: 

$$\frac{\partial\gamma_k(\vec{x})}{\partial\vec{{x}}} = 2^k\pi[-\sin(2^k\pi\vec{x}), \cos(2^k\pi\vec{x})]$$

The problem here is that the gradient signal is amplified by $2^k\pi$, and moreover, its direction changes at the same frequency as the signal from the subsequent MLP. As a result, the gradient signal is incoherent and can even cancel each other out, making it difficult to update the parameters effectively. 

This is where Bundle-Adjustment comes into play. The idea applied in BARF is to apply a coarse-to-fine (low to high) smoothing of the positional encodings at different frequency bands. Basically this means scaling the $k$-th components of the positional encoding by a weight $w_k$ such that it becomes $$\gamma(\vec{x},\alpha) = w_k(\alpha)[\cos(2^k\pi\vec{x}), \sin(2^k\pi\vec{x})]$$
Here $\alpha$ is a parameter in $[0,L]$ proportional to the progress of the optimization and the weight $w_k$ is given by the formula: 

$$ w_k(a) = 
     \begin{cases}
       0 & \quad\text{if } \alpha < k \\
       \frac{1 - \cos((\alpha - k)\pi)}{2} &\quad\text{if } 0 \le \alpha - k< 1 \\
       1 &\quad\text{if } \alpha - k \ge 1 \\ 
     \end{cases}
$$

If we differentiate now, we get:

$$\frac{\partial\gamma_k(\vec{x})}{\partial\vec{{x}}} = w_k(\alpha) 2^k\pi[-\sin(2^k\pi\vec{x}), \cos(2^k\pi\vec{x})]$$

Since the weight term depends on a parameter proportional to the optimisation progress, we can now control how much the gradients contribute to the optimisation depending on its stage. Concretely the process starts by setting $\alpha = 0$ for the 3D input and gradually increasing it till $L$ to allow higher and higher frequencies to contribute to the learning process. As a result, BARF can start with a smoother signal and later refine it more and more to learn the scene.  

## Reproduction 
With the theoretical understanding now in place, we reproduced the results presented in the original paper. To do so, we relied on the [author's published code](https://github.com/chenhsuanlin/bundle-adjusting-NeRF) and ran our experiments in the DelfBlue supercomputer. For the experiments, we used the hyperparameter settings listed in the original paper and used three of the original scenes, the chair, the lego and the drums (shown below).
 
![](https://i.imgur.com/B7aUugI.png)


The results of our reproduction are summarized in the table below. Note that in the table, there are two types of error metrics visible; the *camera pose registration* error which shows the error on registering the camera orientation and position and the *view synthesis* error, which shows how close the generated image is to the target image using different image similarity metrics (PSNR, SSIM, LPIPS). The arrows in the table indicate the direction of improvement. 

![](https://i.imgur.com/AZia3YH.png)


Overall we managed to reproduce the results shown in the paper with reasonable accuracy; thus, we deem BARF to be reproducible. 

## Testing on Visually Uniform scenes

After replicating the original paper's results, we tested the network's ability to recover the camera pose when the scenes are visually uniform. 

To understand what exactly we mean by this, let's start with a look at BARF's objective function given below: 

$$\min_{\vec{p_1},...,\vec{p_M},\Theta}\Sigma_{i=1}^M\Sigma_{\vec{u}}||\hat{I}(\vec{u};\vec{p_i},\Theta) - I_i(\vec{u})||_2^2$$

This loss function gets a set of $M$ images $\{I_i\}^M_{i=1}$ and tries to optimize the camera poses $\vec{p_i}$ and the network parameters $\Theta$ such that the square difference of pixel $\vec{u}$ in the generated image $\hat{I}$ and the ground truth image $I$ is minimized.


As you may notice, this loss function is *purely photometric*, which relies only on the difference between the pixels in the produced and the target images. Yet this function is also used to learn the camera's position. This can create a problem in cases where two pixels in the target image are far away from each other and yet still have similar colours, as it would happen for a visually uniform scene.


We created the two soccer ball scenes below to test our hypothesis. The first has a simple black and white soccer ball, whereas the other has some uniquely coloured patches. 

![](https://i.imgur.com/lrdEOpb.png)


The idea behind this setup is that because of how similar the black-and-white soccer ball looks from different viewpoints, the network will not be able to learn the camera position and rotation correctly. In contrast, we expected the network to perform better for the scene with the uniquely coloured patches since those can be used as "reference points" to establish the camera pose. However, after training and evaluating the network, we observed the opposite behaviour! Our exact results are summarized below:

![](https://i.imgur.com/eOw3nTj.png)

In the next section, we explain why we got those counter-intuitive results. 


The scenes we used for these experiments can be found [here](https://drive.google.com/drive/folders/1oZ3--N5ue-QyZXkU_Oh05zbQCzTM9Is3?usp=share_link).

## Discussion & Conclusions
<!-- We now briefly discuss our results and conclude this blog post with some final remarks. 

Regarding our results on the tests with visually uniform scenes, it is tempting to say that the network proved more robust than expected. However, the results are insufficient to draw that conclusion. First, the noise used to perturb the camera poses was sampled from a distribution with 0.15 standard deviation, which might not be enough to "disorient" the network. Secondly, the results obtained with the coloured soccer ball scene are still close enough to those in the black-and-white scene that the random initialisation of the network parameters could explain it. 

As an extension to this project, we recommend this be investigated more. For starters, it is essential to realise how much the noise distribution impacts the results. For this, we recommend that the impact of the noise is first studied in the standard benchmarking scenes (like the chair) and then compared to the effect on the visually-uniform settings. The relation between the deterioration of the results will provide a clearer view of the effect of visually-uniform scenes. After establishing this relation, comparing the coloured and black-and-white soccer ball scenes will indicate whether the difference in performance we observe is statistically significant. 

In terms of reproducibility, we found BARF to be overall reproducible. The results we obtained by repeating the paper's experiments matched those reported by the authors. Last but certainly not least, the code accompanying the original publication is open-source and well-documented, making it easy to use and expand upon.
     -->
     
We now briefly discuss our results and conclude this blog post with some final remarks. 

As expected, the network performs significantly worse on the visually uniform scenes. Notice that this difference is almost an order of magnitude for both the camera rotation and translation. Despite this, the network performs as well, if not better, for view synthesis. Additionally, we notice that adding coloured patches did not help improve the camera posing.  

The first interesting fact is that even though the performance of the camera pose registration degrades, performance in view synthesis remains high. We attribute this to the symmetry of the pentagonal patches, which make the soccer balls appear the same from different viewpoints. This makes it impossible to know the exact viewpoint for a given image. However, this property allows reconstructing the image without knowing the exact camera pose since the object will appear the same from several viewpoints.

The second interesting finding is that the coloured patches did not improve camera localisation as the network performs similarly badly. This result contradicts our initial assumption that the coloured patches would help the camera localisation. We believe this is because the noise used to perturb the camera poses was sampled from a distribution with 0.15 standard deviation, which might not be enough to "disorient" the network in such a way as to confuse one patch for another. 

As an extension of this project, we recommend further investigating these two findings. For the first finding, it is worth exploring how different symmetries affect the performance. For example, by replacing the pentagonal patches with triangular ones, since triangles have fewer symmetries. We expect that this will improve performance. Regarding the second finding, it could be interesting to investigate if larger amounts of noise make the network perform better in the scene with the coloured patches. 


In terms of reproducibility, we found BARF to be overall reproducible. The results we obtained by repeating the paper's experiments matched those reported by the authors. Last but certainly not least, the code accompanying the original publication is open-source and well-documented, making it easy to use and expand upon.
## References
[1] [Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. In European conference on computer vision, 2020](https://arxiv.org/abs/2003.08934).

[2] [Lin, Chen-Hsuan, et al. "Barf: Bundle-adjusting neural radiance fields." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021](https://arxiv.org/abs/2104.06405).