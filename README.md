
# Person Search by CLSA
This is a reconstruction from scratch of a deep network, based on a technique outlined in the compouter vision paper "Person Search by Multi-Scale Matching". The model is built upon a modified ResNet-50 backbone. The purpose of the technique is to boost the performance of person search by reinforcing semantic alignment across layers.

A friendly suggestion:
perhaps do not attempt to run this code. This code is in transition as I am not satisfied with the condition it ended up in during crunch. One may consider this a proof of experience, rather than a shining example of software design principles I would prefer. I do not have the resources currently (as of 9/2019) to maintain this code so it will likely remain her as proof of my early forrays into deep learning. When I get my hands on a GPU I can continue to improve this implimentation.

There is no formal write-up for this project, only a final presentation (included). I have provided a personal interpretation of the primary technique:

# What is CLSA?
Cross-level Semantic Alignment (CLSA) is a technique to train specific target layers of a network to make predictions similarly to a source layer, by an implimentation of Kullback-Leibler divergence loss. Semantic alignment is achieved by coercing the distribution of class predictions for specific target layers to move towards the distribution of a source layer. Each target layer, with its own distinct features & prediction distribution, is warped by the KL loss to predict similarly to the source layer. KL loss is balanced by classification loss, which enforces that the high level semantics imparted upon the target layers are high quality predictive semantics. Ideally, the target layers retain some aspect of the low level semantics they possess, as they incorporate high level predictive semantics imparted by KL loss from the source layer.

In simpler terms, semantic alignment means to make the target layers predict like the source layer does. We don't want the target layers to be identical to the source layer, only to be flavored by it. Classification loss ensures that, practically speaking, the KL loss will never produce identical distributions across all layers (Unless by some astronomically unlikely miracle the model finds a locally optimal solution in which all distributions are identical across layers). Target layers end up well seasoned by the source layer with plenty of individuality so as not to stray into redundancy.
 
# Interpretation
When this technique is applied to a ResNet based person re-identification network, performance has been shown to vastly improve. There are many ways to describe how / why semantic alignment works.

When we discuss with terms such as vector semantics of this or that layer, we are really talking about the predictive distinctiveness of one layer relative to another or others.  Models which always predict identically to one another are redundant. On the other hand, models that vary wildly from one another are incoherent. Effective models in Machine Learning must strike a delicate balance between redundancy and incoherence. As I can tell, KL loss works by enforcing coherence across layers. The training process enforces a state of coherence across layers, pointing to/suggesting a common direction for all layers to work toward, not unlike files of metal drawn into order by a magnetic field.



