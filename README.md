# Probabilistic Primitive Fitting for Single-View Point Clouds

This repository contains the implementation of a probabilistic method for fitting multiple geometric primitives (superellipsoids, supertoroids, and tapered superparaboloids) to partial point clouds obtained from a single RGB-D viewpoint.  
The approach is based on a Gaussian–Uniform mixture model optimized through an Expectation–Maximization (EM) process that iteratively refines both the probabilistic assignment of points and the parameters of each primitive.  Physical penalties prevent the invasion of visible free space, while support constraints enforce physically plausible reconstructions.  
The sequential combination of primitives enables the reconstruction of complex or composite objects from incomplete observations, providing accurate and interpretable geometric representations suitable for robotic perception and manipulation tasks.
