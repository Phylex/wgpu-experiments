# WGPU experiments
Repository to hold my experiments with WGPU and rust. Currently following the `learn wgpu`_ tutorial with some of my own modifications.

.. _learn wgpu: https://sotrh.github.io/learn-wgpu/


## My understanding thus far

So the GPU acts like a big command processor, that is wrapped by the driver (Mesa or otherwise). The CPU prepares the data and the commands that need to be executed, and when everything is prepared
it shipps everything to the GPU to be processed. The GPU has it's own memory (the VRAM) that stores the data that is to be processed. the Shaders are the programs that are executed on the GPU 'Cores'.

When used for actual rendering (and not for computation) the output is the framebuffer (or more precicely the swapchain). The GPU can also be configured to render the output to a texture (which is essentially a bitmap image in GPU memory)

The shaders get executed in an execution of the render pipeline They end up drawing pixels into the framebuffer. It needs to be decided what things get drawn and what does not get drawn.


### Questions about rendering
* What corresponds to a 'core' inside a GPU. Nvidia always markets 'shader cores' but as cpus are inherently complex (pipelining and hyperthreading and SIMD) I don't know how this exactly maps to compute hardware.
* What is the depth stencil (in the render pipeline creation)
* How is the `clip_position` in the output of the vertex shader generated?
* Describe the entire render pipeline
