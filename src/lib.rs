#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
mod texture;

use wgpu::util::DeviceExt;
use wgpu::{Features, BufferUsages};
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

// Struct to hold the information of the vertices that we want to render
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    color: [f32; 3],
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct TexturedVertex {
    position: [f32; 3],
    tex_coords: [f32; 2],
}

impl Vertex {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
            // The below code uses a macro to do the thing we did by hand in the above definition
            // attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3];
        }
    }
}


impl TexturedVertex {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<TexturedVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
            ]
        }
    }
}

const TRIANGLE: &[Vertex] = &[
    Vertex { position: [0.0, 0.5, 0.0], color: [1.0, 0.0, 0.0] , },
    Vertex { position: [-0.5, -0.5, 0.0], color: [0.0, 1.0, 0.0] },
    Vertex { position: [0.5, -0.5, 0.0], color: [0.0, 0.0, 1.0] },
];

const PENTAGON: &[Vertex] = &[
    Vertex { position: [-0.0868241, 0.49240386, 0.0], color: [0.5, 0.0, 0.5] }, // A
    Vertex { position: [-0.49513406, 0.06958647, 0.0], color: [0.5, 0.0, 0.5] }, // B
    Vertex { position: [-0.21918549, -0.44939706, 0.0], color: [0.5, 0.0, 0.5] }, // C
    Vertex { position: [0.35966998, -0.3473291, 0.0], color: [0.5, 0.0, 0.5] }, // D
    Vertex { position: [0.44147372, 0.2347359, 0.0], color: [0.5, 0.0, 0.5] }, // E
];

const PENTAGON_INDICES: &[u16] = &[
    0, 1, 4,
    1, 2, 4,
    2, 3, 4,
];

const TEXPENTAGON: &[TexturedVertex] = &[
    TexturedVertex { position: [-0.0868241, 0.49240386, 0.0], tex_coords: [0.4131759, 0.99240386], }, // A
    TexturedVertex { position: [-0.49513406, 0.06958647, 0.0], tex_coords: [0.0048659444, 0.56958647], }, // B
    TexturedVertex { position: [-0.21918549, -0.44939706, 0.0], tex_coords: [0.28081453, 0.05060294], }, // C
    TexturedVertex { position: [0.35966998, -0.3473291, 0.0], tex_coords: [0.85967, 0.1526709], }, // D
    TexturedVertex { position: [0.44147372, 0.2347359, 0.0], tex_coords: [0.9414737, 0.7347359], }, // E
];

// structure to hold the state of the graphics system
struct GrapicsSystem {
    window: Window,
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,

    render_pipeline: wgpu::RenderPipeline,
    alternative_render_pipeline: wgpu::RenderPipeline,
    texture_render_pipeline: wgpu::RenderPipeline,
    use_alternate_pipeline: bool,
    use_texture_pipeline: bool,

    textured_pentagon_buffer: wgpu::Buffer,
    pentagon_buffer: wgpu::Buffer,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_vertices: u32,
    use_pentagon: bool,

    diffuse_texture: texture::Texture,
    
    background_color: (f64, f64, f64, f64),
    diffuse_bind_group: wgpu::BindGroup,
}


impl GrapicsSystem {
    // instantiate a new graphics system
    async fn new(window: Window) -> Self {
        let size = window.inner_size();

        // the instance represents the GPU
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            dx12_shader_compiler: Default::default(),
        });

        // the state owns the window and the surface should live as
        // long as the window so this should be safe
        let surface = unsafe { instance.create_surface(&window) }.unwrap();

        // the adapter is the software on the cpu side that takes care of
        // dispatching stuff to the gpu
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        // the adapter then produces handles to the gpu and it's
        // queues that allow the cpu to dispatch work to the gpu
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    // we are not giving the gpu a name
                    label: None,
                    // we don't require any special features
                    features: Features::empty(),

                    // if running on the web we need to disable some features
                    // the 'if cfg!' thing seems to be a compile time macro
                    limits: if cfg!(target_arch = "wasm32") {
                        wgpu::Limits::downlevel_webgl2_defaults()
                    } else {
                        wgpu::Limits::default()
                    },
                },
                None, // Trace path
            )
            .await
            .unwrap();

        // we can now find out what the capabilities are that the 'surface'
        // can provide. The surface is the thing we render to
        let surface_capabilities = surface.get_capabilities(&adapter);
        let surface_format = surface_capabilities.formats.iter()
            .copied()
            // we want an srgb capable surface
            .filter(|f| f.describe().srgb)
            .next()
            .unwrap_or(surface_capabilities.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_capabilities.present_modes[0],
            alpha_mode: surface_capabilities.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        
        // load the image data
        let diffuse_bytes = include_bytes!("happy-tree.png");
        let diffuse_texture = texture::Texture::from_bytes(&device, &queue, diffuse_bytes, "happy-tree").unwrap();

        // build the texture bind group layout
        let texture_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { 
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture { 
                        multisampled: false, 
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
            label: Some("texture_bind_group_layout"),
        });
        let diffuse_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor { 
            label: Some("diffuse_bind_group"),
            layout: &texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry{binding: 0, resource: wgpu::BindingResource::TextureView(&diffuse_texture.view)},
                wgpu::BindGroupEntry{binding: 1, resource: wgpu::BindingResource::Sampler(&diffuse_texture.sampler) },
            ]
        });
        // generate the texture pipeline layout
        let texture_pipeline_layout = 
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Texture Pipeline Layout"),
                bind_group_layouts: &[&texture_bind_group_layout],
                push_constant_ranges: &[],
            });

        // This is shader stuff
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        // this is our original render pipeline layout
        // optionally use the wgpu::include_wgsl! to simplify the above code
        let render_pipeline_layout = 
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[Vertex::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent::REPLACE,
                        alpha: wgpu::BlendComponent::REPLACE,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: Some(wgpu::Face::Back),
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        let alternate_render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[Vertex::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "alt_fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent::REPLACE,
                        alpha: wgpu::BlendComponent::REPLACE,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: Some(wgpu::Face::Back),
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        println!("creating tex pipeline");
        let texture_render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("texture pipeline"),
            layout: Some(&texture_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "tex_vs_main",
                buffers: &[TexturedVertex::desc()],
            },
            primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: Some(wgpu::Face::Back),
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "tex_fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent::REPLACE,
                        alpha: wgpu::BlendComponent::REPLACE,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview: None }
        );


        let vertex_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Triangle Buffer"),
                contents: bytemuck::cast_slice(TRIANGLE),
                usage: wgpu::BufferUsages::VERTEX,
            }
        );

        let index_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Triangle_index_buffer"),
                contents: bytemuck::cast_slice(PENTAGON_INDICES),
                usage: wgpu::BufferUsages::INDEX,
            }
        );
        
        let pentagon_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Pentagon Buffer"),
                contents: bytemuck::cast_slice(PENTAGON),
                usage: wgpu::BufferUsages::VERTEX,
            }
        );

        let pentagon_texture_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Textured Pentagon"),
                contents: bytemuck::cast_slice(TEXPENTAGON),
                usage: BufferUsages::VERTEX 
            }
        );

        let vertex_count = PENTAGON_INDICES.len() as u32;
        let use_pentagon = true;
        let use_alt = false;
        let use_tex = true;

        Self {
            window,
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
            alternative_render_pipeline: alternate_render_pipeline,
            texture_render_pipeline,
            use_alternate_pipeline: use_alt,
            vertex_buffer,
            index_buffer,
            pentagon_buffer,
            textured_pentagon_buffer: pentagon_texture_buffer,
            num_vertices: vertex_count,
            use_pentagon,
            background_color: (0.1, 0.2, 0.3, 1.0),
            diffuse_bind_group,
            use_texture_pipeline: use_tex,
            diffuse_texture,
        }
    }

    pub fn window(&self) -> &Window {
        &self.window
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        };
    }

    #[allow(unused_variables)]
    fn input(&mut self, event: &WindowEvent) -> bool {
        match event {
            // change the color of the background depending on mouse position
            WindowEvent::CursorMoved { position, .. } => {
                self.background_color.1 = position.x / self.config.height as f64;
                self.background_color.2 = position.y / self.config.width as f64;
                true
            },
            WindowEvent::KeyboardInput {
                input: 
                    KeyboardInput {
                        state, 
                        virtual_keycode: Some(VirtualKeyCode::Space),
                        ..
                    },
                ..
            } => {
                match *state {
                    ElementState::Released => {
                        self.use_alternate_pipeline = false;
                        true
                    },
                    ElementState::Pressed => {
                        self.use_alternate_pipeline = !self.use_texture_pipeline;
                        true
                    }
                }
            },
            WindowEvent::KeyboardInput {
                input: 
                    KeyboardInput {
                        state, 
                        virtual_keycode: Some(VirtualKeyCode::P),
                        ..
                    },
                ..
            } => {
                match *state {
                    ElementState::Pressed => {
                        self.use_pentagon = true;
                        self.num_vertices = PENTAGON_INDICES.len() as u32;
                        true
                    },
                    ElementState::Released => {
                        self.use_pentagon = false;
                        self.use_texture_pipeline = false;
                        self.num_vertices = TRIANGLE.len() as u32;
                        true
                    }
                }
            },
            WindowEvent::KeyboardInput {
                input: 
                    KeyboardInput {
                        state, 
                        virtual_keycode: Some(VirtualKeyCode::T),
                        ..
                    },
                ..
            } => {
                match *state {
                    ElementState::Pressed => {
                        match self.use_pentagon {
                            true => {
                                self.use_texture_pipeline = !self.use_alternate_pipeline;
                            },
                            _ => {}, 
                        }
                        true
                    },
                    ElementState::Released => {
                        self.use_texture_pipeline = false;
                        true
                    }
                }
            },
            _ => false,
        }
    }

    fn update(&mut self) {
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor{
                label: Some("Render Encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor { 
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: self.background_color.0,
                            g: self.background_color.1,
                            b: self.background_color.2,
                            a: self.background_color.3,
                        }),
                        store: true,
                    },
                })],
                depth_stencil_attachment: None, 
            });

            // set the pipeline used for rendering
            render_pass.set_pipeline(if self.use_alternate_pipeline {
                &self.alternative_render_pipeline
            } else if self.use_texture_pipeline {
                &self.texture_render_pipeline
            } else {
                &self.render_pipeline
            });
            
            // set the assets used for rendering
            if self.use_texture_pipeline {
                render_pass.set_bind_group(0, &self.diffuse_bind_group, &[]);
                render_pass.set_vertex_buffer(0, self.textured_pentagon_buffer.slice(..));
                render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                render_pass.draw_indexed(0..self.num_vertices, 0, 0..1);
            } else if self.use_pentagon {
                render_pass.set_vertex_buffer(0, self.pentagon_buffer.slice(..));
                render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                render_pass.draw_indexed(0..self.num_vertices, 0, 0..1);
            } else {
                render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
                render_pass.draw(0..self.num_vertices, 0..1);
            };
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        Ok(())
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen(start))]
pub async fn run() {
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("WebGPU-Experiments")
        .build(&event_loop)
        .unwrap();

    #[cfg(target_arch = "wasm32")]
    {
        use winit::dpi::PhysicalSize;
        window.set_inner_size(PhysicalSize::new(450, 400));

        use winit::platform::web::WindowExtWebSys;
        web_sys::window()
            .and_then(|win| win.document())
            .and_then(|doc| {
                let dst = doc.get_element_by_id("wasm-example")?;
                let canvas = web_sys::Element::from(window.canvas());
                dst.append_child(&canvas).ok()?;
                Some(())
            })
            .expect("Could not append canvas to document body");
    }

    let mut state = GrapicsSystem::new(window).await;
    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent {
                window_id,
                ref event,
            } if window_id == state.window().id() => if !state.input(event) {
                match event {
                    WindowEvent::CloseRequested | WindowEvent::KeyboardInput {
                        input:KeyboardInput {
                            state: ElementState::Pressed,
                            virtual_keycode: Some(VirtualKeyCode::Escape),
                            ..
                        },
                        ..
                    } => {
                        *control_flow = ControlFlow::Exit;
                        println!("Closing up");
                    },
                    WindowEvent::Resized(physical_size) => {
                        state.resize(*physical_size);
                    },
                    WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                        state.resize(**new_inner_size);
                    },
                    _ => {}
                }
            },
            Event::RedrawRequested(window_id) if window_id == state.window().id() => {
                state.update();
                match state.render() {
                    Ok(_) => {}
                    // reconfigure the surface if it is lost or outdated
                    Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                    // the system is out of memory. In this case we should Quit the program
                    Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                    // we are ignoring timeouts
                    Err(e) => eprintln!("{:?}", e),
                }
            },
            Event::MainEventsCleared => {
                state.window().request_redraw();
            },
            _ => {}
        }
    });
}
