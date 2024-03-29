use std::iter;
use instant::Instant;
use cgmath::*;
use observer::Camera;
use wgpu::util::DeviceExt;
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};
use egui_winit_platform::{Platform, PlatformDescriptor};
use model::{GPUVertex, DrawModel, Instance, GPUInstance};
use egui::FontDefinitions;
use crate::wgpu_utils::create_render_pipeline;
use std::iter::zip;

#[cfg(target_arch="wasm32")]
use wasm_bindgen::prelude::*;

/// A custom event type for the winit app.
enum REvent {
    RequestRedraw,
}

/// This is the repaint signal type that egui needs for requesting a repaint from another thread.
/// It sends the custom RequestRedraw event to the winit event loop.
struct ExampleRepaintSignal(std::sync::Mutex<winit::event_loop::EventLoopProxy<REvent>>);

impl epi::backend::RepaintSignal for ExampleRepaintSignal {
    fn request_repaint(&self) {
        self.0.lock().unwrap().send_event(REvent::RequestRedraw).ok();
    }
}

mod wgpu_utils;
mod resources;
mod model;
mod texture;
mod observer;
mod light;

const NUM_INSTANCES_PER_ROW: u32 = 10;

struct State {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    render_pipeline: wgpu::RenderPipeline,
    light_render_pipeline: wgpu::RenderPipeline,
    #[allow(dead_code)]
    window: Window,
    observer: observer::Camera, 
    mouse_pressed: bool,
    instances: Vec<Instance>,
    instance_buffer: wgpu::Buffer,
    instance_rot_speed: f32,
    obj_model: model::Object,
    depth_texture: texture::Texture,
    light: light::Light,
    ui_platform: Platform,
    ui_render_pass: egui_wgpu_backend::RenderPass,
    start_time: Instant,
    spacing: f32,
}

impl State {
    async fn new(window: Window) -> Self {
        let spacing: f32 = 2.;
        let size = window.inner_size();

        // The instance is a handle to our GPU
        // BackendBit::PRIMARY => Vulkan + Metal + DX12 + Browser WebGPU
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            dx12_shader_compiler: Default::default(),
        });
        
        // # Safety
        // The surface needs to live as long as the window that created it.
        // State owns the window so this should be safe.
        let surface = unsafe { instance.create_surface(&window) }.unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (mut device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: wgpu::Features::empty(),
                    // WebGL doesn't support all of wgpu's features, so if
                    // we're building for the web we'll have to disable some.
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

        let surface_caps = surface.get_capabilities(&adapter);
        
        // Shader code in this tutorial assumes an Srgb surface texture. Using a different
        // one will result all the colors comming out darker. If you want to support non
        // Srgb surfaces, you'll need to account for that when drawing to the frame.
        let surface_format = surface_caps.formats.iter()
            .copied()
            .filter(|f| f.is_srgb())
            .next()
            .unwrap_or(surface_caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        // initialize the egui platform
         let platform = Platform::new(PlatformDescriptor {
            physical_width: size.width as u32,
            physical_height: size.height as u32,
            scale_factor: window.scale_factor(),
            font_definitions: FontDefinitions::default(),
            style: Default::default(),
        });
        let egui_render_pass = egui_wgpu_backend::RenderPass::new(&device, surface_format, 1);

        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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


        // all the stuff that is needed to initialize the observer of the scene
        let observer = Camera::new(
            (0.0, 5.0, 10.0),
            cgmath::Deg(-90.0),
            cgmath::Deg(-20.0),
            size.width,
            size.height,
            0.1,
            100.0,
            cgmath::Deg(45.0),
            &device,
            4.0, 0.4,
            &queue
        );

        let light = light::Light::new(&mut device);
        
        let depth_texture = texture::Texture::create_depth_texture(&device, &config, "depth texture");

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[
                    &texture_bind_group_layout,
                    &observer.uniform.bind_group_layout,
                    &light.bind_group_layout,
                ],
                push_constant_ranges: &[],
            });

        let render_pipeline = {
            let shader = wgpu::ShaderModuleDescriptor {
                label: Some("Normal Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
            };
            create_render_pipeline(
                &device,
                &render_pipeline_layout,
                config.format,
                Some(texture::Texture::DEPTH_FORMAT),
                &[model::ModelVertex::desc(), GPUInstance::desc()],
                shader,
            )
        };

        let light_render_pipeline = {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Light Render Pipeline"),
                bind_group_layouts: &[&observer.uniform.bind_group_layout, &light.bind_group_layout],
                push_constant_ranges: &[],
            });
            let shader = wgpu::ShaderModuleDescriptor {
                label: Some("Light Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("light.wgsl").into()),
            };
            create_render_pipeline(
                &device,
                &layout,
                config.format,
                Some(texture::Texture::DEPTH_FORMAT),
                &[model::ModelVertex::desc()],
                shader)
        };

        // here we load the model and that we are going to render in this case it is a cube
        let obj_model = resources::load_model("cube.obj", &device, &queue, &texture_bind_group_layout)
            .await
            .unwrap();
        
        // this is effectively a somewhat fancy way of generating a list of instances
        // using generators and iterator mechanics
        let instances = (0..NUM_INSTANCES_PER_ROW).flat_map(|z| {
            (0..NUM_INSTANCES_PER_ROW).map(move |x| {
                let x = spacing * (x as f32 - NUM_INSTANCES_PER_ROW as f32 / 2.0);
                let z = spacing * (z as f32 - NUM_INSTANCES_PER_ROW as f32 / 2.0);
                let position = cgmath::Vector3 { x, y: 0.0, z };

                let rotation = if position.is_zero() {
                    // this is needed so an object at (0, 0, 0) won't get scaled to zero
                    // as Quaternions can effect scale if they're not created correctly
                    cgmath::Quaternion::from_axis_angle(cgmath::Vector3::unit_z(), cgmath::Deg(0.0))
                } else {
                    cgmath::Quaternion::from_axis_angle(position.normalize(), cgmath::Deg(45.0))
                };

                Instance {
                    position, rotation, scale: [1.0, 1.0, 1.0].into()
                }
            })
        }).collect::<Vec<_>>();
        let instance_data = instances.iter().map(Instance::to_shader_format).collect::<Vec<_>>();
        let instance_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Instance Buffer"),
                contents: bytemuck::cast_slice(&instance_data),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            }
        );

        let start_time = Instant::now();
        Self {
            depth_texture,
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
            light_render_pipeline,
            obj_model,
            window,
            observer,
            mouse_pressed: false,
            instances,
            instance_buffer,
            instance_rot_speed: 1.,
            light,
            ui_platform: platform,
            ui_render_pass: egui_render_pass,
            start_time,
            spacing,
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
            self.observer.projection.resize(new_size.width, new_size.height);
            self.depth_texture = texture::Texture::create_depth_texture(&self.device, &self.config, "depth texture")
        }
    }

    #[allow(unused_variables)]
    fn input(&mut self, event: &Event<()>) -> bool {
        let event_processed = self.observer.controlls.process_event(event, self.mouse_pressed, self.window().id());
        if !event_processed {
            match event {
                Event::WindowEvent {event: WindowEvent::MouseInput { state, button: MouseButton::Left, .. }, ..} => {
                    self.mouse_pressed = *state == ElementState::Pressed;
                    true
                }
                _ => false,
            }
        } else {
            true
        }
    }

    fn update_instances(&mut self, dt: instant::Duration) {
        let spacing = self.spacing;
        let positions = (0..NUM_INSTANCES_PER_ROW).flat_map(|z| {
            (0..NUM_INSTANCES_PER_ROW).map(move |x| {
                let x = spacing * (x as f32 - NUM_INSTANCES_PER_ROW as f32 / 2.0);
                let z = spacing * (z as f32 - NUM_INSTANCES_PER_ROW as f32 / 2.0);
                cgmath::Vector3 { x, y: 0.0, z }
            })
        }).collect::<Vec<_>>();
        self.instances = zip(&self.instances, positions).map(|(inst, pos)| {
            let new_rot = inst.rotation * cgmath::Quaternion::from_axis_angle(cgmath::Vector3::unit_z(), cgmath::Deg(10.0 * dt.as_secs_f32() * self.instance_rot_speed));
            Instance {
                position: pos,
                rotation: new_rot,
                scale: [1.0, 1.0, 1.0].into(),
            }
        }).collect::<Vec<_>>();
    }

    fn update(&mut self, dt: instant::Duration) {
        self.observer.update(dt, &self.queue);

        // update the instances to rotate
        self.update_instances(dt);
        let instance_data = self.instances.iter().map(Instance::to_shader_format).collect::<Vec<_>>();
        // write the rotations to the buffer
        self.queue.write_buffer(&self.instance_buffer, 0, bytemuck::cast_slice(&instance_data));

        let old_position: cgmath::Vector3<_> = self.light.uniform.position.into();
        self.light.update(Some((cgmath::Quaternion::from_axis_angle((0.0, 1.0, 0.0).into(), cgmath::Deg(1.0)) * old_position).into()), None, &self.queue);
    }


    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = match self.surface.get_current_texture() {
            Ok(frame) => frame,
            Err(wgpu::SurfaceError::Outdated) => {
                // This error occurs when the app is minimized on Windows.
                // Silently return here to prevent spamming the console with:
                // "The underlying surface has changed, and therefore the swap chain must be updated"
                return Ok(());
            }
            Err(e) => {
                eprintln!("Dropped frame with error: {}", e);
                return Err(e);
            }
        };
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
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
                            r: 0.001,
                            g: 0.001,
                            b: 0.001,
                            a: 1.0,
                        }),
                        store: true,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment { 
                    view: &self.depth_texture.view,
                    depth_ops: Some(wgpu::Operations { 
                        load: wgpu::LoadOp::Clear(1.0),
                        store: true }),
                        stencil_ops: None,
                }),
            });
            render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
            
            use crate::model::DrawLight;
            render_pass.set_pipeline(&self.light_render_pipeline);
            render_pass.draw_light_model(
                &self.obj_model,
                &self.observer.uniform.bind_group,
                &self.light.bind_group
            );

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.draw_model_instanced(
                &self.obj_model,
                0..self.instances.len() as u32,
                &self.observer.uniform.bind_group,
                &self.light.bind_group
            );
        }

        // Render The UI
        self.ui_platform.update_time(self.start_time.elapsed().as_secs_f64());

        let output_view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // Begin to draw the UI frame.
        self.ui_platform.begin_frame();

        // Draw a small windo into the application.
        egui::Window::new("test window")
            .default_size(egui::vec2(200., 200.))
            .show(&self.ui_platform.context(), |ui| {
                ui.label("This is a label");
                ui.hyperlink("https://github.com/emilk/egui");
                ui.add(egui::Slider::new(&mut self.spacing, 2.0..=10.).text("spacing"));
            });

        // End the UI frame. We could now handle the output and draw the UI with the backend.
        let full_output = self.ui_platform.end_frame(Some(&self.window));
        let paint_jobs = self.ui_platform.context().tessellate(full_output.shapes);

        // Upload all resources for the GPU.
        let screen_descriptor = egui_wgpu_backend::ScreenDescriptor {
            physical_width: self.config.width,
            physical_height: self.config.height,
            scale_factor: self.window.scale_factor() as f32,
        };
        let tdelta: egui::TexturesDelta = full_output.textures_delta;
        self.ui_render_pass
            .add_textures(&self.device, &self.queue, &tdelta)
            .expect("add texture ok");
        self.ui_render_pass.update_buffers(&self.device, &self.queue, &paint_jobs, &screen_descriptor);

        // Record all render passes.
        self.ui_render_pass
            .execute(
                &mut encoder,
                &output_view,
                &paint_jobs,
                &screen_descriptor,
                None,
            )
            .unwrap();

        // Submit the commands.
        self.queue.submit(iter::once(encoder.finish()));

        // Redraw egui
        output.present();

        self.ui_render_pass 
            .remove_textures(tdelta)
            .expect("remove texture ok");

        // Support reactive on windows only, but not on linux.
        // if _output.needs_repaint {
        //     *control_flow = ControlFlow::Poll;
        // } else {
        //     *control_flow = ControlFlow::Wait;
        // }

        Ok(())
    }
}

#[cfg_attr(target_arch="wasm32", wasm_bindgen(start))]
pub async fn run() {
    cfg_if::cfg_if! {
        if #[cfg(target_arch = "wasm32")] {
            std::panic::set_hook(Box::new(console_error_panic_hook::hook));
            console_log::init_with_level(log::Level::Warn).expect("Could't initialize logger");
        } else {
            env_logger::init();
        }
    }

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().build(&event_loop).unwrap();

    #[cfg(target_arch = "wasm32")]
    {
        // Winit prevents sizing with CSS, so we have to set
        // the size manually when on web.
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
            .expect("Couldn't append canvas to document body.");
    }
    
    // State::new uses async code, so we're going to wait for it to finish
    let mut state = State::new(window).await;
    let mut last_render_time = Instant::now();

    // This is where events are processed and the resulting frames rendered
    // as the name suggests, this is done in a loop
    event_loop.run(move |event, _, control_flow| {
        state.ui_platform.handle_event(&event);
        let event_handeled_by_ui = state.ui_platform.captures_event(&event);
        if !event_handeled_by_ui {
            let event_handeled_by_app = state.input(&event);
            if !event_handeled_by_app {
                match event {
                    Event::DeviceEvent { event: DeviceEvent::MouseMotion { delta, }, .. } => if state.mouse_pressed {
                        state.observer.controlls.process_mouse_movement(delta.0, delta.1)
                    }
                    Event::WindowEvent {
                        ref event,
                        window_id,
                    } if window_id == state.window().id() => {
                        match event {
                            WindowEvent::CloseRequested
                            | WindowEvent::KeyboardInput {
                                input:
                                    KeyboardInput {
                                        state: ElementState::Pressed,
                                        virtual_keycode: Some(VirtualKeyCode::Escape),
                                        ..
                                    },
                                ..
                            } => *control_flow = ControlFlow::Exit,
                            WindowEvent::Resized(physical_size) => {
                                state.resize(*physical_size);
                            }
                            WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                                // new_inner_size is &mut so w have to dereference it twice
                                state.resize(**new_inner_size);
                            }
                            _ => {}
                        }
                    }
                    Event::RedrawRequested(window_id) if window_id == state.window().id() => {
                        let now = Instant::now();
                        let dt = now - last_render_time;
                        last_render_time = now;
                        state.update(dt);
                        match state.render() {
                            Ok(_) => {}
                            // Reconfigure the surface if it's lost or outdated
                            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => state.resize(state.size),
                            // The system is out of memory, we should probably quit
                            Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                            // We're ignoring timeouts
                            Err(wgpu::SurfaceError::Timeout) => log::warn!("Surface timeout"),
                        }
                    }
                    Event::MainEventsCleared => {
                        // RedrawRequested will only trigger once, unless we manually
                        // request it.
                        state.window().request_redraw();
                    }
                    _ => {}
                }
            }
        }
    });
}
