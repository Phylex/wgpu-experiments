use cgmath::*;
use wgpu::util::DeviceExt;
use winit::event::*;
use winit::dpi::PhysicalPosition;
use instant::Duration;
use std::f32::consts::FRAC_PI_2;
use bytemuck;

const SAFE_FRAC_PI_2: f32 = FRAC_PI_2 - 0.0001;

#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
    1.0, 0.0, 0.0, 0.0, 
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0
);

/// This is the location and viw-direction of the observer It needs to change,
/// when the observer is moved or rotated
#[derive(Debug)]
pub struct Observer {
    pub position: Point3<f32>,
    yaw: Rad<f32>,
    pitch: Rad<f32>,
}

impl Observer {
    pub fn new<V: Into<Point3<f32>>, Y: Into<Rad<f32>>, P: Into<Rad<f32>>>(
        position: V,
        yaw: Y,
        pitch: P,
    ) -> Self {
        Self {
            position: position.into(),
            yaw: yaw.into(),
            pitch: pitch.into(),
        }
    }
    /// Compute the translation and rotation needed to make it appear as if the
    /// observer is at a given location
    pub fn compute_projection_matrix(&self) -> Matrix4<f32> {
        let (sin_pitch, cos_pitch) = self.pitch.0.sin_cos();
        let (sin_yaw, cos_yaw) = self.yaw.0.sin_cos();
        Matrix4::look_to_rh(
            self.position, 
            Vector3::new(
                cos_pitch * cos_yaw,
                sin_pitch,
                cos_pitch * sin_yaw,
            ).normalize(),
            Vector3::unit_y(),
        )
    }
}

/// The Projection describes more of the rendering aspects for what is displayed on screen
/// it determins the clipping of objects and the field of view for of what is shown on screen
pub struct Projection {
    aspect: f32,
    field_of_view: Rad<f32>,
    znear: f32,
    zfar: f32,
}

impl Projection {
    pub fn new<F: Into<Rad<f32>>>(
        width: u32,
        height: u32,
        field_of_view: F,
        znear: f32,
        zfar: f32,
    ) -> Self {
        Self {
            aspect: width as f32 / height as f32,
            field_of_view: field_of_view.into(),
            znear,
            zfar,
        }
    }
    pub fn resize(&mut self, width: u32, height: u32) {
        self.aspect = width as f32 / height as f32;
    }

    pub fn calc_matrix(&self) -> Matrix4<f32> {
        OPENGL_TO_WGPU_MATRIX * perspective(self.field_of_view, self.aspect, self.znear, self.zfar)
    }
}

/// The uniform struct holds the CPU representation of the gpu buffer that stores the transformation matrix
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ObserverViewMatrix {
    pub view_proj: [[f32; 4]; 4],
    pub view_position: [f32; 4],
}

impl ObserverViewMatrix {
    pub fn new() -> Self {
        Self {
            view_proj: Matrix4::identity().into(),
            view_position: [0.0; 4],
        }
    }
    
    pub fn update_projection(&mut self, observer: &Observer, projection: &Projection) {
        self.view_position = observer.position.to_homogeneous().into();
        self.view_proj = (projection.calc_matrix() *  observer.compute_projection_matrix()).into()
    }
}

/// This struct holds the information that is shared between GPU and CPU for the observer
/// it also contains a reference to the gpu buffer and bind group that makes the transformation
/// matrix available to the shaders.
pub struct ObserverUniform {
    view_transformation_matrix: ObserverViewMatrix,
    gpu_buffer: wgpu::Buffer,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub bind_group: wgpu::BindGroup,
}

impl ObserverUniform {
    pub fn new(device: &wgpu::Device) -> Self {
        let projection = ObserverViewMatrix::new();
        let gpu_buffer = create_gpu_buffer(device, projection);
        let bind_group_layout = create_gpu_bind_group_layout(device);
        let bind_group = create_bind_group(device, &bind_group_layout, &gpu_buffer); 
        Self {
            view_transformation_matrix: projection,
            gpu_buffer,
            bind_group_layout,
            bind_group,
        }
    }
    
    pub fn update(&mut self, observer: &Observer, projection: &Projection, queue: &wgpu::Queue) {
        self.view_transformation_matrix.update_projection(observer, projection);
        queue.write_buffer(&self.gpu_buffer, 0, bytemuck::cast_slice(&[self.view_transformation_matrix]));
    }
}

/// Helper function for creating the buffers
fn create_gpu_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { 
        label: Some("observer bind group layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }]
    })
}

fn create_bind_group(device: &wgpu::Device, layout: &wgpu::BindGroupLayout, proj_buffer: &wgpu::Buffer) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &layout,
        label: Some("Observer bind group"),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: proj_buffer.as_entire_binding()
            }
        ],
    })
}

fn create_gpu_buffer(device: &wgpu::Device, projection: ObserverViewMatrix) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Observer projection uniform buffer"),
        contents: bytemuck::cast_slice(&[projection]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    })
}

/// The ObserverControlls are the user interface to an observer it allows the user to
/// move the observer around and look at different objects in the scene/world
#[derive(Debug)]
pub struct ObserverControlls {
    amount_left: f32,
    amount_right: f32,
    amount_forward: f32,
    amount_backward: f32,
    amount_up: f32,
    amount_down: f32,
    rotate_horizontal: f32,
    rotate_vertical: f32,
    scroll: f32,
    speed: f32,
    sensitivity: f32,
}

impl ObserverControlls {
    pub fn new(speed: f32, sensitivity: f32) -> Self {
        Self {
            amount_left: 0.0,
            amount_right: 0.0,
            amount_forward: 0.0,
            amount_backward: 0.0,
            amount_up: 0.0,
            amount_down: 0.0,
            rotate_horizontal: 0.0,
            rotate_vertical: 0.0,
            scroll: 0.0,
            speed,
            sensitivity
        }
    }
    pub fn process_keyboard_input(&mut self, key: VirtualKeyCode, state: ElementState) -> bool {
        let amount: f32 = if state == ElementState::Pressed {1.0} else {0.0};
        match key {
            VirtualKeyCode::R | VirtualKeyCode::Up => {
                self.amount_forward = amount;
                true
            }
            VirtualKeyCode::H | VirtualKeyCode::Down => {
                self.amount_backward = amount;
                true
            }
            VirtualKeyCode::S | VirtualKeyCode::Left => {
                self.amount_left = amount;
                true
            }
            VirtualKeyCode::T | VirtualKeyCode::Right => {
                self.amount_right = amount;
                true
            }
            VirtualKeyCode::Space => {
                self.amount_up = amount;
                true
            }
            VirtualKeyCode::LShift => {
                self.amount_down = amount;
                true 
            }
            _ => false,
        }
    }

    pub fn process_mouse_movement(&mut self, mouse_dx: f64, mouse_dy: f64) {
        self.rotate_horizontal = mouse_dx as f32;
        self.rotate_vertical = mouse_dy as f32;
    }
    
    pub fn process_scroll(&mut self, delta: &MouseScrollDelta) {
        self.scroll = -match delta {
            MouseScrollDelta::LineDelta(_, scroll) => scroll * 100.0,
            MouseScrollDelta::PixelDelta(PhysicalPosition {
                y: scroll,
                ..
            }) => *scroll as f32,
        };
    }
    
    pub fn update_observer(&mut self, observer: &mut Observer, dt: Duration) {
        let dt = dt.as_secs_f32();

        // Process moving forward/backward/left/right/up/down
        let (yaw_sin, yaw_cos) = observer.yaw.0.sin_cos();
        let pitch_sin = observer.pitch.0.sin();
        let forward = Vector3::new(yaw_cos, pitch_sin, yaw_sin).normalize();
        let right = Vector3::new(-yaw_sin, 0.0, yaw_cos).normalize();
        observer.position += forward * (self.amount_forward - self.amount_backward) * self.speed * dt;
        observer.position += right * (self.amount_right - self.amount_left) * self.speed * dt;
        observer.position.y += (self.amount_up - self.amount_down) * self.speed * dt;

        // move in and out (via the scroll wheel
        let (pitch_sin, pitch_cos) = observer.pitch.0.sin_cos();
        let scrollward = Vector3::new(pitch_cos * yaw_cos, pitch_sin, pitch_cos * yaw_sin).normalize();
        observer.position += scrollward * self.scroll * self.speed * self.sensitivity * dt;
        self.scroll = 0.0;

        observer.yaw += Rad(self.rotate_horizontal) * self.sensitivity * dt;
        observer.pitch += Rad(-self.rotate_vertical) * self.sensitivity * dt;

        self.rotate_horizontal = 0.0;
        self.rotate_vertical = 0.0;

        if observer.pitch < -Rad(SAFE_FRAC_PI_2) {
            observer.pitch = -Rad(SAFE_FRAC_PI_2);
        } else if observer.pitch > Rad(SAFE_FRAC_PI_2) {
            observer.pitch = Rad(SAFE_FRAC_PI_2);
        }
    }
}

