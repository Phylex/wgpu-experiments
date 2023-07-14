use cgmath::*;
use winit::event::*;
use winit::dpi::PhysicalPosition;
use instant::Duration;
use winit::window::WindowId;
use std::f32::consts::FRAC_PI_2;
use bytemuck;
use crate::wgpu_utils::*;

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
pub struct Camera {
    pub position: Point3<f32>,
    yaw: Rad<f32>,
    pitch: Rad<f32>,
    pub uniform: CameraUniform,
    pub view: ViewMatrix,
    pub projection: Projection,
    pub controlls: CameraControlls,
}

impl Camera {
    pub fn new<V: Into<Point3<f32>>, Y: Into<Rad<f32>>, P: Into<Rad<f32>>, F: Into<Rad<f32>>>(
        position: V,
        yaw: Y,
        pitch: P,
        screen_width: u32,
        screen_height: u32,
        znear: f32,
        zfar: f32,
        field_of_view: F,
        device: &wgpu::Device,
        speed: f32,
        sensitivity: f32,
        queue: &wgpu::Queue,
    ) -> Self {
        let mut out = Self {
            position: position.into(),
            yaw: yaw.into(),
            pitch: pitch.into(),
            projection: Projection::new(screen_width, screen_height, field_of_view, znear, zfar),
            view: ViewMatrix::new(),
            uniform: CameraUniform::new(&device),
            controlls: CameraControlls::new(speed, sensitivity),
        };
        out.update_gpu_state(&queue);
        out
    }
    /// Compute the translation and rotation needed to make it appear as if the
    /// observer is at a given location
    pub fn compute_view_space_transform_matrix(&self) -> Matrix4<f32> {
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
    
    pub fn update_gpu_state(&mut self, queue: &wgpu::Queue) {
        let view_transform = self.compute_view_space_transform_matrix();
        let projection_matrix = self.projection.compute_matrix();
        self.view.update(view_transform, projection_matrix);
        self.uniform.update_gpu_state(self.view, queue);
    }

    pub fn update(&mut self, dt: Duration, queue: &wgpu::Queue) {
        let dt = dt.as_secs_f32();

        // Process moving forward/backward/left/right/up/down
        let (yaw_sin, yaw_cos) = self.yaw.0.sin_cos();
        let pitch_sin = self.pitch.0.sin();
        let forward = Vector3::new(yaw_cos, pitch_sin, yaw_sin).normalize();
        let right = Vector3::new(-yaw_sin, 0.0, yaw_cos).normalize();
        self.position += forward * (self.controlls.amount_forward - self.controlls.amount_backward) * self.controlls.speed * dt;
        self.position += right * (self.controlls.amount_right - self.controlls.amount_left) * self.controlls.speed * dt;
        self.position.y += (self.controlls.amount_up - self.controlls.amount_down) * self.controlls.speed * dt;

        // move in and out (via the scroll wheel
        let (pitch_sin, pitch_cos) = self.pitch.0.sin_cos();
        let scrollward = Vector3::new(pitch_cos * yaw_cos, pitch_sin, pitch_cos * yaw_sin).normalize();
        self.position += scrollward * self.controlls.scroll * self.controlls.speed * self.controlls.sensitivity * dt;
        self.controlls.scroll = 0.0;

        self.yaw += Rad(self.controlls.rotate_horizontal) * self.controlls.sensitivity * dt;
        self.pitch += Rad(-self.controlls.rotate_vertical) * self.controlls.sensitivity * dt;

        self.controlls.rotate_horizontal = 0.0;
        self.controlls.rotate_vertical = 0.0;

        if self.pitch < -Rad(SAFE_FRAC_PI_2) {
            self.pitch = -Rad(SAFE_FRAC_PI_2);
        } else if self.pitch > Rad(SAFE_FRAC_PI_2) {
            self.pitch = Rad(SAFE_FRAC_PI_2);
        }
        self.update_gpu_state(queue);
    }
}

/// After the world has been transformed into camera space,
/// the coordinates of the model need to be altered in such
/// way as to make the orthographic projection (build in to the gpu
/// look like a perspective view of the world, for this a projection
/// matrix distorts the coordinates of the vertices in view space
#[derive(Debug)]
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

    pub fn compute_matrix(&self) -> Matrix4<f32> {
        OPENGL_TO_WGPU_MATRIX * perspective(self.field_of_view, self.aspect, self.znear, self.zfar)
    }
}

/// Struct that holds the computed transformation from world coordinates to
/// projected coordinates
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ViewMatrix {
    pub view_proj: [[f32; 4]; 4],
    pub view_position: [f32; 4],
}

impl ViewMatrix {
    pub fn new() -> Self {
        Self {
            view_proj: Matrix4::identity().into(),
            view_position: [0.0; 4],
        }
    }
    
    pub fn update(&mut self, view: Matrix4<f32>, projection: Matrix4<f32>) {
        self.view_proj = (projection *  view).into();
    }
}

/// This struct holds the information that is shared between GPU and CPU for the observer
/// it also contains a reference to the gpu buffer and bind group that makes the transformation
/// matrix available to the shaders.
#[derive(Debug)]
pub struct CameraUniform {
    gpu_buffer: wgpu::Buffer,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub bind_group: wgpu::BindGroup,
}

impl CameraUniform {
    pub fn new(device: &wgpu::Device) -> Self {
        let projection = ViewMatrix::new();
        let gpu_buffer = create_gpu_buffer(device, projection);
        let bind_group_layout = create_gpu_bind_group_layout(device);
        let bind_group = create_bind_group(device, &bind_group_layout, &gpu_buffer); 
        Self {
            gpu_buffer,
            bind_group_layout,
            bind_group,
        }
    }
    
    pub fn update_gpu_state(&mut self, view_transform: ViewMatrix, queue: &wgpu::Queue) {
        queue.write_buffer(&self.gpu_buffer, 0, bytemuck::cast_slice(&[view_transform]));
    }
}

/// The ObserverControlls are the user interface to an observer it allows the user to
/// move the observer around and look at different objects in the scene/world
#[derive(Debug)]
pub struct CameraControlls {
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

impl CameraControlls {
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
    pub fn proces_keyboard_input(&mut self, key: VirtualKeyCode, state: ElementState) -> bool {
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

    pub fn process_event(&mut self, ievent: &Event<()>, mouse_pressed: bool, event_window_id: WindowId) -> bool {
        match ievent {
            Event::DeviceEvent {event: DeviceEvent::MouseMotion { delta, }, .. } =>
                if mouse_pressed {
                    self.process_mouse_movement(delta.0, delta.1);
                    true
                } else {
                    false
                }
            Event::WindowEvent { window_id, event } if *window_id == event_window_id => {
                match event {
                    WindowEvent::KeyboardInput { input: KeyboardInput {state, virtual_keycode: Some(key), .. }, ..} => {
                        self.proces_keyboard_input(*key, *state)
                    }
                    WindowEvent::MouseWheel {delta, ..} => {
                        self.process_scroll(delta);
                        true
                    }
                    _ => false
                }
            }
            _ => false
        }
    }
    
}

