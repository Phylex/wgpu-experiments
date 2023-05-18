use cgmath::*;
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
            VirtualKeyCode::W | VirtualKeyCode::Up => {
                self.amount_forward = amount;
                true
            }
            VirtualKeyCode::S | VirtualKeyCode::Down => {
                self.amount_backward = amount;
                true
            }
            VirtualKeyCode::A | VirtualKeyCode::Left => {
                self.amount_left = amount;
                true
            }
            VirtualKeyCode::D | VirtualKeyCode::Right => {
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
        let forward = Vector3::new(yaw_cos, 0.0, yaw_sin).normalize();
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

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ObserverUniform {
    pub view_proj: [[f32; 4]; 4],
    pub view_position: [f32; 4],
}

impl ObserverUniform {
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
