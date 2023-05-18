use bytemuck;
pub struct Observer {
    pub location: cgmath::Point3<f32>,
    pub target: cgmath::Point3<f32>,
    pub up: cgmath::Vector3<f32>,
    pub aspect: f32,
    pub field_of_view: f32,
    pub znear: f32,
    pub zfar: f32
}

#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
    1.0, 0.0, 0.0, 0.0, 
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0
);

impl Observer {
    fn compute_view_projection_matrix(&self) -> cgmath::Matrix4<f32> {
        let view = cgmath::Matrix4::look_at_rh(self.location, self.target, self.up);
        let projection = cgmath::perspective(cgmath::Deg(self.field_of_view), self.aspect, self.znear, self.zfar);
        OPENGL_TO_WGPU_MATRIX * projection * view
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ObserverUniform {
    pub view_proj: [[f32; 4]; 4],
}

impl ObserverUniform {
    pub fn new() -> Self {
        use cgmath::SquareMatrix;
        Self { view_proj: cgmath::Matrix4::identity().into(),
        }
    }
    
    pub fn update_projection(&mut self, observer: &Observer) {
        self.view_proj = observer.compute_view_projection_matrix().into();
    }
}
