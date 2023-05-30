use bytemuck;

// this trait is the thing that allows for the 'device.create_buffer_init' function
use wgpu::util::DeviceExt;
use wgpu;

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LightUniform {
    pub position: [f32; 3],
    _padding: u32,
    pub color: [f32; 3],
    _padding2: u32,
}

pub struct Light {
    pub uniform: LightUniform,
    buffer: wgpu::Buffer,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub bind_group: wgpu::BindGroup,
}

impl Light {
    pub fn new(device: &mut wgpu::Device) -> Self {
        let lu = LightUniform {
                position: [2.0, 2.0, 2.0],
                _padding: 0,
                color: [1., 1., 1.],
                _padding2: 0,
        };
        let light_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor{
                label: Some("light"),
                contents: bytemuck::cast_slice(&[lu]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            }
        );
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None, 
            entries: &[wgpu::BindGroupLayoutEntry{
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None
                },
                count: None,
            }],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: light_buffer.as_entire_binding(),
            }],
        });
        Light { uniform: lu, buffer: light_buffer, bind_group_layout, bind_group }
    }
    pub fn update(&mut self, position: Option<[f32; 3]>, color: Option<[f32; 3]>, dev_queue: &wgpu::Queue) {
        let mut write_buffer = false;
        match position {
            Some(pos) => {
                self.uniform.position = pos;
                write_buffer = true;
            },
            None => {}
        };
        match color {
            Some(col) => {
                self.uniform.color = col;
                write_buffer = true;
            },
            None => {}
        };
        dev_queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(&[self.uniform]));
    }
}
