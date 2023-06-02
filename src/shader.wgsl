// Vertex shader
struct Observer {
    view_proj: mat4x4<f32>,
    position: vec4<f32>,
};
 
@group(1) @binding(0)
var<uniform> observer: Observer;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
    @location(2) normal: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
    // the normal direction in the world reference frame
    @location(1) world_normal: vec3<f32>,
    // the location of the vertex in the world reference frame
    @location(2) position: vec3<f32>,
};

struct InstanceInput {
    @location(5) transform_matrix_0: vec4<f32>,
    @location(6) transform_matrix_1: vec4<f32>,
    @location(7) transform_matrix_2: vec4<f32>,
    @location(8) transform_matrix_3: vec4<f32>,
    @location(9) scale: vec4<f32>,
};

struct Light {
    position: vec3<f32>,
    color: vec3<f32>,
}


@vertex
fn vs_main(
    model: VertexInput,
    instance: InstanceInput,
) -> VertexOutput {
    let instance_transform = mat4x4<f32>(
        instance.transform_matrix_0,
        instance.transform_matrix_1,
        instance.transform_matrix_2,
        instance.transform_matrix_3,
    );
    let inverse_scale_matrix = mat4x4<f32>(
        vec4<f32>(1.0/instance.scale.x, 0.0, 0.0, 0.0),
        vec4<f32>(0.0, 1.0/instance.scale.y, 0.0, 0.0),
        vec4<f32>(0.0, 0.0, 1.0/instance.scale.z, 0.0),
        vec4<f32>(0.0, 0.0, 0.0, 1.0),
    );
    var out: VertexOutput;
    out.tex_coords = model.tex_coords;
    out.world_normal = (inverse_scale_matrix * instance_transform * vec4<f32>(model.normal, 0.0)).xyz;
    var world_position: vec4<f32> = instance_transform * vec4<f32>(model.position, 1.0);
    out.position = world_position.xyz;
    out.clip_position = observer.view_proj * world_position;
    return out;
}

// Fragment shader

@group(0) @binding(0)
var t_diffuse: texture_2d<f32>;
@group(0)@binding(1)
var s_diffuse: sampler;
@group(2) @binding(0)
var<uniform> light: Light;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let object_color: vec4<f32> = textureSample(t_diffuse, s_diffuse, in.tex_coords);
    
    let light_dir = normalize(light.position - in.position);
    let light_distance = length(light.position - in.position);
    let distance_factor = (1.0/(light_distance*light_distance));
    
    let diffuse_strength = 3.0 * max(dot(in.world_normal, light_dir), 0.0) * distance_factor;
    let diffuse_color = light.color * diffuse_strength;

    let view_dir = normalize(observer.position.xyz - in.position);
    let reflect = reflect(-light_dir, in.world_normal);
    let specular_strenght = pow(max(dot(view_dir, reflect), 0.0), 32.0) * distance_factor;
    let specular_color = specular_strenght * light.color;
    
    let ambient_strength = 0.001;
    let ambient_color = light.color * ambient_strength;

    let result = (specular_color + ambient_color + diffuse_color) * object_color.xyz;
    return vec4<f32>(result, object_color.a);
}
