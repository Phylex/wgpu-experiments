use learn_wgpu::run;

fn main() {
    cfg_if::cfg_if! {
    if #[cfg(target_arch = "wasm32")] {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        console_log::init_with_level(log::level::Warn).expect("Could not initialize logger");
    } else {
        tracing_subscriber::fmt::init();
    }
    }
    println!("Hello, world!");
    pollster::block_on(run());
}
