pub const LAYER_WIDTH: usize = 64;
pub const LAYER_HEIGHT: usize = 64;
pub const BIAS: f64 = 0.0;

pub const SHAPE_A: &str = "RECT";
pub const SHAPE_B: &str = "TRIANGLE";

pub const TRAIN_SEED: u64 = 999;
pub const TRAIN_STEP: usize = 1400;
pub const TRAIN_EPOCHS: usize = 20;

pub const VALIDATE_SEED: u64 = 64;
pub const VALIDATE_STEP: usize = TRAIN_STEP * 2;

pub const IMAGE_SCALE: u32 = 32;
pub const IMAGE_EPOCH_SNAPSHOTS: bool = false;
