use std::ops::AddAssign;
use std::ops::SubAssign;
use ndarray::Array2;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

mod perceptron;

fn predict(weights: &Array2<f64>, inputs: &Array2<f64>) -> bool {
	perceptron::utils::sigmoid((weights * inputs).sum()) > perceptron::config::BIAS
	// (weights * inputs).sum() > BIAS
}

fn make_shape(shape: &str, rng: &mut ChaCha8Rng) -> Array2<f64> {
	match shape {
		"RECT" => perceptron::utils::make_random_rect(rng, perceptron::config::LAYER_WIDTH, perceptron::config::LAYER_HEIGHT),
		"CIRCLE" => perceptron::utils::make_random_circle(rng, perceptron::config::LAYER_WIDTH, perceptron::config::LAYER_HEIGHT),
		"TRIANGLE" => perceptron::utils::make_random_triangle(rng, perceptron::config::LAYER_WIDTH, perceptron::config::LAYER_HEIGHT),
		_ => panic!("invalid shape"),
	}
}

fn train(weights: &mut Array2<f64>, epochs: usize) {
	let mut new_weights = weights.clone();

	for epoch in 0..epochs {
		let mut rng = ChaCha8Rng::seed_from_u64(perceptron::config::TRAIN_SEED);
		let mut errors = 0;

		for _ in 0..perceptron::config::TRAIN_STEP {
			let shape_a = make_shape(perceptron::config::SHAPE_A, &mut rng);

			if predict(&new_weights, &shape_a) == true {
				new_weights.sub_assign(&shape_a);
				errors += 1;
			}

			let shape_b = make_shape(perceptron::config::SHAPE_B, &mut rng);

			if predict(&new_weights, &shape_b) == false {
				new_weights.add_assign(&shape_b);
				errors += 1;
			}
		}

		println!(
			"Training Epoch #{} errors {} / {} (accuracy={:.2}%)",
			epoch + 1,
			errors,
			perceptron::config::TRAIN_STEP * 2,
			100.0 - ((errors as f64) / ((perceptron::config::TRAIN_STEP * 2) as f64) * 100.0)
		);

		let snapshot_epoch = if epoch < 9999 {
			[1, 10, 100, 1000][(epoch as f64).log10().floor() as usize]
		} else {
			1000
		};

		if perceptron::config::IMAGE_EPOCH_SNAPSHOTS && epoch % snapshot_epoch == 0 {
			let file_name = format!("epoch_{}", epoch.to_string());

			perceptron::utils::save_as_jpg(
				&file_name,
				&new_weights,
				perceptron::config::LAYER_HEIGHT as u32,
				perceptron::config::LAYER_WIDTH as u32,
				perceptron::config::IMAGE_SCALE,
				perceptron::utils::RED,
				perceptron::utils::GREEN,
				true,
			);

			println!("Snapshot {} saved", file_name);
		}
	}

	weights.clone_from(&new_weights);
}

fn validate(weights: &mut Array2<f64>) {
	let mut rng = ChaCha8Rng::seed_from_u64(perceptron::config::VALIDATE_SEED);
	let mut errors = 0;

	for _ in 0..perceptron::config::VALIDATE_STEP {
		let shape_a = make_shape(perceptron::config::SHAPE_A, &mut rng);

		if predict(&weights, &shape_a) == true {
			errors += 1;
		}

		let shape_b = make_shape(perceptron::config::SHAPE_B, &mut rng);

		if predict(&weights, &shape_b) == false {
			errors += 1;
		}
	}

	println!(
		"Validation errors {} / {} (accuracy={:.2}%)",
		errors,
		perceptron::config::VALIDATE_STEP * 2,
		100.0 - ((errors as f64) / ((perceptron::config::VALIDATE_STEP * 2) as f64) * 100.0)
	);
}

fn create_sample_images(shape: &str, seed: u64, count: usize) {
	let mut rng = ChaCha8Rng::seed_from_u64(seed);

	for i in 1..(count + 1) {
		let file_name = format!("{}_{}", shape, i.to_string());

		let shape = make_shape(shape, &mut rng);
		perceptron::utils::save_as_jpg(
			&file_name,
			&shape,
			perceptron::config::LAYER_HEIGHT as u32,
			perceptron::config::LAYER_WIDTH as u32,
			perceptron::config::IMAGE_SCALE,
			perceptron::utils::WHITE,
			perceptron::utils::BLACK,
			false,
		);
	}
}

fn main() {
	// create_sample_images("RECT", perceptron::config::TRAIN_SEED, 5);
	// create_sample_images("TRIANGLE", perceptron::config::TRAIN_SEED, 5);

	let mut weights = Array2::from_elem((
		perceptron::config::LAYER_HEIGHT,
		perceptron::config::LAYER_WIDTH
	), 0.0);

	validate(&mut weights);
	train(&mut weights, perceptron::config::TRAIN_EPOCHS);
	validate(&mut weights);

	if perceptron::config::IMAGE_EPOCH_SNAPSHOTS {
		perceptron::utils::save_as_jpg(
			"final",
			&weights,
			perceptron::config::LAYER_HEIGHT as u32,
			perceptron::config::LAYER_WIDTH as u32,
			perceptron::config::IMAGE_SCALE,
			perceptron::utils::RED,
			perceptron::utils::GREEN,
			true
		);
	}
}
