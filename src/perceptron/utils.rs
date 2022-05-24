use ndarray::Array2;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rusttype::{Font, Scale};
use imageproc::drawing;
use imageproc::pixelops::interpolate;
use image;
use image::Rgb;

type Point = (f64, f64);

pub const RED: Rgb<u8> = Rgb::<u8>([255, 0, 0]);
pub const GREEN: Rgb<u8> = Rgb::<u8>([0, 255, 0]);
pub const BLACK: Rgb<u8> = Rgb::<u8>([0, 0, 0]);
pub const WHITE: Rgb<u8> = Rgb::<u8>([255, 255, 255]);

fn point_in_triangle(pt: Point, pt1: Point, pt2: Point, pt3: Point) -> bool {
	fn sign(p_a: Point, p_b: Point, p_c: Point) -> f64 {
		(p_a.0 - p_c.0) * (p_b.1 - p_c.1) - (p_b.0 - p_c.0) * (p_a.1 - p_c.1)
	}
	
	let d1 = sign(pt, pt1, pt2);
	let d2 = sign(pt, pt2, pt3);
	let d3 = sign(pt, pt3, pt1);

	let has_neg = (d1 < 0.0) || (d2 < 0.0) || (d3 < 0.0);
	let has_pos = (d1 > 0.0) || (d2 > 0.0) || (d3 > 0.0);

	let is_in_triangle = !(has_neg && has_pos);
	return is_in_triangle;
}

pub fn make_random_rect(rng: &mut ChaCha8Rng, max_width: usize, max_height: usize) -> Array2<f64> {
	let width = rng.gen_range(1..max_width - 1);
	let height = rng.gen_range(1..max_height - 1);
	let x = rng.gen_range(0..max_width - width);
	let y = rng.gen_range(0..max_height - height);

	Array2::from_shape_fn((max_height, max_width), |(cy, cx)| {
		let in_x_axis = cx >= x && cx <= x + width;
		let in_y_axis = cy >= y && cy <= y + height;
		if in_x_axis && in_y_axis { 1.0 } else { 0.0 }
	})
}

pub fn make_random_triangle(rng: &mut ChaCha8Rng, max_width: usize, max_height: usize) -> Array2<f64> {
	let width = rng.gen_range(4..max_width);
	let height = rng.gen_range(4..max_height);
	let x = rng.gen_range(0..max_width - width);
	let y = rng.gen_range(0..max_height - height);

	let pt1 = (x as f64, (y + height) as f64);
	let pt2 = ((x + width / 2) as f64, y as f64);
	let pt3 = ((x + width) as f64, (y + height) as f64);

	Array2::from_shape_fn((max_height, max_width), |(cy, cx)| {
		if point_in_triangle((cx as f64, cy as f64), pt1, pt2, pt3) { 1.0 } else { 0.0 }
	})
}

pub fn make_random_circle(rng: &mut ChaCha8Rng, max_width: usize, max_height: usize) -> Array2<f64> {
	let max_radius = max_width;

	let r = rng.gen_range(1..max_radius);
	let x = rng.gen_range(0..max_width - 0) as f64;
	let y = rng.gen_range(0..max_height - 0) as f64;

	Array2::from_shape_fn((max_height, max_width), |(cy, cx)| {
		let cxf = cx as f64;
		let cyf = cy as f64;

		let d = ((cxf - x).powf(2.0) + (cyf - y).powf(2.0)).sqrt();

		if d < r as f64 { 1.0 } else { 0.0 }
	})
}

pub fn sigmoid(f: f64) -> f64 {
	1.0 / (1.0 + std::f64::consts::E.powf(-f))
}

pub fn find_minmax(weights: &Array2<f64>, layer_height: u32, layer_width: u32) -> (f64, f64) {
	let mut min_value = f64::MAX;
	let mut max_value = f64::MIN;

	for y in 0..layer_height {
		for x in 0..layer_width {
			let val = weights[[y as usize, x as usize]];
			if val < min_value { min_value = val; }
			if val > max_value { max_value = val; }
		}
	}

	return (min_value, max_value);
}

pub fn save_as_jpg(
	file_name: &str,
	weights: &Array2<f64>,
	layer_height: u32,
	layer_width: u32,
	scale: u32,
	color_a: Rgb<u8>,
	color_b: Rgb<u8>,
	annotate: bool,
) {
	let mut rgb_img = image::RgbImage::new(layer_width * scale, layer_height * scale);
	let minmax = find_minmax(weights, layer_height, layer_width);

	for y in 0..layer_height {
		for x in 0..layer_width {
			let weight_value = weights[[y as usize, x as usize]];
			let left_weight = (weight_value + minmax.0.abs()) / (minmax.0.abs() + minmax.1);
			let pixel = interpolate(color_a, color_b, left_weight as f32);

			if scale == 1 {
				rgb_img.put_pixel(x as u32, y as u32, pixel);
			} else {
				drawing::draw_filled_rect_mut(
					&mut rgb_img,
					imageproc::rect::Rect::at(
						(x * scale).try_into().unwrap(),
						(y * scale).try_into().unwrap()
					).of_size(scale, scale),
					pixel
				);
			}
		}
	}

	if annotate {
		let font = Vec::from(include_bytes!("res/DejaVuSans.ttf") as &[u8]);
		let font = Font::try_from_vec(font).unwrap();
	
		let font_size = 8.0;
		let text_scale = Scale {
			x: font_size * (scale as f32),
			y: font_size * (scale as f32),
		};


		let text_y = ((layer_height * scale) as f32) - (font_size * scale as f32);
		drawing::draw_text_mut(&mut rgb_img, BLACK, 0, text_y as i32, text_scale, &font, file_name);
	}

	let current_dir = std::env::current_dir().unwrap().to_str().unwrap().to_owned();
	let path = current_dir + "/training/" + file_name + ".jpg";

	rgb_img.save_with_format(path, image::ImageFormat::Jpeg).unwrap();
}
