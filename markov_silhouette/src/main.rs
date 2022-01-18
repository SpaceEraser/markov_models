#![feature(type_ascription)]

use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::PathBuf;

use image::GrayImage;
use rand::Rng;

mod bitmanip;
mod layered_model;
// mod naive_model;

fn pick_from_weighted_uniform<I: Iterator<Item = f64>>(mut iter: I) -> usize {
    let p: f64 = rand::thread_rng().gen();
    let mut i = 0;
    let mut s = 0.0;
    while let Some(f) = iter.next() {
        s += f;
        if p < s {
            return i;
        }
        i += 1;
    }
    return i - 1;
}

fn get_images() -> impl Iterator<Item = GrayImage> {
    "./silhouettes"
        .parse::<PathBuf>()
        .unwrap()
        .read_dir()
        .expect("Failed to read images")
        .filter(|r| r.is_ok())
        .map(|r| r.unwrap().path())
        .inspect(|p| println!("reading {}", p.to_string_lossy()))
        .map(|p| image::open(p))
        .filter(|r| r.is_ok())
        .map(|r| r.unwrap())
        .map(|im| im.to_luma8())
}

fn image_transpose(im: &GrayImage) -> GrayImage {
    let (width, height) = im.dimensions();
    let mut out = GrayImage::new(height, width);
    for x in 0..width {
        for y in 0..height {
            let p = im.get_pixel(x, y);
            out.put_pixel(y, x, *p);
        }
    }
    return out;
}

fn image_off_transpose(im: &GrayImage) -> GrayImage {
    let (width, height) = im.dimensions();
    let mut out = GrayImage::new(height, width);
    for x in 0..width {
        for y in 0..height {
            let p = im.get_pixel(x, y);
            out.put_pixel(height - y - 1, width - x - 1, *p);
        }
    }
    return out;
}

fn augment_images(iter: impl Iterator<Item = GrayImage>) -> impl Iterator<Item = GrayImage> {
    use image::imageops::{flip_horizontal, flip_vertical, rotate180, rotate270, rotate90};
    iter.flat_map(|im| {
        [
            flip_vertical(&im),
            flip_horizontal(&im),
            rotate90(&im),
            rotate180(&im),
            rotate270(&im),
            image_transpose(&im),
            image_off_transpose(&im),
            im,
        ]
    })
}

fn main() {
    const MODEL_FILENAME: &str = "model.bin";
    let model = if let Ok(file) = File::open(MODEL_FILENAME) {
        println!("Reading model from file");
        bincode::deserialize_from(BufReader::new(file)).unwrap()
    } else {
        let images = augment_images(get_images());

        let start = std::time::Instant::now();
        let model = layered_model::ImageModel::from_images(images, 3, 7);
        bincode::serialize_into(
            BufWriter::new(File::create(MODEL_FILENAME).unwrap()),
            &model,
        )
        .unwrap();
        println!(
            "Generated and saved model in {:?}",
            std::time::Instant::now() - start
        );
        model
    };

    // println!("{:?}", model);
    // let out_image: GrayImage = model.generate_random(512, 512).into();
    // let seed_image = get_images().next().unwrap();
    for i in 0..8 {
        let mut out_image = loop {
            // let im: GrayImage = model.generate_from_image(&seed_image).into();
            let im: GrayImage = model.generate_random(500, 500).into();
            if im.iter().any(|&p| p != 255) {
                break im;
            }
        };
        out_image.save(format!("test_{}.png", i)).expect("Failed to save image");
    }
}
