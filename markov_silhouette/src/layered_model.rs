use std::collections::HashMap;

use image::imageops::resize;
use image::GrayImage;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::bitmanip::BitMatrix;
use crate::pick_from_weighted_uniform;

static RESIZE_FILTER: image::imageops::FilterType = image::imageops::FilterType::Lanczos3;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ImageModel {
    kernel_halfwidth: usize,
    num_layers: usize,
    distributions: Vec<HashMap<BitMatrix, [f64; 2]>>,
}

impl ImageModel {
    pub fn from_images<I: Iterator<Item = GrayImage> + Send>(
        iter: I,
        kernel_halfwidth: usize,
        num_layers: usize,
    ) -> Self {
        assert!(kernel_halfwidth > 0);
        assert!(num_layers > 0);

        let distributions = iter
            .par_bridge()
            .map(|im| Self::analyze_image(&im, kernel_halfwidth, num_layers))
            .reduce_with(|mut a, b| {
                for (distr1, distr2) in a.iter_mut().zip(b.into_iter()) {
                    for (k, v) in distr2.into_iter() {
                        for (t1, t2) in distr1.entry(k).or_default().iter_mut().zip(v.into_iter()) {
                            *t1 += t2;
                        }
                    }
                }
                return a;
            })
            .unwrap();

        let mut this = Self {
            kernel_halfwidth,
            num_layers,
            distributions,
        };
        this.normalize();
        return this;
    }

    fn new_random_bitmat(width: usize, height: usize) -> BitMatrix {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut bitmat = BitMatrix::new(width, height);
        for x in 0..width {
            for y in 0..height {
                bitmat.set([x, y], rng.gen());
            }
        }
        return bitmat;
    }

    fn kernel_buf(kernel_halfwidth: usize, x: usize, y: usize, im: &BitMatrix) -> BitMatrix {
        let kernel_size = 2 * kernel_halfwidth + 1;
        let mut buf = BitMatrix::new(kernel_size, kernel_size);
        for xo in 0..kernel_size {
            for yo in 0..kernel_size {
                if (xo == kernel_halfwidth && xo == yo)
                    || x + xo < kernel_halfwidth
                    || y + yo < kernel_halfwidth
                    || x + xo >= im.width() + kernel_halfwidth
                    || y + yo >= im.height() + kernel_halfwidth
                {
                    buf.set([xo, yo], true);
                } else {
                    buf.set(
                        [xo, yo],
                        im.get([x + xo - kernel_halfwidth, y + yo - kernel_halfwidth]),
                    );
                }
            }
        }
        return buf;
    }

    pub fn generate_random(&self, width: usize, height: usize) -> BitMatrix {
        let s = 2_usize.pow((self.num_layers - 1) as _);
        let bitmat = Self::new_random_bitmat(width / s, height / s);
        return self.generate_raw(bitmat);
    }

    pub fn generate_from_image(&self, im: &GrayImage) -> BitMatrix {
        let s = 2_u32.pow((self.num_layers - 1) as _);
        let im = resize(im, im.width() / s, im.height() / s, RESIZE_FILTER);
        return self.generate_raw((&im).into());
    }

    fn generate_raw(&self, mut bitmat: BitMatrix) -> BitMatrix {
        for i in (0..self.num_layers - 1).rev() {
            let mut new_bitmat = BitMatrix::new(bitmat.width(), bitmat.height());

            for x in 0..bitmat.width() {
                for y in 0..bitmat.height() {
                    let buf = Self::kernel_buf(self.kernel_halfwidth, x, y, &bitmat);

                    let bit = if let Some(distr) = self.distributions[i].get(&buf) {
                        pick_from_weighted_uniform(distr.iter().cloned()) != 0
                    } else {
                        true
                    };

                    new_bitmat.set([x, y], bit);
                }
            }

            let im: GrayImage = new_bitmat.into();
            bitmat = resize(&im, im.width() * 2, im.height() * 2, RESIZE_FILTER).into();
        }

        return bitmat;
    }

    fn analyze_image(
        im: &GrayImage,
        kernel_halfwidth: usize,
        num_layers: usize,
    ) -> Vec<HashMap<BitMatrix, [f64; 2]>> {
        let mut distributions = vec![HashMap::default(); num_layers - 1];
        let scaled_images: Vec<GrayImage> = (0..num_layers)
            .map(|i| 2_u32.pow(i as u32))
            .map(|s| {
                if s == 1 {
                    im.clone()
                } else {
                    resize(im, im.width() / s, im.height() / s, RESIZE_FILTER)
                }
            })
            .collect();

        for i in 0..num_layers - 1 {
            let cur = &scaled_images[i];
            let curmat = cur.into();
            let next = &scaled_images[i + 1];
            let next = resize(next, next.width() * 2, next.height() * 2, RESIZE_FILTER);
            let nextmat = next.into();
            Self::build_predictions(&mut distributions[i], kernel_halfwidth, &nextmat, &curmat);
        }

        return distributions;
    }

    fn build_predictions(
        distribution: &mut HashMap<BitMatrix, [f64; 2]>,
        kernel_halfwidth: usize,
        source: &BitMatrix,
        target: &BitMatrix,
    ) {
        assert!(
            source.width() <= target.width(),
            "{} <= {}",
            source.width(),
            target.width()
        );
        assert!(
            source.height() <= target.height(),
            "{} <= {}",
            source.height(),
            target.height()
        );

        for x in 0..source.width() {
            for y in 0..source.height() {
                let truth = target.get([x, y]);
                let buf = Self::kernel_buf(kernel_halfwidth, x, y, &source);
                distribution.entry(buf).or_default()[truth as usize] += 1.0;
            }
        }
    }

    fn normalize(&mut self) {
        for distribution in &mut self.distributions {
            for (_, v) in distribution.iter_mut() {
                let sum = v[0] + v[1];
                v[0] /= sum;
                v[1] /= sum;
            }
        }
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//
//     #[test]
//     fn test_model_layer1_1x1() {
//     }
// }
